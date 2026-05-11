import time
import os
import docx
import re
from elasticsearch import Elasticsearch, exceptions
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from embed import (
    MedicineInfoStandardizer,
    classify_pharmacy_query,
    connect_elasticsearch,
    extract_drug_info,
    extract_subsections,
    process_and_vectorize,
    verify_data_in_elasticsearch,
    retrieve_vector_and_text,
    retrieve_vector_and_text_for_drug,
    retrieve_drug_subsections,
    retrieve_with_context,
    get_openai_client,
    clear_openai_client_cache,
    quick_intent_hint,
    _score_result_by_drug_name,
)
import os

from config import config

history = []  # 问答记忆列表

# Elasticsearch 客户端懒加载缓存：避免在模块导入阶段就连接 ES，
# 否则 ES 未启动时会静默返回 None，后续请求才在运行时报错。
_es_client = None

# -----------------------------------------------------------------------------
# P4.3 会话级结构化事实缓存（Gradio 每次页面刷新会重置进程，
# 模块级变量即会话级隔离）
# -----------------------------------------------------------------------------
_session_facts = {
    "primary_drug": None,      # 当前会话主要讨论的药品
    "queried_fields": set(),   # 已查询过的字段集合
    "last_intent": None,       # 上一轮意图：pharmacy / chitchat / compare
    "drug_history": [],        # 本轮会话提到过的所有药品（按时间顺序，去重）
}


def _update_session_facts(target_drug, target_fields, intent_hint):
    """每轮问答结束后更新会话事实缓存。"""
    global _session_facts
    if target_drug:
        _session_facts["primary_drug"] = target_drug
        if target_drug not in _session_facts["drug_history"]:
            _session_facts["drug_history"].append(target_drug)
    if target_fields:
        _session_facts["queried_fields"].update(target_fields)
    if intent_hint:
        _session_facts["last_intent"] = intent_hint


def get_session_drug():
    """获取当前会话的主要药品，供指代消解和检索层使用。"""
    return _session_facts.get("primary_drug")


def get_session_fields():
    """获取已查询过的字段集合。"""
    return set(_session_facts.get("queried_fields", set()))


def clear_session_facts():
    """重置会话事实（如用户点击'清空对话'时调用）。"""
    global _session_facts
    _session_facts = {
        "primary_drug": None,
        "queried_fields": set(),
        "last_intent": None,
        "drug_history": [],
    }


def get_es_client():
    """
    懒加载 Elasticsearch 客户端：首次调用时连接并缓存，后续调用直接复用。
    若连接失败返回 None 并打印错误，调用方需自行处理。
    """
    global _es_client
    if _es_client is not None:
        return _es_client
    _es_client = connect_elasticsearch()
    if _es_client is None:
        print("[get_es_client] Elasticsearch 未就绪，请先启动 ES（start_es.bat）后再使用上传/问答功能")
    return _es_client


def clear_es_cache():
    """清除已缓存的 ES 客户端（如配置变更后调用，下次访问会按新配置重连）。"""
    global _es_client
    _es_client = None


def _extract_drug_name_from_query(input_data, standardizer=None):
    """从用户查询中提取药品名。
    复用 MedicineInfoStandardizer.standardize_information + extract_drug_info 解析。
    :param input_data: 用户输入的问题字符串。
    :param standardizer: 可选的 MedicineInfoStandardizer 实例；若传入则复用，避免重复创建。
    :return: 药品名字符串，解析失败或不存在时返回 None。
    """
    if not input_data or not input_data.strip():
        return None
    try:
        if standardizer is None:
            standardizer = MedicineInfoStandardizer(llm=get_openai_client())
        raw = standardizer.standardize_information(input_data)
    except Exception as e:
        print(f"[extract_drug_name] standardize_information 调用失败: {e}")
        return None
    try:
        drugs, _ = extract_drug_info(raw)
    except Exception as e:
        print(f"[extract_drug_name] 解析 LLM 输出失败: {e}")
        return None
    if drugs:
        return drugs[0].strip()
    return None


def _confirm_drug_in_es(drug_name):
    """用 ES 精确查询确认药品名是否存在。
    ES 的 doc_id 就是药品名（章节标题），所以直接用 .get() 查 _id。
    :param drug_name: 要确认的药品名。
    :return: bool，存在返回 True，否则 False。
    """
    if not drug_name:
        return False
    es_instance = get_es_client()
    if es_instance is None:
        return False
    try:
        res = es_instance.get(index=config.ES_INDEX, id=drug_name, ignore=[404])
        return res.get('found', False)
    except Exception as e:
        print(f"[es-confirm] 查询药品 '{drug_name}' 失败: {e}")
        return False


# 指代词集合：用于判断当前问题是否需要结合历史对话做改写
_PRONOUNS = {
    '他', '她', '它', '这个药', '该药', '此药', '刚才', '上面', '之前',
    '之前那个', '刚才那个', '这个', '那个', '其'
}


def _format_history(history, max_turns=3):
    """
    把 Gradio 的 history 列表格式化为文本。
    history 格式：[(user_msg, assistant_msg), ...]
    只取最近 max_turns 轮，避免 prompt 过长。
    """
    if not history:
        return ""
    lines = []
    for item in history[-max_turns:]:
        try:
            user_msg, bot_msg = item
            lines.append(f"User: {user_msg}")
            # 截断助手回复避免 prompt 过长
            bot_snip = bot_msg[:200] if bot_msg else ""
            lines.append(f"Assistant: {bot_snip}...")
        except Exception:
            continue
    return "\n".join(lines)


def _rewrite_query_with_history(current_message, history, llm_client):
    """
    结合对话历史，把包含指代词的当前问题改写成独立、完整的问题。

    规则：
    1. 如果 current_message 中已包含药品名（_extract_drug_name_from_query 能提取到），
       说明问题已自包含，无需改写，直接返回原消息。
    2. 如果 current_message 中不含任何指代词，直接返回原消息。
    3. 否则，调用 LLM 结合 history 做改写。

    返回：改写后的字符串（或原字符串）。
    """
    if not current_message or not current_message.strip():
        return current_message

    # 规则 1：已含药品名 → 自包含，跳过
    try:
        existing_drug = _extract_drug_name_from_query(current_message)
        if existing_drug:
            print(f"[query-rewrite] 消息已含药品名 '{existing_drug}'，跳过改写")
            return current_message
    except Exception:
        pass

    # 规则 2：不含指代词 → 跳过
    text = current_message.strip()
    has_pronoun = any(p in text for p in _PRONOUNS)
    if not has_pronoun:
        return current_message

    # 规则 3：调用 LLM 改写
    if not history:
        return current_message

    formatted_history = _format_history(history, max_turns=3)
    rewrite_prompt = f"""你是一个对话理解助手。请根据以下对话历史，把用户的当前问题改写成独立、完整的问题（消除指代词）。

要求：
- 如果当前问题包含"他/她/它/这个药/该药/此药/刚才/上面/之前"等指代词，请根据历史对话确定指代对象，并替换为具体名称。
- 改写后的问题必须是一个完整的、不依赖上下文也能理解的问题。
- 只输出改写后的问题，不要解释，不要加引号。
- 如果当前问题已经完整（不含指代词），请原样输出。

对话历史：
{formatted_history}

当前问题：{current_message}

改写后的问题："""

    try:
        if llm_client is None:
            llm_client = get_openai_client()
        response = llm_client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.0,
        )
        rewritten = response.choices[0].message.content.strip() if response.choices else ""
        if rewritten and rewritten != current_message:
            print(f"[query-rewrite] '{current_message}' → '{rewritten}'")
            return rewritten
    except Exception as e:
        print(f"[query-rewrite] LLM 改写失败，使用原消息: {e}")

    return current_message

# ... existing imports ...

# web中配置tab页的更新函数
def update_config(es_host, es_port, es_user, es_pass, es_index, vector_db, es_scheme):# 配置页面的更新函数, 用于更新配置信息应用到全局
    """
    更新配置信息
    """
    # Simply update the in-memory config object for this session
    config.ES_HOST = es_host
    config.ES_PORT = int(es_port)
    config.ES_USER = es_user
    config.ES_PASSWORD = es_pass
    config.ES_INDEX = es_index
    config.VECTOR_DB_PATH = vector_db
    config.ES_SCHEME = es_scheme
    # 配置变更后清除缓存：ES 与 OpenAI 客户端下次访问按新配置重连/重建
    clear_es_cache()
    clear_openai_client_cache()
    return "配置已更新! (注意: 重启后将重置为配置文件默认值)"

class UploadDoc:# 上传文档类
    
    def __init__(self, file_input):#初始化类的实例
        self.file_input = file_input  # Path to the uploaded file
        # Use config values
        self.es_host = config.ES_HOST
        self.es_port = config.ES_PORT
        self.es_user = config.ES_USER
        self.es_pass = config.ES_PASSWORD
        self.es_index = config.ES_INDEX
        self.es_scheme = config.ES_SCHEME


    def clean_filename(self,filename):#从文件名中提取中文字符并去除末尾的空格
        return ''.join(re.findall(r'[\u4e00-\u9fff]+', filename)).rstrip()

    # ... (methods extract_titles_and_content, connect_elasticsearch, store_in_elasticsearch remain similar but we can clean them up if needed, but for now we focus on config)
    
    def extract_titles_and_content(self, doc_obj):#从 Word 文档对象中提取标题和内容，并将其存储在一个字典中。
        content_dict = {}
        temp_doc = []

        for paragraph in doc_obj.paragraphs:
            if not paragraph.runs:
                continue
                
            font_size = paragraph.runs[0].font.size
            if font_size is not None:
                # print(f"Font size: {font_size.pt}, Type: {type(font_size.pt)}")
                if isinstance(font_size.pt, (int, float)):
                    if font_size.pt == 12:
                        if temp_doc:
                            title = self.clean_filename(temp_doc[0])
                            if title:
                                content_dict[title] = temp_doc
                            temp_doc = []
            temp_doc.append(paragraph.text)

        if temp_doc:
            title = self.clean_filename(temp_doc[0])
            if title:
                content_dict[title] = temp_doc

        return content_dict

    def store_in_elasticsearch(self, content_dict):#将内容存入es
        # print(f"Content dict to store: {content_dict}")
        # 直接走模块级的懒加载客户端，避免在类内再包一层一行 wrapper（同时屏蔽与 embed.connect_elasticsearch 的同名歧义）
        es_instance = get_es_client()
        if es_instance is None:
            print("无法存储：Elasticsearch 未就绪，请先启动 ES 服务")
            return
        for title, content in content_dict.items():
            try:
                es_instance.index(index=self.es_index, id=title, body={'content': '\n'.join(content)})
                print(f"已存储: {title}到{self.es_index}")
            except exceptions.ConnectionError as e:
                print(f"连接错误：{e}")
            except exceptions.TransportError as e:
                print(f"存储错误：{e}")

    def split_and_index_doc(self):#将文档分割成篇章，然后调用存储到es函数存入
        if not os.path.exists(self.file_input):
            print(f"文件 {self.file_input} 不存在。")
            return

        try:
            doc_obj = docx.Document(self.file_input)
            content_dict = self.extract_titles_and_content(doc_obj)
            self.store_in_elasticsearch(content_dict)
            print(f"已将 {len(content_dict)} 篇章存入 Elasticsearch")
        except Exception as e:
            print(f"处理文件时出错：{e}")

    def upload_doc(self, index_name, vector_db_path): #提交
        self.es_index = index_name
        self.split_and_index_doc()
        # vector_db_path = f"{vector_db_path}/{index_name}.npz" # This logic seems weird in original code, it appended filename to path?
        # If vector_db_path is a directory, append filename. If it's a file, use it?
        # Original: vector_db_path = f"{vector_db_path}/{index_name}.npz"
        # Let's assume input is directory if it has no extension, or we stick to original logic but make it robust
        if not vector_db_path.endswith('.npz'):
             vector_db_path = os.path.join(vector_db_path, f"{index_name}.npz")

        # 用户重新上传文档时，ES 索引刚刚被刷新，必须强制重建 FAISS 以保持同步
        process_and_vectorize(index_name, vector_db_path, force_rebuild=True)

def import_new_documents(uploaded_file, index_name, vector_db_path_input):# 上传文档
    if uploaded_file is not None: 
        file_input = uploaded_file.name  # 获取上传文件的路径
        
        # Build the actual npz path
        if not vector_db_path_input.endswith('.npz'):
            actual_npz = os.path.join(vector_db_path_input, f"{index_name}.npz")
        else:
            actual_npz = vector_db_path_input
        
        config.VECTOR_DB_PATH = actual_npz
        config.ES_INDEX = index_name
        
        uploader = UploadDoc(file_input=file_input)
        uploader.upload_doc(index_name, vector_db_path_input)
        return f"文档上传成功，向量库路径: {actual_npz}"  
    else:
        return "没有上传文件"  

def _score_result_by_fields(title, target_fields):
    """检索结果与 target_fields 的匹配度评分。
    简单实现：title 与某个目标字段精确相等或互相包含 → 命中（1），否则 0。
    返回 int，便于稳定 sort。
    边界：title 为 None / '' / 纯空白时不算命中（避免空串被任何 field 包含的假阳性）。"""
    if not target_fields:
        return 0
    t = (title or '').strip()
    if not t:
        return 0
    for f in target_fields:
        if not f:
            continue
        if f == t or f in t or t in f:
            return 1
    return 0


def slow_echo(message, history, enable_thinking=True):
    """
    问答主流程，在 P1 基础上增加可选的"意图路由层"。
    """
    print(f"\n{'='*60}")
    print(f"[slow_echo] ===== 新问答请求 =====")
    print(f"[slow_echo] 用户原始消息: {message}")
    print(f"[slow_echo] history 长度: {len(history) if history else 0}")

    # ============ 0) 查询改写：消除指代词 ============
    original_message = message
    if history and getattr(config, 'ENABLE_INTENT_ROUTING', False):
        rewritten = _rewrite_query_with_history(message, history, get_openai_client())
        if rewritten and rewritten != message:
            print(f"[query-rewrite] '{message}' → '{rewritten}'")
            message = rewritten
        else:
            print(f"[query-rewrite] 无需改写: '{message}'")
    else:
        print(f"[query-rewrite] 跳过改写（history 为空或意图路由层未启用）")

    if enable_thinking is None:
        enable_thinking = getattr(config, 'ENABLE_THINKING', True)
    current_db_path = config.VECTOR_DB_PATH
    has_kb = os.path.exists(current_db_path)
    print(f"[slow_echo] 向量库路径: {current_db_path}")
    print(f"[slow_echo] 向量库存在: {has_kb}")
    print(f"[slow_echo] ENABLE_INTENT_ROUTING: {getattr(config, 'ENABLE_INTENT_ROUTING', False)}")
    print(f"[slow_echo] ENABLE_THINKING: {enable_thinking}")

    # ============ 1) 意图路由层 ============
    target_fields = []
    target_drug = None
    drug_exists = False
    standardizer = None
    intent_hint = None  # P4.3 用于会话事实缓存
    if has_kb and getattr(config, 'ENABLE_INTENT_ROUTING', False):
        try:
            hint = quick_intent_hint(message)
            intent_hint = hint  # P4.3 记录意图
            print(f"[slow_echo] intent hint: {hint}")
            if hint == 'obvious_chitchat':
                target_fields = []
                print(f"[slow_echo] → 闲聊，跳过 classify + extract")
            elif hint == 'obvious_pharmacy':
                print(f"[slow_echo] → 含字段关键词，跳过 classify，直接抽字段")
                standardizer = MedicineInfoStandardizer(llm=get_openai_client())
                target_fields = standardizer.extract_target_fields(message)
                print(f"[slow_echo] → 提取字段: {target_fields}")
            else:  # ambiguous
                print(f"[slow_echo] → 模糊，调用 classify_pharmacy_query")
                classify_result = classify_pharmacy_query(message)
                print(f"[slow_echo] → classify 结果: {classify_result}")
                if classify_result == 'good':
                    standardizer = MedicineInfoStandardizer(llm=get_openai_client())
                    target_fields = standardizer.extract_target_fields(message)
                    print(f"[slow_echo] → 提取字段: {target_fields}")
                else:
                    target_fields = []
                    print(f"[slow_echo] → classify=bad，跳过字段提取")
        except Exception as e:
            print(f"[slow_echo] [intent-routing] 异常，回退原检索流程: {e}")
            target_fields = []
            standardizer = None

        # 提取药品名
        if standardizer is not None:
            try:
                target_drug = _extract_drug_name_from_query(message, standardizer)
                if target_drug:
                    drug_exists = _confirm_drug_in_es(target_drug)
                    print(f"[slow_echo] → 提取药品名: {target_drug}, ES存在: {drug_exists}")
                    if not drug_exists:
                        print(f"[slow_echo] → [警告] 药品 '{target_drug}' 在 ES 中未找到")
                else:
                    print(f"[slow_echo] → 未提取到药品名")
            except Exception as e:
                print(f"[slow_echo] → [drug-extract] 药品名提取或 ES 确认失败: {e}")
                target_drug = None

        # P4.3：当前轮次未提取到药品名时，尝试从会话事实缓存读取
        if not target_drug:
            session_cached_drug = get_session_drug()
            if session_cached_drug:
                print(f"[session-facts] 当前消息未提取到药品名，使用会话缓存 '{session_cached_drug}'")
                target_drug = session_cached_drug
                drug_exists = _confirm_drug_in_es(target_drug)
    else:
        print(f"[slow_echo] → 意图路由层未启用（知识库不存在或开关关闭）")

    # P4.2 + P4.3：确定 session_drug（优先当前轮次，其次会话缓存）
    session_drug = target_drug or get_session_drug()

    # ============ 2) 检索决策：主路径 ES 精确锁定，降级路径全库向量检索 ============
    results = []
    if has_kb:
        # 主路径：已知药品且 ES 中存在 → 直接查 ES 子段落，跳过全库 FAISS
        if target_drug and drug_exists:
            print(f"[slow_echo] → [主路径] 直接查 ES 药品 '{target_drug}' 的子段落，字段={target_fields}")
            results = retrieve_drug_subsections(target_drug, target_fields, top_k=3)
            if not results:
                print(f"[slow_echo] → [主路径] 字段过滤后无匹配，降级为全库检索")

        # 降级路径：主路径未命中或条件不满足时，走全库向量检索 + 重排
        if not results:
            try:
                retrieve_k = 10 if getattr(config, 'ENABLE_INTENT_ROUTING', False) else 3
                print(f"[slow_echo] → [降级路径] FAISS 全库检索 top_k={retrieve_k}, session_drug={session_drug}")
                results = retrieve_with_context(message, current_db_path, context_drug=session_drug, top_k=retrieve_k)
                print(f"[slow_echo] → FAISS 返回 {len(results)} 条结果（重排前）")
                for rank, r in enumerate(results):
                    print(f"[slow_echo]   raw[{rank}] id='{r[0]}' title='{r[1]}'")
            except Exception as e:
                print(f"[slow_echo] → Retrieval error: {e}")
                import traceback
                traceback.print_exc()
                results = []

        if results:
            scored = []
            for r in results:
                field_score = _score_result_by_fields(r[1], target_fields)
                drug_score = _score_result_by_drug_name(r[0], target_drug) if target_drug else 0
                total_score = drug_score * 2 + field_score * 1
                scored.append((r, total_score))
                print(f"[slow_echo]   score id='{r[0]}' title='{r[1]}' drug_score={drug_score} field_score={field_score} total={total_score}")

            if not target_drug and not any(s > 0 for _, s in scored):
                print(f"[slow_echo] → [警告] top_k={len(results)} 中无 title 匹配 target_fields={target_fields}，保留原顺序")

            scored_sorted = sorted(enumerate(scored), key=lambda kv: (-kv[1][1], kv[0]))
            results = [item[1][0] for item in scored_sorted]

            if target_drug:
                matched_count = sum(1 for _, s in scored if s > 0)
                if matched_count < 3:
                    print(f"[slow_echo] → [drug-aware-rerank] 药品 '{target_drug}' 匹配结果仅 {matched_count} 条，用向量相似度补足")

            results = results[:3]
            print(f"[slow_echo] → 重排后最终 top 3:")
            for rank, r in enumerate(results):
                print(f"[slow_echo]   final[{rank}] id='{r[0]}' title='{r[1]}'")

            # 兜底保险：全库检索后仍未命中目标药品，触发药品专属向量检索
            if target_drug and not any(r[0] == target_drug for r in results):
                print(f"[slow_echo] → [兜底] 全库检索未命中 '{target_drug}'，触发药品专属检索")
                try:
                    drug_results = retrieve_vector_and_text_for_drug(
                        message, current_db_path, target_drug, top_k=3
                    )
                    if drug_results:
                        results = drug_results
                        print(f"[slow_echo] → [兜底] 已替换为 '{target_drug}' 专属检索结果")
                        for rank, r in enumerate(results):
                            print(f"[slow_echo]   fallback[{rank}] id='{r[0]}' title='{r[1]}'")
                except Exception as e:
                    print(f"[slow_echo] → [兜底] 药品专属检索失败: {e}")
                    import traceback
                    traceback.print_exc()

    if results:
        context = "\n".join([f"【{r[1]}】\n{r[2]}" for r in results])
    elif has_kb:
        context = "No context found or error in search."
    else:
        context = "（当前未加载药典知识库，将直接基于模型能力回答。）"

    print(f"[slow_echo] → 组装 context（前200字符）:\n{context[:200]}...")

    # ============ 3) Prompt 组装 ============
    if has_kb:
        base_prompt = "你是一个专业的药典问答助手。请根据提供的上下文回答用户的问题。如果上下文不相关，请根据你自己的知识回答，但要说明上下文不相关。"
        if target_fields:
            system_prompt = (
                base_prompt
                + f"\n用户正在查询以下药品属性：{', '.join(target_fields)}。"
                + "请优先基于提供的上下文回答这些属性，如果上下文中缺少某字段信息，请明确说明。"
            )
        else:
            system_prompt = base_prompt
        user_prompt = f"上下文:\n{context}\n\n问题: {message}"
    else:
        system_prompt = "你是一个专业的药典问答助手。当前未加载药典知识库，请基于你的通用知识回答用户问题。"
        user_prompt = message

    messages = [{"role": "system", "content": system_prompt}]

    for msg in history or []:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
            if role and content:
                messages.append({"role": role, "content": content})
        else:
            try:
                user_msg, bot_msg = msg
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if bot_msg:
                    messages.append({"role": "assistant", "content": bot_msg})
            except Exception:
                print(f"[slow_echo] 无法解析 history 条目: {msg!r}")
                continue

    messages.append({"role": "user", "content": user_prompt})
    print(f"[slow_echo] → 发送给 LLM 的 messages（共 {len(messages)} 条）:")
    for idx, m in enumerate(messages):
        print(f"[slow_echo]   msg[{idx}] role={m['role']} content[:80]={m['content'][:80]}")

    try:
        kwargs = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "stream": True,
        }
        if not enable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            print(f"[slow_echo] → 禁用思考过程")

        print(f"[slow_echo] → 调用 LLM...")
        response = get_openai_client().chat.completions.create(**kwargs)

        partial_message = ""
        in_thinking = False
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning and isinstance(reasoning, str):
                if not in_thinking:
                    in_thinking = True
                    partial_message += "💭 思考中...\n"
                partial_message += reasoning
                yield partial_message
                continue

            if in_thinking and delta.content:
                in_thinking = False
                partial_message += "\n\n---\n\n"

            if delta.content is not None:
                partial_message += delta.content
                yield partial_message
    except Exception as e:
        print(f"[slow_echo] LLM 调用异常: {e}")
        import traceback
        traceback.print_exc()
        yield f"Error calling LLM: {e}"

    # P4.3：更新会话事实缓存，供下一轮使用
    _update_session_facts(target_drug, set(target_fields), intent_hint)
    print(f"[session-facts] 更新: primary_drug={get_session_drug()}, "
          f"fields={get_session_fields()}, history={_session_facts['drug_history']}")

    print(f"{'='*60}\n")

# ...

with gr.Blocks() as demo:# web页面效果主代码
    
    with gr.Tab("药典问答"):#问答页面
        gr.Markdown("**第一次使用请先去配置自己的信息并保存哦**")
        chatbot = gr.Chatbot(height=600)  # 设置高度为600像素
        enable_thinking_checkbox = gr.Checkbox(
            label="显示模型思考过程",
            value=getattr(config, 'ENABLE_THINKING', True),
        )
        qa_interface = gr.ChatInterface(
            fn=slow_echo,
            chatbot=chatbot,  # 将自定义的 Chatbot 传递给 ChatInterface
            additional_inputs=enable_thinking_checkbox,
        )

    
    with gr.Tab("文档导入"):# 文档导入页面
        # Use config.BASE_DIR or config.VECTOR_DB_PATH dir
        default_folder = os.path.dirname(config.VECTOR_DB_PATH)
        folder_input = gr.Textbox(label="保存向量数据库的文件夹路径", value=default_folder)

        upload_interface = gr.Interface(
            fn=import_new_documents,
            inputs=[
                gr.File(label="上传新的药典文档"),
                gr.Textbox(label="数据库和索引名命名", placeholder="请输入", value=config.ES_INDEX),
                folder_input,
            ],
            outputs=gr.Textbox(label="结果"),
            title="文档导入",
            description="上传新的药典文档以更新数据库。"
        )
    
    with gr.Tab("配置"):# 配置页面
        es_host_input = gr.Textbox(label="ES主机地址", value=config.ES_HOST)
        es_port_input = gr.Textbox(label="ES服务端口", value=str(config.ES_PORT))
        es_user_input = gr.Textbox(label="ES用户名", value=config.ES_USER)
        es_pass_input = gr.Textbox(label="ES密码", type="password", value=config.ES_PASSWORD)
        
        es_index_input = gr.Textbox(label="ES索引名（上传的会和向量数据库同名）", value=config.ES_INDEX)
        vector_db_path_input = gr.Textbox(label="向量数据库位置", value=config.VECTOR_DB_PATH)

        es_scheme_input = gr.Textbox(label="ES协议 (http/https)", value=config.ES_SCHEME)

        config_submit = gr.Button("保存配置")
        config_message = gr.Textbox(label="状态", interactive=False)

        config_submit.click(
            fn=update_config,
            inputs=[es_host_input, es_port_input, es_user_input, es_pass_input, es_index_input, vector_db_path_input, es_scheme_input],
            outputs=config_message
        )


# 启动应用
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",share=False) # Share=False is faster/safer for local dev

