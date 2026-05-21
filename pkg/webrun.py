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
    retrieve_product_info,
    get_openai_client,
    clear_openai_client_cache,
    quick_intent_hint,
    _score_result_by_drug_name,
    quick_ecommerce_intent_hint,
    classify_ecommerce_intent,
    model,
)
from orders import (
    find_order,
    get_order_logistics_path,
    get_order_abnormal_status,
    get_order_return_window,
    check_exchange_inventory,
    update_order_field,
    load_orders,
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
_LOGISTICS_STATES = {
    "init", "awaiting_identity", "identity_verified",
    "awaiting_modify_type", "awaiting_new_info",
    "reading_path", "checking_status", "generating_response",
    "writing_back", "completed", "rejected",
}

_MODIFY_TYPE_LABELS = {
    "address": "地址",
    "phone": "电话",
    "recipient": "收件人",
}

_session_facts = {
    "primary_drug": None,       # 当前会话主要讨论的药品（药典兼容）
    "queried_fields": set(),    # 已查询过的字段集合
    "last_intent": None,        # 上一轮意图
    "last_sub_intent": None,    # 上一轮子意图（电商）
    "keywords": None,           # product_info 的模块映射（电商）
    "drug_history": [],         # 本轮会话提到过的所有药品（按时间顺序，去重）
    "verified_identity": False, # 是否已验证身份（电商）
    "bound_order_id": None,     # 已绑定的订单号（电商）
    "bound_phone": None,        # 已绑定的手机号（电商）
    "dialogue_state": "init",   # 对话状态（电商）
    "product_history": [],      # 浏览过的商品历史（电商）
    "modify_type": None,        # 当前修改类型（modify 分支用）
    "price_range": None,        # 价格区间（推荐分支用）
    "recommend_sub_intent": None, # 保存推荐子意图（推荐分支用）
    "matching_scene": None,     # 搭配场景（推荐分支用）
    "matching_usage": None,     # 使用行为（推荐分支用）
    "purchase_stage": None,     # 购买阶段（product_info 分支用）
    "question_repeat_count": 0, # 连续追问计数（转人工用）
    "last_question_embedding": None, # 上轮问题 embedding（转人工用）
    "transfer_requested": False, # 是否已触发转人工（待确认）
    "transfer_confirmed": False, # 用户是否确认转人工
    "operation_reason": None,     # 售后原因（goods_operation 分支用）
    "operation_detail": None,     # 方式/规格/金额（goods_operation 分支用）
}


def _update_session_facts(target_drug=None, target_fields=None, intent_hint=None, sub_intent=None, keywords=None, purchase_stage=None):
    """每轮问答结束后更新会话事实缓存。兼容药典旧参数与电商新参数。"""
    global _session_facts
    if target_drug:
        _session_facts["primary_drug"] = target_drug
        if target_drug not in _session_facts["drug_history"]:
            _session_facts["drug_history"].append(target_drug)
    if target_fields:
        _session_facts["queried_fields"].update(target_fields)
    if intent_hint is not None:
        _session_facts["last_intent"] = intent_hint
    if sub_intent is not None:
        _session_facts["last_sub_intent"] = sub_intent
    if keywords is not None:
        _session_facts["keywords"] = keywords
    if purchase_stage is not None:
        _session_facts["purchase_stage"] = purchase_stage


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
        "last_sub_intent": None,
        "keywords": None,
        "drug_history": [],
        "verified_identity": False,
        "bound_order_id": None,
        "bound_phone": None,
        "dialogue_state": "init",
        "product_history": [],
        "purchase_stage": None,
        "question_repeat_count": 0,
        "last_question_embedding": None,
        "transfer_requested": False,
        "transfer_confirmed": False,
        "operation_reason": None,
        "operation_detail": None,
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


# ---------------------------------------------------------------------------
# Phase 2：物流状态机辅助函数
# ---------------------------------------------------------------------------

def _verify_identity(user_input, session_facts):
    """
    从用户输入中提取订单号和手机号，验证匹配。
    返回 (success: bool, order: dict|None)
    """
    if not user_input:
        return False, None

    order_id_match = re.search(r"(ORD[A-Z0-9]+)", user_input, re.IGNORECASE)
    phone_match = re.search(r"(1[3-9]\d{9})", user_input)

    if not order_id_match or not phone_match:
        print(f"[_verify_identity] 格式解析失败: input={user_input}")
        return False, None

    order_id = order_id_match.group(1).upper()
    phone = phone_match.group(1)
    print(f"[_verify_identity] 解析到 order_id={order_id}, phone={phone}")

    order = find_order(order_id, phone)
    if order:
        session_facts["verified_identity"] = True
        session_facts["bound_order_id"] = order_id
        session_facts["bound_phone"] = phone
        print(f"[_verify_identity] 验证通过: {order_id}")
        return True, order
    else:
        print(f"[_verify_identity] 查无订单: {order_id}, {phone}")
        return False, None


def _parse_modify_type(text):
    """解析用户输入为 address/phone/recipient 或 None。"""
    if not text:
        return None
    t = text.strip().lower()
    if t in ("1", "地址", "addr", "address", "1.地址"):
        return "address"
    if t in ("2", "电话", "手机", "phone", "tel", "2.电话"):
        return "phone"
    if t in ("3", "收件人", "姓名", "名字", "recipient", "name", "3.收件人"):
        return "recipient"
    return None


def _validate_new_info(modify_type, text):
    """校验新信息格式。返回 (bool, error_message)。"""
    if not text or not modify_type:
        return False, "输入不能为空"

    text = text.strip()

    if modify_type == "address":
        if len(text) < 5:
            return False, "地址长度不能少于5个字符"
        if not any(kw in text for kw in ("省", "市", "区", "县", "州", "镇", "街道")):
            return False, "地址需包含省、市、区/县等行政区域信息"
        return True, ""

    elif modify_type == "phone":
        if not re.match(r"^1[3-9]\d{9}$", text):
            return False, "手机号格式不正确，请输入11位有效手机号"
        return True, ""

    elif modify_type == "recipient":
        if len(text) < 2:
            return False, "收件人姓名不能少于2个字符"
        if len(text) > 20:
            return False, "收件人姓名不能超过20个字符"
        return True, ""

    return False, "不支持的修改类型"


def _format_logistics_path(path_nodes):
    """把物流节点列表格式化为 LLM 可读的文本。"""
    if not path_nodes:
        return "暂无物流轨迹信息。"
    lines = []
    for i, node in enumerate(path_nodes, 1):
        time_str = node.get("time", "未知时间")
        location = node.get("node", "未知地点")
        weather = node.get("weather")
        if weather:
            lines.append(f"{i}. {time_str} — {location}（天气：{weather}）")
        else:
            lines.append(f"{i}. {time_str} — {location}")
    return "\n".join(lines)


def _handle_logistics_track(order, message, history, llm_client):
    """
    轨迹追踪处理函数。构造 prompt 调用 LLM，返回语义化回复字符串。
    """
    path_nodes = get_order_logistics_path(order)
    formatted_path = _format_logistics_path(path_nodes)
    logistics = order.get("logistics", {})

    prompt = f"""用户订单 {order.get('order_id')} 的物流状态如下：
承运商：{logistics.get('carrier', '未知')}
当前位置：{logistics.get('current_location', '未知')}
物流路径：
{formatted_path}
预计到达：{logistics.get('estimated_arrival', '未知')}

请用自然语言向用户描述物流进度，包含当前位置和预计时间。
如果路径中有天气信息，请一并描述。保持简洁友好。"""

    try:
        response = llm_client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip() if response.choices else "物流信息查询成功，但生成回复失败。"
    except Exception as e:
        print(f"[_handle_logistics_track] LLM 调用失败: {e}")
        return "抱歉，物流查询服务暂时不可用，请稍后重试。"


def _handle_logistics_abnormal(order, message, history, llm_client):
    """
    异常处理处理函数。根据 abnormal_status 构造不同 prompt，调用 LLM 返回话术。
    """
    abnormal = get_order_abnormal_status(order)
    logistics = order.get("logistics", {})

    if abnormal is None:
        prompt = f"""用户订单 {order.get('order_id')} 物流状态正常，无异常。
承运商：{logistics.get('carrier', '未知')}
当前位置：{logistics.get('current_location', '未知')}
预计到达：{logistics.get('estimated_arrival', '未知')}

请生成安抚话术，告知用户订单正常，请耐心等待。"""
    elif abnormal == "delay":
        prompt = f"""用户订单 {order.get('order_id')} 包裹目前延误。
预计到达时间：{logistics.get('estimated_arrival', '未知')}
当前位置：{logistics.get('current_location', '未知')}

请生成安抚话术，说明延误情况并表达歉意。"""
    elif abnormal == "lost":
        prompt = f"""用户订单 {order.get('order_id')} 包裹确认丢失。
我们将为用户办理自动赔付。

请生成说明话术，告知用户赔付流程和预计到账时间。"""
    elif abnormal == "damaged":
        prompt = f"""用户订单 {order.get('order_id')} 包裹存在损坏。
用户可以选择补发或退款。

请生成说明话术，并询问用户希望选择补发还是退款。"""
    else:
        return "您的订单状态未知，请联系人工客服处理。"

    try:
        response = llm_client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip() if response.choices else "异常处理查询成功，但生成回复失败。"
    except Exception as e:
        print(f"[_handle_logistics_abnormal] LLM 调用失败: {e}")
        return "抱歉，异常处理服务暂时不可用，请稍后重试。"


# ---------------------------------------------------------------------------
# Phase 3：商品推荐分支
# ---------------------------------------------------------------------------
_RECOMMEND_STATES = {
    "init", "awaiting_price_range", "awaiting_matching_scene",
    "awaiting_matching_usage", "retrieving", "generating",
    "completed", "rejected",
}

_PRODUCT_INFO_STATES = {
    "init", "awaiting_confirmation", "completed", "rejected",
}

_REPEAT_SIMILARITY_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Phase 5：货物操作状态机
# ---------------------------------------------------------------------------
_GOODS_OP_STATES = {
    "init", "awaiting_identity", "identity_verified",
    "awaiting_reason", "awaiting_return_method",
    "awaiting_exchange_spec", "awaiting_refund_amount",
    "validating", "writing_back", "generating_response",
    "completed", "rejected",
}

_RETURN_REASONS = {
    "1": "不想要了", "2": "质量问题", "3": "描述不符", "4": "其他"
}

_EXCHANGE_REASONS = {
    "1": "质量问题", "2": "发错货", "3": "尺码颜色不合适", "4": "其他"
}

_REFUND_REASONS = {
    "1": "未收到货", "2": "商品破损", "3": "不想要了", "4": "其他"
}

_RETURN_METHODS = {
    "1": "上门取件", "2": "自行寄回"
}


def _parse_price_range(text):
    """纯解析函数，返回 dict {min, max, text} 或 None。"""
    if not text:
        return None
    text = text.strip()

    # 200-500 / 200~500 / 200到500 / 200至500
    m = re.search(r"(\d+)\s*(?:-|~|到|至)\s*(\d+)", text)
    if m:
        min_v, max_v = int(m.group(1)), int(m.group(2))
        return {"min": min_v, "max": max_v, "text": f"{min_v}-{max_v}元"}

    # 300以内 / 不超过300 / 300以下 / 最多300 / 小于300
    m = re.search(r"(?:不超过|以内|以下|最多|小于)\s*(\d+)|(\d+)\s*(?:以内|以下|最多|小于)", text)
    if m:
        v = int(m.group(1) or m.group(2))
        return {"min": 0, "max": v, "text": f"{v}元以内"}

    # 500左右 / 大概500 / 大约500 / 差不多500
    m = re.search(r"(?:大概|大约|左右|差不多)\s*(\d+)|(\d+)\s*(?:左右|大概|大约|差不多)", text)
    if m:
        v = int(m.group(1) or m.group(2))
        return {"min": int(v * 0.8), "max": int(v * 1.2), "text": f"{v}元左右"}

    # 1000以上 / 至少1000 / 大于1000 / 最少1000
    m = re.search(r"(?:至少|以上|大于|最少)\s*(\d+)|(\d+)\s*(?:以上|最少|至少|大于)", text)
    if m:
        v = int(m.group(1) or m.group(2))
        return {"min": v, "max": None, "text": f"{v}元以上"}

    # 纯数字 500
    m = re.search(r"^(\d+)$", text)
    if m:
        v = int(m.group(1))
        return {"min": 0, "max": v, "text": f"{v}元以内"}

    return None


def _format_price_range(price_dict):
    """将价格区间 dict 格式化为可读文本。"""
    if not price_dict:
        return "未指定"
    return price_dict.get("text", "未指定")


def _get_recommend_type_label(sub_intent):
    """sub_intent → 中文标签。"""
    return {
        "substitute": "平替商品推荐",
        "similar_style": "风格类似商品推荐",
        "matching": "搭配关联推荐",
    }.get(sub_intent, "商品推荐")


# ---------------------------------------------------------------------------
# Phase 4：转人工逻辑 + 商品信息分支
# ---------------------------------------------------------------------------

def _detect_repeat_question(current_message, intent, session_facts):
    """
    检测用户是否在同一 intent 下连续追问相似问题。
    若 embedding cosine_similarity > _REPEAT_SIMILARITY_THRESHOLD，累加计数。
    返回 (is_repeat, repeat_count)。
    """
    global model
    last_intent = session_facts.get("last_intent")
    if intent != last_intent:
        session_facts["question_repeat_count"] = 0
        session_facts["last_question_embedding"] = None
        return False, 0

    try:
        if model is None:
            return False, session_facts.get("question_repeat_count", 0)
        current_emb = model.encode(current_message, normalize_embeddings=True)
        last_emb = session_facts.get("last_question_embedding")
        if last_emb is not None:
            sim = float(np.dot(np.asarray(current_emb).flatten(), np.asarray(last_emb).flatten()))
            if sim > _REPEAT_SIMILARITY_THRESHOLD:
                session_facts["question_repeat_count"] = session_facts.get("question_repeat_count", 0) + 1
                print(f"[repeat-detect] sim={sim:.3f} > threshold, count={session_facts['question_repeat_count']}")
                return True, session_facts["question_repeat_count"]
        session_facts["last_question_embedding"] = current_emb
    except Exception as e:
        print(f"[repeat-detect] embedding 计算失败: {e}")

    return False, session_facts.get("question_repeat_count", 0)


def _transfer_to_human(reason, session_facts):
    """
    触发转人工流程。设置 transfer_requested=True，返回引导话术。
    reason: explicit（明确输入）/ repeat（连续追问）/ out_of_scope（超纲业务）
    """
    session_facts["transfer_requested"] = True
    session_facts["dialogue_state"] = "awaiting_confirmation"
    print(f"[transfer] triggered, reason={reason}")
    if reason == "explicit":
        return "👤 已为您转接人工客服，请稍候……"
    elif reason == "repeat":
        return "🤔 我注意到您似乎在反复询问类似的问题，是否需要转接人工客服为您进一步解答？请回复“是”或“否”。"
    elif reason == "out_of_scope":
        return "😅 这个问题超出了我的服务范围，需要为您转接人工客服吗？请回复“是”或“否”。"
    return "是否需要转接人工客服？请回复“是”或“否”。"


def _build_product_info_prompt(retrieved_context, target_module, purchase_stage, user_message):
    """
    根据购买阶段动态组装 product_info 分支的 LLM prompt。
    """
    stage_desc = {
        "awareness": "了解阶段：用户刚开始了解商品，回答应通俗易懂、突出核心卖点，避免过多技术细节。",
        "evaluation": "评估阶段：用户正在对比和评估，回答应提供详细参数、优缺点对比、真实评价参考。",
        "decision": "决策阶段：用户已有购买意向，回答应突出促销信息、库存状态、售后保障、临门一脚促成转化。",
    }
    stage_text = stage_desc.get(purchase_stage, "请根据用户问题自然回答。")

    module_label = {
        "basic_info": "基础信息",
        "material": "规格材质",
        "design": "设计工艺",
        "care": "护理保养",
        "styling": "场景搭配",
        "shipping_info": "库存物流",
        "after_sales": "售后政策",
        "marketing": "营销评价",
        "fit": "版型身材",
    }.get(target_module, "商品信息")

    return f"""你是专业的电商商品咨询助手。请基于以下商品知识库信息回答用户问题。

【用户购买阶段】
{stage_text}

【用户关注模块】
{module_label}

【商品知识库信息】
{retrieved_context}

【用户问题】
{user_message}

【要求】
1. 优先基于知识库信息回答，若知识库信息不足可适度补充
2. 回答应符合当前购买阶段的特点
3. 语言亲切自然，避免过度营销
4. 若涉及尺码/版型建议，请给出具体参考
"""


# ---------------------------------------------------------------------------
# Phase 5：货物操作（退货 / 换货 / 仅退款）
# ---------------------------------------------------------------------------

def _parse_operation_reason(sub_intent, text):
    """
    解析用户输入的售后原因。
    支持数字 "1"/"2"/"3"/"4"、原因文本模糊匹配、"其他"或任意文本归类为"其他"。
    返回 (reason_text, reason_code) 或 (None, None) 表示解析失败。
    """
    if not text:
        return None, None
    t = text.strip()
    reason_map = {
        "return": _RETURN_REASONS,
        "exchange": _EXCHANGE_REASONS,
        "refund_only": _REFUND_REASONS,
    }.get(sub_intent, {})

    # 直接匹配数字
    if t in reason_map:
        return reason_map[t], t

    # 模糊匹配文本（支持关键词包含）
    for code, reason_text in reason_map.items():
        if reason_text in t or t in reason_text:
            return reason_text, code
    # 更宽松的匹配：原因文本中的关键词出现在用户输入中
    keyword_map = {
        "return": {"质量问题": ["质量"], "不想要了": ["不要", "想退"], "描述不符": ["描述", "不符", "不一致"]},
        "exchange": {"质量问题": ["质量"], "发错货": ["发错", "错发"], "尺码颜色不合适": ["尺码", "大小", "颜色", "不合适"]},
        "refund_only": {"未收到货": ["没收", "没到", "未收"], "商品破损": ["破损", "损坏", "烂"], "不想要了": ["不要", "想退"]},
    }
    kw_map = keyword_map.get(sub_intent, {})
    for reason_text, kws in kw_map.items():
        for kw in kws:
            if kw in t:
                for code, rt in reason_map.items():
                    if rt == reason_text:
                        return reason_text, code

    # "其他"兜底
    if t:
        return "其他", "4"

    return None, None


def _prompt_for_reason(sub_intent):
    labels = {"return": "退货", "exchange": "换货", "refund_only": "退款"}
    reason_map = {
        "return": _RETURN_REASONS,
        "exchange": _EXCHANGE_REASONS,
        "refund_only": _REFUND_REASONS,
    }
    options = "\n".join([f"{k}. {v}" for k, v in reason_map[sub_intent].items()])
    return f"请告诉我们您的{labels[sub_intent]}原因（回复数字或原因）：\n{options}"


def _parse_return_method(text):
    """解析 '1'/'上门取件'/'自行寄回' → method_text 或 None"""
    if not text:
        return None
    t = text.strip()
    if t in ("1", "上门取件"):
        return "上门取件"
    if t in ("2", "自行寄回"):
        return "自行寄回"
    return None


def _prompt_for_return_method():
    return "请选择退货方式：\n1. 上门取件\n2. 自行寄回"


def _prompt_for_exchange_spec(order):
    """
    基于订单商品的 exchange_inventory 生成可选规格提示。
    """
    if not order:
        return "无法获取订单信息，请稍后再试。"
    items = order.get("items", [])
    if not items:
        return "订单中无商品信息。"
    current_spec = items[0].get("spec", {})
    current_label = f"{current_spec.get('color', '未知')} {current_spec.get('size', '未知')}"

    after_sales = order.get("after_sales", {})
    inventory = after_sales.get("exchange_inventory", {})
    available = []
    idx = 1
    for color, sizes in inventory.items():
        for size, stock in sizes.items():
            if stock and stock > 0:
                available.append((str(idx), color, size, stock))
                idx += 1

    if not available:
        return "抱歉，当前所有规格均暂无库存，请联系人工客服处理。"

    lines = [f"您当前商品：{current_label}。可换规格（仅显示有库存）："]
    for num, color, size, stock in available:
        lines.append(f"{num}. {color} {size}")
    lines.append("请回复数字或'颜色 尺码'格式")
    return "\n".join(lines)


def _parse_exchange_spec(text, available_specs):
    """
    解析用户输入为目标规格 dict {'color': ..., 'size': ...} 或 None。
    available_specs: [(num, color, size, stock), ...]
    """
    if not text:
        return None
    t = text.strip()
    # 尝试匹配数字
    for num, color, size, stock in available_specs:
        if t == num:
            return {"color": color, "size": size}
    # 尝试解析 "颜色 尺码"
    parts = t.split()
    if len(parts) >= 2:
        # 最后一部分是尺码，前面是颜色
        size = parts[-1]
        color = " ".join(parts[:-1])
        for _, c, s, _ in available_specs:
            if c == color and s == size:
                return {"color": color, "size": size}
    return None


def _prompt_for_refund_amount(order):
    """基于订单金额生成退款选项。"""
    if not order:
        return "无法获取订单信息。"
    items = order.get("items", [])
    total = sum(item.get("price", 0) * item.get("quantity", 1) for item in items) if items else 0
    if total == 0:
        total = order.get("total_amount", 0)
    return f"您的订单实付金额为 ¥{total}。请选择退款金额：\n1. 全额退款 ¥{total}\n2. 部分退款（请手动输入金额）"


def _parse_refund_amount(text, total_amount):
    """
    解析用户输入。
    '1'/'全额' → total_amount
    数字 '200' → min(200, total_amount)
    超过 total_amount → 失败
    返回: (amount, label) 或 (None, error_msg)
    """
    if not text:
        return None, "请输入退款金额"
    t = text.strip()
    if t in ("1", "全额", "全额退款"):
        return total_amount, f"全额退款 ¥{total_amount}"
    try:
        amount = int(t)
        if amount <= 0:
            return None, "退款金额必须大于0"
        if amount > total_amount:
            return None, f"退款金额不能超过订单实付金额 ¥{total_amount}"
        return amount, f"部分退款 ¥{amount}"
    except ValueError:
        return None, "请输入数字金额或选择全额退款"


def _validate_goods_operation(order, sub_intent, reason, detail, session_facts):
    """
    售后操作前置校验。返回 (is_valid: bool, message: str)。
    """
    if not order:
        return False, "未找到订单信息"

    from datetime import datetime

    if sub_intent == "return":
        deadline = get_order_return_window(order)
        if deadline:
            try:
                if isinstance(deadline, str):
                    deadline_dt = datetime.strptime(deadline, "%Y-%m-%d")
                else:
                    deadline_dt = deadline
                if datetime.now() > deadline_dt:
                    return False, "已超过退货期限（7天），无法退货，建议换货或联系客服"
            except Exception:
                pass
        # 检查是否已有未完成的退货申请
        history = order.get("operations_history", [])
        for h in history:
            if h.get("type") == "return" and h.get("status") == "申请中":
                return False, "您已提交退货申请，请勿重复提交"
        return True, ""

    if sub_intent == "exchange":
        if not detail or not isinstance(detail, dict):
            return False, "请选择要更换的规格"
        items = order.get("items", [])
        current_spec = items[0].get("spec", {}) if items else {}
        if detail.get("color") == current_spec.get("color") and detail.get("size") == current_spec.get("size"):
            return False, "新规格与当前规格相同"
        if not check_exchange_inventory(order, detail):
            return False, "目标规格暂无库存，请选择其他规格"
        return True, ""

    if sub_intent == "refund_only":
        status = order.get("status", "")
        if status == "已签收":
            return False, "已签收订单不支持仅退款，请申请退货退款"
        if status == "已发货":
            return False, "包裹已发出，建议收货后退货，或联系客服拦截"
        return True, ""

    return False, "不支持的售后类型"


def _write_back_goods_operation(order_id, sub_intent, reason, detail, session_facts):
    """
    将售后申请写入订单 JSON。
    返回 (success: bool, message: str)
    """
    import json
    from datetime import datetime
    all_orders = load_orders()
    target_order = None
    for order in all_orders:
        if order.get("order_id") == order_id:
            target_order = order
            break
    if target_order is None:
        return False, "未找到指定订单"

    if "operations_history" not in target_order:
        target_order["operations_history"] = []

    record = {
        "type": sub_intent,
        "reason": reason,
        "detail": detail,
        "status": "申请中",
        "apply_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "update_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    target_order["operations_history"].append(record)
    target_order["status"] = "售后处理中"

    path = getattr(config, "ORDERS_JSON_PATH", "./orders/orders.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_orders, f, ensure_ascii=False, indent=2)
        return True, "申请已提交"
    except Exception as e:
        return False, f"写入文件失败: {e}"


def _build_goods_operation_prompt(sub_intent, reason, detail, order, validation_passed=True, fail_message=None):
    """
    组装售后话术的 LLM messages。
    返回 messages 列表（可直接传给 _yield_llm_stream）。
    """
    labels = {"return": "退货", "exchange": "换货", "refund_only": "仅退款"}
    sub_label = labels.get(sub_intent, "售后申请")
    order_id = order.get("order_id", "未知") if order else "未知"

    if validation_passed:
        detail_str = str(detail) if detail else "未指定"
        user_content = (
            f"用户申请了{sub_label}，原因：{reason}。"
            f"详情：{detail_str}。订单号：{order_id}。"
            f"请生成：1) 确认申请已受理 2) 后续操作指引 3) 预计处理时效 4) 安抚话术"
        )
        return [
            {"role": "system", "content": "你是专业的电商售后客服。请用亲切、清晰的语气告知用户售后申请结果。"},
            {"role": "user", "content": user_content},
        ]
    else:
        user_content = (
            f"用户申请了{sub_label}，但校验未通过：{fail_message}。"
            f"请生成：1) 解释原因 2) 提供替代方案（如换货/联系客服）3) 安抚话术"
        )
        return [
            {"role": "system", "content": "你是专业的电商售后客服。请委婉但清晰地解释无法办理的原因，并提供替代方案。"},
            {"role": "user", "content": user_content},
        ]


def _collect_price_range(user_input, session_facts):
    """
    从用户输入解析价格区间，写入 session_facts。
    返回 (bool, parsed_dict|error_message)
    """
    parsed = _parse_price_range(user_input)
    if parsed:
        session_facts["price_range"] = parsed
        return True, parsed
    return False, "请提供明确的价格区间，例如：200-500元、300以内、500左右"


def _collect_matching_scene(user_input, session_facts):
    """收集搭配场景需求。返回 (bool, scene_text)。"""
    if not user_input or not user_input.strip():
        return False, ""
    scene = user_input.strip()
    session_facts["matching_scene"] = scene
    return True, scene


def _collect_matching_usage(user_input, session_facts):
    """收集使用行为/习惯。返回 (bool, usage_text)。"""
    if not user_input or not user_input.strip():
        return False, ""
    usage = user_input.strip()
    session_facts["matching_usage"] = usage
    return True, usage


def _extract_current_product(message, session_facts):
    """
    提取用户当前关注的商品名称。
    优先级：消息中提取 → product_history 回溯 → None。
    """
    if message:
        # 引号内文本
        m = re.search(r'["\'"`]([^"\'"`]+)["\'"`]', message)
        if m:
            return m.group(1).strip()

        # "这个/那件/刚才那个/这款/那件" + 名词
        m = re.search(r"(?:这个|那件|刚才那个|这款|那件)\s*([\u4e00-\u9fff\w]{2,20})", message)
        if m:
            return m.group(1).strip()

    # 回溯历史
    history = session_facts.get("product_history", [])
    if history:
        return history[-1]
    return None


def _build_recommend_query(current_product, sub_intent, price_range, matching_scene=None, matching_usage=None):
    """构造商品推荐的 RAG 查询文本。"""
    price_text = _format_price_range(price_range)

    if sub_intent == "substitute":
        if current_product:
            return f"{current_product} 平替商品，{price_text}，相似功能，性价比"
        return f"推荐平替商品，{price_text}，相似功能，性价比"

    if sub_intent == "similar_style":
        if current_product:
            return f"{current_product} 风格类似商品，{price_text}，相同风格，相似设计"
        return f"推荐风格类似商品，{price_text}，相同风格，相似设计"

    if sub_intent == "matching":
        scene = matching_scene or ""
        usage = matching_usage or ""
        if current_product:
            return f"{current_product} 搭配商品，{price_text}，{scene}，{usage}，搭配推荐"
        return f"推荐搭配商品，{price_text}，{scene}，{usage}，搭配推荐"

    if current_product:
        return f"{current_product} 商品推荐，{price_text}"
    return f"推荐商品，{price_text}"


def _retrieve_for_recommendation(query, top_k=5):
    """检索商品知识库，返回结果列表。"""
    db_path = getattr(config, "PRODUCT_INDEX_PATH", config.VECTOR_DB_PATH)
    try:
        results = retrieve_with_context(query, db_path, context_drug=None, top_k=top_k)
        print(f"[recommend-retrieve] query={query} | results={len(results)}")
        return results
    except Exception as e:
        print(f"[recommend-retrieve] 检索失败: {e}")
        return []


def _build_recommend_prompt(current_product, sub_intent, price_range, retrieved_context, matching_scene=None, matching_usage=None):
    """构造推荐生成的 LLM prompt。"""
    type_label = _get_recommend_type_label(sub_intent)
    price_text = _format_price_range(price_range)

    matching_block = ""
    if sub_intent == "matching" and (matching_scene or matching_usage):
        matching_block = f"使用场景：{matching_scene or '未指定'}\n使用需求：{matching_usage or '未指定'}\n"

    return f"""你是专业的电商商品推荐助手。请基于以下信息为用户推荐商品：

【用户画像】
当前关注商品：{current_product or "未明确"}
推荐类型：{type_label}
价格区间：{price_text}
{matching_block}
【候选商品信息】
{retrieved_context}

【要求】
1. 推荐 2-3 款商品，说明推荐理由
2. 突出与当前商品的关联（替代性/风格相似性/搭配性）
3. 价格在用户指定区间内
4. 语言亲切自然，避免过度营销
5. 如果候选商品信息不足，请基于你的知识补充，但需标注"基于平台商品库"
"""


def _yield_llm_stream(messages, enable_thinking):
    """统一的 LLM 流式输出 generator，供 slow_echo 各分支复用。"""
    try:
        kwargs = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "stream": True,
        }
        if not enable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            print(f"[llm-stream] 禁用思考过程")

        print(f"[llm-stream] 调用 LLM...")
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
        print(f"[llm-stream] LLM 调用异常: {e}")
        import traceback
        traceback.print_exc()
        yield f"Error calling LLM: {e}"


def slow_echo(message, history, enable_thinking=True):
    """
    问答主流程，在 P1 基础上增加可选的"意图路由层"。
    """
    # 如果上一轮状态机已结束，重置为 init
    if _session_facts.get("dialogue_state") in {"completed", "rejected"}:
        old_state = _session_facts["dialogue_state"]
        _session_facts["dialogue_state"] = "init"
        print(f"[state-machine] {old_state} → init")

    print(f"\n{'='*60}")
    print(f"[slow_echo] ===== 新问答请求 =====")
    print(f"[slow_echo] 用户原始消息: {message}")
    print(f"[slow_echo] history 长度: {len(history) if history else 0}")

    # ============ 0a) 转人工状态拦截 ============
    if _session_facts.get("transfer_confirmed"):
        yield "👤 已为您转接人工客服，请稍候……"
        return

    if _session_facts.get("transfer_requested") and _session_facts.get("dialogue_state") == "awaiting_confirmation":
        msg_lower = message.strip().lower()
        if msg_lower in {"是", "是的", "好", "可以", "ok", "yes", "y"}:
            _session_facts["transfer_confirmed"] = True
            _session_facts["dialogue_state"] = "completed"
            print(f"[transfer-state] awaiting_confirmation → completed (confirmed)")
            yield "👤 已为您转接人工客服，请稍候……"
            return
        elif msg_lower in {"否", "不", "不用", "no", "n", "不需要"}:
            _session_facts["transfer_requested"] = False
            _session_facts["dialogue_state"] = "init"
            _session_facts["question_repeat_count"] = 0
            print(f"[transfer-state] awaiting_confirmation → init (cancelled)")
            yield "好的，我继续为您服务。请问还有什么可以帮您的？"
            return
        else:
            yield "请问是否需要转接人工客服？请回复“是”或“否”。"
            return

    # ============ 0b) 查询改写：消除指代词 ============
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
    # Phase 1：优先使用商品向量库，同时兼容旧版药典向量库路径
    current_db_path = getattr(config, 'PRODUCT_INDEX_PATH', config.VECTOR_DB_PATH)
    has_kb = os.path.exists(current_db_path)
    print(f"[slow_echo] 向量库路径: {current_db_path}")
    print(f"[slow_echo] 向量库存在: {has_kb}")
    print(f"[slow_echo] ENABLE_INTENT_ROUTING: {getattr(config, 'ENABLE_INTENT_ROUTING', False)}")
    print(f"[slow_echo] ENABLE_THINKING: {enable_thinking}")

    # ============ 1) 意图路由层 ============
    intent = "unknown"
    sub_intent = None
    keywords = None
    confidence = 0.0

    purchase_stage = None
    if has_kb and getattr(config, 'ENABLE_INTENT_ROUTING', False):
        try:
            intent_result = quick_ecommerce_intent_hint(message)
            if intent_result == "ambiguous":
                intent_result = classify_ecommerce_intent(message)

            intent = intent_result.get("intent", "unknown")
            sub_intent = intent_result.get("sub_intent")
            keywords = intent_result.get("keywords")
            confidence = intent_result.get("confidence", 0.0)
            purchase_stage = intent_result.get("purchase_stage")
            print(f"[intent] {intent} | sub={sub_intent} | keywords={keywords} | conf={confidence} | stage={purchase_stage}")
        except Exception as e:
            print(f"[slow_echo] [intent-routing] 异常，回退原检索流程: {e}")
            intent = "unknown"
            sub_intent = None
            keywords = None
            confidence = 0.0
            purchase_stage = None
    else:
        print(f"[slow_echo] → 意图路由层未启用（知识库不存在或开关关闭）")

    # ============ 1.5) 转人工触发检查（明确输入/连续追问/超纲业务） ============
    # 1. 明确输入转人工
    msg_normalized = message.strip().lower()
    if msg_normalized in {"转人工", "人工客服", "找人工", "人工", "客服", "人工服务"}:
        yield _transfer_to_human("explicit", _session_facts)
        return

    # 2. 连续追问 ≥3 次
    is_repeat, repeat_count = _detect_repeat_question(message, intent, _session_facts)
    if repeat_count >= 3:
        yield _transfer_to_human("repeat", _session_facts)
        return

    # 3. 超纲业务（unknown 且 confidence < 0.3，连续 2 轮）
    if intent == "unknown" and confidence < 0.3:
        _session_facts["question_repeat_count"] = _session_facts.get("question_repeat_count", 0) + 1
        if _session_facts["question_repeat_count"] >= 2:
            yield _transfer_to_human("out_of_scope", _session_facts)
            return
    elif intent != "unknown":
        _session_facts["question_repeat_count"] = 0

    # ============ 1.6) 状态机延续检查 ============
    current_state = _session_facts.get("dialogue_state", "init")
    last_intent = _session_facts.get("last_intent")
    if current_state in _LOGISTICS_STATES and current_state not in {"init", "completed", "rejected"}:
        if last_intent in (None, "logistics"):
            intent = "logistics"
            sub_intent = sub_intent or _session_facts.get("last_sub_intent")
            print(f"[state-machine] 状态机延续: state={current_state}, intent强制设为logistics, sub={sub_intent}")
    elif current_state in _RECOMMEND_STATES and current_state not in {"init", "completed", "rejected"}:
        if last_intent in (None, "product_recommend"):
            intent = "product_recommend"
            sub_intent = sub_intent or _session_facts.get("last_sub_intent")
            print(f"[state-machine] 状态机延续: state={current_state}, intent强制设为product_recommend, sub={sub_intent}")
    elif current_state in _GOODS_OP_STATES and current_state not in {"init", "completed", "rejected"}:
        if last_intent == "goods_operation":
            intent = "goods_operation"
            sub_intent = sub_intent or _session_facts.get("last_sub_intent")
            print(f"[state-machine] 状态机延续: state={current_state}, intent强制设为goods_operation, sub={sub_intent}")

    # ============ 2) 会话缓存更新 ============
    _update_session_facts(intent_hint=intent, sub_intent=sub_intent, keywords=keywords, purchase_stage=purchase_stage)

    # ============ 3) 分支处理 ============
    results = []
    if intent == "logistics":
        print(f"[branch] 物流查询，子意图={sub_intent}")
        state = _session_facts.get("dialogue_state", "init")

        if state not in _LOGISTICS_STATES:
            state = "init"
            _session_facts["dialogue_state"] = "init"

        if state == "init":
            _session_facts["dialogue_state"] = "awaiting_identity"
            print(f"[logistics-state] init → awaiting_identity")
            yield "📦 为了查询您的物流信息，请提供订单号和手机号。\n格式：订单号 手机号（例如：ORD20250520001 13800138000）"
            return

        elif state == "awaiting_identity":
            success, order = _verify_identity(message, _session_facts)
            if success:
                _session_facts["dialogue_state"] = "identity_verified"
                print(f"[logistics-state] awaiting_identity → identity_verified")
                if sub_intent == "track":
                    _session_facts["dialogue_state"] = "reading_path"
                    print(f"[logistics-state] identity_verified → reading_path")
                    response = _handle_logistics_track(order, message, history, get_openai_client())
                    _session_facts["dialogue_state"] = "completed"
                    print(f"[logistics-state] reading_path → completed")
                    yield response
                elif sub_intent == "abnormal":
                    _session_facts["dialogue_state"] = "checking_status"
                    print(f"[logistics-state] identity_verified → checking_status")
                    response = _handle_logistics_abnormal(order, message, history, get_openai_client())
                    _session_facts["dialogue_state"] = "completed"
                    print(f"[logistics-state] checking_status → completed")
                    yield response
                elif sub_intent == "modify":
                    _session_facts["dialogue_state"] = "awaiting_modify_type"
                    print(f"[logistics-state] identity_verified → awaiting_modify_type")
                    yield "请选择要修改的信息：1.地址 2.电话 3.收件人\n请回复数字或名称"
                    return
                else:
                    # 无子意图默认走 track
                    response = _handle_logistics_track(order, message, history, get_openai_client())
                    _session_facts["dialogue_state"] = "completed"
                    print(f"[logistics-state] identity_verified → completed (默认track)")
                    yield response
            else:
                yield "❌ 订单号或手机号有误，请重新输入。\n格式：订单号 手机号"
                return  # 保持 awaiting_identity

        elif state == "awaiting_modify_type":
            modify_type = _parse_modify_type(message)
            if modify_type:
                _session_facts["modify_type"] = modify_type
                _session_facts["dialogue_state"] = "awaiting_new_info"
                print(f"[logistics-state] awaiting_modify_type → awaiting_new_info (type={modify_type})")
                label = _MODIFY_TYPE_LABELS[modify_type]
                yield f"请输入新的{label}："
                return
            else:
                yield "请选择：1.地址 2.电话 3.收件人"
                return

        elif state == "awaiting_new_info":
            modify_type = _session_facts.get("modify_type")
            valid, error_msg = _validate_new_info(modify_type, message)
            if valid:
                order_id = _session_facts["bound_order_id"]
                _session_facts["dialogue_state"] = "writing_back"
                print(f"[logistics-state] awaiting_new_info → writing_back")
                success, msg = update_order_field(order_id, modify_type, message)
                if success:
                    _session_facts["dialogue_state"] = "completed"
                    print(f"[logistics-state] writing_back → completed")
                    label = _MODIFY_TYPE_LABELS.get(modify_type, modify_type)
                    yield f"✅ 修改成功！您的{label}已更新。"
                else:
                    _session_facts["dialogue_state"] = "rejected"
                    print(f"[logistics-state] writing_back → rejected")
                    yield f"❌ 修改失败：{msg}"
            else:
                yield f"❌ {error_msg}，请重新输入："
                return

        else:
            # 兜底：其他状态直接返回通用提示
            yield "当前物流请求已处理完毕，如需再次查询请重新发起。"
        return  # 物流分支处理完毕，不再走通用 LLM

    # --- goods_operation 分支（Phase 5 状态机）---
    if intent == "goods_operation":
        print(f"[branch] 货物操作，类型={sub_intent}")
        state = _session_facts.get("dialogue_state", "init")
        if state not in _GOODS_OP_STATES:
            state = "init"
            _session_facts["dialogue_state"] = "init"

        # 身份验证阶段
        if not _session_facts.get("verified_identity"):
            if state == "init":
                _session_facts["dialogue_state"] = "awaiting_identity"
                print(f"[goods-op-state] init → awaiting_identity")
                yield "📦 为了处理您的售后申请，请提供订单号和手机号。\n格式：订单号 手机号（例如：ORD20250520001 13800138000）"
                return
            elif state == "awaiting_identity":
                success, order = _verify_identity(message, _session_facts)
                if success:
                    _session_facts["dialogue_state"] = "identity_verified"
                    print(f"[goods-op-state] awaiting_identity → identity_verified")
                    # fall through to reason collection below
                else:
                    yield "❌ 订单号或手机号有误，请重新输入。\n格式：订单号 手机号"
                    return

        # 原因收集阶段
        if _session_facts["dialogue_state"] == "identity_verified":
            _session_facts["dialogue_state"] = "awaiting_reason"
            print(f"[goods-op-state] identity_verified → awaiting_reason")
            yield _prompt_for_reason(sub_intent)
            return

        if _session_facts["dialogue_state"] == "awaiting_reason":
            reason, reason_code = _parse_operation_reason(sub_intent, message)
            if reason:
                _session_facts["operation_reason"] = reason
                next_state = {
                    "return": "awaiting_return_method",
                    "exchange": "awaiting_exchange_spec",
                    "refund_only": "awaiting_refund_amount",
                }.get(sub_intent, "awaiting_reason")
                _session_facts["dialogue_state"] = next_state
                print(f"[goods-op-state] awaiting_reason → {next_state}")
                if sub_intent == "return":
                    yield _prompt_for_return_method()
                    return
                elif sub_intent == "exchange":
                    order = find_order(_session_facts["bound_order_id"], _session_facts["bound_phone"])
                    yield _prompt_for_exchange_spec(order)
                    return
                elif sub_intent == "refund_only":
                    order = find_order(_session_facts["bound_order_id"], _session_facts["bound_phone"])
                    yield _prompt_for_refund_amount(order)
                    return
            else:
                yield _prompt_for_reason(sub_intent)
                return

        # 退货方式收集
        if _session_facts["dialogue_state"] == "awaiting_return_method":
            method = _parse_return_method(message)
            if method:
                _session_facts["operation_detail"] = method
                _session_facts["dialogue_state"] = "validating"
                print(f"[goods-op-state] awaiting_return_method → validating")
                # fall through to validation
            else:
                yield "请选择退货方式：\n1. 上门取件\n2. 自行寄回"
                return

        # 换货规格收集
        if _session_facts["dialogue_state"] == "awaiting_exchange_spec":
            order = find_order(_session_facts["bound_order_id"], _session_facts["bound_phone"])
            after_sales = order.get("after_sales", {}) if order else {}
            inventory = after_sales.get("exchange_inventory", {})
            available_specs = []
            idx = 1
            for color, sizes in inventory.items():
                for size, stock in sizes.items():
                    if stock and stock > 0:
                        available_specs.append((str(idx), color, size, stock))
                        idx += 1
            spec = _parse_exchange_spec(message, available_specs)
            if spec:
                _session_facts["operation_detail"] = spec
                _session_facts["dialogue_state"] = "validating"
                print(f"[goods-op-state] awaiting_exchange_spec → validating")
                # fall through to validation
            else:
                yield "请按格式输入要更换的规格（如：藏青色 M）："
                return

        # 退款金额收集
        if _session_facts["dialogue_state"] == "awaiting_refund_amount":
            order = find_order(_session_facts["bound_order_id"], _session_facts["bound_phone"])
            items = order.get("items", []) if order else []
            total = sum(item.get("price", 0) * item.get("quantity", 1) for item in items) if items else 0
            if total == 0:
                total = order.get("total_amount", 0) if order else 0
            amount, label = _parse_refund_amount(message, total)
            if amount is not None:
                _session_facts["operation_detail"] = {"amount": amount, "label": label}
                _session_facts["dialogue_state"] = "validating"
                print(f"[goods-op-state] awaiting_refund_amount → validating")
                # fall through to validation
            else:
                yield f"❌ {label}\n请重新输入退款金额："
                return

        # 校验 + 写回 + 生成
        if _session_facts["dialogue_state"] == "validating":
            order = find_order(_session_facts["bound_order_id"], _session_facts["bound_phone"])
            reason = _session_facts.get("operation_reason")
            detail = _session_facts.get("operation_detail")
            is_valid, msg = _validate_goods_operation(order, sub_intent, reason, detail, _session_facts)
            if is_valid:
                _session_facts["dialogue_state"] = "writing_back"
                print(f"[goods-op-state] validating → writing_back")
                success, wb_msg = _write_back_goods_operation(
                    _session_facts["bound_order_id"], sub_intent, reason, detail, _session_facts
                )
                if success:
                    _session_facts["dialogue_state"] = "generating_response"
                    print(f"[goods-op-state] writing_back → generating_response")
                    messages = _build_goods_operation_prompt(sub_intent, reason, detail, order, True)
                    yield from _yield_llm_stream(messages, enable_thinking)
                    _session_facts["dialogue_state"] = "completed"
                    print(f"[goods-op-state] generating_response → completed")
                else:
                    _session_facts["dialogue_state"] = "rejected"
                    print(f"[goods-op-state] writing_back → rejected")
                    yield f"❌ 申请提交失败：{wb_msg}"
            else:
                _session_facts["dialogue_state"] = "rejected"
                print(f"[goods-op-state] validating → rejected")
                messages = _build_goods_operation_prompt(sub_intent, reason, detail, order, False, msg)
                yield from _yield_llm_stream(messages, enable_thinking)
            return

        if _session_facts["dialogue_state"] in ("writing_back", "generating_response", "completed", "rejected"):
            yield "当前售后申请已处理完毕，如有其他问题请重新发起。"
            return

    # --- product_recommend 分支（Phase 3 状态机）---
    if intent == "product_recommend":
        print(f"[branch] 商品推荐，类型={sub_intent}")
        state = _session_facts.get("dialogue_state", "init")
        if state not in _RECOMMEND_STATES:
            state = "init"
            _session_facts["dialogue_state"] = "init"

        # 提取当前商品
        current_product = _extract_current_product(message, _session_facts)
        if current_product:
            if current_product not in _session_facts["product_history"]:
                _session_facts["product_history"].append(current_product)
            print(f"[recommend] 当前商品: {current_product}")
        else:
            print(f"[recommend] 当前商品: 未明确（将尝试通用推荐）")

        if state == "init":
            if _session_facts.get("price_range"):
                if sub_intent == "matching" and (not _session_facts.get("matching_scene") or not _session_facts.get("matching_usage")):
                    _session_facts["dialogue_state"] = "awaiting_matching_scene"
                    print(f"[recommend-state] init → awaiting_matching_scene")
                    yield "💡 为了给您更精准的搭配推荐，请描述您的使用场景（例如：上班通勤、周末约会、运动健身）："
                    return
                else:
                    _session_facts["dialogue_state"] = "retrieving"
            else:
                _session_facts["dialogue_state"] = "awaiting_price_range"
                _session_facts["recommend_sub_intent"] = sub_intent
                print(f"[recommend-state] init → awaiting_price_range")
                yield "💰 请告诉我您能接受的价格区间？\n（例如：200-500元、300以内、500左右）"
                return

        elif state == "awaiting_price_range":
            success, result = _collect_price_range(message, _session_facts)
            if success:
                sub_intent = _session_facts.get("recommend_sub_intent", sub_intent)
                if sub_intent == "matching":
                    _session_facts["dialogue_state"] = "awaiting_matching_scene"
                    print(f"[recommend-state] awaiting_price_range → awaiting_matching_scene")
                    yield "💡 为了给您更精准的搭配推荐，请描述您的使用场景（例如：上班通勤、周末约会、运动健身）："
                    return
                else:
                    _session_facts["dialogue_state"] = "retrieving"
                    # 继续执行下面检索逻辑
            else:
                yield f"❌ {result}\n请重新输入价格区间："
                return

        elif state == "awaiting_matching_scene":
            success, scene = _collect_matching_scene(message, _session_facts)
            if success:
                _session_facts["matching_scene"] = scene
                _session_facts["dialogue_state"] = "awaiting_matching_usage"
                print(f"[recommend-state] awaiting_matching_scene → awaiting_matching_usage")
                yield "👤 请描述您的使用习惯或特殊需求（例如：经常出差需要轻便、皮肤敏感需要天然材质）："
                return
            else:
                yield "请描述您的使用场景："
                return

        elif state == "awaiting_matching_usage":
            success, usage = _collect_matching_usage(message, _session_facts)
            if success:
                _session_facts["matching_usage"] = usage
                _session_facts["dialogue_state"] = "retrieving"
                # 继续执行下面检索逻辑
            else:
                yield "请描述您的使用需求："
                return

        # 检索 + 生成（retrieving / generating 统一处理）
        if _session_facts["dialogue_state"] in ("retrieving", "init"):
            _session_facts["dialogue_state"] = "retrieving"
            sub_intent = _session_facts.get("recommend_sub_intent", sub_intent)
            price_range = _session_facts.get("price_range", {})

            query = _build_recommend_query(
                current_product,
                sub_intent,
                price_range,
                _session_facts.get("matching_scene"),
                _session_facts.get("matching_usage"),
            )
            results = _retrieve_for_recommendation(query, top_k=5)

            _session_facts["dialogue_state"] = "generating"
            context = "\n\n".join([f"【{r[1]}】\n{r[2][:300]}" for r in results])

            prompt = _build_recommend_prompt(
                current_product,
                sub_intent,
                price_range,
                context,
                _session_facts.get("matching_scene"),
                _session_facts.get("matching_usage"),
            )

            messages = [
                {"role": "system", "content": "你是专业的电商商品推荐助手。"},
                {"role": "user", "content": prompt},
            ]
            _session_facts["dialogue_state"] = "completed"
            print(f"[recommend-state] generating → completed")
            yield from _yield_llm_stream(messages, enable_thinking)
        return  # 推荐分支处理完毕

    # --- product_info 分支（Phase 4）---
    if intent == "product_info":
        print(f"[branch] 商品信息查询，模块={keywords}, stage={purchase_stage}")
        try:
            results = retrieve_product_info(message, current_db_path, target_module=keywords, top_k=5)
            context = "\n\n".join([f"【{r[1]}】\n{r[2][:400]}" for r in results]) if results else "（未检索到相关商品信息）"
            prompt = _build_product_info_prompt(context, keywords, purchase_stage, message)
            messages = [
                {"role": "system", "content": "你是专业的电商商品咨询助手。"},
                {"role": "user", "content": prompt},
            ]
            _session_facts["dialogue_state"] = "completed"
            print(f"[product-info-state] generating → completed")
            yield from _yield_llm_stream(messages, enable_thinking)
        except Exception as e:
            print(f"[product-info] 处理异常: {e}")
            import traceback
            traceback.print_exc()
            yield "抱歉，商品信息查询出现问题，请稍后再试。"
        return

    # --- 非 logistics / 非 product_recommend / 非 product_info 分支（原 Phase 1 逻辑）---
    if has_kb:
        if intent == "goods_operation":
            print(f"[branch] 货物操作，类型={sub_intent}")
        elif intent == "chitchat":
            print(f"[branch] 闲聊")
        else:
            print(f"[branch] 未知意图，走通用兜底")

        if intent != "chitchat":
            try:
                results = retrieve_with_context(message, current_db_path, context_drug=None, top_k=3)
                print(f"[slow_echo] → FAISS 返回 {len(results)} 条结果")
                for rank, r in enumerate(results):
                    print(f"[slow_echo]   raw[{rank}] id='{r[0]}' title='{r[1]}'")
            except Exception as e:
                print(f"[slow_echo] → Retrieval error: {e}")
                import traceback
                traceback.print_exc()
                results = []
        else:
            results = []  # 闲聊不走 RAG

    if results:
        context = "\n".join([f"【{r[1]}】\n{r[2]}" for r in results])
    elif has_kb:
        context = "No context found or error in search."
    else:
        context = "（当前未加载商品知识库，将直接基于模型能力回答。）"

    print(f"[slow_echo] → 组装 context（前200字符）:\n{context[:200]}...")

    # ============ 4) Prompt 组装 ============
    if has_kb:
        base_prompt = "你是一个专业的电商客服助手。请根据提供的上下文回答用户的问题。如果上下文不相关，请根据你自己的知识回答，但要说明上下文不相关。"
        if keywords:
            system_prompt = (
                base_prompt
                + f"\n用户正在查询以下商品属性：{keywords}。"
                + "请优先基于提供的上下文回答这些属性，如果上下文中缺少某字段信息，请明确说明。"
            )
        else:
            system_prompt = base_prompt
        user_prompt = f"上下文:\n{context}\n\n问题: {message}"
    else:
        system_prompt = "你是一个专业的电商客服助手。当前未加载商品知识库，请基于你的通用知识回答用户问题。"
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

    yield from _yield_llm_stream(messages, enable_thinking)

    # 更新会话事实缓存，供下一轮使用
    _update_session_facts(intent_hint=intent, sub_intent=sub_intent, keywords=keywords, purchase_stage=purchase_stage)
    print(f"[session-facts] 更新: last_intent={_session_facts['last_intent']}, "
          f"last_sub_intent={_session_facts.get('last_sub_intent')}, "
          f"keywords={_session_facts.get('keywords')}, "
          f"purchase_stage={_session_facts.get('purchase_stage')}")

    print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Phase 7：Gradio UI 改造与交互弹窗化
# ---------------------------------------------------------------------------

def get_session_status():
    """提取可展示的会话状态，供状态面板使用。"""
    return {
        "当前意图": _session_facts.get("last_intent", "-"),
        "子意图": _session_facts.get("last_sub_intent", "-"),
        "对话状态": _session_facts.get("dialogue_state", "-"),
        "绑定订单": _session_facts.get("bound_order_id", "-"),
        "验证身份": "✅" if _session_facts.get("verified_identity") else "❌",
        "价格区间": _session_facts.get("price_range", {}).get("text", "-") if _session_facts.get("price_range") else "-",
        "售后原因": _session_facts.get("operation_reason", "-"),
    }


def _get_visible_components(dialogue_state, last_intent):
    """
    根据当前状态返回应该显示的组件。
    返回 dict: {"identity": bool, "reason": bool, "return_method": bool, "price_range": bool, "normal": bool}
    """
    vis = {
        "identity": False,
        "reason": False,
        "return_method": False,
        "price_range": False,
        "normal": True,
    }
    if dialogue_state == "awaiting_identity":
        vis["identity"] = True
        vis["normal"] = False
    elif dialogue_state == "awaiting_reason":
        vis["reason"] = True
        vis["normal"] = False
    elif dialogue_state == "awaiting_return_method":
        vis["return_method"] = True
        vis["normal"] = False
    elif dialogue_state == "awaiting_price_range":
        vis["price_range"] = True
        vis["normal"] = False
    return vis


def _respond(message, history, enable_thinking):
    """
    包装 slow_echo，增加状态面板更新与条件渲染控制。
    返回 generator，yield (chatbot, status, identity_vis, reason_vis, return_method_vis, price_range_vis, normal_vis)
    """
    history = list(history) if history else []
    history.append([message, ""])

    # 初始 yield（使用当前状态）
    status = get_session_status()
    vis = _get_visible_components(_session_facts.get("dialogue_state", "init"), _session_facts.get("last_intent"))
    yield history, status, gr.update(visible=vis["identity"]), gr.update(visible=vis["reason"]), gr.update(visible=vis["return_method"]), gr.update(visible=vis["price_range"]), gr.update(visible=vis["normal"])

    # 流式输出
    for chunk in slow_echo(message, history, enable_thinking):
        history[-1][1] = chunk
        yield history, status, gr.update(visible=vis["identity"]), gr.update(visible=vis["reason"]), gr.update(visible=vis["return_method"]), gr.update(visible=vis["price_range"]), gr.update(visible=vis["normal"])

    # 最终 yield（根据 slow_echo 结束后的新状态更新可见性）
    status = get_session_status()
    vis = _get_visible_components(_session_facts.get("dialogue_state", "init"), _session_facts.get("last_intent"))
    yield history, status, gr.update(visible=vis["identity"]), gr.update(visible=vis["reason"]), gr.update(visible=vis["return_method"]), gr.update(visible=vis["price_range"]), gr.update(visible=vis["normal"])


def on_send(message, history, enable_thinking):
    """普通发送按钮事件。"""
    if not message or not message.strip():
        return
    yield from _respond(message.strip(), history, enable_thinking)


def on_identity_confirm(order_id, phone, history, enable_thinking):
    """身份验证弹窗确认。"""
    message = f"{order_id.strip()} {phone.strip()}"
    yield from _respond(message, history, enable_thinking)


def on_reason_confirm(reason_value, history, enable_thinking):
    """原因选择弹窗确认。"""
    if reason_value and "." in reason_value:
        message = reason_value.split(".")[0].strip()
    else:
        message = reason_value or ""
    yield from _respond(message, history, enable_thinking)


def on_return_method_confirm(method_value, history, enable_thinking):
    """退货方式弹窗确认。"""
    message = method_value or ""
    yield from _respond(message, history, enable_thinking)


def on_price_confirm(min_val, max_val, history, enable_thinking):
    """价格区间弹窗确认。"""
    message = f"{int(min_val)}-{int(max_val)}"
    yield from _respond(message, history, enable_thinking)


def rebuild_product_index(input_dir):
    """调用 build_product_index 重建商品向量索引。"""
    try:
        import build_product_index
        output_path = getattr(config, "PRODUCT_INDEX_PATH", "./products.npz")
        build_product_index.build_product_vector_index(input_dir, output_path)
        return f"索引重建完成：{output_path}"
    except Exception as e:
        return f"索引重建失败：{e}"


def preview_orders(orders_path=None):
    """加载订单 JSON 并返回前 10 条预览。"""
    path = orders_path or getattr(config, "ORDERS_JSON_PATH", "./orders/orders.json")
    try:
        orders = load_orders(path)
        preview = []
        for o in orders[:10]:
            preview.append({
                "订单号": o.get("order_id", ""),
                "手机号": o.get("user_phone", ""),
                "状态": o.get("status", ""),
                "金额": o.get("total_amount", ""),
            })
        return preview
    except Exception as e:
        return [{"订单号": f"加载失败: {e}"}]


def update_ecommerce_config(es_host, es_port, es_user, es_pass, es_index, vector_db, es_scheme,
                            product_kb_path, product_index_path, orders_json_path, return_window_days):
    """
    更新配置信息（Phase 7 扩展版，包含电商配置）。
    """
    config.ES_HOST = es_host
    config.ES_PORT = int(es_port)
    config.ES_USER = es_user
    config.ES_PASSWORD = es_pass
    config.ES_INDEX = es_index
    config.VECTOR_DB_PATH = vector_db
    config.ES_SCHEME = es_scheme
    config.PRODUCT_KB_PATH = product_kb_path
    config.PRODUCT_INDEX_PATH = product_index_path
    config.ORDERS_JSON_PATH = orders_json_path
    config.RETURN_WINDOW_DAYS = int(return_window_days) if return_window_days else 7
    clear_es_cache()
    clear_openai_client_cache()
    return "配置已更新! (注意: 重启后将重置为配置文件默认值)"


# ---------------------------------------------------------------------------
# Gradio UI 定义
# ---------------------------------------------------------------------------

with gr.Blocks(title="智能电商客服系统") as demo:
    
    # --- Tab 1: 智能客服 ---
    with gr.Tab("智能客服"):
        gr.Markdown("## 🤖 智能电商客服系统")
        with gr.Row():
            # 左侧：会话状态面板
            with gr.Column(scale=1):
                gr.Markdown("### 会话状态")
                status_json = gr.JSON(
                    label="当前状态",
                    value=get_session_status,
                )
            # 右侧：聊天区域
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="对话记录")
                enable_thinking_checkbox = gr.Checkbox(
                    label="显示模型思考过程",
                    value=getattr(config, 'ENABLE_THINKING', True),
                )

                # 动态输入区：身份验证弹窗
                with gr.Column(visible=False) as identity_col:
                    gr.Markdown("**📦 请验证身份**")
                    with gr.Row():
                        order_id_input = gr.Textbox(label="订单号", placeholder="ORD20250520001")
                        phone_input = gr.Textbox(label="手机号", placeholder="13800138000")
                    identity_btn = gr.Button("确认身份", variant="primary")

                # 动态输入区：原因选择弹窗
                with gr.Column(visible=False) as reason_col:
                    gr.Markdown("**📝 请选择原因**")
                    reason_radio = gr.Radio(
                        choices=["1. 不想要了", "2. 质量问题", "3. 描述不符", "4. 其他"],
                        label="售后原因"
                    )
                    reason_btn = gr.Button("确认原因", variant="primary")

                # 动态输入区：退货方式弹窗
                with gr.Column(visible=False) as return_method_col:
                    gr.Markdown("**📮 请选择退货方式**")
                    return_method_radio = gr.Radio(
                        choices=["1. 上门取件", "2. 自行寄回"],
                        label="退货方式"
                    )
                    return_method_btn = gr.Button("确认方式", variant="primary")

                # 动态输入区：价格区间弹窗
                with gr.Column(visible=False) as price_range_col:
                    gr.Markdown("**💰 请输入价格区间**")
                    with gr.Row():
                        price_min = gr.Number(label="最低价格", value=0, minimum=0)
                        price_max = gr.Number(label="最高价格", value=1000, minimum=0)
                    price_btn = gr.Button("确认价格", variant="primary")

                # 普通输入区
                with gr.Column(visible=True) as normal_input_col:
                    msg_input = gr.Textbox(label="输入消息", placeholder="请输入您的问题...")
                    with gr.Row():
                        send_btn = gr.Button("发送", variant="primary")
                        clear_btn = gr.Button("清空对话")

        # 事件绑定
        send_outputs = [chatbot, status_json, identity_col, reason_col, return_method_col, price_range_col, normal_input_col]

        send_btn.click(
            fn=on_send,
            inputs=[msg_input, chatbot, enable_thinking_checkbox],
            outputs=send_outputs,
        )

        identity_btn.click(
            fn=on_identity_confirm,
            inputs=[order_id_input, phone_input, chatbot, enable_thinking_checkbox],
            outputs=send_outputs,
        )

        reason_btn.click(
            fn=on_reason_confirm,
            inputs=[reason_radio, chatbot, enable_thinking_checkbox],
            outputs=send_outputs,
        )

        return_method_btn.click(
            fn=on_return_method_confirm,
            inputs=[return_method_radio, chatbot, enable_thinking_checkbox],
            outputs=send_outputs,
        )

        price_btn.click(
            fn=on_price_confirm,
            inputs=[price_min, price_max, chatbot, enable_thinking_checkbox],
            outputs=send_outputs,
        )

        clear_btn.click(
            fn=lambda: (None, get_session_status(), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)),
            outputs=send_outputs,
        )

    # --- Tab 2: 知识库管理 ---
    with gr.Tab("知识库管理"):
        gr.Markdown("## 📚 商品索引管理")
        with gr.Row():
            kb_dir_input = gr.Textbox(
                label="商品文档目录",
                value=getattr(config, "PRODUCT_KB_PATH", "./products/"),
                scale=3,
            )
            rebuild_btn = gr.Button("重建索引", variant="primary", scale=1)
        rebuild_status = gr.Textbox(label="索引状态", interactive=False)
        rebuild_btn.click(fn=rebuild_product_index, inputs=kb_dir_input, outputs=rebuild_status)

        gr.Markdown("## 📋 订单数据预览")
        with gr.Row():
            orders_path_input = gr.Textbox(
                label="订单 JSON 路径",
                value=getattr(config, "ORDERS_JSON_PATH", "./orders/orders.json"),
                scale=3,
            )
            refresh_orders_btn = gr.Button("刷新订单列表", variant="primary", scale=1)
        orders_df = gr.Dataframe(
            headers=["订单号", "手机号", "状态", "金额"],
            label="前10条订单",
            interactive=False,
        )
        refresh_orders_btn.click(fn=preview_orders, inputs=orders_path_input, outputs=orders_df)

    # --- Tab 3: 配置 ---
    with gr.Tab("配置"):
        gr.Markdown("## ⚙️ 系统配置")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ES 与向量库")
                es_host_input = gr.Textbox(label="ES主机地址", value=config.ES_HOST)
                es_port_input = gr.Textbox(label="ES服务端口", value=str(config.ES_PORT))
                es_user_input = gr.Textbox(label="ES用户名", value=config.ES_USER)
                es_pass_input = gr.Textbox(label="ES密码", type="password", value=config.ES_PASSWORD)
                es_index_input = gr.Textbox(label="ES索引名", value=config.ES_INDEX)
                vector_db_path_input = gr.Textbox(label="向量数据库位置", value=config.VECTOR_DB_PATH)
                es_scheme_input = gr.Textbox(label="ES协议 (http/https)", value=config.ES_SCHEME)
            with gr.Column():
                gr.Markdown("### 电商配置")
                product_kb_input = gr.Textbox(label="商品知识库目录", value=getattr(config, "PRODUCT_KB_PATH", "./products/"))
                product_index_input = gr.Textbox(label="商品索引路径", value=getattr(config, "PRODUCT_INDEX_PATH", "./products.npz"))
                orders_json_input = gr.Textbox(label="订单JSON路径", value=getattr(config, "ORDERS_JSON_PATH", "./orders/orders.json"))
                return_window_input = gr.Number(label="退货期限（天）", value=getattr(config, "RETURN_WINDOW_DAYS", 7), minimum=1)

        config_submit = gr.Button("保存配置", variant="primary")
        config_message = gr.Textbox(label="状态", interactive=False)
        config_submit.click(
            fn=update_ecommerce_config,
            inputs=[
                es_host_input, es_port_input, es_user_input, es_pass_input,
                es_index_input, vector_db_path_input, es_scheme_input,
                product_kb_input, product_index_input, orders_json_input, return_window_input,
            ],
            outputs=config_message,
        )


# 启动应用
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)

