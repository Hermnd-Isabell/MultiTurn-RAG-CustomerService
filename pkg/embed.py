import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import json
import hashlib
from langchain.chains import RetrievalQA
# from zhipuai import ZhipuAI # Unused
import re
from elasticsearch import Elasticsearch, exceptions
# from dotenv import load_dotenv, find_dotenv # Handled in config.py
from openai import OpenAI

# load_dotenv(find_dotenv())
from config import config

# OpenAI 客户端懒加载缓存：避免在模块导入阶段就读取 config.OPENAI_API_KEY 等值，
# 否则 Gradio "配置" tab 修改凭证后，旧 client 仍带启动时的 key 发请求。
_openai_client = None


def get_openai_client():
    """懒加载 OpenAI 客户端：首次调用时按当前 config 实例化并缓存。"""
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    _openai_client = OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )
    return _openai_client


def clear_openai_client_cache():
    """清除已缓存的 OpenAI 客户端（如配置变更后调用，下次请求按新凭证重建）。"""
    global _openai_client
    _openai_client = None


model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
_faiss_cache = {}

def clear_faiss_cache(embedding_file_path=None):
    global _faiss_cache
    if embedding_file_path and embedding_file_path in _faiss_cache:
        del _faiss_cache[embedding_file_path]
    elif not embedding_file_path:
        _faiss_cache.clear()

class MedicineInfoStandardizer:# 药物信息标准化器
    field_list=["药物名","类别", "鉴别", "贮藏", "指纹图谱", "功能主治", "规格", #文档中会出现的小标题
                             "含量测定",  "性味与归经", "浸岀物", 
                             "规定", "制法",  "检査", "用法与用量",
                             "用途",  "触藏", "正丁醇提取物", "特征图谱","禁忌", 
                              "效价测定", "正丁醇浸出物", 
                             "注意事项", "功能与主治", "制剂",
                             "性状","挥发油","处方", 
                             "适应症"]
    def __init__(self, llm=None):
        """
        初始化方法，存储语言模型实例。

        :param llm: 可选的语言模型实例；不传则懒加载共享 OpenAI 客户端，
                    保证配置热更新后下次实例化能用到新凭证。
        """
        # 引用类属性，避免触发 NameError（模块级不存在 field_list）
        self.field_list = MedicineInfoStandardizer.field_list
        # 单一来源：默认走懒加载客户端，外部可传入自定义 llm 覆盖（测试 / 多租户场景）
        self.llm = llm if llm is not None else get_openai_client()
    # 药物信息字段列表
    
    def bzh(self,input_data):# 从问题中提取字段列表对应的信息，标准化后输出。
        """
        从问题中提取字段列表对应的信息，如果没有对应信息则为空。

        :param question: 输入的问题。
        :param field_list: 字段列表。
        :return: 一个字典，键为字段名，值为从问题中提取出的对应信息或空字符串。
        """
        text='''问题一般是药品相关的问题，所以字段可能会有近义语句，
        比如制法即是制作方法,有重量的是处方的一部分，“用*制作而成”*也一般是处方，处方中通常只有药材名例如“板蓝根，罂粟壳”，无其他说明
        只有提到的字段才可以出现'''
        all_fields_str = ", ".join(self.field_list)
        extract_template = f"""你是个语意理解大师，你需要充分理解问题中的内容含义，他的问题提到了哪些信息，他的问题通常答案指向一种药物的名称，所以问题中提到的药物有克数的一般为处方中的内容。你需要把问题中的信息分类到字段中提到的内容中
        问题：{input_data}
        字段：{all_fields_str}
        
        用中文回复以及中文字符，回复时参考以下格式，比如 处方：板蓝根1500g,大青叶2250g。将涉及的字段与信息全部输出，顺序为字段名，提到的字段，同时，未提到的字段不需要输出字段名：信息
        额外信息：{text}
        未涉及的字段一定不要提到。一定不要出现“字段：None”的类似句子通常来说问题中只有2-3个字段内容，确保你不会输出超过3个字段，字段间换行输出
        """
        response = self.llm.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": extract_template}]
        )
        
        answer = response.choices[0].message.content.strip() if response.choices else "无法提供答案"
        return answer
    
    def standardize_information(self, input_string):#将输入信息标准化
        """
        使用大模型处理输入信息并进行标准化。

        :param input_string: 输入的药物信息字符串。
        :return: 标准化后的字段信息。
        """
        
        extract_template = f"""从以下药物信息中总结字段，回答这句话需要使用到哪些字段，以及该药品的药品名，不用赘述其他：
        {input_string}
        字段列表：{', '.join(self.field_list)}
        
        请返回以下格式的结果：
        提到的药品名：药品名
        标准化输出：
        字段名
        字段名
        例如：
        提到的药品名：八角茴香
        标准化输出：
        功能主治
        性状
        """
        
        # 调用大模型生成标准化输出
        response = self.llm.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": extract_template}]
        )
        standardized_output = response.choices[0].message.content.strip() if response.choices else "无输出"
        # 提取返回内容
        final_output = standardized_output
        return final_output

    def extract_target_fields(self, input_data):
        """
        从用户问句中提取其关心的字段名列表（仅返回字段名，不含 value）。
        内部复用 standardize_information + extract_drug_info：
          1) LLM 输出 "提到的药品名：X\n标准化输出：\n字段A\n字段B" 形式的字符串；
          2) extract_drug_info 解析出 (drugs, [outputs])；
          3) 扁平化、去重，并与 self.field_list 求交集，过滤 LLM 幻觉出的非法字段。
        返回 list[str]，空列表表示"未识别到任何合法字段"。LLM 失败一律返回 []，由调用方走兜底分支。
        """
        try:
            raw = self.standardize_information(input_data)
        except Exception as e:
            print(f"[extract_target_fields] standardize_information 调用失败: {e}")
            return []

        try:
            _, outputs_list = extract_drug_info(raw)
        except Exception as e:
            print(f"[extract_target_fields] 解析 LLM 输出失败: {e}")
            return []

        valid = set(self.field_list)
        flat = []
        for outputs in outputs_list:
            for line in outputs:
                line = line.strip()
                if line in valid and line not in flat:
                    flat.append(line)
        return flat


# 预编译字段关键词 pattern：从 MedicineInfoStandardizer.field_list 拼一个 OR 正则，
# 用于 quick_intent_hint 在不调用 LLM 的前提下嗅探"用户问的是不是某字段"。
_FIELD_KEYWORD_PATTERN = re.compile(
    '|'.join(re.escape(f) for f in MedicineInfoStandardizer.field_list)
)

# 明显闲聊 / 无信息量短语的小白名单，命中即可短路掉两次 LLM 调用。
_OBVIOUS_CHITCHAT_KEYWORDS = {
    '你好', '您好', 'hi', 'hello', '在吗', '在', '谢谢', '感谢',
    '再见', '拜拜', '晚安', '早上好', '哈喽', 'thx', 'thanks',
}


def quick_intent_hint(input_data):
    """
    用规则做轻量意图预判，避免在每次问答都触发 1~2 次 LLM 调用。

    返回值：
      - 'obvious_chitchat'：明显闲聊（短问候/超短文本），调用方应直接走自由问答；
      - 'obvious_pharmacy'：含 field_list 关键词，调用方可跳过 classify 直接抽字段；
      - 'ambiguous'：规则无法决断，调用方应走完整 classify_pharmacy_query 流程。
    """
    if not input_data:
        return 'obvious_chitchat'
    text = input_data.strip()
    if not text:
        return 'obvious_chitchat'
    if len(text) <= 8 and text.lower() in _OBVIOUS_CHITCHAT_KEYWORDS:
        return 'obvious_chitchat'
    if _FIELD_KEYWORD_PATTERN.search(text):
        return 'obvious_pharmacy'
    return 'ambiguous'


def classify_pharmacy_query(input_data):# 检查用户输入的问题是否与药品或药学相关。
        """
        判断用户输入的问题是否与药品或药学相关。

        :param input_data: 用户输入的问题字符串。
        :return: "good"（药学相关问题）或 "bad"（非药学相关问题）。
        """
        classify_template = f"""你是药学专家，请判断以下问题是否与药品或药学相关：
        {input_data}

        如果这是一个与药品或药学相关的问题（如提问药品的功能、用途、副作用等），返回 "good"；
        如果不是药学相关的问题，返回 "bad"。
        只能返回“good”或者“bad”
        """
        
        response =  get_openai_client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": classify_template}]
        )
        
        query_type = response.choices[0].message.content.strip() if response.choices else "未知类型"
        print(query_type)
        
        # 确保只返回 "good" 或 "bad"
        if query_type not in ["good", "bad"]:
            return "未知类型"
        
        return query_type

def extract_subsections(content):# 提取小标题和内容

    pattern = re.compile(r'(?:【|t)(.+?)(?:】)')
    matches = pattern.finditer(content)
    
    subsections = {}
    last_position = 0
    last_title = None
    
    for match in matches:
        title = match.group().strip('【】t]')
        if last_title:  
            subsections[last_title] = content[last_position:match.start()].strip()
        
        last_title = title
        last_position = match.end()

    if last_title:
        subsections[last_title] = content[last_position:].strip()

    return subsections

def retrieve_data_from_es(index_name):# 从Elasticsearch中检索数据
    es_instance = connect_elasticsearch()
    if not es_instance:
        print("无法连接 ES")
        return []
    res = es_instance.search(index=index_name, body={"query": {"match_all": {}}, "size": 10000})
    return res['hits']['hits']

def _es_index_fingerprint(hits):
    """
    根据 ES 命中列表计算稳定指纹：md5(sorted(_id 列表) + count)。
    纯函数，不发起额外 ES 查询，由调用方一次性 fetch 后复用 hits 即可。
    返回 (fingerprint_hex_or_None, doc_count)。空列表返回 (None, 0)。
    """
    if not hits:
        return None, 0
    try:
        ids = sorted(hit['_id'] for hit in hits)
        payload = json.dumps({'count': len(ids), 'ids': ids}, ensure_ascii=False).encode('utf-8')
        return hashlib.md5(payload).hexdigest(), len(ids)
    except Exception as e:
        print(f"[fingerprint] 计算 ES 指纹失败: {e}")
        return None, len(hits)


def process_and_vectorize(index_name, embedding_file_path, force_rebuild=False):# 处理并向量化Elasticsearch中的数据(存储过程中的）
    """
    处理并向量化 Elasticsearch 中的数据，支持基于 metadata 的脏检测。

    :param index_name: ES 索引名
    :param embedding_file_path: 目标 .npz 路径
    :param force_rebuild: 显式强制重建（如重新上传文档时调用方应传 True）

    行为：
    - 单次 ES 全量拉取，先判定 .npz metadata 是否仍对齐（指纹一致 → 跳过）；
    - 若 metadata 缺失（旧版 .npz）或不一致（ES 已变化），视为脏数据并基于同一份 hits 重建；
    - force_rebuild=True 时无条件重建并刷新 _faiss_cache。
    """
    es_instance = connect_elasticsearch()
    if not es_instance:
        print("无法连接 ES，跳过向量化")
        return

    # 一次 ES 查询完成"指纹比对 + 重建数据源"双重职责，避免 rebuild 路径上的重复请求
    entries = retrieve_data_from_es(index_name)
    current_fp, current_count = _es_index_fingerprint(entries)

    # 1) 已存在 .npz 且未强制重建：做 metadata 校验
    if os.path.exists(embedding_file_path) and not force_rebuild:
        try:
            existing = np.load(embedding_file_path, allow_pickle=True)
            saved_index = str(existing['es_index_name']) if 'es_index_name' in existing.files else None
            saved_fp = str(existing['es_fingerprint']) if 'es_fingerprint' in existing.files else None
            if saved_index == index_name and saved_fp and current_fp and saved_fp == current_fp:
                print(f"FAISS 索引已是最新（fingerprint={saved_fp[:8]}…），跳过重建")
                return
            print("检测到 ES 数据已变化或 .npz 缺少 metadata，将重建向量库")
        except Exception as e:
            print(f"读取 .npz metadata 失败({e})，将重建")

    # 2) 走完整重建流程（复用上面已 fetch 的 entries）
    print("Processing and vectorizing data...")
    if not entries:
        print("No entries found in ES.")
        return

    subsections_list = []
    ids = []
    texts = []  # 存储小标题和对应文本的列表

    # 遍历所有文档条目
    for entry in entries:
        doc_id = entry['_id']  # 文档ID
        content = entry['_source']['content']  # 文档内容
        subsections = extract_subsections(content)  # 提取文档的小节

        print(f"Document ID: {doc_id} - Content Length: {len(content)}")  # 打印文档ID和内容长度

        # 为每个小节存储对应的ID和文本
        for title, text in subsections.items():
            subsections_list.append((doc_id, title, text))  # 文档ID, 标题, 内容
            ids.append(doc_id)  # 存储文档ID
            texts.append((title, text))  # 存储小标题和对应文本

    if not subsections_list:
        print("No valid subsections found to vectorize.")
        return

    # 向量化小节内容
    embeddings = model.encode([text for _, _, text in subsections_list], convert_to_numpy=True)

    # 写入向量 + metadata（向后兼容：旧版 .npz 没有 es_* 字段会被识别为脏并重建）
    np.savez_compressed(
        embedding_file_path,
        embeddings=embeddings,
        ids=ids,
        texts=texts,
        es_index_name=np.array(index_name),
        es_fingerprint=np.array(current_fp or ''),
    )
    clear_faiss_cache(embedding_file_path)
    fp_disp = (current_fp[:8] + '…') if current_fp else 'N/A'
    print(f"Data processed and saved (docs={current_count}, fp={fp_disp}).")

def retrieve_vector_and_text(input_data, embedding_file_path, top_k=1):
    """
    将输入文字向量化并在本地数据库中进行向量检索，同时返回检索到的文本。
    """
    global _faiss_cache
    print(f"[retrieve] input='{input_data}' | top_k={top_k} | file={embedding_file_path}")

    if not os.path.exists(embedding_file_path):
        raise FileNotFoundError(f"Embedding file not found at: {embedding_file_path}")

    query_embedding = model.encode([input_data], convert_to_numpy=True)
    print(f"[retrieve] query_embedding shape={query_embedding.shape} dtype={query_embedding.dtype}")

    if embedding_file_path in _faiss_cache:
        index, ids, texts = _faiss_cache[embedding_file_path]
        print(f"[retrieve] FAISS cache HIT")
    else:
        print(f"[retrieve] FAISS cache MISS — loading npz...")
        data = np.load(embedding_file_path, allow_pickle=True)
        embeddings = data['embeddings']
        ids = data['ids']
        texts = data['texts']
        dimension = embeddings.shape[1]
        print(f"[retrieve] npz loaded: embeddings shape={embeddings.shape} ids count={len(ids)}")
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))
        _faiss_cache[embedding_file_path] = (index, ids, texts)
        print(f"[retrieve] FAISS index built: ntotal={index.ntotal}")

    D, I = index.search(query_embedding.astype(np.float32), top_k)
    print(f"[retrieve] FAISS distances (D)={D[0].tolist()}")
    print(f"[retrieve] FAISS indices  (I)={I[0].tolist()}")

    retrieved_ids = ids[I[0]].tolist()
    retrieved_texts = [texts[i] for i in I[0]]
    results = [(retrieved_ids[i], retrieved_texts[i][0], retrieved_texts[i][1]) for i in range(top_k)]

    for rank, (doc_id, title, text_snip) in enumerate(results):
        print(f"[retrieve] result[{rank}] doc_id='{doc_id}' title='{title}' text[:60]='{text_snip[:60]}'")

    return results



def retrieve_vector_and_text_for_drug(input_data, embedding_file_path, target_drug, top_k=3):
    """
    仅检索目标药品的子段落，用于当全局检索未命中目标药品时的兜底。
    从 npz 中过滤出 target_drug 的所有记录，构建临时 FAISS 索引后搜索。
    """
    print(f"[drug-retrieve] 专属检索 drug='{target_drug}' | query='{input_data}' | top_k={top_k}")
    data = np.load(embedding_file_path, allow_pickle=True)
    embeddings = data['embeddings']
    ids = data['ids']
    texts = data['texts']

    # numpy unicode → python str 后比较
    mask = np.array([str(id_) == target_drug for id_ in ids])
    if not mask.any():
        print(f"[drug-retrieve] 向量库中未找到药品 '{target_drug}'")
        return []

    drug_embeddings = embeddings[mask].astype(np.float32)
    drug_texts = texts[mask]
    actual_k = min(top_k, len(drug_embeddings))

    q_emb = model.encode([input_data], convert_to_numpy=True).astype(np.float32)
    dim = drug_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(drug_embeddings)
    D, I = index.search(q_emb, actual_k)

    results = [(target_drug, drug_texts[i][0], drug_texts[i][1]) for i in I[0]]
    print(f"[drug-retrieve] 返回 {len(results)} 条结果:")
    for rank, r in enumerate(results):
        print(f"[drug-retrieve]   [{rank}] title='{r[1]}' text[:40]='{r[2][:40]}'")
    return results


def retrieve_with_context(input_data, embedding_file_path, context_drug=None, top_k=3):
    """
    带会话上下文的向量检索。

    行为：
    1. 调用 retrieve_vector_and_text 做基础向量检索（top_k 扩大到 15，给过滤留足够候选）
    2. 如果 context_drug 非空，对结果做药品名优先过滤（doc_id 匹配 context_drug 的置顶）
    3. 如果 context_drug 为空，直接返回基础检索结果截断到 top_k
    4. 任何异常都捕获并返回基础检索结果

    返回格式与 retrieve_vector_and_text 完全一致：[(doc_id, title, text), ...]
    """
    base_k = 15
    try:
        base_results = retrieve_vector_and_text(input_data, embedding_file_path, top_k=base_k)
    except Exception as e:
        print(f"[retrieve-with-context] 基础检索失败: {e}")
        return []

    if not context_drug:
        # 无上下文药品，直接截断返回，零额外开销
        return base_results[:top_k]

    try:
        matched = []
        others = []
        for r in base_results:
            if _score_result_by_drug_name(r[0], context_drug):
                matched.append(r)
            else:
                others.append(r)

        if matched:
            print(f"[retrieve-with-context] context_drug='{context_drug}' 命中 {len(matched)} 条，置顶")
            results = matched + others
        else:
            print(f"[retrieve-with-context] context_drug='{context_drug}' 未命中，返回基础检索结果")
            results = base_results

        return results[:top_k]
    except Exception as e:
        print(f"[retrieve-with-context] 药品名过滤异常，返回基础结果: {e}")
        return base_results[:top_k]


def retrieve_drug_subsections(drug_name, target_fields, top_k=3):
    """
    通过 ES 精确查 doc_id=drug_name，获取该药品的全部子段落。
    若 target_fields 非空，优先保留 title 匹配 target_fields 的子段落并按字段顺序排序。
    返回格式与 retrieve_vector_and_text 一致：[(doc_id, title, text), ...]
    """
    try:
        es_instance = connect_elasticsearch()
        if not es_instance:
            print(f"[drug-subsections] ES 未连接，无法查 '{drug_name}'")
            return []
        res = es_instance.get(index=config.ES_INDEX, id=drug_name, ignore=[404])
        if not res.get('found'):
            print(f"[drug-subsections] ES 中未找到药品 '{drug_name}'")
            return []
        content = res['_source']['content']
        subs = extract_subsections(content)
        if not subs:
            return []

        all_items = [(drug_name, title, text.strip()) for title, text in subs.items() if text and text.strip()]

        if target_fields:
            matched = []
            seen_titles = set()
            # 第一轮：按 target_fields 顺序收集匹配项
            for field in target_fields:
                for item in all_items:
                    title = item[1]
                    if title in seen_titles:
                        continue
                    t = title.strip()
                    f = (field or '').strip()
                    if not t or not f:
                        continue
                    if f == t or f in t or t in f:
                        matched.append(item)
                        seen_titles.add(title)
            # 第二轮：追加未匹配的其他子段落（补足 top_k）
            others = [item for item in all_items if item[1] not in seen_titles]
            results = matched + others
            print(f"[drug-subsections] 药品 '{drug_name}' 字段匹配 {len(matched)} 条，其余 {len(others)} 条")
        else:
            results = all_items
            print(f"[drug-subsections] 药品 '{drug_name}' 全部子段落 {len(results)} 条")

        results = results[:top_k]
        for rank, r in enumerate(results):
            print(f"[drug-subsections]   [{rank}] title='{r[1]}' text[:40]='{r[2][:40]}'")
        return results
    except Exception as e:
        print(f"[drug-subsections] 查 '{drug_name}' 失败: {e}")
        return []


def connect_elasticsearch():# 连接Elasticsearch
    es = None
    try:
        es = Elasticsearch(
            [{'host': config.ES_HOST, 'port': config.ES_PORT, 'scheme': config.ES_SCHEME}],
            basic_auth=(config.ES_USER, config.ES_PASSWORD),
            verify_certs=False  # 在开发时禁用 SSL 验证，生产环境中请谨慎使用
        )
        if es.ping():
            print(f'成功连接到 Elasticsearch: {config.ES_HOST}')
            return es
        else:
            print(f'无法连接到 Elasticsearch: {config.ES_HOST}')
    except exceptions.ConnectionError as e:
        print(f"连接错误：{e}")

    print('连接失败')
    return None
# 移除全局 es = connect_elasticsearch() 以避免启动时的死锁
def extract_drug_info(text):# 将标准化后的信息切分

    # 匹配多个药品名和标准化输出
    drug_pattern = re.compile(r'提到的药品名：(.+?)\s+标准化输出：\s*(.+?)(?=(提到的药品名：|$))', re.DOTALL)
    matches = drug_pattern.findall(text)
    
    drugs = []
    standard_outputs = []
    
    for match in matches:
        # 提取药品名，可能包含多个药品，以逗号或其他标点分隔
        drug_names = [name.strip() for name in re.split(r'[、,，]', match[0]) if name.strip()]
        # 提取并分割标准化输出的每一行
        outputs = [line.strip() for line in match[1].strip().split('\n') if line.strip()]
        
        for drug_name in drug_names:
            drugs.append(drug_name)
            standard_outputs.append(outputs)

    return drugs, standard_outputs


def _score_result_by_drug_name(doc_id, target_drug):
    """判断向量结果的 doc_id 是否匹配目标药品名。
    doc_id 是药品名（章节标题），如 "瞿胆丸"。
    做模糊匹配：target_drug in doc_id 或 doc_id in target_drug。
    返回 1（匹配）或 0（不匹配）。
    """
    if not target_drug:
        return 0
    d = (doc_id or '').strip()
    t = target_drug.strip()
    if not d or not t:
        return 0
    if t == d or t in d or d in t:
        return 1
    return 0


def verify_data_in_elasticsearch(es, index_name, doc_id, sub_titles):# 验证数据是否在Elasticsearch中，并且返回结果

    output = []  # 存储输出结果

    try:
        #    检查索引是否存在
        response = es.get(index=index_name, id=doc_id)
        content = response['_source']['content']
        
        #    提取所有小标题及其内容
        subsections = extract_subsections(content)
        
        found_content = []
        
        #    检查每个小标题是否存在于提取的内容中
        for sub_title in sub_titles:
            if sub_title in subsections and subsections[sub_title]:
                found_content.append(f"小标题 '{sub_title}' 的内容:\n{subsections[sub_title]}")
        
        #    输出结果
        if found_content:
            output.append("\n".join(found_content))
        else:
            #    没有找到任何小标题，输出完整内容
            output.append(f"未找到任何小标题，输出完整内容:\n{content}")
        
    except exceptions.NotFoundError:
        output.append(f"文档 ID '{doc_id}' 或索引 '{index_name}' 不存在。")
    except exceptions.TransportError as e:
        output.append(f"查询错误：{e}")

    return "\n".join(output)  # 以换行符连接输出结果

