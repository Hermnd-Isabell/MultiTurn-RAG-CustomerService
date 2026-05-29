import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import json
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
    if embedding_file_path:
        key = os.path.normpath(os.path.abspath(embedding_file_path))
        if key in _faiss_cache:
            del _faiss_cache[key]
        elif embedding_file_path in _faiss_cache:
            # 兼容未规范化的旧缓存键
            del _faiss_cache[embedding_file_path]
    else:
        _faiss_cache.clear()



# ---------------------------------------------------------------------------
# E-commerce 意图识别体系（Phase 1）
# ---------------------------------------------------------------------------
_VALID_INTENTS = {
    "product_info", "logistics", "product_recommend",
    "goods_operation", "chitchat", "unknown"
}

_VALID_SUB_INTENTS = {
    "track", "abnormal", "modify",           # logistics
    "substitute", "similar_style", "matching", # product_recommend
    "return", "exchange", "refund_only"      # goods_operation
}

_PRODUCT_INFO_KEYWORDS = {
    "basic_info", "material", "design", "care",
    "styling", "shipping_info", "after_sales", "marketing", "fit"
}

_PRODUCT_INFO_MODULES = {
    "basic_info": "基础信息",
    "material": "规格材质",
    "design": "设计工艺",
    "care": "护理保养",
    "styling": "场景搭配",
    "shipping_info": "库存物流",
    "after_sales": "售后政策",
    "marketing": "营销评价",
    "fit": "版型身材",
}

_MODULE_TITLE_KEYWORDS = {
    "basic_info": ["基础信息", "商品标识", "价格", "品牌", "名称"],
    "material": ["规格材质", "面料成分", "面料特性", "颜色", "尺码", "厚度", "尺码表", "版型说明", "身材建议"],
    "design": ["设计工艺", "风格", "领型", "袖型", "图案", "工艺细节", "特殊细节"],
    "care": ["护理保养", "洗涤方式", "晾晒方式", "熨烫方式", "特殊说明"],
    "styling": ["场景搭配", "适用场景", "搭配建议"],
    "shipping_info": ["库存物流", "库存状态", "发货地", "发货与运费", "运费"],
    "after_sales": ["售后政策", "退换货", "质保"],
    "marketing": ["营销评价", "促销活动", "核心卖点", "评价摘要"],
    "fit": ["版型说明", "身材建议", "尺码表", "版型", "身材"],
}

_VALID_PURCHASE_STAGES = {"awareness", "evaluation", "decision"}

# 规则预筛关键词库
_EC_CHITCHAT_KEYWORDS = {
    '你好', '您好', 'hi', 'hello', '在吗', '谢谢', '感谢',
    '再见', '拜拜', '晚安', '早上好', '哈喽', 'thx', 'thanks',
    '吃饭了吗', '天气', '心情', '聊聊', '讲个笑话',
}

_EC_LOGISTICS_KEYWORDS = {
    '快递': 'track', '到哪了': 'track', '发货': 'track', '物流': 'track',
    '丢了': 'abnormal', '损坏': 'abnormal', '破损': 'abnormal',
    '改地址': 'modify', '改收件人': 'modify',
}

_EC_RETURN_KEYWORDS = {
    '退货': 'return', '换货': 'exchange', '退款': 'refund_only',
    '退钱': 'refund_only',
}

_EC_RECOMMEND_KEYWORDS = {
    '搭配': 'matching', '类似': 'similar_style', '平替': 'substitute',
    '相似': 'similar_style', '推荐': 'matching',
}

_EC_PRODUCT_KEYWORDS = {
    '多少钱', '价格', '材质', '面料', '尺码', '怎么洗', '洗涤',
    '评价', '活动', '促销', '库存', '重量', '颜色', '款式',
}


def quick_ecommerce_intent_hint(input_data):
    """
    电商意图规则预筛。返回 dict 或 "ambiguous"。

    规则：
    - 闲聊关键词 → chitchat
    - 物流关键词（快递/到哪了/发货/丢了/损坏/改地址） → logistics + 对应 sub_intent
    - 退货/换货/退款关键词 → goods_operation + 对应 sub_intent
    - 推荐关键词（搭配/类似/平替） → product_recommend + 对应 sub_intent
    - 商品信息关键词（多少钱/材质/尺码/怎么洗/评价/活动） → product_info
    - 其余 → ambiguous
    """
    if not input_data:
        return "ambiguous"
    text = input_data.strip()
    if not text:
        return "ambiguous"

    # 1) 闲聊短路
    if len(text) <= 8 and text.lower() in {k.lower() for k in _EC_CHITCHAT_KEYWORDS}:
        return {"intent": "chitchat", "sub_intent": None, "keywords": None, "confidence": 1.0}
    # 长句中包含强闲聊词也判闲聊
    for kw in _EC_CHITCHAT_KEYWORDS:
        if kw in text and len(text) <= 12:
            return {"intent": "chitchat", "sub_intent": None, "keywords": None, "confidence": 1.0}

    # 2) 物流
    for kw, sub in _EC_LOGISTICS_KEYWORDS.items():
        if kw in text:
            return {"intent": "logistics", "sub_intent": sub, "keywords": None, "confidence": 1.0}

    # 3) 售后
    for kw, sub in _EC_RETURN_KEYWORDS.items():
        if kw in text:
            return {"intent": "goods_operation", "sub_intent": sub, "keywords": None, "confidence": 1.0}

    # 4) 推荐
    for kw, sub in _EC_RECOMMEND_KEYWORDS.items():
        if kw in text:
            return {"intent": "product_recommend", "sub_intent": sub, "keywords": None, "confidence": 1.0}

    # 5) 商品信息
    for kw in _EC_PRODUCT_KEYWORDS:
        if kw in text:
            # 简单映射：尝试嗅探 module 关键词
            mapped = _map_product_info_keyword(text)
            return {"intent": "product_info", "sub_intent": None, "keywords": mapped, "confidence": 1.0}

    return "ambiguous"


def _map_product_info_keyword(text):
    """辅助：从商品信息问句中嗅探 9 大模块映射。"""
    keyword_rules = [
        ("basic_info", ["价格", "多少钱", "名称", "品牌", "基本信息"]),
        ("material", ["材质", "面料", "成分"]),
        ("design", ["设计", "款式", "版型"]),
        ("care", ["洗涤", "保养", "怎么洗", "护理"]),
        ("styling", ["搭配", "穿搭", "风格"]),
        ("shipping_info", ["发货", "物流", "快递", "运费"]),
        ("after_sales", ["售后", "保修", "退换", "退货", "换货"]),
        ("marketing", ["活动", "促销", "优惠", "满减", "折扣"]),
        ("fit", ["尺码", "大小", "合身", "版型", "体重", "身高"]),
    ]
    for module, kws in keyword_rules:
        for kw in kws:
            if kw in text:
                return module
    return None


def classify_ecommerce_intent(input_data, model=None, client=None):
    """
    LLM 分类器。返回 dict:
    {
        "intent": "product_info|logistics|product_recommend|goods_operation|chitchat|unknown",
        "sub_intent": str|null,
        "confidence": float,
        "keywords": str|null  # 仅 product_info 时有值，映射到 9 大模块之一
    }

    Prompt 要求：
    1. 明确列出 5 大类 + chitchat + unknown
    2. logistics/goods_operation/product_recommend 要求给出 sub_intent
    3. product_info 要求同时识别 keywords（映射到 9 大模块：basic_info/material/design/care/styling/shipping_info/after_sales/marketing/fit）
    4. 要求返回纯 JSON，不要 markdown
    5. confidence < 0.6 时降级为 unknown
    6. 解析失败时返回 {"intent": "unknown", "confidence": 0.0, "sub_intent": null, "keywords": null}
    """
    llm_client = client if client is not None else get_openai_client()

    classify_template = f"""你是电商客服意图识别专家。请判断以下用户问题属于哪一类意图，并按要求返回 JSON。

用户问题：{input_data}

可选意图（intent）：
- product_info：商品信息咨询（价格、材质、尺码、怎么洗、评价、活动等）
- logistics：物流相关（快递到哪了、发货、丢了、损坏、改地址等）
- product_recommend：商品推荐（搭配、类似款式、平替等）
- goods_operation：货物操作（退货、换货、退款）
- chitchat：闲聊
- unknown：无法识别

对于 logistics / goods_operation / product_recommend，必须给出 sub_intent：
- logistics 子意图：track（查物流）/ abnormal（异常）/ modify（修改地址）
- goods_operation 子意图：return（退货）/ exchange（换货）/ refund_only（仅退款）
- product_recommend 子意图：substitute（平替）/ similar_style（类似款）/ matching（搭配）

对于 product_info，请同时识别 keywords（映射到 9 大模块之一：basic_info/material/design/care/styling/shipping_info/after_sales/marketing/fit）和 purchase_stage（购买阶段：awareness 了解 / evaluation 评估 / decision 决策）。

要求：
1. 只返回纯 JSON，不要 markdown 代码块
2. confidence 为 0~1 的置信度
3. confidence < 0.6 时 intent 降级为 unknown

返回格式示例：
{{"intent": "product_info", "sub_intent": null, "confidence": 0.95, "keywords": "material", "purchase_stage": "evaluation"}}
"""

    default_result = {"intent": "unknown", "confidence": 0.0, "sub_intent": None, "keywords": None}

    try:
        response = llm_client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": classify_template}],
        )
        raw = response.choices[0].message.content.strip() if response.choices else ""
    except Exception as e:
        print(f"[classify_ecommerce_intent] LLM 调用失败: {e}")
        return default_result

    # 尝试提取 JSON
    try:
        # 有时模型会包裹在 markdown 代码块中，做简单清洗
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        result = json.loads(cleaned)
    except Exception:
        print(f"[classify_ecommerce_intent] JSON 解析失败，原始输出: {raw}")
        return default_result

    if not isinstance(result, dict):
        return default_result

    intent = result.get("intent", "unknown")
    if intent not in _VALID_INTENTS:
        intent = "unknown"

    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.6:
        intent = "unknown"

    sub_intent = result.get("sub_intent")
    if sub_intent and sub_intent not in _VALID_SUB_INTENTS:
        sub_intent = None

    keywords = result.get("keywords")
    if keywords and keywords not in _PRODUCT_INFO_KEYWORDS:
        keywords = None

    purchase_stage = result.get("purchase_stage")
    if purchase_stage and purchase_stage not in _VALID_PURCHASE_STAGES:
        purchase_stage = None

    # 非 product_info 意图下清除商品专属字段，防止跨意图污染
    if intent != "product_info":
        keywords = None
        purchase_stage = None

    return {
        "intent": intent,
        "sub_intent": sub_intent,
        "confidence": confidence,
        "keywords": keywords,
        "purchase_stage": purchase_stage,
    }


def _score_result_by_module(title, target_module):
    """
    判断检索结果的 title 是否匹配目标模块。
    返回 0 或 1（模糊匹配：精确匹配、包含关系、关键词在 title 中）。
    """
    if not title or not target_module:
        return 0
    t = title.strip()
    if not t:
        return 0

    keywords = _MODULE_TITLE_KEYWORDS.get(target_module, [])
    for kw in keywords:
        if kw in t or t in kw:
            return 1
    return 0


def _extract_text_from_npz_item(t):
    """统一从 npz texts 数组元素中提取 (display_title, text_content)。"""
    if isinstance(t, np.ndarray) and t.shape[0] >= 3:
        display_title = f"{t[0]} - {t[1]}" if t[1] else t[0]
        text_content = t[2]
    elif isinstance(t, (list, tuple)) and len(t) >= 3:
        display_title = f"{t[0]} - {t[1]}" if t[1] else t[0]
        text_content = t[2]
    else:
        display_title = t[0]
        text_content = t[1]
    return display_title, text_content


def _direct_extract_product_chunks(embedding_file_path, context_product, target_module=None):
    """
    直接从已加载的 npz 索引中按 SKU + 模块提取 chunk，绕过向量检索。
    用于 strict_filter 未命中或需要确保召回关键模块的场景。
    """
    cache_key = os.path.normpath(os.path.abspath(embedding_file_path))
    if cache_key not in _faiss_cache:
        return []
    _, ids, texts = _faiss_cache[cache_key]
    results = []
    for idx, doc_id in enumerate(ids):
        if not _score_result_by_product_name(doc_id, context_product):
            continue
        display_title, text_content = _extract_text_from_npz_item(texts[idx])
        if target_module and not _score_result_by_module(display_title, target_module):
            continue
        results.append((doc_id, display_title, text_content))
    return results


def retrieve_product_info(query, embedding_file_path, target_module=None, top_k=5, context_product=None, strict_filter=False):
    """
    商品信息专用检索。
    1. 基础向量检索 top_k=50（给过滤留足候选）
    2. 如果 context_product 非空，按 SKU 匹配过滤；strict 未命中时直接从索引提取该商品 chunk
    3. 如果 target_module 非空，按模块匹配度重排，并补充直接提取的模块 chunk 到最前
    4. 截断到 top_k
    """
    base_k = 50
    try:
        base_results = retrieve_vector_and_text(query, embedding_file_path, top_k=base_k)
    except Exception as e:
        print(f"[product-info-retrieve] 基础检索失败: {e}")
        base_results = []

    # 应用 context_product 过滤
    if context_product:
        matched, others = [], []
        for r in base_results:
            if _score_result_by_product_name(r[0], context_product):
                matched.append(r)
            else:
                others.append(r)
        if matched:
            base_results = matched if strict_filter else matched + others
        elif strict_filter:
            # 向量检索未命中该商品，直接从索引中提取该商品所有 chunk
            base_results = _direct_extract_product_chunks(embedding_file_path, context_product)
            print(f"[product-info-retrieve] strict_filter 向量未命中，直接从索引提取 {context_product} 的 {len(base_results)} 条 chunk")
        print(f"[product-info-retrieve] context_product={context_product} | strict={strict_filter} | matched={len(matched)} | after_filter={len(base_results)}")

    # 如果已知 context_product + target_module，直接从索引中补充提取目标模块 chunk 置顶
    direct_module_results = []
    if context_product and target_module:
        direct_module_results = _direct_extract_product_chunks(embedding_file_path, context_product, target_module=target_module)
        # 去重：避免与 base_results 中的相同 chunk 重复
        existing_ids = {r[0] for r in base_results}
        direct_module_results = [r for r in direct_module_results if r[0] not in existing_ids]
        print(f"[product-info-retrieve] 直接提取目标模块 chunk: {len(direct_module_results)} 条")

    if not target_module:
        results = direct_module_results + base_results
        print(f"[product-info-retrieve] module=None | results={len(results)} | top_k={top_k}")
        return results[:top_k]

    # 按 module_score 重排 base_results
    scored = []
    for r in base_results:
        module_score = _score_result_by_module(r[1], target_module)
        scored.append((r, module_score))
    scored_sorted = sorted(enumerate(scored), key=lambda kv: (-kv[1][1], kv[0]))
    base_results_sorted = [item[1][0] for item in scored_sorted]

    # 直接提取的目标模块 chunk 置顶，然后是向量检索重排结果
    results = direct_module_results + base_results_sorted
    matched = len(direct_module_results) + sum(1 for _, s in scored if s > 0)
    print(f"[product-info-retrieve] module={target_module} | direct={len(direct_module_results)} | matched={matched} | results={len(results)} | top_k={top_k}")
    return results[:top_k]


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

    cache_key = os.path.normpath(os.path.abspath(embedding_file_path))
    if cache_key in _faiss_cache:
        index, ids, texts = _faiss_cache[cache_key]
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
        _faiss_cache[cache_key] = (index, ids, texts)
        print(f"[retrieve] FAISS index built: ntotal={index.ntotal}")

    D, I = index.search(query_embedding.astype(np.float32), top_k)
    print(f"[retrieve] FAISS distances (D)={D[0].tolist()}")
    print(f"[retrieve] FAISS indices  (I)={I[0].tolist()}")

    # 过滤 FAISS 可能返回的越界索引（含 -1 填充值）
    valid_mask = (I[0] >= 0) & (I[0] < len(ids))
    valid_I = I[0][valid_mask]
    retrieved_ids = ids[valid_I].tolist() if len(valid_I) else []
    retrieved_texts = [texts[i] for i in valid_I]
    actual_k = min(top_k, len(retrieved_ids))
    results = []
    for i in range(actual_k):
        t = retrieved_texts[i]
        # 兼容 numpy ndarray（np.savez 保存 list of tuples 可能变成 2D str array）
        if isinstance(t, np.ndarray) and t.shape[0] >= 3:
            display_title = f"{t[0]} - {t[1]}" if t[1] else t[0]
            text_content = t[2]
        elif isinstance(t, (list, tuple)) and len(t) >= 3:
            # 电商格式: (module, sub_module, text)
            display_title = f"{t[0]} - {t[1]}" if t[1] else t[0]
            text_content = t[2]
        else:
            # 药典格式: (title, text)
            display_title = t[0]
            text_content = t[1]
        results.append((retrieved_ids[i], display_title, text_content))

    for rank, (doc_id, title, text_snip) in enumerate(results):
        print(f"[retrieve] result[{rank}] doc_id='{doc_id}' title='{title}' text[:60]='{text_snip[:60]}'")

    return results



def retrieve_with_context(input_data, embedding_file_path, context_product=None, top_k=3, strict_filter=False):
    """
    带会话上下文的向量检索。

    行为：
    1. 调用 retrieve_vector_and_text 做基础向量检索（top_k 扩大到 50，给过滤留足够候选）
    2. 如果 context_product 非空：
       - strict_filter=True：只保留匹配该商品的结果；若向量为空则从索引直接提取该商品全部 chunk
       - strict_filter=False：匹配结果置顶，其余结果保留在后
    3. 如果 context_product 为空，直接返回基础检索结果截断到 top_k
    4. 任何异常都捕获并返回基础检索结果

    返回格式与 retrieve_vector_and_text 完全一致：[(doc_id, title, text), ...]
    """
    base_k = 50
    try:
        base_results = retrieve_vector_and_text(input_data, embedding_file_path, top_k=base_k)
    except Exception as e:
        print(f"[retrieve-with-context] 基础检索失败: {e}")
        return []

    if not context_product:
        # 无上下文商品，直接截断返回，零额外开销
        return base_results[:top_k]

    try:
        matched = []
        others = []
        for r in base_results:
            if _score_result_by_product_name(r[0], context_product):
                matched.append(r)
            else:
                others.append(r)

        if matched:
            if strict_filter:
                print(f"[retrieve-with-context] context_product='{context_product}' 严格过滤，保留 {len(matched)} 条")
                results = matched
            else:
                print(f"[retrieve-with-context] context_product='{context_product}' 命中 {len(matched)} 条，置顶")
                results = matched + others
        else:
            if strict_filter:
                # 向量检索未命中，直接从索引中提取该商品所有 chunk
                direct_results = _direct_extract_product_chunks(embedding_file_path, context_product)
                print(f"[retrieve-with-context] context_product='{context_product}' 向量未命中，直接提取 {len(direct_results)} 条")
                results = direct_results
            else:
                print(f"[retrieve-with-context] context_product='{context_product}' 未命中，返回基础检索结果")
                results = base_results

        return results[:top_k]
    except Exception as e:
        print(f"[retrieve-with-context] 商品名过滤异常，返回基础结果: {e}")
        return base_results[:top_k]


def _score_result_by_product_name(doc_id, target_product):
    """判断向量结果的 doc_id 是否匹配目标商品名。
    做模糊匹配：target_product in doc_id 或 doc_id in target_product。
    返回 1（匹配）或 0（不匹配）。
    """
    if not target_product:
        return 0
    d = (doc_id or '').strip()
    t = target_product.strip()
    if not d or not t:
        return 0
    if t == d or t in d or d in t:
        return 1
    return 0


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
