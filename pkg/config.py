import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

class Config:
    # Elasticsearch
    ES_HOST = os.getenv("ES_HOST", "127.0.0.1")
    ES_PORT = int(os.getenv("ES_PORT", 9200))
    ES_USER = os.getenv("ES_USER", "elastic")
    ES_PASSWORD = os.getenv("ES_PASSWORD", "changeme")
    ES_INDEX = os.getenv("ES_INDEX", "zhyd")
    ES_SCHEME = os.getenv("ES_SCHEME", "http")
    
    # Paths
    # Use absolute path relative to project root if needed, or rely on CWD
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", os.path.join(BASE_DIR, "embeddings2.npz"))

    # Model
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

    # E-commerce paths
    # 商品知识库：优先 data/products/，若不存在则回退到 data/
    _product_kb_default = os.path.join(BASE_DIR, "data", "products")
    PRODUCT_KB_PATH = os.getenv("PRODUCT_KB_PATH", _product_kb_default)
    if not os.path.exists(PRODUCT_KB_PATH) and PRODUCT_KB_PATH == _product_kb_default:
        PRODUCT_KB_PATH = os.path.join(BASE_DIR, "data")

    PRODUCT_INDEX_PATH = os.getenv("PRODUCT_INDEX_PATH", os.path.join(BASE_DIR, "data", "products.npz"))
    PRODUCT_META_PATH = os.getenv("PRODUCT_META_PATH", os.path.join(BASE_DIR, "data", "product_meta.json"))
    ORDERS_JSON_PATH = os.getenv("ORDERS_JSON_PATH", os.path.join(BASE_DIR, "data", "orders.json"))
    RETURN_WINDOW_DAYS = int(os.getenv("RETURN_WINDOW_DAYS", "7"))

    # Feature flags
    # 是否启用问答前置的意图路由层（regex 预筛 + LLM 分类 + 字段抽取）。
    # 关闭时 slow_echo 与 P1 行为完全一致；开启后会按 quick_intent_hint 的判断
    # 决定是否额外调用 1~2 次 LLM。
    ENABLE_INTENT_ROUTING = os.getenv("ENABLE_INTENT_ROUTING", "1").strip().lower() in ("1", "true", "yes", "on")
    # 是否启用推理模型的思考过程输出（仅对 kimi-k2.6 / kimi-k2.5 等有效）。
    # 关闭时通过 extra_body={"thinking": {"type": "disabled"}} 禁用思考。
    ENABLE_THINKING = os.getenv("ENABLE_THINKING", "1").strip().lower() in ("1", "true", "yes", "on")

config = Config()

# 启动时打印实际使用的数据路径，方便调试
print(f"[config] BASE_DIR: {config.BASE_DIR}")
print(f"[config] PRODUCT_KB_PATH: {config.PRODUCT_KB_PATH}")
print(f"[config] PRODUCT_INDEX_PATH: {config.PRODUCT_INDEX_PATH}")
print(f"[config] PRODUCT_META_PATH: {config.PRODUCT_META_PATH}")
print(f"[config] ORDERS_JSON_PATH: {config.ORDERS_JSON_PATH}")
print(f"[config] VECTOR_DB_PATH: {config.VECTOR_DB_PATH}")
