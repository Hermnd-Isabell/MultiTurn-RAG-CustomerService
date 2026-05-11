"""
pytest 公共配置与 fixture：
1. 在任何 pkg.* 导入之前，stub 掉 sentence_transformers / faiss / gradio /
   docx / langchain / elasticsearch / openai 等"加载即重"或"加载即触网"的依赖；
2. 把 pkg/ 加入 sys.path（pkg 内部用 bare import，例如 `from config import config`）；
3. 注入安全的环境变量默认值，确保 pkg.config 模块能稳定 import；
4. 暴露一组 reset 类 fixture，保证测试间无副作用串扰。
"""
from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


# -----------------------------------------------------------------------------
# 1. 路径 & 环境变量准备（必须最早执行）
# -----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
PKG_DIR = os.path.join(PROJECT_ROOT, "pkg")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

# pkg/config.py 在 import 阶段就读取这些 env，给一组稳定的默认值
os.environ.setdefault("ES_HOST", "localhost")
os.environ.setdefault("ES_PORT", "9200")
os.environ.setdefault("ES_USER", "elastic")
os.environ.setdefault("ES_PASSWORD", "test-pwd")
os.environ.setdefault("ES_INDEX", "test-index")
os.environ.setdefault("ES_SCHEME", "http")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://test.example.com/v1")
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("VECTOR_DB_PATH", os.path.join(PROJECT_ROOT, "test_vector.npz"))
os.environ.setdefault("ENABLE_INTENT_ROUTING", "1")


# -----------------------------------------------------------------------------
# 2. 在 pkg.* 第一次被 import 之前，把"重依赖"全部塞进 sys.modules
# -----------------------------------------------------------------------------
def _install_heavy_module_stubs() -> None:
    """
    把 sentence_transformers / faiss / gradio / docx / langchain / elasticsearch / openai
    的模块 stub 注入 sys.modules，避免：
      - 加载真实 SentenceTransformer 触发模型下载（数百 MB / 数十秒）；
      - faiss / gradio 在某些环境上不可装；
      - openai/elasticsearch 客户端在 import 阶段就尝试连接。
    被测代码不应该感知到 stub 存在。
    """
    # ---- sentence_transformers --------------------------------------------------
    if "sentence_transformers" not in sys.modules:
        st_mod = types.ModuleType("sentence_transformers")

        def _fake_encode(texts, convert_to_numpy=True):
            if isinstance(texts, str):
                texts = [texts]
            # 任意稳定形状即可，下游只看 .shape 与 dtype
            return np.zeros((len(texts), 384), dtype=np.float32)

        def _SentenceTransformer(*_args, **_kwargs):
            inst = MagicMock(name="FakeSentenceTransformer")
            inst.encode.side_effect = _fake_encode
            return inst

        st_mod.SentenceTransformer = _SentenceTransformer
        sys.modules["sentence_transformers"] = st_mod

    # ---- faiss ------------------------------------------------------------------
    if "faiss" not in sys.modules:
        faiss_mod = types.ModuleType("faiss")

        def _IndexFlatL2(_dimension):
            idx = MagicMock(name="FakeFaissIndex")
            # 默认 search 返回 top-3：距离矩阵 + 索引矩阵
            idx.search = MagicMock(
                return_value=(
                    np.array([[0.0, 0.1, 0.2]], dtype=np.float32),
                    np.array([[0, 1, 2]], dtype=np.int64),
                )
            )
            idx.add = MagicMock()
            return idx

        faiss_mod.IndexFlatL2 = _IndexFlatL2
        sys.modules["faiss"] = faiss_mod

    # ---- gradio ----------------------------------------------------------------
    if "gradio" not in sys.modules:
        gr = types.ModuleType("gradio")

        class _CtxMock(MagicMock):
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        # Blocks / Tab / Row / Column 都需要支持 with 语法
        for name in ["Blocks", "Tab", "Row", "Column"]:
            setattr(gr, name, lambda *a, _n=name, **kw: _CtxMock(name=f"gr.{_n}"))

        # 其它组件返回 MagicMock 即可（被 with 块内部赋值给变量）
        for name in [
            "Markdown", "Chatbot", "ChatInterface", "Interface",
            "File", "Textbox", "Button", "Checkbox",
        ]:
            setattr(gr, name, MagicMock(name=f"gr.{name}"))
        sys.modules["gradio"] = gr

    # ---- docx ------------------------------------------------------------------
    if "docx" not in sys.modules:
        docx_mod = types.ModuleType("docx")
        docx_mod.Document = MagicMock(name="docx.Document")
        sys.modules["docx"] = docx_mod

    # ---- langchain.chains -----------------------------------------------------
    if "langchain" not in sys.modules:
        lc_mod = types.ModuleType("langchain")
        chains_mod = types.ModuleType("langchain.chains")
        chains_mod.RetrievalQA = MagicMock(name="RetrievalQA")
        lc_mod.chains = chains_mod
        sys.modules["langchain"] = lc_mod
        sys.modules["langchain.chains"] = chains_mod

    # ---- elasticsearch --------------------------------------------------------
    if "elasticsearch" not in sys.modules:
        es_mod = types.ModuleType("elasticsearch")

        class _ConnectionError(Exception):
            pass

        class _TransportError(Exception):
            pass

        class _NotFoundError(Exception):
            pass

        exc_mod = types.ModuleType("elasticsearch.exceptions")
        exc_mod.ConnectionError = _ConnectionError
        exc_mod.TransportError = _TransportError
        exc_mod.NotFoundError = _NotFoundError

        es_mod.Elasticsearch = MagicMock(name="Elasticsearch")
        es_mod.exceptions = exc_mod
        sys.modules["elasticsearch"] = es_mod
        sys.modules["elasticsearch.exceptions"] = exc_mod

    # ---- openai ----------------------------------------------------------------
    if "openai" not in sys.modules:
        oai_mod = types.ModuleType("openai")
        oai_mod.OpenAI = MagicMock(name="OpenAI")
        sys.modules["openai"] = oai_mod

    # ---- python-dotenv -------------------------------------------------------
    # config.py: `from dotenv import load_dotenv, find_dotenv`，直接给一对 no-op 函数即可
    if "dotenv" not in sys.modules:
        dotenv_mod = types.ModuleType("dotenv")
        dotenv_mod.load_dotenv = lambda *_a, **_kw: False
        dotenv_mod.find_dotenv = lambda *_a, **_kw: ""
        sys.modules["dotenv"] = dotenv_mod


_install_heavy_module_stubs()


# -----------------------------------------------------------------------------
# 3. 通用 fixture
# -----------------------------------------------------------------------------
@pytest.fixture
def restore_config():
    """
    每个测试在 setup 阶段对 config 做属性快照，teardown 阶段恢复。
    用于测试 update_config / 配置热更新而不污染其他用例。
    """
    from config import config

    snapshot = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    yield config
    for k, v in snapshot.items():
        setattr(config, k, v)


@pytest.fixture(autouse=True)
def _reset_module_caches():
    """
    autouse：每个测试前后都清空 embed/_faiss_cache、embed/_openai_client、webrun/_es_client
    以及 webrun.history。避免测试间共享 mock 客户端造成断言串扰。
    """
    import embed
    import webrun

    embed._openai_client = None
    embed._faiss_cache.clear()
    webrun._es_client = None
    webrun.history = []

    yield

    embed._openai_client = None
    embed._faiss_cache.clear()
    webrun._es_client = None
    webrun.history = []


@pytest.fixture
def mock_openai_client():
    """
    返回一个可控的 fake OpenAI 客户端：
      - .chat.completions.create() 默认返回一个生成器，吐 3 个 chunk；
      - 测试用例可覆盖 .chat.completions.create 的 return_value 自己造场景。
    """
    client = MagicMock(name="FakeOpenAIClient")

    def _make_chunk(text):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = text
        return chunk

    def _stream_create(*_args, **_kwargs):
        for piece in ["你好", "，这是", "测试回复"]:
            yield _make_chunk(piece)

    client.chat.completions.create.side_effect = _stream_create
    return client


@pytest.fixture
def sample_faiss_data(tmp_path):
    """
    构造一份合法的 .npz 文件并返回路径，用于 retrieve_vector_and_text 端到端流转。
    内容：3 条 (doc_id, title, text)。
    """
    npz_path = tmp_path / "fake.npz"
    embeddings = np.zeros((3, 384), dtype=np.float32)
    ids = np.array(["drug_a", "drug_b", "drug_c"])
    texts = np.array([("性状", "白色粉末"), ("功能主治", "解表"), ("处方", "板蓝根")], dtype=object)
    np.savez_compressed(
        npz_path,
        embeddings=embeddings,
        ids=ids,
        texts=texts,
        es_index_name=np.array("zhyd"),
        es_fingerprint=np.array("deadbeef"),
    )
    return str(npz_path)


@pytest.fixture
def sample_hits():
    """构造一组 ES 查询结果，用于 _es_index_fingerprint / process_and_vectorize 测试。"""
    return [
        {"_id": "drug_a", "_source": {"content": "【性状】白色粉末\n【功能主治】解表"}},
        {"_id": "drug_b", "_source": {"content": "【处方】板蓝根 1500g"}},
    ]
