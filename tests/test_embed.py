"""
pkg/embed.py 测试覆盖：
- get_openai_client + clear_openai_client_cache 的"懒加载 + 失效"全生命周期；
- clear_faiss_cache 的两种调用形态；
- retrieve_vector_and_text 的检索流程与 _faiss_cache 命中行为；
- quick_ecommerce_intent_hint + classify_ecommerce_intent 的电商意图识别；
- retrieve_with_context 的上下文商品优先过滤。

所有外部依赖（sentence_transformers / faiss / openai / elasticsearch）已在 conftest.py
预先 stub，因此本文件不需要再 patch 它们。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# -----------------------------------------------------------------------------
# get_openai_client / clear_openai_client_cache
# -----------------------------------------------------------------------------
class TestOpenAIClientLifecycle:
    """懒加载 + 显式失效，确保 Gradio 配置 tab 修改 API key 后能重建客户端。"""

    def test_first_call_creates_instance(self):
        import embed

        assert embed._openai_client is None
        client = embed.get_openai_client()
        assert client is not None
        assert embed._openai_client is client

    def test_second_call_returns_cached_instance(self):
        import embed

        first = embed.get_openai_client()
        second = embed.get_openai_client()
        assert first is second, "二次调用必须命中缓存，避免反复 new OpenAI()"

    def test_clear_then_recreate_returns_new_instance(self):
        """clear 后再次 get 必须重新调用 OpenAI(...) 构造新实例。
        conftest 的全局 OpenAI MagicMock 会复用同一个 .return_value，所以这里改用
        patch + side_effect 让两次调用返回明显不同的对象，断言构造函数被调用了 2 次。"""
        import embed

        with patch.object(embed, "OpenAI") as mock_openai_cls:
            mock_openai_cls.side_effect = [
                MagicMock(name="ClientFirst"),
                MagicMock(name="ClientSecond"),
            ]
            first = embed.get_openai_client()
            embed.clear_openai_client_cache()
            assert embed._openai_client is None
            second = embed.get_openai_client()

        assert first is not second, "clear 后必须重建，否则配置热更新无意义"
        assert mock_openai_cls.call_count == 2, "OpenAI() 应被调用 2 次（首建 + 重建）"


# -----------------------------------------------------------------------------
# clear_faiss_cache
# -----------------------------------------------------------------------------
class TestClearFaissCache:
    """两种调用形态：传 path 精准失效；不传则全量清空。"""

    def test_clear_specific_path(self):
        import embed

        embed._faiss_cache["/tmp/a.npz"] = ("idx_a", "ids_a", "txt_a")
        embed._faiss_cache["/tmp/b.npz"] = ("idx_b", "ids_b", "txt_b")

        embed.clear_faiss_cache("/tmp/a.npz")
        assert "/tmp/a.npz" not in embed._faiss_cache
        assert "/tmp/b.npz" in embed._faiss_cache, "未指定的路径不应受影响"

    def test_clear_all_when_no_path(self):
        import embed

        embed._faiss_cache["/tmp/a.npz"] = ("idx_a", "ids_a", "txt_a")
        embed._faiss_cache["/tmp/b.npz"] = ("idx_b", "ids_b", "txt_b")

        embed.clear_faiss_cache()
        assert embed._faiss_cache == {}, "不传 path 必须清空全部"

    def test_clear_nonexistent_path_is_noop(self):
        """传一个不存在的 key 不应抛错。"""
        import embed

        embed._faiss_cache["/tmp/exist.npz"] = ("idx", "ids", "txt")
        embed.clear_faiss_cache("/tmp/missing.npz")
        assert "/tmp/exist.npz" in embed._faiss_cache, "不存在的 key 不应触发清空全部"


# -----------------------------------------------------------------------------
# retrieve_vector_and_text
# -----------------------------------------------------------------------------
class TestRetrieveVectorAndText:
    """端到端：np.load + faiss.IndexFlatL2 走通；缓存命中第二次免重建。"""

    def test_returns_tuples_of_id_title_text(self, sample_faiss_data):
        from embed import retrieve_vector_and_text

        results = retrieve_vector_and_text("白色T恤", sample_faiss_data, top_k=3)

        assert len(results) == 3
        for doc_id, title, text in results:
            # ids 与 texts 由 sample_faiss_data fixture 控制
            assert any(sku in doc_id for sku in ("TX2026001", "CS2026001", "NZ2026001"))
            assert isinstance(title, str)
            assert isinstance(text, str)

    def test_second_call_hits_cache(self, sample_faiss_data):
        """第二次以相同路径调用应命中 _faiss_cache，不再触发 np.load。"""
        import embed

        # 第一次：触发 np.load 与 IndexFlatL2 构建
        embed.retrieve_vector_and_text("q1", sample_faiss_data, top_k=3)
        assert sample_faiss_data in embed._faiss_cache

        # 替换 np.load 为爆炸函数：若被调用则一定失败
        with patch.object(np, "load", side_effect=AssertionError("不应再次加载 .npz")):
            embed.retrieve_vector_and_text("q2", sample_faiss_data, top_k=3)

    def test_raises_when_file_missing(self, tmp_path):
        from embed import retrieve_vector_and_text

        missing_path = str(tmp_path / "missing.npz")
        with pytest.raises(FileNotFoundError):
            retrieve_vector_and_text("anything", missing_path, top_k=3)


# -----------------------------------------------------------------------------
# E-commerce 意图识别（Phase 1）
# -----------------------------------------------------------------------------
class TestEcommerceIntent:
    """quick_ecommerce_intent_hint + classify_ecommerce_intent 覆盖 5 大类 + chitchat + unknown。"""

    # ---- quick_ecommerce_intent_hint ----

    def test_chitchat(self):
        from embed import quick_ecommerce_intent_hint

        result = quick_ecommerce_intent_hint("你好")
        assert isinstance(result, dict)
        assert result["intent"] == "chitchat"
        assert result["confidence"] == 1.0

    def test_logistics_track(self):
        from embed import quick_ecommerce_intent_hint

        result = quick_ecommerce_intent_hint("快递到哪了")
        assert result["intent"] == "logistics"
        assert result["sub_intent"] == "track"

    def test_goods_operation_return(self):
        from embed import quick_ecommerce_intent_hint

        result = quick_ecommerce_intent_hint("我要退货")
        assert result["intent"] == "goods_operation"
        assert result["sub_intent"] == "return"

    def test_product_recommend_similar(self):
        from embed import quick_ecommerce_intent_hint

        result = quick_ecommerce_intent_hint("有类似款式吗")
        assert result["intent"] == "product_recommend"
        assert result["sub_intent"] == "similar_style"

    def test_product_info_material(self):
        from embed import quick_ecommerce_intent_hint

        result = quick_ecommerce_intent_hint("什么材质")
        assert result["intent"] == "product_info"
        assert result["keywords"] == "material"

    def test_ambiguous(self):
        from embed import quick_ecommerce_intent_hint

        result = quick_ecommerce_intent_hint("随便问问")
        assert result == "ambiguous"

    # ---- classify_ecommerce_intent ----

    def test_classify_success(self):
        from embed import classify_ecommerce_intent

        fake_client = MagicMock(name="FakeLLM")
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = '{"intent":"product_info","sub_intent":null,"confidence":0.95,"keywords":"basic_info"}'
        fake_client.chat.completions.create.return_value = fake_response

        result = classify_ecommerce_intent("多少钱", client=fake_client)
        assert result["intent"] == "product_info"
        assert result["confidence"] == 0.95
        assert result["keywords"] == "basic_info"

    def test_classify_low_confidence_becomes_unknown(self):
        from embed import classify_ecommerce_intent

        fake_client = MagicMock(name="FakeLLM")
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = '{"intent":"product_info","confidence":0.3}'
        fake_client.chat.completions.create.return_value = fake_response

        result = classify_ecommerce_intent("xxx", client=fake_client)
        assert result["intent"] == "unknown"
        assert result["confidence"] == 0.3

    def test_classify_parse_failure_returns_unknown(self):
        from embed import classify_ecommerce_intent

        fake_client = MagicMock(name="FakeLLM")
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = "not json at all"
        fake_client.chat.completions.create.return_value = fake_response

        result = classify_ecommerce_intent("xxx", client=fake_client)
        assert result["intent"] == "unknown"
        assert result["confidence"] == 0.0

    def test_classify_invalid_intent_filtered(self):
        from embed import classify_ecommerce_intent

        fake_client = MagicMock(name="FakeLLM")
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = '{"intent":"hacker","confidence":0.99}'
        fake_client.chat.completions.create.return_value = fake_response

        result = classify_ecommerce_intent("xxx", client=fake_client)
        assert result["intent"] == "unknown"


class TestRetrieveWithContext:
    """带会话上下文的向量检索：商品名优先过滤。"""

    def test_with_context_product(self, monkeypatch):
        """context_product 匹配的结果置顶，不匹配的后移。"""
        from embed import retrieve_with_context

        base = [
            ("TX2026001", "基础信息", "T恤基础信息"),
            ("CS2026001", "规格材质", "衬衫材质"),
            ("NZ2026001", "规格材质", "牛仔裤材质"),
        ]
        monkeypatch.setattr("embed.retrieve_vector_and_text", lambda *a, **k: base)

        # 只有 "CS2026001" 匹配
        def mock_score(doc_id, target):
            return 1 if "CS2026001" in (doc_id or "") else 0
        monkeypatch.setattr("embed._score_result_by_product_name", mock_score)

        results = retrieve_with_context("query", "fake.npz", context_product="CS2026001", top_k=3)
        assert results[0][0] == "CS2026001"
        assert len(results) == 3

    def test_without_context(self, monkeypatch):
        """context_product=None 时直接返回基础结果截断。"""
        from embed import retrieve_with_context

        base = [("a", "t1", "x"), ("b", "t2", "y")]
        monkeypatch.setattr("embed.retrieve_vector_and_text", lambda *a, **k: base)

        results = retrieve_with_context("query", "fake.npz", context_product=None, top_k=2)
        assert results == base

    def test_context_product_not_in_results(self, monkeypatch):
        """context_product 不在结果中时返回基础结果。"""
        from embed import retrieve_with_context

        base = [("TX2026001", "规格材质", "x")]
        monkeypatch.setattr("embed.retrieve_vector_and_text", lambda *a, **k: base)
        monkeypatch.setattr("embed._score_result_by_product_name", lambda d, t: 0)

        results = retrieve_with_context("query", "fake.npz", context_product="不存在", top_k=1)
        assert results == base

    def test_context_product_exception_fallback(self, monkeypatch):
        """匹配过程异常时返回基础结果，不抛异常。"""
        from embed import retrieve_with_context

        base = [("a", "t", "x")]
        monkeypatch.setattr("embed.retrieve_vector_and_text", lambda *a, **k: base)
        monkeypatch.setattr("embed._score_result_by_product_name", lambda d, t: (_ for _ in ()).throw(RuntimeError("boom")))

        results = retrieve_with_context("query", "fake.npz", context_product="a", top_k=1)
        assert results == base
