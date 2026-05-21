"""
商品推荐分支（substitute / similar_style / matching）测试：
- 价格区间解析与收集
- 当前商品提取
- RAG 查询构造
- 完整状态机流转
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _parse_price_range
# ---------------------------------------------------------------------------
class TestParsePriceRange:
    def test_hyphen(self):
        from webrun import _parse_price_range
        result = _parse_price_range("200-500")
        assert result == {"min": 200, "max": 500, "text": "200-500元"}

    def test_tilde(self):
        from webrun import _parse_price_range
        result = _parse_price_range("200~500")
        assert result["min"] == 200
        assert result["max"] == 500

    def test_within(self):
        from webrun import _parse_price_range
        result = _parse_price_range("300以内")
        assert result == {"min": 0, "max": 300, "text": "300元以内"}

    def test_around(self):
        from webrun import _parse_price_range
        result = _parse_price_range("500左右")
        assert result["min"] == 400
        assert result["max"] == 600
        assert result["text"] == "500元左右"

    def test_above(self):
        from webrun import _parse_price_range
        result = _parse_price_range("1000以上")
        assert result == {"min": 1000, "max": None, "text": "1000元以上"}

    def test_pure_number(self):
        from webrun import _parse_price_range
        result = _parse_price_range("500")
        assert result == {"min": 0, "max": 500, "text": "500元以内"}

    def test_invalid(self):
        from webrun import _parse_price_range
        assert _parse_price_range("便宜点") is None
        assert _parse_price_range("") is None
        assert _parse_price_range(None) is None


# ---------------------------------------------------------------------------
# _collect_price_range
# ---------------------------------------------------------------------------
class TestCollectPriceRange:
    def test_success(self):
        from webrun import _collect_price_range

        facts = {}
        success, result = _collect_price_range("200-500", facts)
        assert success is True
        assert facts["price_range"]["min"] == 200

    def test_fail(self):
        from webrun import _collect_price_range

        facts = {}
        success, msg = _collect_price_range("随便", facts)
        assert success is False
        assert "请提供明确的价格区间" in msg


# ---------------------------------------------------------------------------
# _extract_current_product
# ---------------------------------------------------------------------------
class TestExtractCurrentProduct:
    def test_from_quoted_message(self):
        from webrun import _extract_current_product
        assert _extract_current_product('"法式茶歇连衣裙"有平替吗', {}) == "法式茶歇连衣裙"

    def test_from_history(self):
        from webrun import _extract_current_product
        facts = {"product_history": ["牛仔裤", "连衣裙"]}
        assert _extract_current_product("有平替吗", facts) == "连衣裙"

    def test_none(self):
        from webrun import _extract_current_product
        assert _extract_current_product("有平替吗", {}) is None


# ---------------------------------------------------------------------------
# _build_recommend_query
# ---------------------------------------------------------------------------
class TestBuildRecommendQuery:
    def test_substitute(self):
        from webrun import _build_recommend_query
        q = _build_recommend_query("连衣裙", "substitute", {"text": "200-500元"})
        assert "连衣裙" in q
        assert "平替" in q
        assert "200-500元" in q

    def test_similar_style(self):
        from webrun import _build_recommend_query
        q = _build_recommend_query("T恤", "similar_style", {"text": "100元以内"})
        assert "风格类似" in q
        assert "100元以内" in q

    def test_matching(self):
        from webrun import _build_recommend_query
        q = _build_recommend_query("西装", "matching", {"text": "500元左右"}, "上班通勤", "需要轻便")
        assert "搭配" in q
        assert "上班通勤" in q
        assert "需要轻便" in q

    def test_no_product(self):
        from webrun import _build_recommend_query
        q = _build_recommend_query(None, "substitute", {"text": "300元以内"})
        assert "推荐平替商品" in q
        assert "连衣裙" not in q


# ---------------------------------------------------------------------------
# _build_recommend_prompt
# ---------------------------------------------------------------------------
class TestBuildRecommendPrompt:
    def test_prompt_contains_type_and_price(self):
        from webrun import _build_recommend_prompt
        prompt = _build_recommend_prompt("连衣裙", "substitute", {"text": "200-500元"}, "context_here")
        assert "平替商品推荐" in prompt
        assert "200-500元" in prompt
        assert "context_here" in prompt

    def test_matching_block(self):
        from webrun import _build_recommend_prompt
        prompt = _build_recommend_prompt("西装", "matching", {"text": "500元"}, "ctx", "上班通勤", "需要轻便")
        assert "使用场景：上班通勤" in prompt
        assert "使用需求：需要轻便" in prompt


# ---------------------------------------------------------------------------
# slow_echo 状态机流转
# ---------------------------------------------------------------------------
class TestRecommendStateMachine:
    def _run_echo(self, webrun_mod, message, history=None):
        return list(webrun_mod.slow_echo(message, history or []))

    def _fake_streaming_client(self):
        """构造一个能返回流式 chunk 的 fake OpenAI 客户端。"""
        def _make_chunk(text):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = text
            return chunk

        def _stream(*_a, **_k):
            for piece in ["测试", "推荐", "回复"]:
                yield _make_chunk(piece)

        client = MagicMock(name="FakeLLM")
        client.chat.completions.create.side_effect = _stream
        return client

    def test_init_to_awaiting_price(self, sample_faiss_data, restore_config):
        """init → awaiting_price_range"""
        import webrun

        webrun.config.PRODUCT_INDEX_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = True
        webrun.clear_session_facts()

        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "product_recommend", "sub_intent": "substitute", "confidence": 1.0}):
            chunks = self._run_echo(webrun, "有平替吗")

        assert webrun._session_facts["dialogue_state"] == "awaiting_price_range"
        assert "价格区间" in chunks[-1]
        assert webrun._session_facts["recommend_sub_intent"] == "substitute"

    def test_price_to_retrieve_substitute(self, sample_faiss_data, restore_config):
        """awaiting_price_range → retrieving → generating → completed（substitute）"""
        import webrun

        webrun.config.PRODUCT_INDEX_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = True
        webrun.clear_session_facts()

        # 先进入 awaiting_price_range
        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "product_recommend", "sub_intent": "substitute", "confidence": 1.0}):
            self._run_echo(webrun, "有平替吗")

        # 再提供价格，直接走完检索+生成
        fake_client = self._fake_streaming_client()
        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "product_recommend", "sub_intent": "substitute", "confidence": 1.0}), \
             patch.object(webrun, "retrieve_with_context", return_value=[("id", "标题", "内容" * 10)]), \
             patch.object(webrun, "get_openai_client", return_value=fake_client):
            chunks = self._run_echo(webrun, "200-500", [("有平替吗", "...")])

        assert webrun._session_facts["dialogue_state"] == "completed"
        assert webrun._session_facts["price_range"]["min"] == 200
        assert chunks  # 应有 LLM 流式输出

    def test_matching_scene_to_usage(self, sample_faiss_data, restore_config):
        """matching 路径：price → scene → usage → retrieve"""
        import webrun

        webrun.config.PRODUCT_INDEX_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = True
        webrun.clear_session_facts()

        # Turn 1: 触发 matching
        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "product_recommend", "sub_intent": "matching", "confidence": 1.0}):
            self._run_echo(webrun, "怎么搭配")
        assert webrun._session_facts["dialogue_state"] == "awaiting_price_range"

        # Turn 2: 提供价格 → 进入 awaiting_matching_scene
        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "product_recommend", "sub_intent": "matching", "confidence": 1.0}):
            chunks = self._run_echo(webrun, "300以内", [("怎么搭配", "...")])
        assert webrun._session_facts["dialogue_state"] == "awaiting_matching_scene"
        assert "使用场景" in chunks[-1]

        # Turn 3: 提供场景 → 进入 awaiting_matching_usage
        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "product_recommend", "sub_intent": "matching", "confidence": 1.0}):
            chunks = self._run_echo(webrun, "上班通勤", [("怎么搭配", "..."), ("300以内", "...")])
        assert webrun._session_facts["dialogue_state"] == "awaiting_matching_usage"
        assert "使用习惯" in chunks[-1] or "使用需求" in chunks[-1]

        # Turn 4: 提供使用行为 → 检索生成
        fake_client = self._fake_streaming_client()
        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "product_recommend", "sub_intent": "matching", "confidence": 1.0}), \
             patch.object(webrun, "retrieve_with_context", return_value=[("id", "搭配", "内容" * 10)]), \
             patch.object(webrun, "get_openai_client", return_value=fake_client):
            chunks = self._run_echo(webrun, "需要轻便", [
                ("怎么搭配", "..."), ("300以内", "..."), ("上班通勤", "..."),
            ])
        assert webrun._session_facts["dialogue_state"] == "completed"
        assert webrun._session_facts["matching_scene"] == "上班通勤"
        assert webrun._session_facts["matching_usage"] == "需要轻便"
        assert chunks

    def test_collect_price_range_fail_then_success(self, sample_faiss_data, restore_config):
        """价格解析失败时保持 awaiting_price_range，重新输入后成功。"""
        import webrun

        webrun.config.PRODUCT_INDEX_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = True
        webrun.clear_session_facts()

        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "product_recommend", "sub_intent": "substitute", "confidence": 1.0}):
            self._run_echo(webrun, "有平替吗")

        # 第一次输入无效
        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "product_recommend", "sub_intent": "substitute", "confidence": 1.0}):
            chunks = self._run_echo(webrun, "便宜点", [("有平替吗", "...")])
        assert webrun._session_facts["dialogue_state"] == "awaiting_price_range"
        assert "请重新输入" in chunks[-1]

        # 第二次输入有效
        fake_client = self._fake_streaming_client()
        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "product_recommend", "sub_intent": "substitute", "confidence": 1.0}), \
             patch.object(webrun, "retrieve_with_context", return_value=[]), \
             patch.object(webrun, "get_openai_client", return_value=fake_client):
            chunks = self._run_echo(webrun, "200-500", [("有平替吗", "..."), ("便宜点", "...")])
        assert webrun._session_facts["dialogue_state"] == "completed"

    def test_extract_product_and_update_history(self, sample_faiss_data, restore_config):
        """消息中包含商品名时，应写入 product_history。"""
        import webrun

        webrun.config.PRODUCT_INDEX_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = True
        webrun.clear_session_facts()

        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "product_recommend", "sub_intent": "similar_style", "confidence": 1.0}):
            self._run_echo(webrun, '"法式茶歇连衣裙"类似款')

        assert "法式茶歇连衣裙" in webrun._session_facts["product_history"]

    def test_state_machine_continuation_from_other_intent(self, sample_faiss_data, restore_config):
        """当处于 awaiting_price_range 时，即使新消息被分为 unknown，也应延续推荐状态机。"""
        import webrun

        webrun.config.PRODUCT_INDEX_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = True
        webrun.clear_session_facts()

        # Turn 1
        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "product_recommend", "sub_intent": "substitute", "confidence": 1.0}):
            self._run_echo(webrun, "有平替吗")

        # Turn 2: 模拟一个会被判为 ambiguous/unknown 的输入，但状态机应强制延续
        fake_client = self._fake_streaming_client()
        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value="ambiguous"), \
             patch.object(webrun, "classify_ecommerce_intent", return_value={"intent": "unknown", "confidence": 0.0}), \
             patch.object(webrun, "retrieve_with_context", return_value=[]), \
             patch.object(webrun, "get_openai_client", return_value=fake_client):
            chunks = self._run_echo(webrun, "200-500", [("有平替吗", "...")])

        # 由于状态机延续检查，即使 classify=unknown，也应强制进入 product_recommend 流程并完成
        assert webrun._session_facts["dialogue_state"] == "completed"
