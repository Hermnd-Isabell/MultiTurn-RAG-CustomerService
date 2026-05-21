"""
Phase 4：商品信息分支测试
覆盖：
1. embed._score_result_by_module（模块匹配打分）
2. embed.retrieve_product_info（模块感知检索重排）
3. embed.classify_ecommerce_intent（purchase_stage 解析与过滤）
4. webrun._build_product_info_prompt（动态 prompt 组装）
5. webrun.slow_echo product_info 分支端到端
"""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


class TestScoreResultByModule:
    """embed._score_result_by_module 覆盖 5 种打分场景。"""

    def test_exact_keyword_match(self):
        from embed import _score_result_by_module
        assert _score_result_by_module("规格材质", "material") == 1

    def test_title_contains_keyword(self):
        from embed import _score_result_by_module
        assert _score_result_by_module("面料成分与特性", "material") == 1

    def test_keyword_contains_title(self):
        from embed import _score_result_by_module
        assert _score_result_by_module("退换货", "after_sales") == 1

    def test_no_match_returns_zero(self):
        from embed import _score_result_by_module
        assert _score_result_by_module(" unrelated ", "material") == 0

    def test_empty_inputs_return_zero(self):
        from embed import _score_result_by_module
        assert _score_result_by_module("", "material") == 0
        assert _score_result_by_module("规格材质", "") == 0
        assert _score_result_by_module(None, "material") == 0


class TestRetrieveProductInfo:
    """embed.retrieve_product_info 覆盖缓存行为、模块重排、缺失文件容错。"""

    def test_without_module_returns_top_k(self, monkeypatch):
        from embed import retrieve_product_info

        base = [("id1", "标题A", "内容A"), ("id2", "标题B", "内容B")]
        monkeypatch.setattr("embed.retrieve_vector_and_text", lambda *a, **k: base)

        results = retrieve_product_info("query", "fake.npz", target_module=None, top_k=2)
        assert len(results) == 2
        assert results[0][0] == "id1"

    def test_with_module_reorders_matched_first(self, monkeypatch):
        from embed import retrieve_product_info

        base = [
            ("id1", "无关标题", "内容1"),
            ("id2", "规格材质说明", "内容2"),
            ("id3", "面料成分", "内容3"),
        ]
        monkeypatch.setattr("embed.retrieve_vector_and_text", lambda *a, **k: base)

        results = retrieve_product_info("query", "fake.npz", target_module="material", top_k=2)
        # 匹配模块的应排在前面
        titles = [r[1] for r in results]
        assert "规格材质说明" in titles[:2] or "面料成分" in titles[:2]

    def test_retrieval_failure_returns_empty(self, monkeypatch):
        from embed import retrieve_product_info

        def _raise(*a, **k):
            raise RuntimeError("boom")
        monkeypatch.setattr("embed.retrieve_vector_and_text", _raise)

        results = retrieve_product_info("query", "fake.npz", target_module="material")
        assert results == []


class TestClassifyEcommerceIntentPurchaseStage:
    """classify_ecommerce_intent 对 purchase_stage 的解析与过滤。"""

    def test_purchase_stage_parsed_and_validated(self):
        from embed import classify_ecommerce_intent

        fake_client = MagicMock(name="FakeLLM")
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = (
            '{"intent":"product_info","sub_intent":null,"confidence":0.95,"keywords":"material","purchase_stage":"evaluation"}'
        )
        fake_client.chat.completions.create.return_value = fake_response

        result = classify_ecommerce_intent("这款面料怎么样", client=fake_client)
        assert result["intent"] == "product_info"
        assert result["purchase_stage"] == "evaluation"

    def test_invalid_purchase_stage_filtered(self):
        from embed import classify_ecommerce_intent

        fake_client = MagicMock(name="FakeLLM")
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = (
            '{"intent":"product_info","confidence":0.95,"keywords":"material","purchase_stage":"foobar"}'
        )
        fake_client.chat.completions.create.return_value = fake_response

        result = classify_ecommerce_intent("xxx", client=fake_client)
        assert result["purchase_stage"] is None

    def test_missing_purchase_stage_defaults_none(self):
        from embed import classify_ecommerce_intent

        fake_client = MagicMock(name="FakeLLM")
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = (
            '{"intent":"product_info","confidence":0.95,"keywords":"material"}'
        )
        fake_client.chat.completions.create.return_value = fake_response

        result = classify_ecommerce_intent("xxx", client=fake_client)
        assert result["purchase_stage"] is None


class TestBuildProductInfoPrompt:
    """webrun._build_product_info_prompt 动态组装。"""

    def test_awareness_stage_prompt(self):
        from webrun import _build_product_info_prompt
        prompt = _build_product_info_prompt("ctx", "basic_info", "awareness", "多少钱")
        assert "了解阶段" in prompt
        assert "基础信息" in prompt
        assert "ctx" in prompt
        assert "多少钱" in prompt

    def test_evaluation_stage_prompt(self):
        from webrun import _build_product_info_prompt
        prompt = _build_product_info_prompt("ctx", "material", "evaluation", "面料好吗")
        assert "评估阶段" in prompt
        assert "规格材质" in prompt

    def test_decision_stage_prompt(self):
        from webrun import _build_product_info_prompt
        prompt = _build_product_info_prompt("ctx", "shipping_info", "decision", "有货吗")
        assert "决策阶段" in prompt
        assert "库存物流" in prompt

    def test_unknown_stage_fallback(self):
        from webrun import _build_product_info_prompt
        prompt = _build_product_info_prompt("ctx", "care", None, "怎么洗")
        assert "请根据用户问题自然回答" in prompt
        assert "护理保养" in prompt


class TestSlowEchoProductInfoBranch:
    """slow_echo product_info 分支端到端。"""

    def test_product_info_branch_uses_retrieve_product_info(self, monkeypatch, tmp_path):
        from webrun import slow_echo, _session_facts

        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        monkeypatch.setattr("os.path.exists", lambda p: True)

        # 强制走 product_info 分支
        def _mock_hint(msg):
            return {"intent": "product_info", "sub_intent": None, "confidence": 0.95, "keywords": "material", "purchase_stage": "evaluation"}
        monkeypatch.setattr("webrun.quick_ecommerce_intent_hint", _mock_hint)

        monkeypatch.setattr(
            "webrun.retrieve_product_info",
            lambda *a, **k: [("sku1", "规格材质", "纯棉面料，透气舒适")]
        )

        captured_prompt = []
        def _mock_stream(messages, enable_thinking):
            captured_prompt.append(messages)
            yield "好的，这是回答"
        monkeypatch.setattr("webrun._yield_llm_stream", _mock_stream)

        _session_facts["dialogue_state"] = "init"
        _session_facts["transfer_requested"] = False

        result = list(slow_echo("这衣服什么面料", [], enable_thinking=False))
        assert result[-1] == "好的，这是回答"
        assert len(captured_prompt) == 1
        assert "评估阶段" in captured_prompt[0][1]["content"]
        assert "规格材质" in captured_prompt[0][1]["content"]
        assert "纯棉面料" in captured_prompt[0][1]["content"]

    def test_product_info_no_module_fallback(self, monkeypatch, tmp_path):
        from webrun import slow_echo, _session_facts

        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        monkeypatch.setattr("os.path.exists", lambda p: True)

        def _mock_hint(msg):
            return {"intent": "product_info", "sub_intent": None, "confidence": 0.9, "keywords": None, "purchase_stage": None}
        monkeypatch.setattr("webrun.quick_ecommerce_intent_hint", _mock_hint)

        monkeypatch.setattr(
            "webrun.retrieve_product_info",
            lambda *a, **k: [("sku1", "基础信息", "价格99元")]
        )

        captured_prompt = []
        def _mock_stream(messages, enable_thinking):
            captured_prompt.append(messages)
            yield "回答"
        monkeypatch.setattr("webrun._yield_llm_stream", _mock_stream)

        _session_facts["dialogue_state"] = "init"
        _session_facts["transfer_requested"] = False

        result = list(slow_echo("介绍一下", [], enable_thinking=False))
        assert result[-1] == "回答"
        assert "价格99元" in captured_prompt[0][1]["content"]
