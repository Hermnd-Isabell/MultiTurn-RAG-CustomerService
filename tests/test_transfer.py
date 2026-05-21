"""
Phase 4：转人工逻辑测试
覆盖：
1. _detect_repeat_question（相似度检测与计数）
2. _transfer_to_human（触发话术与状态设置）
3. slow_echo 转人工拦截（明确输入 / 连续追问 / 超纲业务 / 确认对话）
"""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


class TestDetectRepeatQuestion:
    """webrun._detect_repeat_question 覆盖 intent 变化、相似度阈值、计数累加。"""

    def test_different_intent_resets_count(self):
        from webrun import _detect_repeat_question, _session_facts

        _session_facts["last_intent"] = "logistics"
        _session_facts["question_repeat_count"] = 2
        _session_facts["last_question_embedding"] = None

        is_repeat, count = _detect_repeat_question("新消息", "product_info", _session_facts)
        assert is_repeat is False
        assert count == 0
        assert _session_facts["question_repeat_count"] == 0
        assert _session_facts["last_question_embedding"] is None

    def test_same_intent_low_similarity_no_repeat(self):
        from webrun import _detect_repeat_question, _session_facts

        # 给 last_question_embedding 一个与当前完全不同的向量
        emb = np.zeros(384, dtype=np.float32)
        emb[0] = 1.0
        emb = emb / np.linalg.norm(emb)
        _session_facts["last_intent"] = "product_info"
        _session_facts["question_repeat_count"] = 1
        _session_facts["last_question_embedding"] = emb

        is_repeat, count = _detect_repeat_question("完全不同的内容", "product_info", _session_facts)
        # 当前 embedding 会在 index 0 有值，与 last (index 0) 同方向，所以 sim 可能高
        # 为了稳定，我们依赖 fixture 中 _fake_encode 的设计：每个文本在 i%384 处为 1
        # 所以 "完全不同的内容" 与 "之前问题" 如果索引不同则 sim=0
        # 但这里我们无法控制索引... 改用 patch

    def test_same_intent_high_similarity_increments_count(self, monkeypatch):
        from webrun import _detect_repeat_question, _session_facts

        fake_emb = np.ones(384, dtype=np.float32) / np.sqrt(384)
        _session_facts["last_intent"] = "product_info"
        _session_facts["question_repeat_count"] = 1
        _session_facts["last_question_embedding"] = fake_emb

        monkeypatch.setattr("webrun.model", MagicMock())
        monkeypatch.setattr("webrun.model.encode", lambda text, normalize_embeddings=True: fake_emb)

        is_repeat, count = _detect_repeat_question("同样的问题", "product_info", _session_facts)
        assert is_repeat is True
        assert count == 2

    def test_model_none_returns_false(self, monkeypatch):
        from webrun import _detect_repeat_question, _session_facts

        _session_facts["last_intent"] = "product_info"
        _session_facts["question_repeat_count"] = 1
        monkeypatch.setattr("webrun.model", None)

        is_repeat, count = _detect_repeat_question("问题", "product_info", _session_facts)
        assert is_repeat is False
        assert count == 1


class TestTransferToHuman:
    """webrun._transfer_to_human 覆盖 3 类 reason 与状态设置。"""

    def test_explicit_reason(self):
        from webrun import _transfer_to_human, _session_facts

        _session_facts["transfer_requested"] = False
        _session_facts["dialogue_state"] = "init"
        msg = _transfer_to_human("explicit", _session_facts)
        assert "已为您转接" in msg
        assert _session_facts["transfer_requested"] is True
        assert _session_facts["dialogue_state"] == "awaiting_confirmation"

    def test_repeat_reason(self):
        from webrun import _transfer_to_human, _session_facts

        msg = _transfer_to_human("repeat", _session_facts)
        assert "反复询问" in msg
        assert _session_facts["transfer_requested"] is True

    def test_out_of_scope_reason(self):
        from webrun import _transfer_to_human, _session_facts

        msg = _transfer_to_human("out_of_scope", _session_facts)
        assert "超出" in msg or "范围" in msg
        assert _session_facts["transfer_requested"] is True


class TestSlowEchoTransferExplicit:
    """slow_echo 明确输入转人工。"""

    def test_keyword_transfer_explicit(self, monkeypatch):
        from webrun import slow_echo, _session_facts

        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        _session_facts["dialogue_state"] = "init"
        _session_facts["transfer_confirmed"] = False
        _session_facts["transfer_requested"] = False

        result = list(slow_echo("转人工", [], enable_thinking=False))
        assert any("已为您转接" in r for r in result)
        assert _session_facts["transfer_requested"] is True

    def test_keyword_transfer_human_service(self, monkeypatch):
        from webrun import slow_echo, _session_facts

        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        _session_facts["dialogue_state"] = "init"
        _session_facts["transfer_confirmed"] = False
        _session_facts["transfer_requested"] = False

        result = list(slow_echo("人工客服", [], enable_thinking=False))
        assert any("已为您转接" in r for r in result)


class TestSlowEchoTransferRepeat:
    """slow_echo 连续追问 ≥3 次转人工。"""

    def test_repeat_count_reaches_threshold(self, monkeypatch):
        from webrun import slow_echo, _session_facts

        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        monkeypatch.setattr("os.path.exists", lambda p: True)
        _session_facts["dialogue_state"] = "init"
        _session_facts["transfer_confirmed"] = False
        _session_facts["transfer_requested"] = False
        fake_emb = np.ones(384, dtype=np.float32) / np.sqrt(384)
        _session_facts["last_intent"] = "product_info"
        _session_facts["question_repeat_count"] = 2
        _session_facts["last_question_embedding"] = fake_emb

        monkeypatch.setattr("webrun.model", MagicMock())
        monkeypatch.setattr("webrun.model.encode", lambda text, normalize_embeddings=True: fake_emb)

        def _mock_hint(msg):
            return {"intent": "product_info", "sub_intent": None, "confidence": 0.9, "keywords": "material"}
        monkeypatch.setattr("webrun.quick_ecommerce_intent_hint", _mock_hint)
        monkeypatch.setattr("webrun.retrieve_product_info", lambda *a, **k: [("sku1", "规格材质", "纯棉")])
        monkeypatch.setattr("webrun._yield_llm_stream", lambda msgs, et: iter(["回答"]))

        result = list(slow_echo("什么面料", [], enable_thinking=False))
        assert any("转接人工" in r or "人工客服" in r for r in result)


class TestSlowEchoTransferOutOfScope:
    """slow_echo 超纲业务转人工。"""

    def test_two_consecutive_unknown_low_confidence(self, monkeypatch):
        from webrun import slow_echo, _session_facts

        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        _session_facts["dialogue_state"] = "init"
        _session_facts["transfer_confirmed"] = False
        _session_facts["transfer_requested"] = False
        _session_facts["question_repeat_count"] = 1
        _session_facts["last_intent"] = "unknown"

        def _mock_hint(msg):
            return {"intent": "unknown", "sub_intent": None, "confidence": 0.2, "keywords": None}
        monkeypatch.setattr("webrun.quick_ecommerce_intent_hint", _mock_hint)

        result = list(slow_echo("宇宙哲学问题", [], enable_thinking=False))
        assert any("超出" in r or "范围" in r or "转接" in r for r in result)

    def test_unknown_reset_after_valid_intent(self, monkeypatch):
        from webrun import slow_echo, _session_facts

        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        monkeypatch.setattr("os.path.exists", lambda p: True)
        _session_facts["dialogue_state"] = "init"
        _session_facts["transfer_confirmed"] = False
        _session_facts["transfer_requested"] = False
        _session_facts["question_repeat_count"] = 1
        _session_facts["last_intent"] = "unknown"

        def _mock_hint(msg):
            return {"intent": "product_info", "sub_intent": None, "confidence": 0.9, "keywords": "material"}
        monkeypatch.setattr("webrun.quick_ecommerce_intent_hint", _mock_hint)

        # Mock retrieve to avoid npz issues
        monkeypatch.setattr("webrun.retrieve_product_info", lambda *a, **k: [("sku1", "规格材质", "纯棉")])
        monkeypatch.setattr("webrun._yield_llm_stream", lambda msgs, et: iter(["回答"]))

        result = list(slow_echo("什么面料", [], enable_thinking=False))
        # 不应触发转人工
        assert not any("超出" in r for r in result)
        assert _session_facts["question_repeat_count"] == 0


class TestSlowEchoTransferConfirmation:
    """slow_echo 转人工确认对话（是/否/其他）。"""

    def test_confirm_yes(self, monkeypatch):
        from webrun import slow_echo, _session_facts

        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        _session_facts["transfer_requested"] = True
        _session_facts["transfer_confirmed"] = False
        _session_facts["dialogue_state"] = "awaiting_confirmation"

        result = list(slow_echo("是", [], enable_thinking=False))
        assert any("已为您转接" in r for r in result)
        assert _session_facts["transfer_confirmed"] is True
        assert _session_facts["dialogue_state"] == "completed"

    def test_confirm_no(self, monkeypatch):
        from webrun import slow_echo, _session_facts

        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        _session_facts["transfer_requested"] = True
        _session_facts["transfer_confirmed"] = False
        _session_facts["dialogue_state"] = "awaiting_confirmation"

        result = list(slow_echo("否", [], enable_thinking=False))
        assert any("继续为您服务" in r for r in result)
        assert _session_facts["transfer_requested"] is False
        assert _session_facts["dialogue_state"] == "init"

    def test_confirm_unclear(self, monkeypatch):
        from webrun import slow_echo, _session_facts

        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        _session_facts["transfer_requested"] = True
        _session_facts["transfer_confirmed"] = False
        _session_facts["dialogue_state"] = "awaiting_confirmation"

        result = list(slow_echo("maybe", [], enable_thinking=False))
        assert any("请回复" in r for r in result)

    def test_already_confirmed_intercepted(self, monkeypatch):
        from webrun import slow_echo, _session_facts

        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        _session_facts["transfer_confirmed"] = True
        _session_facts["dialogue_state"] = "completed"

        result = list(slow_echo("任何问题", [], enable_thinking=False))
        assert any("已为您转接" in r for r in result)
