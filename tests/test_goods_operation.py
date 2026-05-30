"""
Phase 5：货物操作决策树测试（退货 / 换货 / 仅退款）
覆盖：
1. 原因/方式/规格/金额解析
2. 系统校验逻辑
3. 写回订单 JSON
4. slow_echo 状态机完整流程
"""
import json
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# 1. 解析函数测试
# ---------------------------------------------------------------------------
class TestParseOperationReason:
    def test_parse_return_reason_by_number(self):
        from webrun import _parse_operation_reason
        r, c = _parse_operation_reason("return", "1")
        assert r == "不想要了"
        assert c == "1"

    def test_parse_return_reason_by_text(self):
        from webrun import _parse_operation_reason
        r, c = _parse_operation_reason("return", "质量有问题")
        assert r == "质量问题"
        assert c == "2"

    def test_parse_exchange_reason(self):
        from webrun import _parse_operation_reason
        r, c = _parse_operation_reason("exchange", "3")
        assert r == "尺码颜色不合适"
        assert c == "3"

    def test_parse_refund_reason(self):
        from webrun import _parse_operation_reason
        r, c = _parse_operation_reason("refund_only", "未收到货")
        assert r == "未收到货"
        assert c == "1"

    def test_parse_unknown_text_fallback_other(self):
        from webrun import _parse_operation_reason
        r, c = _parse_operation_reason("return", "随便什么")
        assert r == "其他"
        assert c == "4"

    def test_parse_empty_returns_none(self):
        from webrun import _parse_operation_reason
        assert _parse_operation_reason("return", "") == (None, None)
        assert _parse_operation_reason("return", None) == (None, None)


class TestParseReturnMethod:
    def test_number_1(self):
        from webrun import _parse_return_method
        assert _parse_return_method("1") == "上门取件"

    def test_text_door(self):
        from webrun import _parse_return_method
        assert _parse_return_method("上门取件") == "上门取件"

    def test_number_2(self):
        from webrun import _parse_return_method
        assert _parse_return_method("2") == "自行寄回"

    def test_invalid_returns_none(self):
        from webrun import _parse_return_method
        assert _parse_return_method("3") is None
        assert _parse_return_method("") is None


class TestParseExchangeSpec:
    def test_by_number(self):
        from webrun import _parse_exchange_spec
        specs = [("1", "藏青色", "M", 10), ("2", "米白色", "S", 5)]
        result = _parse_exchange_spec("1", specs)
        assert result == {"color": "藏青色", "size": "M"}

    def test_by_text(self):
        from webrun import _parse_exchange_spec
        specs = [("1", "藏青色", "M", 10)]
        result = _parse_exchange_spec("藏青色 M", specs)
        assert result == {"color": "藏青色", "size": "M"}

    def test_invalid_returns_none(self):
        from webrun import _parse_exchange_spec
        specs = [("1", "藏青色", "M", 10)]
        assert _parse_exchange_spec("红色 XL", specs) is None


class TestParseRefundAmount:
    def test_full_refund(self):
        from webrun import _parse_refund_amount
        amount, label = _parse_refund_amount("1", 299)
        assert amount == 299
        assert "全额" in label

    def test_full_refund_text(self):
        from webrun import _parse_refund_amount
        amount, label = _parse_refund_amount("全额", 299)
        assert amount == 299

    def test_partial_refund(self):
        from webrun import _parse_refund_amount
        amount, label = _parse_refund_amount("200", 299)
        assert amount == 200
        assert "200" in label

    def test_over_total_fails(self):
        from webrun import _parse_refund_amount
        result = _parse_refund_amount("500", 299)
        assert result[0] is None
        assert "不能超过" in result[1]

    def test_zero_fails(self):
        from webrun import _parse_refund_amount
        result = _parse_refund_amount("0", 100)
        assert result[0] is None


# ---------------------------------------------------------------------------
# 2. 校验逻辑测试
# ---------------------------------------------------------------------------
class TestValidateGoodsOperation:
    def test_return_within_window(self):
        from webrun import _validate_goods_operation
        future = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
        order = {"after_sales": {"return_deadline": future}, "operations_history": []}
        valid, msg = _validate_goods_operation(order, "return", "质量问题", "上门取件", {})
        assert valid is True

    def test_return_overdue(self):
        from webrun import _validate_goods_operation
        past = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        order = {"after_sales": {"return_deadline": past}, "operations_history": []}
        valid, msg = _validate_goods_operation(order, "return", "质量问题", "上门取件", {})
        assert valid is False
        assert "超过退货期限" in msg

    def test_return_duplicate(self):
        from webrun import _validate_goods_operation
        future = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
        order = {
            "after_sales": {"return_deadline": future},
            "operations_history": [{"type": "return", "status": "申请中"}]
        }
        valid, msg = _validate_goods_operation(order, "return", "质量问题", "上门取件", {})
        assert valid is False
        assert "重复提交" in msg

    def test_exchange_same_spec(self):
        from webrun import _validate_goods_operation
        order = {
            "items": [{"spec": {"color": "米白色", "size": "M"}}],
            "after_sales": {"exchange_inventory": {"米白色": {"M": 5}}}
        }
        valid, msg = _validate_goods_operation(order, "exchange", "质量问题", {"color": "米白色", "size": "M"}, {})
        assert valid is False
        assert "相同" in msg

    def test_exchange_no_stock(self):
        from webrun import _validate_goods_operation
        order = {
            "items": [{"spec": {"color": "米白色", "size": "M"}}],
            "after_sales": {"exchange_inventory": {"藏青色": {"L": 0}}}
        }
        valid, msg = _validate_goods_operation(order, "exchange", "质量问题", {"color": "藏青色", "size": "L"}, {})
        assert valid is False
        assert "库存" in msg

    def test_refund_unshipped(self):
        from webrun import _validate_goods_operation
        order = {"status": "未发货"}
        valid, msg = _validate_goods_operation(order, "refund_only", "未收到货", {"amount": 299}, {})
        assert valid is True

    def test_refund_delivered(self):
        from webrun import _validate_goods_operation
        order = {"status": "已签收"}
        valid, msg = _validate_goods_operation(order, "refund_only", "不想要了", {"amount": 299}, {})
        assert valid is False
        assert "不支持仅退款" in msg


# ---------------------------------------------------------------------------
# 3. 写回订单测试
# ---------------------------------------------------------------------------
class TestWriteBackGoodsOperation:
    def test_appends_history(self, tmp_path, monkeypatch):
        from webrun import _write_back_goods_operation
        orders_path = tmp_path / "orders.json"
        orders_data = [{"order_id": "ORD001", "user_phone": "13800138000", "operations_history": [], "status": "已完成"}]
        orders_path.write_text(json.dumps(orders_data, ensure_ascii=False), encoding="utf-8")
        # 同时 patch webrun 和 orders 模块引用的 config
        monkeypatch.setattr("webrun.config.ORDERS_JSON_PATH", str(orders_path))
        monkeypatch.setattr("orders.config.ORDERS_JSON_PATH", str(orders_path))

        success, msg = _write_back_goods_operation("ORD001", "return", "质量问题", "上门取件", {})
        assert success is True

        loaded = json.loads(orders_path.read_text(encoding="utf-8"))
        assert len(loaded[0]["operations_history"]) == 1
        assert loaded[0]["operations_history"][0]["type"] == "return"
        assert loaded[0]["status"] == "售后处理中"

    def test_order_not_found(self, monkeypatch):
        from webrun import _write_back_goods_operation
        monkeypatch.setattr("webrun.config.ORDERS_JSON_PATH", "/nonexistent/orders.json")
        monkeypatch.setattr("orders.config.ORDERS_JSON_PATH", "/nonexistent/orders.json")
        success, msg = _write_back_goods_operation("ORD999", "return", "质量问题", "上门取件", {})
        assert success is False


# ---------------------------------------------------------------------------
# 4. slow_echo 状态机端到端测试
# ---------------------------------------------------------------------------
def _make_order():
    """构造一个标准测试订单。"""
    future = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    return {
        "order_id": "ORD20250520001",
        "user_phone": "13800138000",
        "status": "未发货",
        "items": [{"sku_id": "SKU001", "name": "T恤", "price": 299, "quantity": 1, "spec": {"color": "米白色", "size": "M"}}],
        "total_amount": 299,
        "after_sales": {
            "return_deadline": future,
            "exchange_inventory": {
                "米白色": {"S": 5, "M": 3, "L": 0},
                "藏青色": {"S": 10, "M": 10, "L": 10}
            }
        },
        "operations_history": [],
        "logistics": {"address": "北京市"},
        "contact": {"name": "张三"}
    }


class TestStateMachineReturnFlow:
    def test_full_return_flow(self, monkeypatch, tmp_path):
        from webrun import slow_echo, _session_facts

        orders_path = tmp_path / "orders.json"
        orders_path.write_text(json.dumps([_make_order()], ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr("webrun.config.ORDERS_JSON_PATH", str(orders_path))
        monkeypatch.setattr("orders.config.ORDERS_JSON_PATH", str(orders_path))
        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        monkeypatch.setattr("os.path.exists", lambda p: True)

        def _mock_hint(msg):
            return {"intent": "goods_operation", "sub_intent": "return", "confidence": 0.95, "keywords": None}
        monkeypatch.setattr("webrun.quick_ecommerce_intent_hint", _mock_hint)

        stream_captured = []
        def _mock_stream(messages, enable_thinking):
            stream_captured.append(messages)
            yield "申请已受理"
        monkeypatch.setattr("webrun._yield_llm_stream", _mock_stream)

        _session_facts["dialogue_state"] = "init"
        _session_facts["verified_identity"] = False
        _session_facts["transfer_requested"] = False
        _session_facts["last_intent"] = None

        # Step 1: init → awaiting_identity
        result = list(slow_echo("我要退货", [], enable_thinking=False))
        assert any("订单号和手机号" in r for r in result)
        assert _session_facts["dialogue_state"] == "awaiting_identity"
        assert _session_facts.get("last_intent") == "goods_operation", f"last_intent={_session_facts.get('last_intent')}"

        # Step 2: awaiting_identity → identity_verified → awaiting_reason
        result = list(slow_echo("ORD20250520001 13800138000", [], enable_thinking=False))
        assert any("退货原因" in r for r in result)
        assert _session_facts["dialogue_state"] == "awaiting_reason"

        # Step 3: awaiting_reason → awaiting_return_method
        result = list(slow_echo("1", [], enable_thinking=False))
        assert any("退货方式" in r for r in result)
        assert _session_facts["dialogue_state"] == "awaiting_return_method"
        assert _session_facts["operation_reason"] == "不想要了"

        # Step 4: awaiting_return_method → validating → writing_back → completed
        result = list(slow_echo("1", [], enable_thinking=False))
        assert result[-1] == "申请已受理"
        assert _session_facts["dialogue_state"] == "completed"
        assert _session_facts["operation_detail"] == "上门取件"


class TestStateMachineExchangeFlow:
    def test_full_exchange_flow(self, monkeypatch, tmp_path):
        from webrun import slow_echo, _session_facts

        orders_path = tmp_path / "orders.json"
        orders_path.write_text(json.dumps([_make_order()], ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr("webrun.config.ORDERS_JSON_PATH", str(orders_path))
        monkeypatch.setattr("orders.config.ORDERS_JSON_PATH", str(orders_path))
        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        monkeypatch.setattr("os.path.exists", lambda p: True)

        def _mock_hint(msg):
            return {"intent": "goods_operation", "sub_intent": "exchange", "confidence": 0.95, "keywords": None}
        monkeypatch.setattr("webrun.quick_ecommerce_intent_hint", _mock_hint)

        stream_captured = []
        def _mock_stream(messages, enable_thinking):
            stream_captured.append(messages)
            yield "换货申请已受理"
        monkeypatch.setattr("webrun._yield_llm_stream", _mock_stream)

        _session_facts["dialogue_state"] = "init"
        _session_facts["verified_identity"] = False
        _session_facts["transfer_requested"] = False
        _session_facts["last_intent"] = None

        # Step 1: identity
        result = list(slow_echo("换货", [], enable_thinking=False))
        assert any("订单号" in r for r in result)

        result = list(slow_echo("ORD20250520001 13800138000", [], enable_thinking=False))
        assert any("换货原因" in r for r in result)

        # Step 2: reason
        result = list(slow_echo("3", [], enable_thinking=False))
        assert any("可换规格" in r for r in result)
        assert _session_facts["dialogue_state"] == "awaiting_exchange_spec"

        # Step 3: spec
        result = list(slow_echo("藏青色 M", [], enable_thinking=False))
        assert result[-1] == "换货申请已受理"
        assert _session_facts["dialogue_state"] == "completed"
        assert _session_facts["operation_detail"] == {"color": "藏青色", "size": "M"}


class TestStateMachineRefundFlow:
    def test_full_refund_flow(self, monkeypatch, tmp_path):
        from webrun import slow_echo, _session_facts

        orders_path = tmp_path / "orders.json"
        orders_path.write_text(json.dumps([_make_order()], ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr("webrun.config.ORDERS_JSON_PATH", str(orders_path))
        monkeypatch.setattr("orders.config.ORDERS_JSON_PATH", str(orders_path))
        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        monkeypatch.setattr("os.path.exists", lambda p: True)

        def _mock_hint(msg):
            return {"intent": "goods_operation", "sub_intent": "refund_only", "confidence": 0.95, "keywords": None}
        monkeypatch.setattr("webrun.quick_ecommerce_intent_hint", _mock_hint)

        stream_captured = []
        def _mock_stream(messages, enable_thinking):
            stream_captured.append(messages)
            yield "退款申请已受理"
        monkeypatch.setattr("webrun._yield_llm_stream", _mock_stream)

        _session_facts["dialogue_state"] = "init"
        _session_facts["verified_identity"] = False
        _session_facts["transfer_requested"] = False
        _session_facts["last_intent"] = None

        # identity
        list(slow_echo("退款", [], enable_thinking=False))
        list(slow_echo("ORD20250520001 13800138000", [], enable_thinking=False))
        # reason
        result = list(slow_echo("1", [], enable_thinking=False))
        assert any("退款金额" in r for r in result)
        assert _session_facts["dialogue_state"] == "awaiting_refund_amount"

        # amount
        result = list(slow_echo("1", [], enable_thinking=False))
        assert result[-1] == "退款申请已受理"
        assert _session_facts["dialogue_state"] == "completed"


class TestRejectThenFallback:
    def test_return_overdue_fallback(self, monkeypatch, tmp_path):
        from webrun import slow_echo, _session_facts

        past = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        order = _make_order()
        order["after_sales"]["return_deadline"] = past
        orders_path = tmp_path / "orders.json"
        orders_path.write_text(json.dumps([order], ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr("webrun.config.ORDERS_JSON_PATH", str(orders_path))
        monkeypatch.setattr("orders.config.ORDERS_JSON_PATH", str(orders_path))
        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        monkeypatch.setattr("os.path.exists", lambda p: True)

        def _mock_hint(msg):
            return {"intent": "goods_operation", "sub_intent": "return", "confidence": 0.95, "keywords": None}
        monkeypatch.setattr("webrun.quick_ecommerce_intent_hint", _mock_hint)

        stream_captured = []
        def _mock_stream(messages, enable_thinking):
            stream_captured.append(messages)
            yield "很抱歉"
        monkeypatch.setattr("webrun._yield_llm_stream", _mock_stream)

        _session_facts["dialogue_state"] = "init"
        _session_facts["verified_identity"] = False
        _session_facts["transfer_requested"] = False
        _session_facts["last_intent"] = None

        list(slow_echo("退货", [], enable_thinking=False))
        list(slow_echo("ORD20250520001 13800138000", [], enable_thinking=False))
        list(slow_echo("1", [], enable_thinking=False))  # reason
        list(slow_echo("1", [], enable_thinking=False))  # method
        assert _session_facts["dialogue_state"] == "rejected"
        # 校验失败时 prompt 应包含 fail_message
        assert "超过退货期限" in stream_captured[0][1]["content"]


class TestStateMachineWithUnknownIntent:
    """
    模拟真实场景：用户进入售后状态机后，第二轮输入（如订单号）被意图识别为 unknown，
    但系统应根据 dialogue_state 正确延续状态机，而不是走通用兜底。
    """
    def test_return_flow_with_unknown_second_intent(self, monkeypatch, tmp_path):
        from webrun import slow_echo, _session_facts

        orders_path = tmp_path / "orders.json"
        orders_path.write_text(json.dumps([_make_order()], ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr("webrun.config.ORDERS_JSON_PATH", str(orders_path))
        monkeypatch.setattr("orders.config.ORDERS_JSON_PATH", str(orders_path))
        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        monkeypatch.setattr("os.path.exists", lambda p: True)

        call_count = [0]
        def _mock_hint(msg):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"intent": "goods_operation", "sub_intent": "return", "confidence": 0.95, "keywords": None}
            # 第二轮输入订单号，规则预筛返回 ambiguous，LLM 返回 unknown
            return "ambiguous"
        monkeypatch.setattr("webrun.quick_ecommerce_intent_hint", _mock_hint)

        monkeypatch.setattr(
            "webrun.classify_ecommerce_intent",
            lambda msg: {"intent": "unknown", "sub_intent": None, "confidence": 0.35, "keywords": None}
        )

        stream_captured = []
        def _mock_stream(messages, enable_thinking):
            stream_captured.append(messages)
            yield "申请已受理"
        monkeypatch.setattr("webrun._yield_llm_stream", _mock_stream)

        _session_facts["dialogue_state"] = "init"
        _session_facts["verified_identity"] = False
        _session_facts["transfer_requested"] = False
        _session_facts["last_intent"] = None

        # Step 1: 用户说"我要退货" → 进入 awaiting_identity
        result = list(slow_echo("我要退货", [], enable_thinking=False))
        assert any("订单号和手机号" in r for r in result)
        assert _session_facts["dialogue_state"] == "awaiting_identity"

        # Step 2: 用户提供订单号，意图识别返回 unknown
        # 但 dialogue_state=awaiting_identity 应强制延续 goods_operation 意图
        result = list(slow_echo("ORD20250520001 13800138000", [], enable_thinking=False))
        assert any("退货原因" in r for r in result), f"实际返回: {result}"
        assert _session_facts["dialogue_state"] == "awaiting_reason"

        # Step 3: 后续流程正常推进
        result = list(slow_echo("1", [], enable_thinking=False))
        assert any("退货方式" in r for r in result)
        assert _session_facts["dialogue_state"] == "awaiting_return_method"

        result = list(slow_echo("1", [], enable_thinking=False))
        assert result[-1] == "申请已受理"
        assert _session_facts["dialogue_state"] == "completed"

    def test_return_flow_with_missing_last_intent(self, monkeypatch, tmp_path):
        """
        极端场景：last_intent 意外丢失（如被设为 None），但 dialogue_state 处于 goods_operation
        独有的状态（awaiting_reason），且 last_sub_intent 仍保留。系统应根据 dialogue_state
        兜底推断 goods_operation 意图，并恢复 sub_intent。
        """
        from webrun import slow_echo, _session_facts

        orders_path = tmp_path / "orders.json"
        orders_path.write_text(json.dumps([_make_order()], ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr("webrun.config.ORDERS_JSON_PATH", str(orders_path))
        monkeypatch.setattr("orders.config.ORDERS_JSON_PATH", str(orders_path))
        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        monkeypatch.setattr("os.path.exists", lambda p: True)

        monkeypatch.setattr("webrun.quick_ecommerce_intent_hint", lambda msg: "ambiguous")
        monkeypatch.setattr(
            "webrun.classify_ecommerce_intent",
            lambda msg: {"intent": "unknown", "sub_intent": None, "confidence": 0.35, "keywords": None}
        )

        _session_facts["dialogue_state"] = "awaiting_reason"
        _session_facts["verified_identity"] = True
        _session_facts["bound_order_id"] = "ORD20250520001"
        _session_facts["bound_phone"] = "13800138000"
        _session_facts["transfer_requested"] = False
        _session_facts["last_intent"] = None  # 模拟丢失
        _session_facts["last_sub_intent"] = "return"  # 但子意图仍保留

        result = list(slow_echo("1", [], enable_thinking=False))
        assert any("退货方式" in r for r in result), f"实际返回: {result}"
        assert _session_facts["dialogue_state"] == "awaiting_return_method"

    def test_logistics_flow_with_unknown_second_intent(self, monkeypatch, tmp_path):
        """
        物流场景：第二轮输入被意图识别为 unknown，应根据 dialogue_state 兜底。
        """
        from webrun import slow_echo, _session_facts
        import webrun

        orders_path = tmp_path / "orders.json"
        orders_path.write_text(json.dumps([_make_order()], ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr("webrun.config.ORDERS_JSON_PATH", str(orders_path))
        monkeypatch.setattr("orders.config.ORDERS_JSON_PATH", str(orders_path))
        monkeypatch.setattr("config.config.ENABLE_INTENT_ROUTING", True)
        monkeypatch.setattr("os.path.exists", lambda p: True)

        call_count = [0]
        def _mock_hint(msg):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"intent": "logistics", "sub_intent": "track", "confidence": 0.95, "keywords": None}
            return "ambiguous"
        monkeypatch.setattr("webrun.quick_ecommerce_intent_hint", _mock_hint)

        monkeypatch.setattr(
            "webrun.classify_ecommerce_intent",
            lambda msg: {"intent": "unknown", "sub_intent": None, "confidence": 0.35, "keywords": None}
        )

        llm_captured = []
        def _mock_llm_track(order, message, history, llm_client):
            llm_captured.append("track")
            return "物流轨迹回复"
        monkeypatch.setattr(webrun, "_handle_logistics_track", _mock_llm_track)

        _session_facts["dialogue_state"] = "init"
        _session_facts["verified_identity"] = False
        _session_facts["transfer_requested"] = False
        _session_facts["last_intent"] = None

        # Step 1: 查询物流
        result = list(slow_echo("快递到哪了", [], enable_thinking=False))
        assert any("订单号和手机号" in r for r in result)
        assert _session_facts["dialogue_state"] == "awaiting_identity"

        # Step 2: 提供订单号，意图识别返回 unknown，但应兜底到 logistics
        result = list(slow_echo("ORD20250520001 13800138000", [], enable_thinking=False))
        assert result[-1] == "物流轨迹回复"
        assert _session_facts["dialogue_state"] == "completed"
        assert llm_captured == ["track"]
