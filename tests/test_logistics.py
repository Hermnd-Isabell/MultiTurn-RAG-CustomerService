"""
物流信息决策树与状态机测试：
- _verify_identity：正则提取 + 订单匹配
- _handle_logistics_track / _handle_logistics_abnormal：LLM prompt 构造
- _parse_modify_type / _validate_new_info：输入解析与校验
- update_order_field：JSON 写回
- slow_echo 完整状态机流转：track / modify
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _verify_identity
# ---------------------------------------------------------------------------
class TestVerifyIdentity:
    def test_verify_identity_success(self):
        from webrun import _verify_identity, clear_session_facts
        import webrun

        clear_session_facts()
        mock_order = {"order_id": "ORD001", "user_phone": "13800138000"}

        with patch.object(webrun, "find_order", return_value=mock_order):
            success, order = _verify_identity("ORD20250520001 13800138000", webrun._session_facts)

        assert success is True
        assert order == mock_order
        assert webrun._session_facts["verified_identity"] is True
        assert webrun._session_facts["bound_order_id"] == "ORD20250520001"
        assert webrun._session_facts["bound_phone"] == "13800138000"

    def test_verify_identity_fail(self):
        from webrun import _verify_identity, clear_session_facts
        import webrun

        clear_session_facts()
        with patch.object(webrun, "find_order", return_value=None):
            success, order = _verify_identity("ORD001 13800138000", webrun._session_facts)

        assert success is False
        assert order is None
        assert webrun._session_facts["verified_identity"] is False

    def test_verify_identity_regex_variants(self):
        from webrun import _verify_identity
        import webrun

        mock_order = {"order_id": "ORD001", "user_phone": "13800138000"}

        variants = [
            "ORD001 13800138000",
            "订单号ORD001，手机13800138000",
            "ord001, 13800138000",
            "我的订单是ORD001，电话13800138000",
        ]

        for text in variants:
            with patch.object(webrun, "find_order", return_value=mock_order):
                success, order = _verify_identity(text, {"verified_identity": False})
            assert success is True, f"应能解析: {text}"


# ---------------------------------------------------------------------------
# _handle_logistics_track
# ---------------------------------------------------------------------------
class TestHandleLogisticsTrack:
    def _make_llm(self, return_text):
        client = MagicMock(name="FakeLLM")
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = return_text
        client.chat.completions.create.return_value = resp
        return client

    def test_prompt_contains_path_nodes(self):
        from webrun import _handle_logistics_track

        order = {
            "order_id": "ORD001",
            "logistics": {
                "carrier": "中通快递",
                "current_location": "上海转运中心",
                "estimated_arrival": "2025-05-20T18:00:00",
                "path": [
                    {"node": "杭州发货仓", "time": "2025-05-15T14:00:00", "weather": "晴, 28°C"},
                    {"node": "上海转运中心", "time": "2025-05-16T03:00:00", "weather": "多云, 26°C"},
                ],
            },
        }
        llm = self._make_llm("测试物流回复")
        result = _handle_logistics_track(order, "快递到哪了", [], llm)

        assert result == "测试物流回复"
        call_args = llm.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "中通快递" in prompt
        assert "杭州发货仓" in prompt
        assert "晴, 28°C" in prompt
        assert "预计到达" in prompt


# ---------------------------------------------------------------------------
# _handle_logistics_abnormal
# ---------------------------------------------------------------------------
class TestHandleLogisticsAbnormal:
    def _make_llm(self, return_text):
        client = MagicMock(name="FakeLLM")
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = return_text
        client.chat.completions.create.return_value = resp
        return client

    def test_abnormal_delay(self):
        from webrun import _handle_logistics_abnormal

        order = {
            "order_id": "ORD001",
            "logistics": {
                "abnormal_status": "delay",
                "estimated_arrival": "2025-05-22T18:00:00",
                "current_location": "北京转运中心",
            },
        }
        llm = self._make_llm("延误安抚话术")
        result = _handle_logistics_abnormal(order, "我的快递怎么了", [], llm)

        assert result == "延误安抚话术"
        prompt = llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "延误" in prompt
        assert "2025-05-22" in prompt

    def test_abnormal_lost(self):
        from webrun import _handle_logistics_abnormal

        order = {
            "order_id": "ORD001",
            "logistics": {"abnormal_status": "lost"},
        }
        llm = self._make_llm("赔付说明话术")
        result = _handle_logistics_abnormal(order, "我的快递丢了", [], llm)

        prompt = llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "丢失" in prompt
        assert "赔付" in prompt

    def test_abnormal_none(self):
        from webrun import _handle_logistics_abnormal

        order = {
            "order_id": "ORD001",
            "logistics": {"abnormal_status": None, "carrier": "顺丰"},
        }
        llm = self._make_llm("订单正常话术")
        result = _handle_logistics_abnormal(order, "我的快递有问题吗", [], llm)

        prompt = llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "正常" in prompt


# ---------------------------------------------------------------------------
# _parse_modify_type
# ---------------------------------------------------------------------------
class TestParseModifyType:
    def test_valid_inputs(self):
        from webrun import _parse_modify_type

        assert _parse_modify_type("1") == "address"
        assert _parse_modify_type("地址") == "address"
        assert _parse_modify_type("2") == "phone"
        assert _parse_modify_type("电话") == "phone"
        assert _parse_modify_type("3") == "recipient"
        assert _parse_modify_type("收件人") == "recipient"

    def test_invalid_inputs(self):
        from webrun import _parse_modify_type

        assert _parse_modify_type("垃圾") is None
        assert _parse_modify_type("") is None
        assert _parse_modify_type(None) is None


# ---------------------------------------------------------------------------
# _validate_new_info
# ---------------------------------------------------------------------------
class TestValidateNewInfo:
    def test_address(self):
        from webrun import _validate_new_info

        assert _validate_new_info("address", "浙江省杭州市西湖区xx街道") == (True, "")
        assert _validate_new_info("address", "短") == (False, "地址长度不能少于5个字符")
        assert _validate_new_info("address", "abcdefg") == (False, "地址需包含省、市、区/县等行政区域信息")

    def test_phone(self):
        from webrun import _validate_new_info

        assert _validate_new_info("phone", "13800138000") == (True, "")
        valid, msg = _validate_new_info("phone", "123456")
        assert valid is False
        assert "手机号格式不正确" in msg

    def test_recipient(self):
        from webrun import _validate_new_info

        assert _validate_new_info("recipient", "张三") == (True, "")
        valid, msg = _validate_new_info("recipient", "a")
        assert valid is False
        assert "不能少于2个字符" in msg
        valid2, msg2 = _validate_new_info("recipient", "a" * 21)
        assert valid2 is False
        assert "不能超过20个字符" in msg2


# ---------------------------------------------------------------------------
# update_order_field（通过 orders 模块）
# ---------------------------------------------------------------------------
class TestUpdateOrderField:
    def test_update_address(self, tmp_path):
        from orders import update_order_field

        orders_data = [
            {"order_id": "ORD001", "user_phone": "13800138000", "logistics": {}},
        ]
        path = tmp_path / "orders.json"
        path.write_text(json.dumps(orders_data), encoding="utf-8")

        success, msg = update_order_field("ORD001", "address", "浙江省杭州市", str(path))
        assert success is True

        updated = json.loads(path.read_text(encoding="utf-8"))
        assert updated[0]["logistics"]["address"] == "浙江省杭州市"

    def test_update_phone(self, tmp_path):
        from orders import update_order_field

        orders_data = [{"order_id": "ORD001", "user_phone": "13800138000"}]
        path = tmp_path / "orders.json"
        path.write_text(json.dumps(orders_data), encoding="utf-8")

        success, msg = update_order_field("ORD001", "phone", "13900139000", str(path))
        assert success is True

        updated = json.loads(path.read_text(encoding="utf-8"))
        assert updated[0]["user_phone"] == "13900139000"

    def test_order_not_found(self, tmp_path):
        from orders import update_order_field

        path = tmp_path / "orders.json"
        path.write_text(json.dumps([{"order_id": "ORD001", "user_phone": "13800138000"}]), encoding="utf-8")

        success, msg = update_order_field("ORD999", "address", "xxx", str(path))
        assert success is False
        assert "未找到" in msg


# ---------------------------------------------------------------------------
# slow_echo 状态机完整流转
# ---------------------------------------------------------------------------
class TestStateMachineFlow:
    def _mock_llm_stream(self):
        """构造 slow_echo 可用的 stream mock。"""
        def _stream(*_a, **_k):
            for piece in ["测试", "回复"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = piece
                yield chunk
        client = MagicMock()
        client.chat.completions.create.side_effect = _stream
        return client

    def test_state_machine_flow_track(self, sample_faiss_data, restore_config):
        """完整 track 流程：init → awaiting_identity → completed。"""
        import webrun

        webrun.config.PRODUCT_INDEX_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = True
        webrun.clear_session_facts()

        # Turn 1: 触发 logistics track
        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "logistics", "sub_intent": "track", "confidence": 1.0}):
            chunks = list(webrun.slow_echo("快递到哪了", []))

        assert webrun._session_facts["dialogue_state"] == "awaiting_identity"
        assert "订单号" in chunks[-1]

        # Turn 2: 提供身份信息
        mock_order = {
            "order_id": "ORD001",
            "user_phone": "13800138000",
            "logistics": {
                "carrier": "中通",
                "current_location": "上海",
                "estimated_arrival": "2025-05-20",
                "path": [{"node": "杭州", "time": "2025-05-15", "weather": "晴"}],
            },
        }
        with patch.object(webrun, "_verify_identity", return_value=(True, mock_order)), \
             patch.object(webrun, "_handle_logistics_track", return_value="您的包裹已到上海。"):
            chunks = list(webrun.slow_echo("ORD001 13800138000", [("快递到哪了", "...")]))

        assert "上海" in chunks[-1]
        assert webrun._session_facts["dialogue_state"] == "completed"

    def test_state_machine_flow_modify(self, sample_faiss_data, restore_config):
        """完整 modify 流程：init → awaiting_identity → awaiting_modify_type → awaiting_new_info → completed。"""
        import webrun

        webrun.config.PRODUCT_INDEX_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = True
        webrun.clear_session_facts()

        mock_order = {
            "order_id": "ORD001",
            "user_phone": "13800138000",
            "logistics": {},
        }

        # Turn 1: 触发 logistics modify
        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "logistics", "sub_intent": "modify", "confidence": 1.0}):
            chunks = list(webrun.slow_echo("改地址", []))

        assert webrun._session_facts["dialogue_state"] == "awaiting_identity"

        # Turn 2: 身份验证
        with patch.object(webrun, "_verify_identity", return_value=(True, mock_order)):
            chunks = list(webrun.slow_echo("ORD001 13800138000", [("改地址", "...")]))

        assert webrun._session_facts["dialogue_state"] == "awaiting_modify_type"
        assert "地址" in chunks[-1] or "电话" in chunks[-1]

        # Turn 3: 选择修改类型
        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "logistics", "sub_intent": "modify", "confidence": 1.0}):
            chunks = list(webrun.slow_echo("地址", [("改地址", "..."), ("ORD001...", "...")]))

        assert webrun._session_facts["dialogue_state"] == "awaiting_new_info"
        assert "请输入新的" in chunks[-1]

        # Turn 4: 提供新地址并写回
        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "logistics", "sub_intent": "modify", "confidence": 1.0}), \
             patch.object(webrun, "update_order_field", return_value=(True, "更新成功")):
            chunks = list(webrun.slow_echo("浙江省杭州市西湖区xx街道", [
                ("改地址", "..."), ("ORD001...", "..."), ("地址", "..."),
            ]))

        assert webrun._session_facts["dialogue_state"] == "completed"
        assert "修改成功" in chunks[-1]

    def test_state_machine_awaiting_identity_fail(self, sample_faiss_data, restore_config):
        """身份验证失败时保持 awaiting_identity 并提示重新输入。"""
        import webrun

        webrun.config.PRODUCT_INDEX_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = True
        webrun.clear_session_facts()

        # Turn 1
        with patch.object(webrun, "quick_ecommerce_intent_hint", return_value={"intent": "logistics", "sub_intent": "track", "confidence": 1.0}):
            list(webrun.slow_echo("快递", []))

        assert webrun._session_facts["dialogue_state"] == "awaiting_identity"

        # Turn 2: 错误身份
        with patch.object(webrun, "_verify_identity", return_value=(False, None)):
            chunks = list(webrun.slow_echo("ORD999 13999999999", [("快递", "...")]))

        assert webrun._session_facts["dialogue_state"] == "awaiting_identity"
        assert "有误" in chunks[-1]
