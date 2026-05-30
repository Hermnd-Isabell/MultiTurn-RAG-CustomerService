"""
pkg/orders.py 测试覆盖：
- load_orders：mock JSON 文件加载与容错
- find_order：精确匹配订单号+手机号
- get_order_sku / get_order_abnormal_status：字段提取
"""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# load_orders
# ---------------------------------------------------------------------------
class TestLoadOrders:
    def test_load_valid_json(self, tmp_path):
        from orders import load_orders

        orders_data = [
            {"order_id": "ORD001", "user_phone": "13800138000"},
            {"order_id": "ORD002", "user_phone": "13800138001"},
        ]
        path = tmp_path / "orders.json"
        path.write_text(json.dumps(orders_data), encoding="utf-8")

        with patch("orders.config.ORDERS_JSON_PATH", str(path)):
            result = load_orders()

        assert len(result) == 2
        assert result[0]["order_id"] == "ORD001"

    def test_file_not_found_returns_empty(self):
        from orders import load_orders

        with patch("orders.config.ORDERS_JSON_PATH", "/nonexistent/path/orders.json"):
            result = load_orders()

        assert result == []

    def test_invalid_json_returns_empty(self, tmp_path):
        from orders import load_orders

        path = tmp_path / "bad_orders.json"
        path.write_text("not json", encoding="utf-8")

        with patch("orders.config.ORDERS_JSON_PATH", str(path)):
            result = load_orders()

        assert result == []


# ---------------------------------------------------------------------------
# find_order
# ---------------------------------------------------------------------------
class TestFindOrder:
    def test_find_order_hit(self):
        from orders import find_order

        orders = [
            {"order_id": "ORD20250520001", "user_phone": "13800138000", "items": []},
            {"order_id": "ORD20250520002", "user_phone": "13900139000", "items": []},
        ]
        result = find_order("ORD20250520001", "13800138000", orders)
        assert result is not None
        assert result["order_id"] == "ORD20250520001"

    def test_find_order_not_found(self):
        from orders import find_order

        orders = [
            {"order_id": "ORD20250520001", "user_phone": "13800138000", "items": []},
        ]
        result = find_order("ORD20250520001", "99999999999", orders)
        assert result is None

    def test_find_order_uses_load_orders_when_none_passed(self, tmp_path):
        from orders import find_order

        orders_data = [
            {"order_id": "ORD001", "user_phone": "13800138000"},
        ]
        path = tmp_path / "orders.json"
        path.write_text(json.dumps(orders_data), encoding="utf-8")

        with patch("orders.config.ORDERS_JSON_PATH", str(path)):
            result = find_order("ORD001", "13800138000")

        assert result is not None


# ---------------------------------------------------------------------------
# get_order_sku
# ---------------------------------------------------------------------------
class TestGetOrderSku:
    def test_extract_first_item_sku(self):
        from orders import get_order_sku

        order = {
            "items": [
                {"sku_id": "SKU-2025-DRESS-001", "product_name": "连衣裙"},
            ]
        }
        assert get_order_sku(order) == "SKU-2025-DRESS-001"

    def test_empty_items_returns_none(self):
        from orders import get_order_sku

        assert get_order_sku({"items": []}) is None
        assert get_order_sku({}) is None


# ---------------------------------------------------------------------------
# get_order_abnormal_status
# ---------------------------------------------------------------------------
class TestGetOrderAbnormalStatus:
    def test_normal_order(self):
        from orders import get_order_abnormal_status

        order = {"logistics": {"abnormal_status": None}}
        assert get_order_abnormal_status(order) is None

    def test_delay_order(self):
        from orders import get_order_abnormal_status

        order = {"logistics": {"abnormal_status": "delay"}}
        assert get_order_abnormal_status(order) == "delay"

    def test_missing_logistics(self):
        from orders import get_order_abnormal_status

        assert get_order_abnormal_status({}) is None


# ---------------------------------------------------------------------------
# get_order_logistics_path
# ---------------------------------------------------------------------------
class TestGetOrderLogisticsPath:
    def test_extract_path(self):
        from orders import get_order_logistics_path

        order = {
            "logistics": {
                "path": [
                    {"node": "杭州发货仓", "time": "2025-05-15T14:00:00"},
                ]
            }
        }
        path = get_order_logistics_path(order)
        assert len(path) == 1
        assert path[0]["node"] == "杭州发货仓"

    def test_missing_path_returns_empty(self):
        from orders import get_order_logistics_path

        assert get_order_logistics_path({}) == []
        assert get_order_logistics_path({"logistics": {}}) == []


# ---------------------------------------------------------------------------
# get_order_return_window
# ---------------------------------------------------------------------------
class TestGetOrderReturnWindow:
    def test_return_deadline(self):
        from orders import get_order_return_window

        order = {"after_sales": {"return_deadline": "2025-05-27T23:59:59"}}
        assert get_order_return_window(order) == "2025-05-27T23:59:59"

    def test_missing_returns_none(self):
        from orders import get_order_return_window

        assert get_order_return_window({}) is None


# ---------------------------------------------------------------------------
# check_exchange_inventory
# ---------------------------------------------------------------------------
class TestCheckExchangeInventory:
    def test_in_stock(self):
        from orders import check_exchange_inventory

        order = {
            "after_sales": {
                "exchange_inventory": {
                    "米白色": {"S": 5, "M": 3, "L": 0},
                }
            }
        }
        assert check_exchange_inventory(order, {"color": "米白色", "size": "M"}) is True

    def test_out_of_stock(self):
        from orders import check_exchange_inventory

        order = {
            "after_sales": {
                "exchange_inventory": {
                    "米白色": {"S": 5, "M": 0, "L": 0},
                }
            }
        }
        assert check_exchange_inventory(order, {"color": "米白色", "size": "M"}) is False

    def test_invalid_spec(self):
        from orders import check_exchange_inventory

        assert check_exchange_inventory({}, None) is False
        assert check_exchange_inventory({}, "bad") is False
