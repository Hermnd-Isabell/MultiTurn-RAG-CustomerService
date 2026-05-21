import json
import os
from config import config


def load_orders(orders_path=None):
    """加载订单 JSON 文件，返回订单列表。文件不存在时返回 []。"""
    path = orders_path or getattr(config, "ORDERS_JSON_PATH", "./orders/orders.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except (json.JSONDecodeError, OSError):
        return []


def find_order(order_id, phone, orders=None):
    """
    根据订单号 + 手机号精确匹配订单。
    返回订单 dict 或 None。
    """
    if orders is None:
        orders = load_orders()
    for order in orders:
        if order.get("order_id") == order_id and order.get("user_phone") == phone:
            return order
    return None


def get_order_sku(order):
    """从订单中提取 sku_id（假设订单中有 items[0].sku_id）"""
    if not order:
        return None
    items = order.get("items", [])
    if items and isinstance(items, list):
        return items[0].get("sku_id")
    return None


def get_order_logistics_path(order):
    """从订单的 logistics.path 中提取物流节点列表"""
    if not order:
        return []
    logistics = order.get("logistics", {})
    return logistics.get("path", []) or []


def get_order_abnormal_status(order):
    """返回订单异常状态：None / delay / lost / damaged"""
    if not order:
        return None
    logistics = order.get("logistics", {})
    return logistics.get("abnormal_status")


def get_order_return_window(order):
    """返回退货截止日期（字符串或 datetime）"""
    if not order:
        return None
    after_sales = order.get("after_sales", {})
    return after_sales.get("return_deadline")


def check_exchange_inventory(order, new_spec):
    """检查换货目标规格是否有库存。返回 bool。"""
    if not order or not new_spec:
        return False
    after_sales = order.get("after_sales", {})
    inventory = after_sales.get("exchange_inventory", {})
    # new_spec 格式示例：{"color": "米白色", "size": "M"}
    if not isinstance(new_spec, dict):
        return False
    color = new_spec.get("color")
    size = new_spec.get("size")
    if color is None or size is None:
        return False
    color_inventory = inventory.get(color, {})
    if not isinstance(color_inventory, dict):
        return False
    stock = color_inventory.get(size, 0)
    return bool(stock and stock > 0)


def update_order_field(order_id, field, new_value, orders_path=None):
    """
    修改订单 JSON 中的指定字段。
    field 支持: "address", "phone", "recipient"
    返回 (success: bool, message: str)
    """
    path = orders_path or getattr(config, "ORDERS_JSON_PATH", "./orders/orders.json")

    all_orders = load_orders(path)
    if not all_orders:
        return False, "订单数据为空或文件不存在"

    target_order = None
    for order in all_orders:
        if order.get("order_id") == order_id:
            target_order = order
            break

    if target_order is None:
        return False, "未找到指定订单"

    if field == "address":
        if "logistics" not in target_order:
            target_order["logistics"] = {}
        target_order["logistics"]["address"] = new_value
    elif field == "phone":
        target_order["user_phone"] = new_value
        if "contact" in target_order:
            target_order["contact"]["phone"] = new_value
    elif field == "recipient":
        if "contact" not in target_order:
            target_order["contact"] = {}
        target_order["contact"]["name"] = new_value
        target_order["recipient"] = new_value
    else:
        return False, f"不支持的字段: {field}"

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_orders, f, ensure_ascii=False, indent=2)
        return True, "更新成功"
    except Exception as e:
        return False, f"写入文件失败: {e}"
