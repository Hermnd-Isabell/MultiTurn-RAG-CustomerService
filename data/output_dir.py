import pandas as pd
import json
import os

# ========== 配置 ==========
excel_path = r'C:\Users\19929\Desktop\goods_operation.xlsx'
output_dir = r'C:\Users\19929\Desktop\output_json'
# ==========================

os.makedirs(output_dir, exist_ok=True)

def clean_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, str):
        v = v.replace('\\|', '|').strip()
        if v == '':
            return None
        if (v.startswith('{') and v.endswith('}')) or (v.startswith('[') and v.endswith(']')):
            try:
                return json.loads(v)
            except:
                pass
        return v
    if isinstance(v, (int, float)):
        return v
    return str(v)

# 读取Excel
xls = pd.ExcelFile(excel_path)
sheet_names = {name.strip(): name for name in xls.sheet_names}

# 读取intent_classification
intent_sheet = sheet_names.get('intent_classification')
df_intent = pd.read_excel(xls, intent_sheet, header=0, dtype=str)
df_intent.columns = [str(c).strip() for c in df_intent.columns]

if len(df_intent) > 0 and str(df_intent.iloc[0, 0]) == str(df_intent.columns[0]):
    df_intent = df_intent.iloc[1:].reset_index(drop=True)

intents = []
for _, row in df_intent.iterrows():
    record = {col: clean_value(row[col]) for col in df_intent.columns if clean_value(row[col]) is not None}
    if record:
        intents.append(record)

with open(os.path.join(output_dir, 'intent_classification.json'), 'w', encoding='utf-8') as f:
    json.dump(intents, f, ensure_ascii=False, indent=2)
print(f"✓ intent_classification.json: {len(intents)} 条")

# 读取phone_to_order，phone强制文本
order_sheet = sheet_names.get('phone_to_order')
df_orders = pd.read_excel(xls, order_sheet, header=0, dtype={'phone': str})
df_orders.columns = [str(c).strip() for c in df_orders.columns]

if len(df_orders) > 0 and str(df_orders.iloc[0, 0]) == str(df_orders.columns[0]):
    df_orders = df_orders.iloc[1:].reset_index(drop=True)

# 构建所有订单列表
all_orders = []

for _, row in df_orders.iterrows():
    phone = str(row['phone']) if pd.notna(row['phone']) else None
    order_id = clean_value(row['order_id'])
    
    item = {
        "sku_id": clean_value(row['item_sku_id']),
        "product_name": clean_value(row['item_name']),
        "specs": {
            "color": clean_value(row['product_info_color']),
            "size": clean_value(row['product_info_size'])
        },
        "price": float(clean_value(row['item_unit_price'])),
        "quantity": int(clean_value(row['item_quantity']))
    }
    
    logistics = {
        "status": clean_value(row['status']),
        "carrier": clean_value(row['logistics_carrier']),
        "tracking_no": clean_value(row['logistics_tracking_no']) or "",
        "path": clean_value(row['logistics_path']) or [],
        "estimated_arrival": clean_value(row['logistics_estimated_arrival']),
        "current_location": clean_value(row['logistics_current_location']),
        "abnormal_status": clean_value(row['logistics_abnormal_status']),
        "signed_time": clean_value(row['deliver_time']),
        "address": clean_value(row['logistics_address']),
        "recipient": clean_value(row['logistics_recipient']),
        "phone": phone
    }
    
    after_sales = {
        "return_window_days": int(clean_value(row['after_sales_return_window_days'])),
        "return_deadline": clean_value(row['after_sales_return_deadline']),
        "exchange_inventory": clean_value(row['after_sales_exchange_inventory']) or {},
        "refund_rules": {
            "unshipped": clean_value(row['after_sales_refund_rules_unshipped']),
            "shipped": clean_value(row['after_sales_refund_rules_shipped']),
            "delivered": clean_value(row['after_sales_refund_rules_delivered'])
        }
    }
    
    ops = clean_value(row['operations_history'])
    if not ops:
        ops = []
    
    order = {
        "order_id": order_id,
        "user_phone": phone,
        "create_time": clean_value(row['create_time']),
        "items": [item],
        "logistics": logistics,
        "after_sales": after_sales,
        "operations_history": ops
    }
    
    all_orders.append(order)

# 写入单个orders.json
with open(os.path.join(output_dir, 'orders.json'), 'w', encoding='utf-8') as f:
    json.dump(all_orders, f, ensure_ascii=False, indent=2)

print(f"✓ orders.json: {len(all_orders)} 个订单")
print(f"\n全部完成！JSON保存在: {output_dir}")
