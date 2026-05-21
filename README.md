# 电商智能客服大模型 — 意图识别与多轮对话优化

## 项目简介

基于大语言模型（LLM）与检索增强生成（RAG）技术的电商智能客服系统，支持五类用户意图识别、多轮对话状态机管理、商品知识库检索、订单售后处理等核心功能。

**核心特性**：
- **五类意图识别**：商品信息、物流信息、商品推荐、货物操作、闲聊，覆盖电商客服全链路
- **多轮对话状态机**：物流查询、商品推荐、售后处理等复杂流程支持多轮交互与状态记忆
- **模块化 RAG 检索**：商品知识库按 9 大模块切分索引，支持模块级精准召回
- **动态提示词**：根据用户购买意向阶段（了解/评估/决策）调整 LLM 回复策略
- **转人工兜底**：明确输入、连续追问、超纲业务三类场景自动触发转人工
- **Gradio 可视化 UI**：支持弹窗交互、实时状态面板、知识库管理

---

## 技术架构

```text
用户输入
    │
    ├─→ Gradio UI（智能客服 Tab）
    │       ├─→ 条件渲染弹窗（身份验证 / 原因选择 / 退货方式 / 价格输入）
    │       └─→ 实时状态面板（意图 / 状态 / 订单 / 价格区间 / 售后原因）
    │
    ├─→ 查询改写层（多轮指代消解：他/她/这个/刚才 → 具体商品名）
    │
    ├─→ 意图识别层
    │       ├─→ 规则预筛（quick_ecommerce_intent_hint）
    │       │       ├─→ 闲聊白名单 → chitchat
    │       │       ├─→ 物流关键词 → logistics
    │       │       ├─→ 操作关键词 → goods_operation
    │       │       ├─→ 推荐关键词 → product_recommend
    │       │       ├─→ 信息关键词 → product_info
    │       │       └─→ 模糊 → ambiguous
    │       └─→ LLM 分类器（classify_ecommerce_intent）
    │               → {intent, sub_intent, confidence, keywords, purchase_stage}
    │
    ├─→ 意图分发层
    │       ├─→ product_info → 模块级 RAG + 阶段化 Prompt（awareness/evaluation/decision）
    │       ├─→ logistics → 状态机（track / abnormal / modify）
    │       ├─→ product_recommend → 状态机（substitute / similar_style / matching）
    │       ├─→ goods_operation → 状态机（return / exchange / refund_only）
    │       ├─→ chitchat → 直接 LLM 回复
    │       └─→ unknown → 通用 RAG 兜底 / 转人工
    │
    ├─→ 检索层
    │       ├─→ 商品知识库（FAISS 向量检索 + 9 大模块重排）
    │       └─→ 订单 JSON（按 order_id + phone 精确查询）
    │
    └─→ LLM 生成层（流式输出，支持思考过程展示）
```

---

## 五类意图决策树

### 1. 商品信息（product_info）

**触发问法**："这件衣服什么材质"、"多少钱"、"包邮吗"、"怎么洗"

**子意图/维度**：
- `keywords`：基础信息 / 规格材质 / 设计工艺 / 护理保养 / 场景搭配 / 库存物流 / 售后政策 / 营销评价 / 版型身材
- `purchase_stage`：了解阶段(awareness) / 评估阶段(evaluation) / 决策阶段(decision)

**处理流程**：
1. 意图识别 → 提取 `keywords` 和 `purchase_stage`
2. 模块级 RAG 检索：按 `keywords` 对商品知识库做优先重排
3. 动态系统提示词：根据 `purchase_stage` 切换回复策略
   - awareness：通俗易懂、突出核心卖点
   - evaluation：详细参数、优缺点对比
   - decision：促销信息、库存状态、售后保障
4. LLM 流式生成回答

**状态节点**：无多轮状态机，单次交互完成

---

### 2. 物流信息（logistics）

**触发问法**："快递到哪了"、"包裹丢了"、"改地址"、"物流异常"

**子意图**：
- `track`（轨迹追踪）
- `abnormal`（异常处理：延误/丢失/损坏）
- `modify`（订单修改：地址/电话/收件人）

**处理流程（以 track 为例）**：
1. 触发 `logistics/track` 意图
2. 状态机进入 `awaiting_identity`
3. **弹窗收集**：订单号 + 手机号
4. 身份验证 → 绑定订单（`_verify_identity`）
5. 读取订单 `logistics.path`
6. LLM 整合路径+天气，生成语义化进度描述

**状态流转**：
```
init → awaiting_identity → identity_verified → reading_path → completed
```

---

### 3. 商品推荐（product_recommend）

**触发问法**："有没有类似的"、"搭配什么好"、"平替推荐"

**子意图**：
- `substitute`（平替商品）
- `similar_style`（风格类似）
- `matching`（搭配关联）

**处理流程（以 matching 为例）**：
1. 触发 `product_recommend/matching`
2. 状态机进入 `awaiting_price_range`
3. **弹窗收集**：价格区间（最低-最高）
4. 状态机进入 `awaiting_matching_scene`
5. 对话收集：使用场景（如"上班通勤"）
6. 状态机进入 `awaiting_matching_usage`
7. 对话收集：使用需求（如"经常出差需要轻便"）
8. 构造查询 → RAG 检索 → LLM 生成推荐话术

**状态流转**：
```
init → awaiting_price_range → awaiting_matching_scene → awaiting_matching_usage → retrieving → generating → completed
```

---

### 4. 货物操作（goods_operation）

**触发问法**："我要退货"、"换尺码"、"仅退款"

**子意图**：
- `return`（退货）
- `exchange`（换货）
- `refund_only`（仅退款）

**处理流程（以 return 为例）**：
1. 触发 `goods_operation/return`
2. 状态机进入 `awaiting_identity`
3. **弹窗收集**：订单号 + 手机号
4. 身份验证 → 绑定订单
5. 状态机进入 `awaiting_reason`
6. **弹窗选择**：不想要了 / 质量问题 / 描述不符 / 其他
7. 状态机进入 `awaiting_return_method`
8. **弹窗选择**：上门取件 / 自行寄回
9. 系统校验：是否在退货期内？是否重复申请？
10. 校验通过 → 写回订单 JSON → LLM 生成确认话术
11. 校验失败 → LLM 生成替代方案话术（如"超期建议换货"）

**状态流转**：
```
init → awaiting_identity → identity_verified → awaiting_reason
  → awaiting_return_method → validating → writing_back → completed/rejected
```

---

### 5. 转人工逻辑

**触发条件**（满足任一即触发）：
1. **明确输入**：用户发送"转人工"、"人工客服"、"投诉"
2. **连续追问**：同一问题（语义相似度>0.85）在同一意图下连续追问 ≥3 次
3. **超纲业务**：LLM 连续 2 轮识别为 `unknown` 且 `confidence<0.3`

**处理流程**：
1. 触发转人工标记（`transfer_requested=True`）
2. 系统询问："是否需要转接人工客服？请回复'是'或'否'。"
3. 用户确认"是" → 标记 `transfer_confirmed=True`，提示"正在为您转接人工客服……"
4. 用户拒绝"否" → 清除标记，恢复 `dialogue_state=init`，继续自助服务

---

## 快速开始

### 环境要求
- Python 3.10+
- Windows / Linux / macOS

### 安装依赖

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\pip install -r requirements-dev.txt   # 如需运行测试
```

### 配置环境变量

创建 `.env` 文件：

```env
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat

ES_HOST=127.0.0.1
ES_PORT=9200
ES_INDEX=products

PRODUCT_KB_PATH=./products/
PRODUCT_INDEX_PATH=./products.npz
ORDERS_JSON_PATH=./orders/orders.json
RETURN_WINDOW_DAYS=7
```

### 启动步骤

1. **构建商品知识库索引**（首次运行或商品文档更新后）：

```bash
python build_product_index.py
```

2. **启动智能客服系统**：

```bash
set KMP_DUPLICATE_LIB_OK=TRUE
python pkg\webrun.py
```

3. **打开浏览器访问**：http://localhost:7860

### 运行测试

```bash
venv\Scripts\python.exe -m pytest tests\ -v --tb=short
```

预期输出：`268 passed`

---

## 项目结构

```
RAG_project_zhyd/
├── pkg/
│   ├── config.py              # 配置管理（.env 加载 + 电商配置项）
│   ├── embed.py               # 向量化、FAISS 检索、意图识别、模块重排
│   ├── orders.py              # 订单 JSON 查询与更新（原子读写）
│   ├── webrun.py              # Gradio UI + 多轮状态机 + 流式生成
│   └── ...
├── tests/
│   ├── test_embed.py          # 意图识别测试
│   ├── test_orders.py         # 订单操作测试
│   ├── test_logistics.py      # 物流状态机测试
│   ├── test_recommend.py      # 推荐状态机测试
│   ├── test_product_info.py   # 商品信息检索 + 阶段化 Prompt 测试
│   ├── test_transfer.py       # 转人工触发 + 确认对话测试
│   ├── test_goods_operation.py # 售后状态机完整流程测试
│   └── conftest.py            # pytest fixtures（重依赖 stub + 缓存清理）
├── products/                  # 商品 Markdown 文档目录
├── orders/
│   └── orders.json            # 订单数据（结构化 JSON）
├── build_product_index.py     # 商品索引构建脚本
├── requirements.txt           # 生产依赖
├── requirements-dev.txt       # 开发依赖
└── README.md                  # 本文档
```

---

## 测试报告

| 测试模块 | 测试数量 | 覆盖内容 |
|---|---|---|
| 配置解析（TestConfig） | 10+ | 布尔矩阵 / 默认值 / 路径配置 |
| 意图识别（TestEcommerceIntent） | 12+ | 规则预筛 5 类 + LLM 分类 + 置信度降级 + purchase_stage |
| 订单操作（TestOrders） | 12 | 加载 / 查询 / 更新 / 校验库存 |
| 物流状态机（TestLogistics） | 18 | 身份验证 / 轨迹追踪 / 异常处理 / 订单修改 |
| 推荐状态机（TestRecommend） | 24 | 价格解析 5 种格式 / 搭配追问 / RAG 检索 |
| 商品信息检索（TestProductInfo） | 16 | 模块重排 / 阶段化 Prompt / 端到端分支 |
| 转人工逻辑（TestTransfer） | 17 | 明确输入 / 连续追问 / 超纲业务 / 确认对话 |
| 售后状态机（TestGoodsOperation） | 31 | 原因解析 / 校验规则 / 写回订单 / 完整流程 |
| **总计** | **268** | **全部通过** |

运行命令：

```bash
pytest tests/ -v --tb=short
# 预期输出：268 passed in ~0.5s
```

---

## 技术栈

| 层级 | 技术/库 | 用途 |
|---|---|---|
| 前端 UI | Gradio | 三 Tab 交互界面、条件渲染弹窗、实时状态面板 |
| 大语言模型 | OpenAI SDK（兼容接口） | 意图分类、查询改写、语义生成 |
| 向量模型 | sentence-transformers | `paraphrase-multilingual-MiniLM-L12-v2` 编码 |
| 向量检索 | FAISS (IndexFlatL2) | 商品知识库语义检索 + 模块重排 |
| 全文存储 | Elasticsearch 7.17 | 商品全文索引（预留扩展） |
| 订单数据 | JSON 文件 | 结构化订单查询与更新 |
| 测试框架 | pytest + pytest-mock | 自动化单元测试（零外部依赖） |

---

## 已知限制与后续优化

1. **会话隔离**：当前 `_session_facts` 为模块级变量，Gradio 单进程模式下天然会话隔离；多 worker 部署时需改为 per-session 存储（如 Redis）
2. **订单数据持久化**：当前使用本地 JSON 文件，高并发场景建议迁移至关系型数据库
3. **情绪识别**：已实现转人工兜底，但未在每条回复前做实时情绪检测（可作为后续优化）
4. **商品推荐多样性**：当前 RAG 检索基于向量相似度，推荐结果多样性可通过引入 MMR（最大边际相关性）算法提升
5. **弹窗交互粒度**：Phase 7 已实现原因/方式/身份/价格弹窗，后续可细化为规格选择下拉框、日期选择器等更精细组件

---

## 作者

- 项目：电商智能客服大模型的意图识别与多轮对话优化
- 课程：大语言模型应用开发
