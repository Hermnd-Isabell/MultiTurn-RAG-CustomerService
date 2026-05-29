# AGENTS.md — RAG_project_zhyd 项目代理指南

> 本文件面向 AI 编码代理。项目的主要自然语言为**中文**（注释、文档、提交信息、UI 文案均以中文为主）。

---

## 项目概述

`RAG_project_zhyd` 是一个**智能电商客服问答系统**，基于 RAG（检索增强生成）架构，支持商品信息咨询、物流查询、售后操作（退货/换货/退款）、商品推荐等多轮对话能力。

核心特性：
- **商品向量检索**：FAISS 保存商品知识库子段落稠密向量（语义检索），支持按模块（规格材质、设计工艺等）精准召回。
- **意图路由层**：通过规则预筛（`quick_ecommerce_intent_hint`）+ LLM 分类（`classify_ecommerce_intent`）减少无效 LLM 调用，支持 5 大电商意图（商品信息 / 物流 / 推荐 / 售后 / 闲聊）。
- **状态机驱动的多轮对话**：物流查询、售后操作、商品推荐等复杂流程内置状态机，支持身份验证、信息收集、确认等完整交互链路。
- **Gradio Web UI**：三标签页界面（智能客服 / 知识库管理 / 配置），支持运行时热更新配置。
- **多轮对话**：`slow_echo` 将 Gradio 的 `history` 参数转换为 OpenAI `messages` 格式，支持上下文记忆与指代消解。

数据流：
```
.md / .docx（商品知识库文档）
    │
    ▼
build_product_index.py（切分 Chunk → 向量化）
    │
    ▼
products.npz（FAISS 向量库，格式：sku_id|module|sub_module）
    │
    ▼
Gradio 问答界面（用户提问 → 意图识别 → 检索 → LLM 生成答案）
```

---

## 技术栈

- **语言**：Python 3.10+
- **向量模型**：`sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`)
- **向量检索**：`faiss-cpu` (`IndexFlatL2`)
- **LLM 接口**：OpenAI SDK，兼容任意 OpenAI 格式 API（默认 `https://api.deepseek.com`，模型 `deepseek-chat`）
- **Web UI**：Gradio (`gradio.Blocks`，messages 格式)
- **配置管理**：`python-dotenv` 读取 `.env`
- **测试框架**：pytest + pytest-mock

---

## 目录结构与模块划分

```
├── pkg/
│   ├── config.py          # 配置单例：从 .env 加载，支持运行时热更新
│   ├── embed.py           # 向量化、FAISS 检索、电商意图识别、ES 连接
│   ├── webrun.py          # Gradio UI、问答主流程 slow_echo、状态机逻辑
│   └── orders.py          # 订单数据模型与售后校验逻辑
├── tests/
│   ├── conftest.py        # 全局 fixture：stub 重依赖、注入 sys.path、默认 env、缓存清理
│   ├── test_config.py     # 配置解析测试
│   ├── test_embed.py      # embed.py 单元测试（意图、缓存、检索）
│   ├── test_webrun.py     # webrun.py 单元测试（slow_echo 分支、打分、配置更新）
│   ├── test_orders.py     # 订单与售后逻辑测试
│   ├── test_goods_operation.py  # 售后操作解析测试
│   ├── test_logistics.py  # 物流状态机测试
│   ├── test_product_info.py     # 商品信息检索测试
│   └── test_transfer.py   # 转人工逻辑测试
├── data/
│   ├── 商品知识库.docx    # 示例商品知识库（build_product_index.py 可读取）
│   ├── products.npz       # 商品向量库（由 build_product_index.py 生成）
│   ├── orders.json        # 示例订单数据
│   └── goods_operation.xlsx     # 售后操作规则表
├── build_product_index.py # 商品知识库向量化脚本（遍历 .md/.docx → 切分 → 编码 → .npz）
├── .env                   # 环境变量（ES 连接、OpenAI Key、功能开关等）
├── requirements.txt       # 生产依赖
├── requirements-dev.txt   # 测试依赖（pytest、pytest-mock）
├── run_rag.bat            # 启动 Gradio 应用（Windows）
├── run_tests.bat          # 运行 pytest 测试套件（Windows）
└── pytest.ini             # pytest 配置
```

---

## 构建与运行命令

项目脚本均为 **Windows 批处理（.bat）**，且假设 `venv/` 位于仓库根目录。

### 环境初始化

```bat
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
.\venv\Scripts\pip install -r requirements-dev.txt
```

### 启动应用

```bat
:: 启动 Gradio 应用
run_rag.bat
```

手动启动（等价于 `run_rag.bat`）：

```bat
set KMP_DUPLICATE_LIB_OK=TRUE
python pkg/webrun.py
```

应用默认绑定 `0.0.0.0`，在浏览器中打开 Gradio 输出的本地地址即可使用。

> **注意**：`KMP_DUPLICATE_LIB_OK=TRUE` 是为了避免底层 OpenMP 动态库冲突导致崩溃，必须设置。

### 构建商品向量索引

```bat
python build_product_index.py
```

或指定输入目录与输出路径：

```bat
python -c "from build_product_index import build_product_vector_index; build_product_vector_index('./data', './data/products.npz')"
```

### 运行测试

```bat
run_tests.bat
```

或手动：

```bat
.\venv\Scripts\python.exe -m pytest tests -v --tb=short
```

---

## 测试策略与说明

- **215 个自动化测试**，运行耗时约 **0.4 秒**。
- **零外部服务依赖**：无需启动 ES、无需有效 OpenAI Key、无需真实 `.npz` 文件。
- `tests/conftest.py` 在 `pkg.*` 首次 import **之前**全局 stub 掉 `sentence_transformers` / `faiss` / `gradio` / `elasticsearch` / `openai` 等重依赖。
- 测试间通过 `autouse` fixture `_reset_module_caches` 清空 `_faiss_cache`、`_openai_client`、`_es_client`、`history`，避免副作用串扰。
- `restore_config` fixture 用于需要修改 `config` 的测试，在 teardown 阶段恢复原始属性。

### 测试覆盖要点

| 测试文件 | 覆盖内容 |
|----------|----------|
| `tests/test_config.py` | `ENABLE_INTENT_ROUTING` 布尔解析矩阵、ES/OpenAI/VECTOR_DB_PATH 配置读取与默认值 |
| `tests/test_embed.py` | `quick_ecommerce_intent_hint` 分支（chitchat/logistics/goods_operation/recommend/product_info）、`get_openai_client` 懒加载生命周期、`clear_faiss_cache` 两种形态、`retrieve_vector_and_text` 缓存行为、`classify_ecommerce_intent` JSON 解析与降级、`retrieve_with_context` 商品名置顶过滤 |
| `tests/test_webrun.py` | `slow_echo` 意图分支（关闭路由、异常自降级、检索异常兜底）、`_score_result_by_fields` 打分场景、`update_config` 写回与双清缓存、`get_es_client` 懒加载生命周期、多轮对话指代消解、session facts 更新与清空 |
| `tests/test_orders.py` | 订单查找、物流轨迹、异常状态、退换货窗口期校验 |
| `tests/test_goods_operation.py` | 售后原因解析、退货方式解析、退款金额解析、售后校验逻辑 |
| `tests/test_logistics.py` | 物流状态机流转、身份验证、修改地址/电话/收件人分支 |
| `tests/test_product_info.py` | 商品信息检索、模块匹配重排、购买阶段识别 |
| `tests/test_transfer.py` | 转人工触发（明确输入/连续追问/超纲业务）、确认/取消转人工 |

---

## 代码风格与开发约定

### 导入约定（极其重要）

`pkg/` 内部模块使用**裸导入（bare import）**，而非包限定导入：

```python
# ✅ 正确（项目惯例）
from config import config
from embed import retrieve_vector_and_text

# ❌ 错误（会破坏运行，因为 sys.path 未包含 pkg/ 作为包根）
from pkg.config import config
from pkg.embed import retrieve_vector_and_text
```

因此，**入口脚本必须从仓库根目录运行**，例如：

```bat
python pkg/webrun.py
```

从其他目录运行会导致 `ModuleNotFoundError`。

### 配置读取惯例

`config` 是一个**可变单例**（`Config` 类的实例）。Gradio "配置" 标签页会直接修改 `config.XXX` 属性。

- **禁止**在模块导入阶段捕获配置值（如 `es_host = config.ES_HOST`），否则配置热更新后不会生效。
- **必须**在请求处理函数内部实时读取 `config.ES_HOST`、`config.VECTOR_DB_PATH` 等属性。

### 客户端懒加载惯例

为避免模块导入阶段就连接外部服务（导致 ES 未启动时静默返回 `None`、OpenAI Key 被旧值缓存），项目统一使用懒加载 + 显式缓存失效模式：

| 模块 | 缓存变量 | get 函数 | clear 函数 |
|------|----------|----------|------------|
| `pkg/embed.py` | `_openai_client` | `get_openai_client()` | `clear_openai_client_cache()` |
| `pkg/webrun.py` | `_es_client` | `get_es_client()` | `clear_es_cache()` |

配置变更后必须调用 `clear_es_cache()` + `clear_openai_client_cache()`，确保下次请求按新配置重建客户端。

### 全局状态管理

- `_faiss_cache`（`embed.py`）：按 `.npz` 路径缓存 `(index, ids, texts)`，避免每次问答都执行硬盘 I/O。
- `history`（`webrun.py`）：模块级问答记忆列表，目前用于多轮对话拼接。
- `_session_facts`（`webrun.py`）：会话级结构化事实缓存，包含当前商品、已查询字段、对话状态机、绑定订单等。
- 重构时应**避免引入新的 `global` 变量**；若必须共享状态，优先使用模块级字典/列表，并通过 `clear_*` 函数提供显式失效接口。

---

## 核心模块职责

### `pkg/config.py`

- 加载 `.env` 中的环境变量。
- 提供单一可变 `config` 实例。

关键环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ES_HOST` | `127.0.0.1` | Elasticsearch 主机 |
| `ES_PORT` | `9200` | ES 端口 |
| `ES_USER` | `elastic` | ES 用户名 |
| `ES_PASSWORD` | `changeme` | ES 密码 |
| `ES_INDEX` | `zhyd` | ES 索引名 |
| `ES_SCHEME` | `http` | ES 协议 |
| `OPENAI_API_KEY` | — | API 密钥 |
| `OPENAI_BASE_URL` | `https://api.deepseek.com` | API 基础地址 |
| `LLM_MODEL` | `deepseek-chat` | 模型名称 |
| `VECTOR_DB_PATH` | `./embeddings2.npz` | 向量库存储路径（旧版兼容） |
| `PRODUCT_KB_PATH` | `./data/products` | 商品知识库目录 |
| `PRODUCT_INDEX_PATH` | `./data/products.npz` | 商品向量库路径 |
| `ORDERS_JSON_PATH` | `./data/orders.json` | 订单 JSON 路径 |
| `ENABLE_INTENT_ROUTING` | `1` | 意图路由开关 |

### `pkg/embed.py`

- **向量化与检索引擎**：`SentenceTransformer` 编码、`faiss.IndexFlatL2` 检索、`_faiss_cache` 内存缓存。
- **电商意图识别**：
  - `quick_ecommerce_intent_hint`：0-LLM 规则预筛（闲聊白名单 + 物流/售后/推荐/商品信息关键词正则）。
  - `classify_ecommerce_intent`：LLM 判断问题电商意图，返回 `intent` / `sub_intent` / `keywords` / `confidence` / `purchase_stage`。
- **检索函数**：`retrieve_vector_and_text`（基础向量检索）、`retrieve_with_context`（支持商品名置顶过滤）、`retrieve_product_info`（按模块匹配度重排）。
- **ES 连接**：`connect_elasticsearch`（懒加载，开发环境禁用 SSL 验证）。

### `pkg/webrun.py`

- **Gradio Web UI**：三标签页（智能客服 / 知识库管理 / 配置）。
- **问答主流程 `slow_echo`**：流式 generator，集成意图路由、商品检索、字段重排、历史记忆、LLM 调用。
  - `top_k=3` 召回商品知识库内容。
  - 若存在 `target_fields`，按 `_score_result_by_fields` 进行字段感知重排。
  - 无匹配时保留原 FAISS 顺序并打印 warning，不丢弃结果。
- **状态机**：物流查询（身份验证 → 轨迹/异常/修改）、售后操作（身份验证 → 原因 → 方式/规格/金额）、商品推荐（价格区间 → 搭配场景 → 使用行为）。
- **配置更新 `update_config`**：写回 config 并双清 ES/OpenAI 客户端缓存。
- **商品索引重建 `rebuild_product_index`**：调用 `build_product_index.py` 重建向量索引。

### `build_product_index.py`

- 遍历输入目录下的 `.md` 和 `.docx` 文件。
- 按一级标题（`# SKU-XXXX`）和二级标题（`## 模块名`）切分 Chunk。
- 使用 `embed.py` 中的全局 `model` 做 `encode`。
- 保存为 `.npz`：
  - `embeddings`: `(N, 384) float32`
  - `ids`: `List[str]` — 格式 `sku_id|module|sub_module`
  - `texts`: `List[Tuple[str, str, str]]` — `(module, sub_module, text)`

---

## 安全与风险注意事项

1. **ES 连接禁用 SSL 验证**
   `embed.py:connect_elasticsearch` 中设置了 `verify_certs=False`，仅在本地开发环境安全，生产环境部署时务必谨慎或移除。

2. **API 密钥存储**
   `OPENAI_API_KEY` 通过 `.env` 文件管理，该文件被 `.gitignore` 排除（不应提交到版本控制）。Gradio "配置" 标签页以明文/密文形式在浏览器端展示密钥，属于单用户本地场景设计。

3. **`.env` 文件不可读**
   系统层面将 `.env` 标记为敏感文件，代理无法直接读取；所有涉及密钥的操作应通过 `config.OPENAI_API_KEY` 间接引用，不要在代码中硬编码密钥。

4. **并发安全**
   当前设计基于 Gradio 的单进程/单线程事件循环，模块级缓存（`_faiss_cache`、`_es_client`、`_openai_client`）在该模式下工作正常。若未来切换到多 worker 部署，需将缓存改为线程安全结构或使用外部存储。

5. **输入安全**
   长文档可能导致内存峰值。ES 存储时对单条文档大小没有额外限制。

---

## 常见故障与调试提示

| 现象 | 可能原因 | 排查方向 |
|------|----------|----------|
| 启动时 `ModuleNotFoundError: No module named 'config'` | 运行目录不是仓库根目录 | 确保 CWD 是项目根，使用 `python pkg/webrun.py` |
| `无法连接到 Elasticsearch` / ES 返回 `None` | ES 未启动或配置错误 | 检查 `.env` 中的 `ES_HOST`/`ES_PORT`，或确认是否需要 ES |
| 每次问答都很慢 | `_faiss_cache` 未命中或 `.npz` 过大 | 首次加载后会缓存，检查日志是否有 "Embedding file path" 重复打印 |
| 商品索引重建失败 | `build_product_index.py` 导入失败或输入路径错误 | 确保从仓库根目录运行，且输入路径为存在 `.md` / `.docx` 的目录 |
| LLM 返回流式内容拼接异常 | OpenAI SDK 版本不兼容 | 当前锁定 `openai>=0.27.0`，新版 SDK 的 chunk 结构可能不同 |
| 测试报错 `pytest not found` | 未安装开发依赖 | 运行 `pip install -r requirements-dev.txt` |
| Gradio 界面消息为空 | `_respond` 未创建新 history 列表导致前端不刷新 | 已修复：每次 yield 前创建新的 history 对象 |

---

## 修改 checklist

在对本项目做任何代码修改前，请确认：

- [ ] 如果修改了配置相关逻辑，同步更新 `tests/test_config.py` 中的布尔矩阵或默认值断言。
- [ ] 如果修改了 `slow_echo` 的意图分支或字段重排逻辑，同步更新 `tests/test_webrun.py` 中对应的 mock 断言。
- [ ] 如果修改了 `embed.py` 中的缓存/检索逻辑，同步更新 `tests/test_embed.py`。
- [ ] 如果新增了对重依赖（如 `sentence_transformers`、`faiss`、`gradio`、`elasticsearch`、`openai`）的直接调用，检查 `tests/conftest.py` 中是否需要补充 stub。
- [ ] 修改后运行 `run_tests.bat` 或通过 `pytest tests -v --tb=short` 验证全部测试通过。
- [ ] 如果修改了 `.bat` 脚本或 `setup.py` 中的依赖列表，同步更新 `README.md` 和本文档的"构建与运行命令"章节。
