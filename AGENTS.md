# AGENTS.md — RAG_project_zhyd 项目代理指南

> 本文件面向 AI 编码代理。项目的主要自然语言为**中文**（注释、文档、提交信息、UI 文案均以中文为主）。

---

## 项目概述

`RAG_project_zhyd` 是一个面向《中华人民共和国药典》的**中文 RAG（检索增强生成）智能问答系统**。

核心特性：
- **双存储检索**：Elasticsearch 保存完整药品章节原文（全文检索/溯源），FAISS 保存子段落稠密向量（语义检索）。
- **意图路由层**：通过规则预筛（`quick_intent_hint`）+ LLM 分类（`classify_pharmacy_query`）减少无效 LLM 调用。
- **字段级精准问答**：`MedicineInfoStandardizer.extract_target_fields` 识别用户关心的药品属性（如性状、功能主治、处方等），对检索结果进行字段感知重排。
- **Gradio Web UI**：三标签页界面（药典问答 / 文档导入 / 配置），支持运行时热更新配置。
- **多轮对话**：`slow_echo` 将 Gradio 的 `history` 参数转换为 OpenAI `messages` 格式，支持上下文记忆。

数据流：
```
.docx（药典文档）
    │
    ▼
ES 索引（整章存储，以药品名为 doc_id）
    │
    ▼
FAISS 向量库（子段落向量化，按章节切分为【性状】【功能主治】等）
    │
    ▼
Gradio 问答界面（用户提问 → 检索 → LLM 生成答案）
```

---

## 技术栈

- **语言**：Python 3.10+
- **向量模型**：`sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`)
- **向量检索**：`faiss-cpu` (`IndexFlatL2`)
- **全文存储**：Elasticsearch 7.17.21（项目根目录已捆绑 `elasticsearch-7.17.21/`）
- **LLM 接口**：OpenAI SDK，兼容任意 OpenAI 格式 API（默认 `https://api.deepseek.com`，模型 `deepseek-chat`）
- **Web UI**：Gradio (`gradio.Blocks` + `ChatInterface`)
- **文档解析**：`python-docx`
- **配置管理**：`python-dotenv` 读取 `.env`
- **测试框架**：pytest + pytest-mock

---

## 目录结构与模块划分

```
├── pkg/
│   ├── config.py          # 配置单例：从 .env 加载，支持运行时热更新
│   ├── embed.py           # 向量化、FAISS 检索、意图路由、字段抽取、ES 交互
│   └── webrun.py          # Gradio UI、问答主流程 slow_echo、文档上传 UploadDoc
├── tests/
│   ├── conftest.py        # 全局 fixture：stub 重依赖、注入 sys.path、默认 env、缓存清理
│   ├── test_config.py     # 配置解析测试（ENABLE_INTENT_ROUTING 布尔矩阵等）
│   ├── test_embed.py      # embed.py 单元测试（意图、指纹、缓存、检索、解析）
│   └── test_webrun.py     # webrun.py 单元测试（slow_echo 分支、打分、配置更新、ES 存储）
├── .env                   # 环境变量（ES 连接、OpenAI Key、功能开关等）
├── requirements.txt       # 生产依赖
├── requirements-dev.txt   # 测试依赖（pytest、pytest-mock）
├── setup.py               # setuptools 打包配置
├── run_rag.bat            # 启动 Gradio 应用（Windows）
├── start_es.bat           # 启动捆绑的 Elasticsearch（Windows）
├── run_tests.bat          # 运行 pytest 测试套件（Windows）
├── pytest.ini             # pytest 配置（扫描 tests/、短回溯、详细输出）
└── 2020年药典一部.docx    # 示例/默认药典文档（.docx 分章逻辑与该格式强耦合）
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
:: 第 1 步：启动 Elasticsearch（不要关闭窗口）
start_es.bat

:: 第 2 步：启动 Gradio 应用
run_rag.bat
```

手动启动（等价于 `run_rag.bat`）：

```bat
set KMP_DUPLICATE_LIB_OK=TRUE
python pkg/webrun.py
```

应用默认绑定 `0.0.0.0`，在浏览器中打开 Gradio 输出的本地地址即可使用。

> **注意**：`KMP_DUPLICATE_LIB_OK=TRUE` 是为了避免底层 OpenMP 动态库冲突导致崩溃，必须设置。

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

- **89 个自动化测试**，运行耗时约 **0.35 秒**。
- **零外部服务依赖**：无需启动 ES、无需有效 OpenAI Key、无需真实 `.npz` 文件。
- `tests/conftest.py` 在 `pkg.*` 首次 import **之前**全局 stub 掉 `sentence_transformers` / `faiss` / `gradio` / `elasticsearch` / `openai` / `python-docx` 等重依赖。
- 测试间通过 `autouse` fixture `_reset_module_caches` 清空 `_faiss_cache`、`_openai_client`、`_es_client`、`history`，避免副作用串扰。
- `restore_config` fixture 用于需要修改 `config` 的测试，在 teardown 阶段恢复原始属性。

### 测试覆盖要点

| 测试文件 | 覆盖内容 |
|----------|----------|
| `tests/test_config.py` | `ENABLE_INTENT_ROUTING` 布尔解析矩阵、ES/OpenAI/VECTOR_DB_PATH 配置读取与默认值 |
| `tests/test_embed.py` | `quick_intent_hint` 三分支（chitchat/pharmacy/ambiguous）、`_es_index_fingerprint` 幂等性与稳定性、`get_openai_client` 懒加载生命周期、`clear_faiss_cache` 两种形态、`MedicineInfoStandardizer.extract_target_fields` 正常/异常/幻觉过滤、`retrieve_vector_and_text` 缓存行为与缺失文件容错、`extract_subsections` / `extract_drug_info` 解析 |
| `tests/test_webrun.py` | `slow_echo` 7 条意图分支（含关闭路由、异常自降级、检索异常兜底）、`_score_result_by_fields` 5 种打分场景、`update_config` 写回与双清缓存、`UploadDoc.store_in_elasticsearch` ES 未就绪降级与单条异常不阻断、`get_es_client` 懒加载生命周期 |

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
- 重构时应**避免引入新的 `global` 变量**；若必须共享状态，优先使用模块级字典/列表，并通过 `clear_*` 函数提供显式失效接口。

### 文档分章耦合

`.docx` 分章逻辑与**文档格式强耦合**：
- 章节边界检测依赖**首 run 字体大小为 12pt** 的段落（`UploadDoc.extract_titles_and_content`）。
- 子段落切分依赖 `【】` 标头（`embed.py:extract_subsections`，正则 `(?:【|t)(.+?)(?:】)`）。
- 中文标题清洗使用 `re.findall(r'[\u4e00-\u9fff]+', filename)`，只保留 CJK 字符。

若使用非标准药典格式文档，必须修改上述两个函数的分章/切分策略。

---

## 核心模块职责

### `pkg/config.py`

- 加载 `.env` 中的环境变量。
- 提供单一可变 `config` 实例。
- `ENABLE_INTENT_ROUTING` 为功能开关：默认 `1`（开启），设为 `0`/`false`/`no`/`off` 时完全关闭意图路由层。

关键环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ES_HOST` | `127.0.0.1` | Elasticsearch 主机 |
| `ES_PORT` | `9200` | ES 端口 |
| `ES_USER` | `elastic` | ES 用户名 |
| `ES_PASSWORD` | `changeme` | ES 密码 |
| `ES_INDEX` | `zhyd` | ES 索引名（上传时向量库同名） |
| `ES_SCHEME` | `http` | ES 协议 |
| `OPENAI_API_KEY` | — | API 密钥 |
| `OPENAI_BASE_URL` | `https://api.deepseek.com` | API 基础地址 |
| `LLM_MODEL` | `deepseek-chat` | 模型名称 |
| `VECTOR_DB_PATH` | `./embeddings2.npz` | 向量库存储路径 |
| `ENABLE_INTENT_ROUTING` | `1` | 意图路由开关 |

### `pkg/embed.py`

- **向量化与检索引擎**：`SentenceTransformer` 编码、`faiss.IndexFlatL2` 检索、`_faiss_cache` 内存缓存。
- **ES-FAISS 指纹脏检测**：`_es_index_fingerprint(hits)` 基于 ES `_id` 列表计算 MD5 指纹，`process_and_vectorize` 在重建前比对 `.npz` metadata，一致则跳过。
- **意图与标准化**：
  - `quick_intent_hint`：0-LLM 规则预筛（闲聊白名单 + 字段关键词正则）。
  - `classify_pharmacy_query`：LLM 判断问题是否与药学相关（`good`/`bad`）。
  - `MedicineInfoStandardizer`：字段提取与标准化，内置约 30 个药典字段（`field_list`）。
- **工具函数**：`extract_subsections`（按 `【】` 切分）、`extract_drug_info`（解析 LLM 输出）。

### `pkg/webrun.py`

- **Gradio Web UI**：三标签页（药典问答 / 文档导入 / 配置）。
- **问答主流程 `slow_echo`**：流式 generator，集成意图路由、字段重排、历史记忆、LLM 调用。
  - `top_k=3` 召回。
  - 若存在 `target_fields`，按 `_score_result_by_fields` 进行字段感知重排（精确匹配/互相包含得 1 分）。
  - 无匹配时保留原 FAISS 顺序并打印 warning，不丢弃结果。
- **文档上传 `UploadDoc`**：解析 `.docx`（按 12pt 字体分章）→ ES 存储 → 强制重建 FAISS（`force_rebuild=True`）。
- **配置更新 `update_config`**：写回 config 并双清 ES/OpenAI 客户端缓存。

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
   `UploadDoc.extract_titles_and_content` 和 `extract_subsections` 对 `.docx` 内容做纯文本提取，没有执行任意代码的风险，但长文档可能导致内存峰值。ES 存储时对单条文档大小没有额外限制。

---

## 常见故障与调试提示

| 现象 | 可能原因 | 排查方向 |
|------|----------|----------|
| 启动时 `ModuleNotFoundError: No module named 'config'` | 运行目录不是仓库根目录 | 确保 CWD 是项目根，使用 `python pkg/webrun.py` |
| `无法连接到 Elasticsearch` / ES 返回 `None` | ES 未启动或配置错误 | 先运行 `start_es.bat`，再检查 `.env` 中的 `ES_HOST`/`ES_PORT` |
| 每次问答都很慢 | `_faiss_cache` 未命中或 `.npz` 过大 | 首次加载后会缓存，检查日志是否有 "Embedding file path" 重复打印 |
| 上传文档后问答结果还是旧的 | FAISS 缓存未刷新 | `process_and_vectorize` 内部已调用 `clear_faiss_cache`，若手动替换 `.npz` 需手动调用 |
| `NameError: name 'field_list' is not defined` | 旧代码残留 | 已在 P0 修复：`MedicineInfoStandardizer.__init__` 改为引用类属性 `MedicineInfoStandardizer.field_list` |
| 测试报错 `pytest not found` | 未安装开发依赖 | 运行 `pip install -r requirements-dev.txt` |
| LLM 返回流式内容拼接异常 | OpenAI SDK 版本不兼容 | 当前锁定 `openai>=0.27.0`，新版 SDK 的 chunk 结构可能不同 |

---

## 修改 checklist

在对本项目做任何代码修改前，请确认：

- [ ] 如果修改了配置相关逻辑，同步更新 `tests/test_config.py` 中的布尔矩阵或默认值断言。
- [ ] 如果修改了 `slow_echo` 的意图分支或字段重排逻辑，同步更新 `tests/test_webrun.py` 中对应的 mock 断言。
- [ ] 如果修改了 `embed.py` 中的缓存/指纹/检索逻辑，同步更新 `tests/test_embed.py`。
- [ ] 如果新增了对重依赖（如 `sentence_transformers`、`faiss`、`gradio`、`elasticsearch`、`openai`）的直接调用，检查 `tests/conftest.py` 中是否需要补充 stub。
- [ ] 修改后运行 `run_tests.bat` 或通过 `pytest tests -v --tb=short` 验证全部测试通过。
- [ ] 如果修改了 `.bat` 脚本或 `setup.py` 中的依赖列表，同步更新 `README.md` 和本文档的"构建与运行命令"章节。
