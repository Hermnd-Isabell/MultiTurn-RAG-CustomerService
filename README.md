# RAG_project_zhyd — 中文药典智能问答系统

基于 **Elasticsearch + FAISS** 的双存储检索增强生成（RAG）系统，面向《中华人民共和国药典》提供中文药品知识问答。支持文档上传、向量检索、意图路由与字段级精准回答。

---

## 1. 项目简介

本项目是一个面向中文药典的 RAG 问答系统，核心特性包括：

- **双存储检索**：Elasticsearch 保存完整章节原文（全文检索），FAISS 保存子段落稠密向量（语义检索）
- **意图路由层**：通过规则预筛 + LLM 分类快速判断用户问题类型，减少无效 LLM 调用
- **字段级精准问答**：识别用户关心的药品属性（如性状、功能主治、处方等），对检索结果进行字段感知重排
- **Gradio Web UI**：三标签页界面（药典问答 / 文档导入 / 配置），支持运行时热更新配置

---

## 2. 快速开始

确保已安装 Python 3.10+，并完成[环境准备](#3-环境准备)。

```bat
:: 第 1 步：启动 Elasticsearch（不要关闭窗口）
start_es.bat

:: 第 2 步：启动 Gradio 应用
run_rag.bat

:: 第 3 步：（可选）运行自动化测试套件
run_tests.bat
```

手动启动方式（等价于 `run_rag.bat`）：

```bat
set KMP_DUPLICATE_LIB_OK=TRUE
set HF_HUB_OFFLINE=1
python pkg/webrun.py
```

应用默认绑定 `0.0.0.0`，在浏览器中打开 Gradio 输出的本地地址即可使用。

---

## 3. 环境准备

### 3.1 创建虚拟环境并安装依赖

```bat
python -m venv venv

:: 生产依赖
.\venv\Scripts\pip install -r requirements.txt

:: 开发依赖（跑测试需要）
.\venv\Scripts\pip install -r requirements-dev.txt
```

### 3.2 配置 .env 文件

在项目根目录创建 `.env` 文件，参考以下变量：

```env
# Elasticsearch Configuration
ES_HOST=127.0.0.1
ES_PORT=9200
ES_USER=elastic
ES_PASSWORD=changeme
ES_INDEX=zhyd
ES_SCHEME=http

# Vector Database
VECTOR_DB_PATH=./embeddings2.npz

# OpenAI-Compatible API (DeepSeek / Moonshot / OpenAI)
OPENAI_BASE_URL=https://api.moonshot.cn/v1
OPENAI_API_KEY=sk-your-key-here
LLM_MODEL=kimi-k2.6

# Feature Flags
ENABLE_INTENT_ROUTING=1
ENABLE_THINKING=1
```

**变量说明**：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ES_HOST` | `127.0.0.1` | Elasticsearch 主机地址 |
| `ES_PORT` | `9200` | ES 端口 |
| `ES_USER` | `elastic` | ES 用户名 |
| `ES_PASSWORD` | `changeme` | ES 密码 |
| `ES_INDEX` | `zhyd` | ES 索引名（上传时向量库同名） |
| `ES_SCHEME` | `http` | ES 协议（http/https） |
| `OPENAI_API_KEY` | — | API 密钥（DeepSeek / Moonshot / OpenAI 等） |
| `OPENAI_BASE_URL` | `https://api.deepseek.com` | API 基础地址 |
| `LLM_MODEL` | `deepseek-chat` | 模型名称（如 `deepseek-chat`、`kimi-k2.6`、`gpt-4o-mini`） |
| `VECTOR_DB_PATH` | `./embeddings2.npz` | 向量库存储路径 |
| `ENABLE_INTENT_ROUTING` | `1` | 意图路由开关：`1/true/yes/on` 开启，`0/false/no/off` 关闭 |
| `ENABLE_THINKING` | `1` | 推理模型思考过程输出（仅对 kimi-k2.6 / kimi-k2.5 等有效）：`1` 开启，`0` 通过 `extra_body` 禁用 |

> **提示**：`ENABLE_INTENT_ROUTING=0` 时系统完全退化为 P1 行为，不再调用 `classify_pharmacy_query` 和 `extract_target_fields`，适合对延迟极度敏感的场景。

---

## 4. 架构概览

### 4.1 数据流

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

### 4.2 检索链路（P2.1 意图路由版）

1. **意图预筛**（`quick_intent_hint`）：
   - 明显闲聊（"你好"/"thanks" 等）→ 跳过全部 LLM 路由调用，直接走自由问答
   - 含字段关键词（"性状"/"处方" 等）→ 跳过 `classify`，直接抽取字段
   - 无法判断 → 进入 LLM 分类流程

2. **LLM 分类与字段提取**（可选）：
   - `classify_pharmacy_query`：判断问题是否与药学相关（`good`/`bad`）
   - `MedicineInfoStandardizer.extract_target_fields`：从问句中提取用户关心的字段列表

3. **FAISS 向量检索**（`retrieve_vector_and_text`）：
   - 取 `top_k=3` 最相关子段落
   - 若存在 `target_fields`，按 `_score_result_by_fields` 进行字段感知重排

4. **LLM 生成**：
   - 组装 system prompt（含字段感知指令）+ 上下文 + 历史对话
   - 流式返回回答

---

## 5. 核心模块说明

| 文件 | 职责 |
|------|------|
| `pkg/config.py` | 配置单例。从 `.env` 加载环境变量，支持运行时热更新（Gradio 配置标签页直接修改属性）。包含 `ENABLE_INTENT_ROUTING` 功能开关。 |
| `pkg/embed.py` | **向量化与检索引擎**：Sentence-Transformers 编码、`faiss` 向量检索、`_es_index_fingerprint` ES 指纹脏检测、`process_and_vectorize` 建库流程。 **意图与标准化**：`quick_intent_hint` 规则预筛、`classify_pharmacy_query` LLM 分类、`MedicineInfoStandardizer` 字段提取。 **工具函数**：`extract_subsections` 按 `【】` 切分、`extract_drug_info` 解析 LLM 输出。 |
| `pkg/webrun.py` | **Gradio Web UI**：三标签页（问答 / 文档导入 / 配置）。 **问答主流程**（`slow_echo`）：流式 generator，集成意图路由、字段重排、历史记忆、LLM 调用。 **文档上传**（`UploadDoc`）：解析 `.docx`（按 12pt 字体分章）→ ES 存储 → 强制重建 FAISS。 **配置更新**（`update_config`）：写回 config 并双清 ES/OpenAI 客户端缓存。 |

---

## 6. 测试说明

- **框架**：pytest + pytest-mock
- **运行命令**：
  ```bat
  .\venv\Scripts\python.exe -m pytest tests -v --tb=short
  ```
  或直接双击：
  ```bat
  run_tests.bat
  ```

### 测试覆盖

| 测试文件 | 覆盖内容 |
|----------|----------|
| `tests/test_config.py` | `ENABLE_INTENT_ROUTING` 布尔解析矩阵、ES/OpenAI/VECTOR_DB_PATH 配置读取与默认值 |
| `tests/test_embed.py` | `quick_intent_hint` 三分支（chitchat/pharmacy/ambiguous）、`_es_index_fingerprint` 幂等性与稳定性、`get_openai_client` 懒加载生命周期、`clear_faiss_cache` 两种形态、`MedicineInfoStandardizer.extract_target_fields` 正常/异常/幻觉过滤、`retrieve_vector_and_text` 缓存行为与缺失文件容错、`extract_subsections` / `extract_drug_info` 解析 |
| `tests/test_webrun.py` | `slow_echo` 7 条意图分支（含关闭路由、异常自降级、检索异常兜底）、`_score_result_by_fields` 5 种打分场景、`update_config` 写回与双清缓存、`UploadDoc.store_in_elasticsearch` ES 未就绪降级与单条异常不阻断、`get_es_client` 懒加载生命周期 |

### 测试特点

- **89 个测试**，运行耗时约 **0.35 秒**
- **零外部服务依赖**：无需启动 ES、无需 OpenAI Key、无需真实 `.npz` 文件
- `conftest.py` 在 `pkg.*` 首次 import 前全局 stub 掉 `sentence_transformers` / `faiss` / `gradio` / `elasticsearch` / `openai` 等重依赖

---

## 7. 变更日志

### P0 — 缺陷修复（2026-05-04）

- **修复 `MedicineInfoStandardizer` 实例化时的 `NameError`**
  - 问题：`__init__` 中引用不存在的模块级 `field_list`
  - 修复：改为引用类属性 `MedicineInfoStandardizer.field_list`
  - 文件：`pkg/embed.py`

- **ES 连接改为懒加载**
  - 问题：模块导入阶段调用 `connect_elasticsearch()`，ES 未启动时静默返回 `None`，后续请求才报错
  - 修复：引入 `get_es_client()` 懒加载 + `clear_es_cache()` 显式失效
  - 文件：`pkg/webrun.py`

- **清理 `retrieve_vector_and_text` 不可达代码**
  - 问题：函数末尾存在第二个 `return`，造成误导
  - 修复：删除死代码
  - 文件：`pkg/embed.py`

### P1 — 架构加固（2026-05-04）

- **ES-FAISS 指纹脏检测同步**
  - 新增 `_es_index_fingerprint(hits)`：基于 ES `_id` 列表计算 MD5 指纹
  - `process_and_vectorize` 在重建前比对 `.npz` 中保存的 `es_fingerprint`，一致则跳过
  - 避免 ES 数据已更新但 FAISS 未重建的 stale index 问题
  - 文件：`pkg/embed.py`

- **新增 `force_rebuild` 强制刷新**
  - `UploadDoc.upload_doc` 重新上传文档时传 `force_rebuild=True`，无条件重建向量库
  - 确保 `.docx → ES → FAISS` 链路始终一致
  - 文件：`pkg/embed.py`、`pkg/webrun.py`

- **OpenAI 客户端懒加载 + 显式缓存失效**
  - 新增 `get_openai_client()` / `clear_openai_client_cache()`
  - 避免模块导入阶段读取 `OPENAI_API_KEY`，支持 Gradio 配置 tab 修改凭证后即时生效
  - 文件：`pkg/embed.py`

- **配置热更新双清缓存**
  - `update_config` 在写回 config 后调用 `clear_es_cache()` + `clear_openai_client_cache()`
  - 下次请求自动按新配置重建客户端
  - 文件：`pkg/webrun.py`

### P2.1 — 功能增强（2026-05-05）

- **引入意图路由层**
  - `quick_intent_hint`：基于规则（闲聊白名单 + 字段关键词正则）做 0 次 LLM 预筛
  - `classify_pharmacy_query`：LLM 判断问题是否与药学相关（`good`/`bad`）
  - `MedicineInfoStandardizer.extract_target_fields`：从问句中提取用户关心的字段名列表
  - 文件：`pkg/embed.py`

- **字段感知重排**
  - 新增 `_score_result_by_fields(title, target_fields)`：title 与目标字段精确匹配或互相包含则得 1 分
  - `slow_echo` 在检索后对 `top_k=3` 结果按字段匹配度重排，高分优先
  - 无匹配时保留原 FAISS 顺序并打印 warning，不丢弃结果
  - 文件：`pkg/webrun.py`

- **增强版 system prompt**
  - 当 `target_fields` 非空时，在 system prompt 中注入字段列表，引导 LLM 优先基于上下文回答这些属性
  - 上下文缺失某字段时，要求 LLM 明确说明
  - 文件：`pkg/webrun.py`

- **新增 `ENABLE_INTENT_ROUTING` 配置开关**
  - 默认开启（`1`）；设为 `0` 时 `slow_echo` 行为与 P1 完全一致
  - 路由层任何环节异常均自动降级到 P1 行为，不会阻塞回答
  - 文件：`pkg/config.py`

### P3 — 测试套件（2026-05-05）

- **建立 `tests/` 目录，89 个 pytest 自动化测试**
  - `tests/conftest.py`：全局 stub 8 个重依赖，注入 `sys.path` + 默认 env，提供 `restore_config` / `_reset_module_caches` / `sample_faiss_data` / `sample_hits` 等共享 fixture
  - `tests/test_config.py`：配置解析覆盖（9 个测试）
  - `tests/test_embed.py`：向量化、检索、意图、标准化覆盖（≈24 个测试项）
  - `tests/test_webrun.py`：问答流、打分、配置更新、ES 存储覆盖（≈20 个测试项）

- **零外部服务依赖设计**
  - 无需启动 Elasticsearch、无需有效 OpenAI Key、无需真实 `.npz`
  - 全部 LLM / ES / FAISS 调用通过 `unittest.mock` / `pytest-mock` 替换

---

## 注意事项

- **模型加载与 HuggingFace 离线模式**：`sentence-transformers` 在 import 阶段会尝试连接 HuggingFace 检查模型更新。如果网络不通，将反复重试 5 次（等待时间 1s→2s→4s→8s→16s），导致启动极慢。`run_rag.bat` 与 `run_tests.bat` 已内置 `HF_HUB_OFFLINE=1`，强制使用本地缓存的模型，跳过全部网络请求。若你手动启动，请务必带上该环境变量。

- `.docx` 分章逻辑与文档格式强耦合：章节边界检测依赖 **首 run 字体大小为 12pt** 的段落，子段落切分依赖 `【】` 标头。若使用非标准药典格式文档，需修改 `pkg/webrun.py:UploadDoc.extract_titles_and_content` 与 `pkg/embed.py:extract_subsections`。

- 首次使用前请先在 **配置** 标签页填写自己的 API Key 并保存，或直接修改 `.env` 文件。

- 向量库路径（`VECTOR_DB_PATH`）默认指向项目根目录下的 `embeddings2.npz`，上传新文档时会自动重建。
