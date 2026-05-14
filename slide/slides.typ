#set text(font: "Microsoft YaHei", size: 11pt)
#set page(paper: "presentation-16-9", margin: (x: 1.5cm, y: 1.2cm))

// ===== 标题页 =====
#align(center + horizon)[
  #text(size: 28pt, weight: "bold")[MultiTurn-RAG-CustomerService]
  #v(0.5em)
  #text(size: 16pt, fill: rgb("2c5aa0"))[《中华人民共和国药典》智能问答系统架构梳理]
  #v(1.5em)
  #text(size: 12pt)[Hermnd-Isabell]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[2025年7月]
]

#pagebreak()

// ===== 目录 =====
#text(size: 18pt, weight: "bold", fill: rgb("2c5aa0"))[目录]
#v(1em)
- 1. 项目概述
- 2. 已实现功能
- 3. 实现机制详解
- 4. 当前问题与改进方向

#pagebreak()

// ===== 1. 项目概述 =====
#text(size: 18pt, weight: "bold", fill: rgb("2c5aa0"))[1. 项目概述]
#v(0.8em)

#text(size: 14pt, weight: "bold")[项目定位]
#v(0.3em)
MultiTurn-RAG-CustomerService 是一个基于《中华人民共和国药典》的中文智能问答系统。

#v(0.5em)
#table(
  columns: (1fr, 3fr),
  inset: 8pt,
  stroke: 0.5pt + rgb("cccccc"),
  [*Retrieval*], [Elasticsearch + FAISS 双路检索],
  [*Augmented*], [检索结果作为 LLM 上下文],
  [*Generation*], [OpenAI 兼容 API（默认 DeepSeek）],
  [*MultiTurn*], [支持多轮对话、指代消解、会话上下文],
)

#pagebreak()

// ===== 核心数据流 =====
#text(size: 18pt, weight: "bold", fill: rgb("2c5aa0"))[1. 项目概述 — 核心数据流]
#v(0.8em)

#table(
  columns: (2fr, 3fr),
  inset: 8pt,
  stroke: 0.5pt + rgb("cccccc"),
  [*阶段*], [*处理*],
  [文档导入], [.docx → ES（整章）→ FAISS（子段落）],
  [用户提问], [查询改写 → 意图路由 → 检索决策],
  [检索阶段], [ES 精确锁定（主）/ FAISS 向量检索（降）],
  [生成阶段], [上下文组装 → LLM 流式回答],
)

#pagebreak()

// ===== 2. 已实现功能 =====
#text(size: 18pt, weight: "bold", fill: rgb("2c5aa0"))[2. 已实现功能]
#v(0.8em)

#text(size: 14pt, weight: "bold")[P0：基础架构]
#v(0.3em)
- *Elasticsearch*：存储整章原文，doc_id = 药品名
- *FAISS*：存储子段落向量（20857 条，384 维）
- 同一份文档存两份，目的不同

#v(0.5em)
#text(size: 14pt, weight: "bold")[P1：意图路由层]
#v(0.3em)
- `quick_intent_hint`：规则预筛（闲聊 / 药学 / 模糊）
- `classify_pharmacy_query`：LLM 二分类 good/bad
- `MedicineInfoStandardizer`：提取 target_fields + target_drug

#pagebreak()

// ===== P2 =====
#text(size: 18pt, weight: "bold", fill: rgb("2c5aa0"))[2. 已实现功能 — P2 药品感知重排]
#v(0.8em)

全局 FAISS 检索 top-10 后，双层打分重排：
#v(0.3em)
#align(center)[
  #box(fill: rgb("f0f0f0"), inset: 12pt, radius: 4pt)[
    #text(size: 13pt, weight: "bold")[
      total_score = drug_score × 2 + field_score × 1
    ]
  ]
]
#v(0.5em)
- `_score_result_by_drug_name`：doc_id 与目标药品名匹配
- `_score_result_by_fields`：段落 title 与 target_fields 匹配
- 匹配小于 3 条时用向量相似度补足

#pagebreak()

// ===== P2.2 =====
#text(size: 18pt, weight: "bold", fill: rgb("2c5aa0"))[2. 已实现功能 — P2.2 ES 精确锁定]
#v(0.8em)

当 target_drug 非空且 ES 中存在时：

#v(0.3em)
- *跳过* 全库 FAISS 盲搜
- 直接 `es.get(index, id=target_drug)` 取整章原文
- `extract_subsections` 切分后按 target_fields 过滤
- 匹配字段置顶，其余子段落补足 top_k

#v(0.5em)
#box(fill: rgb("e8f4e8"), inset: 10pt, radius: 4pt)[
  #text(weight: "bold", fill: rgb("2c5aa0"))[效果：「川射干的性状」直接命中【性状】段落，零漂移。]
]

#pagebreak()

// ===== P4.1 =====
#text(size: 18pt, weight: "bold", fill: rgb("2c5aa0"))[2. 已实现功能 — P4.1 查询改写]
#v(0.8em)

解决多轮对话中的*指代消解*问题。

#v(0.3em)
#text(size: 13pt, weight: "bold")[规则预筛]
- 已含药名 / 无指代词 → 跳过改写，节省 LLM 调用

#v(0.3em)
#text(size: 13pt, weight: "bold")[指代词集合]
\{他, 她, 它, 这个药, 该药, 刚才, 上面, 之前, ...\}

#v(0.3em)
#text(size: 13pt, weight: "bold")[LLM 改写]
- 结合最近 3 轮 history，消除指代词

#v(0.5em)
#box(fill: rgb("e8f4e8"), inset: 10pt, radius: 4pt)[
  Q1: 川射干的性状是什么？ #linebreak()
  Q2: #text(weight: "bold")[他的副作用是什么？] #linebreak()
  #text(weight: "bold", fill: rgb("2c5aa0"))[改写后：川射干的副作用是什么？]
]

#pagebreak()

// ===== P4.2 / P4.3 =====
#text(size: 18pt, weight: "bold", fill: rgb("2c5aa0"))[2. 已实现功能 — P4.2 / P4.3]
#v(0.8em)

#text(size: 14pt, weight: "bold")[P4.2：上下文感知检索]
#v(0.3em)
- `retrieve_with_context`：基础检索 top-15 + context_drug 硬过滤
- 匹配药品的结果置顶，不足时补足
- 改写层的*双保险*：即使 LLM 改写失败，检索层仍能硬约束

#v(0.8em)
#text(size: 14pt, weight: "bold")[P4.3：会话事实缓存]
#v(0.3em)
模块级 `_session_facts` 字典（Gradio 进程隔离 = 会话隔离）：
- `primary_drug`：当前会话主要讨论药品
- `queried_fields`：已查询字段集合（去重）
- `last_intent`：上一轮意图
- `drug_history`：会话中所有药品（按时间顺序）

#pagebreak()

// ===== 测试覆盖 =====
#text(size: 18pt, weight: "bold", fill: rgb("2c5aa0"))[2. 已实现功能 — 测试覆盖]
#v(0.8em)

#table(
  columns: (3fr, 1fr, 3fr),
  inset: 8pt,
  stroke: 0.5pt + rgb("cccccc"),
  [*测试文件*], [*数量*], [*覆盖范围*],
  [test_config.py], [17], [配置读取 / 默认值],
  [test_embed.py], [48], [嵌入 / 检索 / 字段提取],
  [test_webrun.py], [45], [slow_echo 分支 / 改写 / 缓存],
  [test_drug_aware_rerank.py], [18], [药品名打分 / 提取 / ES 确认],
)

#v(0.8em)
#align(center)[
  #box(fill: rgb("2c5aa0"), inset: 12pt, radius: 4pt)[
    #text(size: 16pt, weight: "bold", fill: white)[总计：128 个测试全部通过]
  ]
]

#pagebreak()

// ===== 3. 实现机制详解 =====
#text(size: 18pt, weight: "bold", fill: rgb("2c5aa0"))[3. 实现机制详解]
#v(0.8em)

#text(size: 14pt, weight: "bold")[检索决策流程]
#v(0.3em)
#table(
  columns: (4fr, 4fr),
  inset: 8pt,
  stroke: 0.5pt + rgb("cccccc"),
  [*条件*], [*路径*],
  [target_drug 非空 且 drug_exists], [主路径：ES 精确锁定],
  [target_drug 为空 且 session_drug 非空], [降级：retrieve_with_context],
  [target_drug 为空 且 session_drug 为空], [降级：FAISS 全库 top-3],
  [全库未命中目标药品], [兜底：retrieve_vector_and_text_for_drug],
)

#v(0.8em)
#text(size: 14pt, weight: "bold")[关键模块]
#v(0.3em)
- `config.py` — 全局配置单例（运行时热更新）
- `embed.py` — 向量 / ES / LLM 工具函数
- `webrun.py` — Gradio UI + slow_echo 主流程
- `tests/` — pytest + MagicMock 全覆盖

#pagebreak()

// ===== 4. 当前问题 =====
#text(size: 18pt, weight: "bold", fill: rgb("2c5aa0"))[4. 当前问题与改进方向]
#v(0.8em)

#text(size: 14pt, weight: "bold", fill: rgb("c44"))[结构性问题]
#v(0.3em)
- *向量语义漂移*：FAISS 只编码子段落内容，不含药品名
  → "川射干的性状"编码后失去"川射干"信号
- *切分规则硬编码*：依赖 12pt 字体 + 固定正则
  → 非标准 .docx 无法正确分章
- *模型不可切换*：Sentence-Transformer 固定加载

#v(0.8em)
#text(size: 14pt, weight: "bold", fill: rgb("c44"))[工程性问题]
#v(0.3em)
- *LLM 依赖*：改写/分类/提取均走外部 API → 延迟高、成本高
- *会话状态易失*：`_session_facts` 模块级变量，进程重启即丢失
- *无对话持久化*：没有数据库记录历史问答
- *Windows-only*：启动脚本为 .bat，跨平台性差
- *无权限管理*：配置页密码明文，无用户隔离

#pagebreak()

// ===== 改进方向 =====
#text(size: 18pt, weight: "bold", fill: rgb("2c5aa0"))[4. 改进方向]
#v(0.8em)

#text(size: 14pt, weight: "bold", fill: rgb("2c5aa0"))[短期]
#v(0.3em)
- 向量库编码时 prepend 药品名（`[川射干] 本品为...`）
- 引入本地小模型做意图分类（减少 LLM 调用）
- 添加 SQLite 对话历史持久化

#v(0.8em)
#text(size: 14pt, weight: "bold", fill: rgb("2c5aa0"))[中长期]
#v(0.3em)
- 支持多药品对比查询
- 动态 field_list 扩展（用户自定义字段）
- 跨平台启动脚本（sh + Docker）
- 引入重排序模型（Cross-Encoder）替代规则打分
- 向量库支持增量更新（无需全量重建）

#pagebreak()

// ===== 结束页 =====
#align(center + horizon)[
  #text(size: 32pt, weight: "bold")[谢谢！]
  #v(1em)
  #text(size: 11pt)[
    GitHub: https://github.com/Hermnd-Isabell/MultiTurn-RAG-CustomerService
  ]
]
