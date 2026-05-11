# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`RAG_project_zhyd` is a Chinese-language RAG (Retrieval-Augmented Generation) Q&A system over the **Chinese Pharmacopoeia** (《中华人民共和国药典》). It combines Elasticsearch (full-text store) with FAISS (dense-vector retrieval), serves a Gradio web UI with three tabs (问答 / 文档导入 / 配置), and calls an OpenAI-compatible LLM (default: DeepSeek) for answer generation.

## Common Commands

All scripts are Windows-targeted (.bat) and assume the bundled `venv/` exists at the repo root.

```bat
:: Start Elasticsearch (required before answering questions)
start_es.bat                        :: launches elasticsearch-7.17.21\bin\elasticsearch.bat

:: Start the Gradio app (sets KMP_DUPLICATE_LIB_OK=TRUE, then runs pkg/webrun.py)
run_rag.bat

:: Install dependencies into a fresh environment
pip install -r requirements.txt
:: or, as a package
pip install -e .
```

Run the app manually (equivalent to `run_rag.bat`):

```bat
set KMP_DUPLICATE_LIB_OK=TRUE
python pkg/webrun.py
```

The app binds to `0.0.0.0` on Gradio's default port. There is **no test suite** in this repo.

## Architecture

### Two-tier storage, not one

The system deliberately stores the same documents twice for different purposes — when changing ingestion or retrieval, both stores must stay in sync:

1. **Elasticsearch** (`zhyd` index by default) holds whole-chapter raw text, keyed by drug name (the chapter title). It is the source of truth for re-vectorization.
2. **FAISS / `.npz`** (`zhyd.npz` / `embeddings2.npz`) holds per-subsection dense vectors plus their `(title, text)` metadata, persisted via `np.savez_compressed`. Created by reading from ES and re-encoding — **never built directly from the original .docx**.

The flow is `.docx → ES (chapters) → FAISS (subsections per chapter)`.

### Document chunking is template-coupled

`pkg/webrun.py:UploadDoc.extract_titles_and_content` segments a `.docx` by detecting paragraphs whose **first run has `font.size.pt == 12`** as chapter boundaries (drug titles). Subsection extraction inside each chapter uses the regex `(?:【|t)(.+?)(?:】)` over the joined chapter text (`pkg/embed.py:extract_subsections`). Both rules are tightly bound to the format of `2020年药典一部.docx`; arbitrary Word documents will not chunk correctly without changing these heuristics.

Chinese title cleaning: `''.join(re.findall(r'[\u4e00-\u9fff]+', filename))` strips everything except CJK characters.

### Module layout (`pkg/`)

- `config.py` — Loads `.env` via `python-dotenv` and exposes a single mutable `config` instance (class attributes, not dataclass). The Gradio "配置" tab mutates this at runtime, so `config.VECTOR_DB_PATH`, `config.ES_INDEX`, etc. should always be read fresh inside request handlers, never captured at import time.
- `embed.py` — Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`) is loaded **once at import** as the module-global `model`. FAISS indexes are cached in the module-global dict `_faiss_cache` keyed by the `.npz` path; mutate it via `clear_faiss_cache(path=None)` whenever an `.npz` is rewritten. `process_and_vectorize` early-returns if the target `.npz` already exists, so to re-ingest you must delete the `.npz` first.
- `webrun.py` — Gradio Blocks UI, the `slow_echo` chat handler (which streams via `yield` and folds Gradio history into OpenAI `messages`), and `UploadDoc` orchestrating docx → ES → FAISS.

Imports inside `pkg/` use bare names (`from config import config`, `from embed import ...`) rather than `pkg.config`. Running entrypoints from anywhere other than `python pkg/webrun.py` will break imports — keep the working directory at the repo root.

### LLM and retrieval contract

- `slow_echo` always retrieves `top_k=3` from FAISS, formats context as `【title】\n{text}` blocks, prepends a fixed Chinese system prompt, then forwards full Gradio history as alternating `user`/`assistant` messages before the new question.
- `MedicineInfoStandardizer` and `classify_pharmacy_query` in `embed.py` exist as LLM-driven helpers (intent classification: `"good"`/`"bad"`; field extraction over a fixed `field_list` of ~30 pharmacopoeia headings) but are **not currently wired into `slow_echo`**. Treat them as available utilities, not part of the active path.
- LLM credentials and endpoint come from `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL` in `.env`. The default base URL is DeepSeek's API.

### Known fragile spots (read before editing)

- `MedicineInfoStandardizer.__init__` assigns `self.field_list = field_list` referencing a non-existent module-level `field_list`; the class-level `field_list` works, but instantiating the class as written will raise `NameError`.
- `retrieve_vector_and_text` has unreachable code (a second `return` after the first) — harmless but easy to misread.
- `webrun.py` calls `connect_elasticsearch()` at module import. If ES is not running when the app starts, the call returns `None` and prints an error but does not abort startup; the upload and chat paths will then fail at request time.
- `process_and_vectorize` skips work when the `.npz` exists, even if the underlying ES index has changed. Delete the `.npz` (and call `clear_faiss_cache`) when re-ingesting the same index name.
