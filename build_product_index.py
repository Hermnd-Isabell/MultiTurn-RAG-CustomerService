import os
import sys
import re
import numpy as np

# 确保 pkg/ 在 sys.path 中，以便复用 embed.py 的 model（SentenceTransformer）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(PROJECT_ROOT, "pkg")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import config
from embed import model

try:
    from docx import Document
except ImportError:
    print("[error] 读取 .docx 需要 python-docx，请运行: pip install python-docx")
    sys.exit(1)


# 中文数字模块标题，如 "一、基础信息" / "二、规格材质"
_CN_MODULE_RE = re.compile(r'^[一二三四五六七八九十]+、(.+)')
# 商品条目编号，如 "1. T 恤" / "2. 衬衫"
_ITEM_NUM_RE = re.compile(r'^(\d+)\.\s*(.+)')


def read_docx_text(docx_path):
    """
    读取 .docx 文件，返回 Markdown 风格的文本字符串。

    策略：
    1. 遍历所有段落，提取 paragraph.text，过滤空段落
    2. 识别中文编号格式并自动转换为 Markdown 标题前缀：
       - 商品条目（"1. T 恤"）→ "# T 恤"
       - 中文数字模块（"一、基础信息"）→ "## 基础信息"
    3. Word 原生标题样式（Heading 1/2/3）降级补 # 标记
    4. 段落间用 \n\n 分隔，保留段落结构

    返回：str，失败时返回空字符串
    """
    try:
        doc = Document(docx_path)
    except Exception as e:
        print(f"[warning] 读取 .docx 失败: {docx_path} ({e})，跳过")
        return ""

    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # 1. 中文数字模块标题 → ##
        m = _CN_MODULE_RE.match(text)
        if m:
            text = "## " + m.group(1)
        else:
            # 2. 商品条目编号 → #
            m = _ITEM_NUM_RE.match(text)
            if m:
                text = "# " + m.group(2)
            else:
                # 3. Word 原生标题样式降级
                style_name = getattr(para.style, "name", "")
                if not text.startswith("#"):
                    if "Heading 1" in style_name:
                        text = "# " + text
                    elif "Heading 2" in style_name:
                        text = "## " + text
                    elif "Heading 3" in style_name:
                        text = "### " + text

        paragraphs.append(text)

    print(f"[build] 读取 .docx: {os.path.basename(docx_path)} | 段落数: {len(paragraphs)}")
    return "\n\n".join(paragraphs)


def chunk_product_document(text):
    """
    按一级/二级/三级标题切分商品文档（兼容 Markdown 与 .docx 转换后的文本）。

    输入示例：
        # SKU-2025-DRESS-001: 法式茶歇连衣裙
        ## 基础信息
        ### 商品标识
        - 名称: ...
        ### 价格与图片
        - 价格: ¥299
        ## 规格材质
        ### 面料成分
        ...

    切分规则：
    1. 一级标题 # 提取 sku_id（格式 SKU-XXXX）和商品名称
    2. 二级标题 ## 作为 Chunk 边界，每个 ## 下的内容为一个 Chunk
    3. 如果二级标题下有三级标题 ###，把 ### 作为 sub_module，否则 sub_module 为空
    4. 每个 Chunk 的 text 包含该二级标题下的全部内容（含三级标题和正文）

    返回: List[dict]，每个 dict 格式：
    {
        "sku_id": "SKU-2025-DRESS-001",
        "module": "基础信息",           # 二级标题名
        "sub_module": "商品标识",       # 三级标题名（若无则为 ""）
        "text": "### 商品标识\n- 名称: 法式茶歇连衣裙\n..."  # 完整文本
    }
    """
    chunks = []
    lines = text.splitlines()
    sku_id = ""
    product_header = ""    # 一级标题内容（如 "TX2026001: 云感T恤"），用于添加到每个 chunk
    current_module = ""
    current_sub_module = ""
    current_lines = []

    def _flush():
        if current_module and current_lines:
            text = "\n".join(current_lines).strip()
            # 为每个 chunk 前缀商品标识（SKU + 名称），确保向量检索能通过商品名匹配
            prefix = ""
            if product_header:
                prefix = f"商品：{product_header}\n"
            elif sku_id:
                prefix = f"商品编号：{sku_id}\n"
            chunks.append({
                "sku_id": sku_id,
                "module": current_module,
                "sub_module": current_sub_module,
                "text": prefix + text,
            })

    for line in lines:
        stripped = line.strip()

        # 从正文中动态提取编号作为 sku_id 兜底（兼容 "编号：TX2026001"）
        if "编号" in stripped and ("：" in stripped or ":" in stripped):
            match = re.search(r"编号[：:]\s*([A-Za-z0-9\-]+)", stripped)
            if match:
                sku_id = match.group(1)

        # 从基础信息模块中提取完整商品名称，丰富前缀语义
        if current_module == "基础信息" and "名称" in stripped and ("：" in stripped or ":" in stripped):
            match = re.search(r"名称[：:]\s*(.+)", stripped)
            if match:
                product_header = match.group(1).strip()

        if stripped.startswith("# ") and not stripped.startswith("## "):
            # 一级标题：先 flush 上一个商品的最后 chunk，再更新当前商品信息
            _flush()
            current_module = ""
            current_sub_module = ""
            current_lines = []
            product_header = stripped[2:].strip()
            match = re.search(r"(SKU-[A-Z0-9\-]+)", stripped)
            if match:
                sku_id = match.group(1)
            continue

        if stripped.startswith("## ") and not stripped.startswith("### "):
            _flush()
            current_module = stripped[3:].strip()
            current_sub_module = ""
            current_lines = []
        elif stripped.startswith("### "):
            current_sub_module = stripped[4:].strip()
            current_lines.append(line)
        else:
            if current_module:
                current_lines.append(line)

    _flush()
    return chunks


# 向后兼容：保留旧函数名
chunk_product_markdown = chunk_product_document


def build_product_vector_index(input_dir, output_npz):
    """
    遍历 input_dir 下所有 .md 和 .docx 文件，切分 Chunk → 向量化 → 保存为 .npz。

    与现有 zhyd.npz 格式兼容：
    - embeddings: (N, 384) float32
    - ids: List[str]  — 格式 "sku_id|module|sub_module"
    - texts: List[Tuple[str, str, str]]  — (module, sub_module, text)

    要求：
    1. 复用 embed.py 中的全局 model（SentenceTransformer）做 encode
    2. 如果 output_npz 已存在，强制重建（商品库更新频繁，不保留旧索引）
    3. 打印进度
    """
    all_chunks = []

    if not os.path.isdir(input_dir):
        print(f"[build_product_vector_index] 输入目录不存在: {input_dir}")
        return

    # 收集所有 .md 和 .docx 文件，同一 basename 优先 .md
    basenames = {}
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".md") or filename.endswith(".docx"):
            base = os.path.splitext(filename)[0]
            # .md 优先级高于 .docx
            if base not in basenames or filename.endswith(".md"):
                basenames[base] = filename

    for base, filename in sorted(basenames.items()):
        filepath = os.path.join(input_dir, filename)

        if filename.endswith(".md"):
            with open(filepath, "r", encoding="utf-8") as f:
                file_text = f.read()
        elif filename.endswith(".docx"):
            file_text = read_docx_text(filepath)
            if not file_text:
                continue
        else:
            continue

        chunks = chunk_product_document(file_text)
        if chunks:
            sku_id = chunks[0]["sku_id"]
            print(f"[build] 切分 Chunk: {filename} | sku={sku_id} | chunks: {len(chunks)}")
            all_chunks.extend(chunks)

    if not all_chunks:
        print("[build_product_vector_index] 未找到任何商品 Chunk，跳过构建")
        return 0

    texts_for_encode = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts_for_encode, convert_to_numpy=True)

    ids = [f"{c['sku_id']}|{c['module']}|{c['sub_module']}" for c in all_chunks]
    texts = [(c["module"], c["sub_module"], c["text"]) for c in all_chunks]

    # 强制重建：直接覆盖
    # texts 是 list of tuples，必须保存为 object array，否则 numpy 可能将其解析为 2D str array
    texts_arr = np.empty(len(texts), dtype=object)
    texts_arr[:] = texts
    np.savez_compressed(
        output_npz,
        embeddings=embeddings.astype(np.float32),
        ids=ids,
        texts=texts_arr,
    )
    print(f"[build_product_vector_index] 已保存 {len(all_chunks)} 条向量到 {output_npz}")
    return len(all_chunks)


if __name__ == "__main__":
    if not os.path.exists(config.PRODUCT_KB_PATH):
        print(f"商品知识库目录未找到: {config.PRODUCT_KB_PATH}，请在 data/products/ 或 data/ 下放置 .md/.docx 文件")
    else:
        count = build_product_vector_index(config.PRODUCT_KB_PATH, config.PRODUCT_INDEX_PATH)
        if count:
            print(f"索引已生成: {config.PRODUCT_INDEX_PATH}，共 {count} 个 chunks")
