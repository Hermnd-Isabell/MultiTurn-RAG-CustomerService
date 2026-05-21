import os
import sys
import re
import numpy as np

# 确保 pkg/ 在 sys.path 中，以便复用 embed.py 的 model（SentenceTransformer）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(PROJECT_ROOT, "pkg")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from config import config
from embed import model


def chunk_product_markdown(md_text):
    """
    按一级/二级标题切分商品 Markdown 文档。

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
    lines = md_text.splitlines()
    sku_id = ""
    current_module = ""
    current_sub_module = ""
    current_lines = []

    # 先提取 sku_id：从一级标题中匹配 SKU-XXXX
    for line in lines:
        if line.strip().startswith("# ") and not line.strip().startswith("## "):
            match = re.search(r"(SKU-[A-Z0-9\-]+)", line.strip())
            if match:
                sku_id = match.group(1)
            break

    def _flush():
        if current_module and current_lines:
            text = "\n".join(current_lines).strip()
            chunks.append({
                "sku_id": sku_id,
                "module": current_module,
                "sub_module": current_sub_module,
                "text": text,
            })

    for line in lines:
        stripped = line.strip()
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


def build_product_vector_index(input_dir, output_npz):
    """
    遍历 input_dir 下所有 .md 文件，切分 Chunk → 向量化 → 保存为 .npz。

    与现有 zhyd.npz 格式兼容：
    - embeddings: (N, 384) float32
    - ids: List[str]  — 格式 "sku_id|module|sub_module"
    - texts: List[Tuple[str, str, str]]  — (module, sub_module, text)

    要求：
    1. 复用 embed.py 中的全局 model（SentenceTransformer）做 encode
    2. 如果 output_npz 已存在，强制重建（商品库更新频繁，不保留旧索引）
    3. 打印进度：Processing SKU-XXX, total chunks: N
    """
    all_chunks = []

    if not os.path.isdir(input_dir):
        print(f"[build_product_vector_index] 输入目录不存在: {input_dir}")
        return

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".md"):
            continue
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            md_text = f.read()

        chunks = chunk_product_markdown(md_text)
        if chunks:
            sku_id = chunks[0]["sku_id"]
            print(f"Processing {sku_id}, total chunks: {len(chunks)}")
            all_chunks.extend(chunks)

    if not all_chunks:
        print("[build_product_vector_index] 未找到任何商品 Chunk，跳过构建")
        return

    texts_for_encode = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts_for_encode, convert_to_numpy=True)

    ids = [f"{c['sku_id']}|{c['module']}|{c['sub_module']}" for c in all_chunks]
    texts = [(c["module"], c["sub_module"], c["text"]) for c in all_chunks]

    # 强制重建：直接覆盖
    np.savez_compressed(
        output_npz,
        embeddings=embeddings.astype(np.float32),
        ids=ids,
        texts=texts,
    )
    print(f"[build_product_vector_index] 已保存 {len(all_chunks)} 条向量到 {output_npz}")


if __name__ == "__main__":
    build_product_vector_index(config.PRODUCT_KB_PATH, config.PRODUCT_INDEX_PATH)
