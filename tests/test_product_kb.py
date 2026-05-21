"""
build_product_index.py 测试覆盖：
- chunk_product_markdown：验证 Markdown 切分逻辑正确
- build_product_vector_index：mock model.encode，验证 npz 格式正确
"""
from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

import build_product_index


# ---------------------------------------------------------------------------
# chunk_product_markdown
# ---------------------------------------------------------------------------
class TestChunkProductMarkdown:
    def test_basic_chunking(self):
        md_text = """# SKU-2025-DRESS-001: 法式茶歇连衣裙
## 基础信息
### 商品标识
- 名称: 法式茶歇连衣裙
### 价格与图片
- 价格: ¥299
## 规格材质
### 面料成分
100% 棉
"""
        chunks = build_product_index.chunk_product_markdown(md_text)
        assert len(chunks) == 2

        # 第一个 Chunk：基础信息
        c1 = chunks[0]
        assert c1["sku_id"] == "SKU-2025-DRESS-001"
        assert c1["module"] == "基础信息"
        assert c1["sub_module"] == "价格与图片"
        assert "商品标识" in c1["text"]
        assert "价格: ¥299" in c1["text"]

        # 第二个 Chunk：规格材质
        c2 = chunks[1]
        assert c2["sku_id"] == "SKU-2025-DRESS-001"
        assert c2["module"] == "规格材质"
        assert c2["sub_module"] == "面料成分"
        assert "100% 棉" in c2["text"]

    def test_no_sku_id(self):
        md_text = """## 基础信息
内容
"""
        chunks = build_product_index.chunk_product_markdown(md_text)
        assert len(chunks) == 1
        assert chunks[0]["sku_id"] == ""
        assert chunks[0]["module"] == "基础信息"

    def test_empty_markdown(self):
        assert build_product_index.chunk_product_markdown("") == []


# ---------------------------------------------------------------------------
# build_product_vector_index
# ---------------------------------------------------------------------------
class TestBuildProductVectorIndex:
    def test_build_and_save_npz(self, tmp_path):
        input_dir = tmp_path / "products"
        input_dir.mkdir()

        md_file = input_dir / "dress.md"
        md_file.write_text(
            "# SKU-2025-DRESS-001: 法式茶歇连衣裙\n"
            "## 基础信息\n"
            "### 商品标识\n- 名称: 连衣裙\n"
            "## 规格材质\n"
            "100% 棉\n",
            encoding="utf-8",
        )

        output_npz = tmp_path / "products.npz"

        # mock encode 返回固定形状
        fake_embeddings = np.ones((2, 384), dtype=np.float32)
        with patch.object(build_product_index.model, "encode", return_value=fake_embeddings):
            build_product_index.build_product_vector_index(str(input_dir), str(output_npz))

        assert os.path.exists(output_npz)

        data = np.load(output_npz, allow_pickle=True)
        assert data["embeddings"].shape == (2, 384)
        assert len(data["ids"]) == 2
        assert len(data["texts"]) == 2

        # 验证 id 格式
        assert "SKU-2025-DRESS-001" in str(data["ids"][0])
        assert "基础信息" in str(data["ids"][0])

    def test_empty_dir_skips_build(self, tmp_path, capsys):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        output_npz = tmp_path / "out.npz"

        build_product_index.build_product_vector_index(str(empty_dir), str(output_npz))
        assert not os.path.exists(output_npz)

    def test_force_rebuild_overwrite(self, tmp_path):
        input_dir = tmp_path / "products"
        input_dir.mkdir()
        md_file = input_dir / "sku.md"
        md_file.write_text("# SKU-001\n## A\ntext\n", encoding="utf-8")

        output_npz = tmp_path / "products.npz"
        # 先写一份旧数据
        np.savez_compressed(output_npz, embeddings=np.zeros((1, 384)), ids=["old"], texts=[("","","")])

        fake_embeddings = np.ones((1, 384), dtype=np.float32)
        with patch.object(build_product_index.model, "encode", return_value=fake_embeddings):
            build_product_index.build_product_vector_index(str(input_dir), str(output_npz))

        data = np.load(output_npz, allow_pickle=True)
        assert "SKU-001" in str(data["ids"][0])
