"""
build_product_index.py 测试覆盖：
- chunk_product_markdown：验证 Markdown 切分逻辑正确
- build_product_vector_index：mock model.encode，验证 npz 格式正确
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

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



# ---------------------------------------------------------------------------
# read_docx_text
# ---------------------------------------------------------------------------
class TestReadDocxText:
    def test_read_docx_text_with_hash_markers(self):
        """方案 A：.docx 段落文本自带 # 标记，直接提取"""
        from unittest.mock import MagicMock, patch

        para1 = MagicMock()
        para1.text = "# SKU-001: 测试商品"
        para1.style.name = "Normal"

        para2 = MagicMock()
        para2.text = "这是一段普通描述"
        para2.style.name = "Normal"

        para3 = MagicMock()
        para3.text = "## 规格材质"
        para3.style.name = "Normal"

        fake_doc = MagicMock()
        fake_doc.paragraphs = [para1, para2, para3]

        with patch.object(build_product_index, "Document", return_value=fake_doc):
            text = build_product_index.read_docx_text("fake.docx")
            assert "# SKU-001: 测试商品" in text
            assert "## 规格材质" in text
            assert "这是一段普通描述" in text

    def test_read_docx_text_with_heading_styles(self):
        """方案 B：Word 原生标题样式无 # 标记，自动降级补前缀"""
        from unittest.mock import MagicMock, patch

        para1 = MagicMock()
        para1.text = "SKU-002: 另一商品"
        para1.style.name = "Heading 1"

        para2 = MagicMock()
        para2.text = "基础信息"
        para2.style.name = "Heading 2"

        para3 = MagicMock()
        para3.text = "商品标识"
        para3.style.name = "Heading 3"

        fake_doc = MagicMock()
        fake_doc.paragraphs = [para1, para2, para3]

        with patch.object(build_product_index, "Document", return_value=fake_doc):
            text = build_product_index.read_docx_text("fake2.docx")
            assert "# SKU-002: 另一商品" in text
            assert "## 基础信息" in text
            assert "### 商品标识" in text

    def test_read_docx_text_skips_empty_paragraphs(self):
        """空段落应被过滤"""
        from unittest.mock import MagicMock, patch

        para1 = MagicMock()
        para1.text = "# SKU-003: 商品"
        para1.style.name = "Normal"

        para2 = MagicMock()
        para2.text = "   "
        para2.style.name = "Normal"

        para3 = MagicMock()
        para3.text = "## 基础信息"
        para3.style.name = "Normal"

        fake_doc = MagicMock()
        fake_doc.paragraphs = [para1, para2, para3]

        with patch.object(build_product_index, "Document", return_value=fake_doc):
            text = build_product_index.read_docx_text("fake.docx")
            assert "## 基础信息" in text
            # 空段落不应产生独立段落
            lines = [ln for ln in text.splitlines() if ln.strip()]
            assert len(lines) == 2


# ---------------------------------------------------------------------------
# build_product_vector_index with .docx
# ---------------------------------------------------------------------------
class TestBuildProductVectorIndexDocx:
    def test_build_with_docx(self, tmp_path):
        """纯 .docx 目录应能正确生成 npz"""
        input_dir = tmp_path / "products"
        input_dir.mkdir()

        # 创建占位 .docx 文件，确保 build_product_vector_index 能发现它
        (input_dir / "dress.docx").write_text("dummy", encoding="utf-8")

        para1 = MagicMock()
        para1.text = "# SKU-DOCX-001: 连衣裙"
        para1.style.name = "Normal"

        para2 = MagicMock()
        para2.text = "## 基础信息"
        para2.style.name = "Normal"

        para3 = MagicMock()
        para3.text = "### 商品标识"
        para3.style.name = "Normal"

        para4 = MagicMock()
        para4.text = "- 名称: 连衣裙"
        para4.style.name = "Normal"

        fake_doc = MagicMock()
        fake_doc.paragraphs = [para1, para2, para3, para4]

        output_npz = tmp_path / "products.npz"
        fake_embeddings = np.ones((1, 384), dtype=np.float32)

        with patch.object(build_product_index.model, "encode", return_value=fake_embeddings):
            with patch.object(build_product_index, "Document", return_value=fake_doc):
                build_product_index.build_product_vector_index(str(input_dir), str(output_npz))

        assert os.path.exists(output_npz)
        data = np.load(output_npz, allow_pickle=True)
        assert data["embeddings"].shape == (1, 384)
        assert "SKU-DOCX-001" in str(data["ids"][0])

    def test_build_mixed_md_docx(self, tmp_path):
        """混合目录：.md 和 .docx 同时存在，且同 basename 时 .md 优先"""
        input_dir = tmp_path / "products"
        input_dir.mkdir()

        # .md 文件
        md_file = input_dir / "dress.md"
        md_file.write_text(
            "# SKU-MD-001: 连衣裙\n## 基础信息\n内容\n",
            encoding="utf-8",
        )

        # 同 basename 的 .docx（应被 .md 覆盖，不处理）
        docx_same = input_dir / "dress.docx"
        docx_same.write_text("dummy", encoding="utf-8")

        # 另一个独立的 .docx（需要真实文件存在才会被扫描到）
        (input_dir / "shirt.docx").write_text("dummy", encoding="utf-8")

        para1 = MagicMock()
        para1.text = "# SKU-DOCX-002: 衬衫"
        para1.style.name = "Normal"

        para2 = MagicMock()
        para2.text = "## 规格"
        para2.style.name = "Normal"

        para3 = MagicMock()
        para3.text = "棉"
        para3.style.name = "Normal"

        fake_doc = MagicMock()
        fake_doc.paragraphs = [para1, para2, para3]

        output_npz = tmp_path / "products.npz"
        fake_embeddings = np.ones((2, 384), dtype=np.float32)

        with patch.object(build_product_index.model, "encode", return_value=fake_embeddings):
            with patch.object(build_product_index, "Document", return_value=fake_doc):
                build_product_index.build_product_vector_index(str(input_dir), str(output_npz))

        assert os.path.exists(output_npz)
        data = np.load(output_npz, allow_pickle=True)
        assert data["embeddings"].shape == (2, 384)

        ids = [str(i) for i in data["ids"]]
        assert any("SKU-MD-001" in i for i in ids)
        assert any("SKU-DOCX-002" in i for i in ids)
        # dress.docx 被 dress.md 覆盖，不应出现 SKU-DOCX-002 以外的 dress 内容

    def test_docx_read_failure_skips_gracefully(self, tmp_path, capsys):
        """.docx 读取失败时不阻断其他文件"""
        input_dir = tmp_path / "products"
        input_dir.mkdir()

        # 一个损坏的 .docx（真实文件存在但内容不是 docx 格式）
        bad_docx = input_dir / "bad.docx"
        bad_docx.write_text("not a docx", encoding="utf-8")

        # 一个正常的 .md
        md_file = input_dir / "good.md"
        md_file.write_text("# SKU-001\n## A\ntext\n", encoding="utf-8")

        output_npz = tmp_path / "products.npz"
        fake_embeddings = np.ones((1, 384), dtype=np.float32)

        with patch.object(build_product_index.model, "encode", return_value=fake_embeddings):
            build_product_index.build_product_vector_index(str(input_dir), str(output_npz))

        assert os.path.exists(output_npz)
        data = np.load(output_npz, allow_pickle=True)
        assert "SKU-001" in str(data["ids"][0])
