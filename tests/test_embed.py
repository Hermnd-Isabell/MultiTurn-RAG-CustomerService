"""
pkg/embed.py 测试覆盖：
- quick_intent_hint：闲聊 / 字段命中 / 模糊 / 空输入；
- _es_index_fingerprint：空 hits / 正常 hits / 相同 hits 同指纹 / id 顺序无关；
- get_openai_client + clear_openai_client_cache 的"懒加载 + 失效"全生命周期；
- clear_faiss_cache 的两种调用形态；
- MedicineInfoStandardizer.extract_target_fields 的成功 / 异常 / 幻觉过滤路径；
- retrieve_vector_and_text 的检索流程与 _faiss_cache 命中行为。

所有外部依赖（sentence_transformers / faiss / openai / elasticsearch）已在 conftest.py
预先 stub，因此本文件不需要再 patch 它们。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# -----------------------------------------------------------------------------
# quick_intent_hint
# -----------------------------------------------------------------------------
class TestQuickIntentHint:
    """quick_intent_hint 仅做正则 / 关键词预筛，绝不应触发任何 LLM 调用。"""

    @pytest.mark.parametrize("text", ["你好", "您好", "hi", "Hello", "thanks", "thx", "再见"])
    def test_obvious_chitchat_keywords(self, text):
        from embed import quick_intent_hint

        assert quick_intent_hint(text) == "obvious_chitchat", f"{text!r} 应被识别为闲聊"

    @pytest.mark.parametrize("text", ["", "   ", None])
    def test_empty_or_none_returns_chitchat(self, text):
        """空字符串 / None / 全空白都视作"无意图"，归到闲聊兜底。"""
        from embed import quick_intent_hint

        assert quick_intent_hint(text) == "obvious_chitchat"

    @pytest.mark.parametrize(
        "text",
        [
            "性状",
            "当归的性状是什么",
            "请问这味药的功能主治",
            "用法与用量怎么写",
            "处方里有哪些药材",
        ],
    )
    def test_obvious_pharmacy_when_field_keyword_present(self, text):
        """命中 field_list 关键词的均应直接判为 obvious_pharmacy，跳过 classify。"""
        from embed import quick_intent_hint

        assert quick_intent_hint(text) == "obvious_pharmacy", f"{text!r} 应被识别为药学查询"

    @pytest.mark.parametrize(
        "text",
        [
            "它的副作用是什么",
            "请问这种药能治疗高血压吗",
            "这个能长期吃吗",
            "孕妇可以服用吗",
        ],
    )
    def test_ambiguous_returns_ambiguous(self, text):
        """既非闲聊白名单、又不含字段关键词 → 必须交给完整 classify 流程。"""
        from embed import quick_intent_hint

        assert quick_intent_hint(text) == "ambiguous"

    def test_long_chitchat_keyword_does_not_trigger_short_circuit(self):
        """长度阈值保护：>8 字符的句子即使包含 '你好' 也不会被白名单短路。"""
        from embed import quick_intent_hint

        # 长度 > 8 且不含字段关键词 → ambiguous（既不是 chitchat 也不是 pharmacy）
        assert quick_intent_hint("你好我想问一个药品相关问题") == "ambiguous"


# -----------------------------------------------------------------------------
# _es_index_fingerprint
# -----------------------------------------------------------------------------
class TestEsIndexFingerprint:
    """指纹函数是 process_and_vectorize 的脏检测基石，必须保证稳定 & 可复现。"""

    def test_empty_hits_returns_none(self):
        from embed import _es_index_fingerprint

        fp, count = _es_index_fingerprint([])
        assert fp is None
        assert count == 0

    def test_normal_hits_returns_md5_and_count(self, sample_hits):
        from embed import _es_index_fingerprint

        fp, count = _es_index_fingerprint(sample_hits)
        assert isinstance(fp, str) and len(fp) == 32, "MD5 hex 应为 32 字符"
        assert count == len(sample_hits)

    def test_idempotent_same_hits_same_fingerprint(self, sample_hits):
        from embed import _es_index_fingerprint

        fp1, _ = _es_index_fingerprint(sample_hits)
        fp2, _ = _es_index_fingerprint(sample_hits)
        assert fp1 == fp2, "纯函数：同输入必须输出同指纹"

    def test_id_order_does_not_matter(self):
        """函数内部对 _id 排序，因此 hits 顺序不应影响结果。"""
        from embed import _es_index_fingerprint

        hits1 = [
            {"_id": "drug_a", "_source": {"content": "x"}},
            {"_id": "drug_b", "_source": {"content": "y"}},
        ]
        hits2 = list(reversed(hits1))
        fp1, _ = _es_index_fingerprint(hits1)
        fp2, _ = _es_index_fingerprint(hits2)
        assert fp1 == fp2, "排序后的 ids 决定指纹，输入顺序应被规范化掉"

    def test_different_hits_different_fingerprint(self):
        from embed import _es_index_fingerprint

        hits1 = [{"_id": "drug_a", "_source": {"content": "x"}}]
        hits2 = [{"_id": "drug_b", "_source": {"content": "x"}}]
        fp1, _ = _es_index_fingerprint(hits1)
        fp2, _ = _es_index_fingerprint(hits2)
        assert fp1 != fp2


# -----------------------------------------------------------------------------
# get_openai_client / clear_openai_client_cache
# -----------------------------------------------------------------------------
class TestOpenAIClientLifecycle:
    """懒加载 + 显式失效，确保 Gradio 配置 tab 修改 API key 后能重建客户端。"""

    def test_first_call_creates_instance(self):
        import embed

        assert embed._openai_client is None
        client = embed.get_openai_client()
        assert client is not None
        assert embed._openai_client is client

    def test_second_call_returns_cached_instance(self):
        import embed

        first = embed.get_openai_client()
        second = embed.get_openai_client()
        assert first is second, "二次调用必须命中缓存，避免反复 new OpenAI()"

    def test_clear_then_recreate_returns_new_instance(self):
        """clear 后再次 get 必须重新调用 OpenAI(...) 构造新实例。
        conftest 的全局 OpenAI MagicMock 会复用同一个 .return_value，所以这里改用
        patch + side_effect 让两次调用返回明显不同的对象，断言构造函数被调用了 2 次。"""
        import embed

        with patch.object(embed, "OpenAI") as mock_openai_cls:
            mock_openai_cls.side_effect = [
                MagicMock(name="ClientFirst"),
                MagicMock(name="ClientSecond"),
            ]
            first = embed.get_openai_client()
            embed.clear_openai_client_cache()
            assert embed._openai_client is None
            second = embed.get_openai_client()

        assert first is not second, "clear 后必须重建，否则配置热更新无意义"
        assert mock_openai_cls.call_count == 2, "OpenAI() 应被调用 2 次（首建 + 重建）"


# -----------------------------------------------------------------------------
# clear_faiss_cache
# -----------------------------------------------------------------------------
class TestClearFaissCache:
    """两种调用形态：传 path 精准失效；不传则全量清空。"""

    def test_clear_specific_path(self):
        import embed

        embed._faiss_cache["/tmp/a.npz"] = ("idx_a", "ids_a", "txt_a")
        embed._faiss_cache["/tmp/b.npz"] = ("idx_b", "ids_b", "txt_b")

        embed.clear_faiss_cache("/tmp/a.npz")
        assert "/tmp/a.npz" not in embed._faiss_cache
        assert "/tmp/b.npz" in embed._faiss_cache, "未指定的路径不应受影响"

    def test_clear_all_when_no_path(self):
        import embed

        embed._faiss_cache["/tmp/a.npz"] = ("idx_a", "ids_a", "txt_a")
        embed._faiss_cache["/tmp/b.npz"] = ("idx_b", "ids_b", "txt_b")

        embed.clear_faiss_cache()
        assert embed._faiss_cache == {}, "不传 path 必须清空全部"

    def test_clear_nonexistent_path_is_noop(self):
        """传一个不存在的 key 不应抛错（避免在 process_and_vectorize 中误伤）。"""
        import embed

        embed._faiss_cache["/tmp/exist.npz"] = ("idx", "ids", "txt")
        embed.clear_faiss_cache("/tmp/missing.npz")
        assert "/tmp/exist.npz" in embed._faiss_cache, "不存在的 key 不应触发清空全部"


# -----------------------------------------------------------------------------
# MedicineInfoStandardizer.extract_target_fields
# -----------------------------------------------------------------------------
class TestExtractTargetFields:
    """字段抽取：必须容错（异常 → []），且严格过滤 LLM 幻觉出的字段。"""

    def _make_standardizer_with_llm_response(self, llm_text):
        """构造一个会让 standardize_information 返回指定字符串的 fake llm。"""
        from embed import MedicineInfoStandardizer

        fake_llm = MagicMock(name="FakeLLM")
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = llm_text
        fake_llm.chat.completions.create.return_value = fake_response
        return MedicineInfoStandardizer(llm=fake_llm)

    def test_returns_intersection_with_field_list(self):
        """LLM 输出 '提到的药品名 / 标准化输出' 格式时，应提取并去重保留合法字段。"""
        llm_text = (
            "提到的药品名：当归\n"
            "标准化输出：\n"
            "性状\n"
            "功能主治\n"
        )
        standardizer = self._make_standardizer_with_llm_response(llm_text)
        result = standardizer.extract_target_fields("当归的性状和功能主治")
        assert "性状" in result
        assert "功能主治" in result
        assert all(f in standardizer.field_list for f in result), "结果必须是 field_list 子集"

    def test_filters_out_hallucinated_fields(self):
        """LLM 幻觉出 field_list 之外的字段（如 '玄学'）必须被丢弃。"""
        llm_text = (
            "提到的药品名：当归\n"
            "标准化输出：\n"
            "玄学\n"
            "性状\n"
            "胡说八道\n"
        )
        standardizer = self._make_standardizer_with_llm_response(llm_text)
        result = standardizer.extract_target_fields("xxx")
        assert result == ["性状"], f"非法字段必须被过滤，得到 {result}"

    def test_returns_empty_when_llm_raises(self):
        """standardize_information 抛错（如网络 / 鉴权）必须降级为 []，不能向上抛。"""
        from embed import MedicineInfoStandardizer

        fake_llm = MagicMock(name="FakeLLM")
        fake_llm.chat.completions.create.side_effect = RuntimeError("boom")
        standardizer = MedicineInfoStandardizer(llm=fake_llm)

        # 不应抛异常
        result = standardizer.extract_target_fields("xxx")
        assert result == []

    def test_returns_empty_when_llm_output_unparseable(self):
        """LLM 返回乱七八糟的字符串（无 '提到的药品名' 标头）→ extract_drug_info 解析为空。"""
        standardizer = self._make_standardizer_with_llm_response("hello world, no structure here")
        result = standardizer.extract_target_fields("xxx")
        assert result == []

    def test_init_without_llm_uses_lazy_client(self):
        """不传 llm 时应回退到 get_openai_client() 的懒加载客户端。"""
        from embed import MedicineInfoStandardizer, get_openai_client

        standardizer = MedicineInfoStandardizer()
        # 懒加载客户端在 conftest 中是 MagicMock；这里只断言确实回退到了它
        assert standardizer.llm is get_openai_client()


# -----------------------------------------------------------------------------
# retrieve_vector_and_text
# -----------------------------------------------------------------------------
class TestRetrieveVectorAndText:
    """端到端：np.load + faiss.IndexFlatL2 走通；缓存命中第二次免重建。"""

    def test_returns_tuples_of_id_title_text(self, sample_faiss_data):
        from embed import retrieve_vector_and_text

        results = retrieve_vector_and_text("白色粉末", sample_faiss_data, top_k=3)

        assert len(results) == 3
        for doc_id, title, text in results:
            # ids 与 texts 由 sample_faiss_data fixture 控制
            assert doc_id in {"drug_a", "drug_b", "drug_c"}
            assert isinstance(title, str)
            assert isinstance(text, str)

    def test_second_call_hits_cache(self, sample_faiss_data):
        """第二次以相同路径调用应命中 _faiss_cache，不再触发 np.load。"""
        import embed

        # 第一次：触发 np.load 与 IndexFlatL2 构建
        embed.retrieve_vector_and_text("q1", sample_faiss_data, top_k=3)
        assert sample_faiss_data in embed._faiss_cache

        # 替换 np.load 为爆炸函数：若被调用则一定失败
        with patch.object(np, "load", side_effect=AssertionError("不应再次加载 .npz")):
            embed.retrieve_vector_and_text("q2", sample_faiss_data, top_k=3)

    def test_raises_when_file_missing(self, tmp_path):
        from embed import retrieve_vector_and_text

        missing_path = str(tmp_path / "missing.npz")
        with pytest.raises(FileNotFoundError):
            retrieve_vector_and_text("anything", missing_path, top_k=3)


# -----------------------------------------------------------------------------
# extract_subsections / extract_drug_info（轻量校验，确保 P0 重构未改坏行为）
# -----------------------------------------------------------------------------
class TestSubsectionAndDrugInfoParsers:
    """这两个纯文本解析器是上传链路的核心，留一个 smoke 测试做防回归。"""

    def test_extract_subsections_basic(self):
        from embed import extract_subsections

        content = "【性状】白色粉末，无臭【功能主治】解表，清热【用法与用量】6~9g"
        sections = extract_subsections(content)
        # extract_subsections 在最后一个标题后只能拿到剩余内容，前面的小节正常切
        assert "性状" in sections
        assert "白色粉末" in sections.get("性状", "")
        assert "功能主治" in sections

    def test_extract_drug_info_parses_format(self):
        from embed import extract_drug_info

        text = (
            "提到的药品名：当归\n"
            "标准化输出：\n"
            "性状\n"
            "功能主治\n"
        )
        drugs, outputs = extract_drug_info(text)
        assert drugs == ["当归"]
        assert outputs == [["性状", "功能主治"]]


# -----------------------------------------------------------------------------
# retrieve_drug_subsections
# -----------------------------------------------------------------------------
class TestRetrieveDrugSubsections:
    """ES 精确锁定药品子段落：无需 FAISS，直接查 ES 原文并字段过滤。"""

    def _make_fake_es(self, doc_id, content, found=True):
        """构造一个按 id 返回 content 的 fake ES 客户端。"""
        class FakeES:
            def get(self, index, id, ignore=None):
                if id == doc_id and found:
                    return {"found": True, "_source": {"content": content}}
                return {"found": False}
        return FakeES()

    def test_hit(self, monkeypatch):
        """正常药品 + 字段命中 → 匹配子段落置顶，others 补足 top_k。"""
        from embed import retrieve_drug_subsections

        fake_es = self._make_fake_es(
            "川射干",
            "【性状】\n本品为不规则薄片。\n【鉴别】\n显微观察。"
        )
        monkeypatch.setattr("embed.connect_elasticsearch", lambda: fake_es)
        monkeypatch.setattr("embed.config.ES_INDEX", "zhyd")

        results = retrieve_drug_subsections("川射干", ["性状"], top_k=3)
        # matched(1) + others(1) = 2，截断到 top_k=3 仍为 2
        assert len(results) == 2
        # 匹配字段必须排在第一位
        assert results[0][0] == "川射干"
        assert results[0][1] == "性状"
        assert "不规则薄片" in results[0][2]
        # others 排在后面
        assert results[1][1] == "鉴别"

    def test_not_found(self, monkeypatch):
        """药品不存在 → 返回 []。"""
        from embed import retrieve_drug_subsections

        fake_es = self._make_fake_es("不存在药品", "", found=False)
        monkeypatch.setattr("embed.connect_elasticsearch", lambda: fake_es)
        monkeypatch.setattr("embed.config.ES_INDEX", "zhyd")

        assert retrieve_drug_subsections("不存在药品", ["性状"], 3) == []

    def test_no_fields(self, monkeypatch):
        """target_fields=[] → 返回全部子段落（截断到 top_k）。"""
        from embed import retrieve_drug_subsections

        fake_es = self._make_fake_es(
            "川射干",
            "【性状】A\n【鉴别】B\n【功能与主治】C"
        )
        monkeypatch.setattr("embed.connect_elasticsearch", lambda: fake_es)
        monkeypatch.setattr("embed.config.ES_INDEX", "zhyd")

        results = retrieve_drug_subsections("川射干", [], top_k=3)
        assert len(results) == 3
        titles = [r[1] for r in results]
        assert "性状" in titles
        assert "鉴别" in titles
        assert "功能与主治" in titles

    def test_no_match_field(self, monkeypatch):
        """字段在药典中不存在 → 返回该药品全部子段落（matched=[] 时 others 补足）。"""
        from embed import retrieve_drug_subsections

        fake_es = self._make_fake_es(
            "川射干",
            "【性状】A\n【鉴别】B"
        )
        monkeypatch.setattr("embed.connect_elasticsearch", lambda: fake_es)
        monkeypatch.setattr("embed.config.ES_INDEX", "zhyd")

        results = retrieve_drug_subsections("川射干", ["副作用"], top_k=3)
        # 实现中 matched=[] 但 others=[性状,鉴别] 会 appended
        assert len(results) == 2
        assert results[0][1] == "性状"
        assert results[1][1] == "鉴别"
