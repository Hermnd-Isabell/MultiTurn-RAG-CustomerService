"""
冒烟测试：验证药品名感知重排修复的关键函数。
通过 pytest 运行，复用 conftest.py 中的重依赖 stub。
"""
import pytest
from embed import _score_result_by_drug_name, extract_drug_info
from webrun import _extract_drug_name_from_query, _confirm_drug_in_es


class TestScoreResultByDrugName:
    """验证 _score_result_by_drug_name 的模糊匹配逻辑。"""

    def test_exact_match_returns_one(self):
        assert _score_result_by_drug_name("瞿胆丸", "瞿胆丸") == 1

    def test_no_match_returns_zero(self):
        assert _score_result_by_drug_name("瞿胆丸", "黄藤素片") == 0

    def test_doc_id_contains_target_returns_one(self):
        assert _score_result_by_drug_name("复方瞿胆丸", "瞿胆丸") == 1

    def test_target_contains_doc_id_returns_one(self):
        assert _score_result_by_drug_name("瞿胆丸", "复方瞿胆丸") == 1

    def test_empty_doc_id_returns_zero(self):
        assert _score_result_by_drug_name("", "瞿胆丸") == 0

    def test_none_doc_id_returns_zero(self):
        assert _score_result_by_drug_name(None, "瞿胆丸") == 0

    def test_empty_target_returns_zero(self):
        assert _score_result_by_drug_name("瞿胆丸", "") == 0

    def test_none_target_returns_zero(self):
        assert _score_result_by_drug_name("瞿胆丸", None) == 0


class FakeMedicineInfoStandardizer:
    """Mock MedicineInfoStandardizer，控制 standardize_information 的返回值。"""
    def __init__(self, fixed_raw):
        self._raw = fixed_raw

    def standardize_information(self, _input_data):
        return self._raw


class TestExtractDrugNameFromQuery:
    """验证 _extract_drug_name_from_query 能正确解析 LLM 输出中的药品名。"""

    def test_single_drug(self):
        std = FakeMedicineInfoStandardizer(
            "提到的药品名：当归\n标准化输出：\n性状\n鉴别"
        )
        name = _extract_drug_name_from_query("当归的性状", standardizer=std)
        assert name == "当归"

    def test_multiple_drugs_take_first(self):
        std = FakeMedicineInfoStandardizer(
            "提到的药品名：瞿胆丸、黄藤素片\n标准化输出：\n性状"
        )
        name = _extract_drug_name_from_query("瞿胆丸和黄藤素片的性状", standardizer=std)
        assert name == "瞿胆丸"

    def test_no_drug_returns_none(self):
        std = FakeMedicineInfoStandardizer(
            "标准化输出：\n性状"
        )
        name = _extract_drug_name_from_query("性状是什么", standardizer=std)
        assert name is None

    def test_exception_fallback(self):
        class BrokenStd:
            def standardize_information(self, _input_data):
                raise RuntimeError("LLM 挂了")
        name = _extract_drug_name_from_query("随便问问", standardizer=BrokenStd())
        assert name is None

    def test_empty_input_returns_none(self):
        std = FakeMedicineInfoStandardizer("")
        name = _extract_drug_name_from_query("", standardizer=std)
        assert name is None


class FakeEsClient:
    """Mock Elasticsearch 客户端。"""
    def __init__(self, found_map):
        self._found_map = found_map

    def get(self, index, id, ignore=None):
        return {"found": self._found_map.get(id, False)}


class TestConfirmDrugInEs:
    """验证 _confirm_drug_in_es 能正确通过 ES 确认药品存在性。"""

    def test_drug_exists(self, monkeypatch):
        fake_es = FakeEsClient({"瞿胆丸": True})
        monkeypatch.setattr(
            "webrun.get_es_client", lambda: fake_es
        )
        assert _confirm_drug_in_es("瞿胆丸") is True

    def test_drug_not_exists(self, monkeypatch):
        fake_es = FakeEsClient({"瞿胆丸": True, "黄藤素片": False})
        monkeypatch.setattr(
            "webrun.get_es_client", lambda: fake_es
        )
        assert _confirm_drug_in_es("黄藤素片") is False

    def test_empty_drug_name(self, monkeypatch):
        fake_es = FakeEsClient({})
        monkeypatch.setattr(
            "webrun.get_es_client", lambda: fake_es
        )
        assert _confirm_drug_in_es("") is False

    def test_es_none_returns_false(self, monkeypatch):
        monkeypatch.setattr(
            "webrun.get_es_client", lambda: None
        )
        assert _confirm_drug_in_es("瞿胆丸") is False

    def test_es_exception_returns_false(self, monkeypatch):
        class BadEs:
            def get(self, index, id, ignore=None):
                raise RuntimeError("ES 炸了")
        monkeypatch.setattr(
            "webrun.get_es_client", lambda: BadEs()
        )
        assert _confirm_drug_in_es("瞿胆丸") is False
