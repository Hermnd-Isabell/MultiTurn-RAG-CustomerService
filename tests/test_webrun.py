"""
pkg/webrun.py 测试覆盖：
- slow_echo 的 6 个关键分支：缺 .npz 报错、ENABLE_INTENT_ROUTING 关闭、obvious_chitchat、
  obvious_pharmacy、ambiguous→good、ambiguous→bad、意图路由层异常自降级；
- _score_result_by_fields 的精确 / 包含 / 空 title / 空 fields 四种打分；
- update_config：把入参写回 config，且会触发 ES + OpenAI 客户端缓存失效；
- UploadDoc.store_in_elasticsearch：ES 未就绪时优雅降级，不抛异常。

webrun.slow_echo 是流式生成器（yield 字符串片段），测试中通过 list(...) 消费。
所有 LLM / ES / FAISS 调用均被替换为 MagicMock，保证测试无副作用。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# -----------------------------------------------------------------------------
# 工具：把 OpenAI streaming 响应造成可被 slow_echo 消费的 generator
# -----------------------------------------------------------------------------
def _make_streaming_chunks(pieces=("hello", " from ", "test")):
    """构造若干个 .choices[0].delta.content 为字符串的 chunk 对象。"""
    chunks = []
    for piece in pieces:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = piece
        chunks.append(chunk)
    return chunks


def _patch_streaming_llm(webrun_mod):
    """
    给 webrun 模块内的 get_openai_client 打桩：
    返回的 client.chat.completions.create() 会 yield 一组固定 chunk。
    返回 (mock_client, mock_create) 便于断言被调用次数。
    """
    fake_client = MagicMock(name="FakeOpenAIClient")

    def _stream(*_args, **_kwargs):
        for c in _make_streaming_chunks():
            yield c

    fake_client.chat.completions.create.side_effect = _stream
    return fake_client


# -----------------------------------------------------------------------------
# slow_echo 分支测试
# -----------------------------------------------------------------------------
class TestSlowEchoBranches:
    """覆盖 slow_echo 各意图路径，每个用例只验证对应 LLM 子调用是否被触发。"""

    def _run_slow_echo(self, webrun_mod, message="测试消息", history=None):
        """消费 slow_echo 的 generator，返回最终累积字符串。"""
        history = history or []
        chunks = list(webrun_mod.slow_echo(message, history))
        return chunks[-1] if chunks else ""

    def test_missing_npz_skips_retrieval_and_calls_llm(self, tmp_path, restore_config):
        """VECTOR_DB_PATH 不存在时应跳过检索，直接走 LLM 自由问答。"""
        import webrun

        # 关键：必须改 webrun.config（slow_echo 闭包看到的 config 单例）
        webrun.config.VECTOR_DB_PATH = str(tmp_path / "definitely_missing.npz")
        webrun.config.ENABLE_INTENT_ROUTING = False

        fake_client = _patch_streaming_llm(webrun)
        with patch.object(webrun, "retrieve_vector_and_text") as mock_retrieve, \
             patch.object(webrun, "get_openai_client", return_value=fake_client):
            chunks = list(webrun.slow_echo("anything", []))

        # 不应包含错误提示
        assert not any("not found" in c.lower() or "vector database" in c.lower() for c in chunks), (
            f"不应提示向量库缺失，得到: {chunks}"
        )
        mock_retrieve.assert_not_called()
        fake_client.chat.completions.create.assert_called_once()
        # 验证 messages 中没有包含上下文
        call_args = fake_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert user_msg["content"] == "anything"

    def test_routing_disabled_skips_intent_layer(self, sample_faiss_data, restore_config):
        """ENABLE_INTENT_ROUTING=False 时 quick_intent_hint / classify 都不应被调用。"""
        import webrun

        webrun.config.VECTOR_DB_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = False

        fake_client = _patch_streaming_llm(webrun)
        with patch.object(webrun, "quick_intent_hint") as mock_hint, \
             patch.object(webrun, "classify_pharmacy_query") as mock_classify, \
             patch.object(webrun, "MedicineInfoStandardizer") as mock_std_cls, \
             patch.object(webrun, "retrieve_vector_and_text", return_value=[]), \
             patch.object(webrun, "get_openai_client", return_value=fake_client):
            self._run_slow_echo(webrun)

        mock_hint.assert_not_called()
        mock_classify.assert_not_called()
        mock_std_cls.assert_not_called()
        # 主路径仍应调用 LLM
        fake_client.chat.completions.create.assert_called_once()

    def test_obvious_chitchat_skips_classify_and_extract(self, sample_faiss_data, restore_config):
        """命中 obvious_chitchat 时 classify 和 extract 都不应触发，省 2 次 LLM。"""
        import webrun

        webrun.config.VECTOR_DB_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = True

        fake_client = _patch_streaming_llm(webrun)
        with patch.object(webrun, "quick_intent_hint", return_value="obvious_chitchat") as mock_hint, \
             patch.object(webrun, "classify_pharmacy_query") as mock_classify, \
             patch.object(webrun, "MedicineInfoStandardizer") as mock_std_cls, \
             patch.object(webrun, "retrieve_vector_and_text", return_value=[]), \
             patch.object(webrun, "get_openai_client", return_value=fake_client):
            self._run_slow_echo(webrun, message="你好")

        mock_hint.assert_called_once()
        mock_classify.assert_not_called()
        mock_std_cls.assert_not_called()

    def test_obvious_pharmacy_skips_classify_but_calls_extract(self, sample_faiss_data, restore_config):
        """命中 obvious_pharmacy 时跳过 classify，直接走 extract_target_fields，省 1 次 LLM。"""
        import webrun

        webrun.config.VECTOR_DB_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = True

        fake_standardizer = MagicMock(name="FakeStandardizer")
        fake_standardizer.extract_target_fields.return_value = ["性状"]

        fake_client = _patch_streaming_llm(webrun)
        with patch.object(webrun, "quick_intent_hint", return_value="obvious_pharmacy"), \
             patch.object(webrun, "classify_pharmacy_query") as mock_classify, \
             patch.object(webrun, "MedicineInfoStandardizer", return_value=fake_standardizer) as mock_std_cls, \
             patch.object(webrun, "retrieve_vector_and_text", return_value=[("d1", "性状", "白色粉末")]), \
             patch.object(webrun, "get_openai_client", return_value=fake_client):
            self._run_slow_echo(webrun, message="当归的性状")

        mock_classify.assert_not_called(), "obvious_pharmacy 必须跳过 classify"
        mock_std_cls.assert_called_once()
        fake_standardizer.extract_target_fields.assert_called_once_with("当归的性状")

    def test_ambiguous_good_calls_classify_and_extract(self, sample_faiss_data, restore_config):
        """ambiguous + classify=good：应当先后调用 classify 与 extract。"""
        import webrun

        webrun.config.VECTOR_DB_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = True

        fake_standardizer = MagicMock(name="FakeStandardizer")
        fake_standardizer.extract_target_fields.return_value = ["功能主治"]

        fake_client = _patch_streaming_llm(webrun)
        with patch.object(webrun, "quick_intent_hint", return_value="ambiguous"), \
             patch.object(webrun, "classify_pharmacy_query", return_value="good") as mock_classify, \
             patch.object(webrun, "MedicineInfoStandardizer", return_value=fake_standardizer), \
             patch.object(webrun, "retrieve_vector_and_text", return_value=[("d1", "功能主治", "解表")]), \
             patch.object(webrun, "get_openai_client", return_value=fake_client):
            self._run_slow_echo(webrun, message="它能治什么")

        mock_classify.assert_called_once()
        fake_standardizer.extract_target_fields.assert_called_once()

    def test_ambiguous_bad_skips_extract(self, sample_faiss_data, restore_config):
        """ambiguous + classify=bad：不应触发 extract_target_fields，target_fields=[]。"""
        import webrun

        webrun.config.VECTOR_DB_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = True

        fake_client = _patch_streaming_llm(webrun)
        with patch.object(webrun, "quick_intent_hint", return_value="ambiguous"), \
             patch.object(webrun, "classify_pharmacy_query", return_value="bad") as mock_classify, \
             patch.object(webrun, "MedicineInfoStandardizer") as mock_std_cls, \
             patch.object(webrun, "retrieve_vector_and_text", return_value=[]), \
             patch.object(webrun, "get_openai_client", return_value=fake_client):
            self._run_slow_echo(webrun)

        mock_classify.assert_called_once()
        mock_std_cls.assert_not_called(), "classify=bad 时必须跳过 extract"

    def test_intent_layer_exception_falls_back_gracefully(self, sample_faiss_data, restore_config):
        """quick_intent_hint 抛异常时主流程不应中断，最终仍能拿到 LLM 流式返回。"""
        import webrun

        webrun.config.VECTOR_DB_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = True

        fake_client = _patch_streaming_llm(webrun)
        with patch.object(webrun, "quick_intent_hint", side_effect=RuntimeError("boom")), \
             patch.object(webrun, "retrieve_vector_and_text", return_value=[("d1", "X", "Y")]), \
             patch.object(webrun, "get_openai_client", return_value=fake_client):
            chunks = list(webrun.slow_echo("anything", []))

        # 应仍然产出 LLM 流式拼接结果（非空，且不是 'Error: ...'）
        assert chunks, "意图层异常不应阻塞主流程"
        assert not chunks[-1].startswith("Error:"), f"不应进 LLM 调用失败分支：{chunks[-1]!r}"

    def test_retrieval_exception_yields_no_context_message(self, sample_faiss_data, restore_config):
        """retrieve_vector_and_text 抛错时仍应正常调用 LLM（context = 'No context found'）。"""
        import webrun

        webrun.config.VECTOR_DB_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = False

        fake_client = _patch_streaming_llm(webrun)
        with patch.object(webrun, "retrieve_vector_and_text", side_effect=RuntimeError("faiss boom")), \
             patch.object(webrun, "get_openai_client", return_value=fake_client):
            chunks = list(webrun.slow_echo("anything", []))

        assert chunks, "检索失败也应继续调 LLM"
        # 检查传入 LLM 的 messages 含 'No context found'
        call_args = fake_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "No context found" in user_msg["content"]

    def test_gradio6_message_dict_history(self, sample_faiss_data, restore_config):
        """Gradio 6.x 使用 openai-style MessageDict 列表作为 history，slow_echo 应正确解析。"""
        import webrun

        webrun.config.VECTOR_DB_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = False

        fake_client = _patch_streaming_llm(webrun)
        with patch.object(webrun, "retrieve_vector_and_text", return_value=[]), \
             patch.object(webrun, "get_openai_client", return_value=fake_client):
            # Gradio 6.x 格式：dict 列表
            history = [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！我是药典助手。"},
            ]
            chunks = list(webrun.slow_echo("再见", history))

        assert chunks, "MessageDict 格式 history 不应导致异常"
        # 验证 messages 中包含了历史记录
        call_args = fake_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles

    def test_disable_thinking_sends_extra_body(self, sample_faiss_data, restore_config):
        """enable_thinking=False 时应在 LLM 请求中附加 extra_body 禁用思考。"""
        import webrun

        webrun.config.VECTOR_DB_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = False

        fake_client = _patch_streaming_llm(webrun)
        with patch.object(webrun, "retrieve_vector_and_text", return_value=[]), \
             patch.object(webrun, "get_openai_client", return_value=fake_client):
            list(webrun.slow_echo("test", [], enable_thinking=False))

        call_args = fake_client.chat.completions.create.call_args
        kwargs = call_args.kwargs
        assert "extra_body" in kwargs, "应包含 extra_body 参数"
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}, (
            f"extra_body 内容不匹配: {kwargs['extra_body']}"
        )

    def test_enable_thinking_shows_reasoning_content(self, sample_faiss_data, restore_config):
        """enable_thinking=True 且流式返回包含 reasoning_content 时应输出思考过程。"""
        import webrun

        webrun.config.VECTOR_DB_PATH = sample_faiss_data
        webrun.config.ENABLE_INTENT_ROUTING = False

        def _stream_with_reasoning(*_args, **_kwargs):
            # 先输出 reasoning，再输出 content
            pieces = [
                ("先考虑", None),
                ("再验证", None),
                (None, "最终回答"),
            ]
            for reasoning, content in pieces:
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                delta = chunk.choices[0].delta
                delta.reasoning_content = reasoning
                delta.content = content
                yield chunk

        fake_client = MagicMock(name="FakeOpenAIClient")
        fake_client.chat.completions.create.side_effect = _stream_with_reasoning

        with patch.object(webrun, "retrieve_vector_and_text", return_value=[]), \
             patch.object(webrun, "get_openai_client", return_value=fake_client):
            chunks = list(webrun.slow_echo("test", [], enable_thinking=True))

        final = chunks[-1]
        assert "💭 思考中..." in final, f"应包含思考标记，得到: {final}"
        assert "先考虑" in final, f"应包含 reasoning 内容，得到: {final}"
        assert "最终回答" in final, f"应包含 content 内容，得到: {final}"


# -----------------------------------------------------------------------------
# _score_result_by_fields
# -----------------------------------------------------------------------------
class TestScoreResultByFields:
    """这是 slow_echo 字段重排的打分函数（在 webrun.py 中实现）。"""

    def test_exact_match_returns_one(self):
        from webrun import _score_result_by_fields

        assert _score_result_by_fields("性状", ["性状"]) == 1

    def test_partial_contains_returns_one(self):
        """title 包含 field 或被 field 包含都算命中。"""
        from webrun import _score_result_by_fields

        assert _score_result_by_fields("性状描述", ["性状"]) == 1
        assert _score_result_by_fields("味", ["性味与归经"]) == 1

    def test_no_match_returns_zero(self):
        from webrun import _score_result_by_fields

        assert _score_result_by_fields("功能主治", ["性状"]) == 0

    def test_empty_target_fields_returns_zero(self):
        from webrun import _score_result_by_fields

        assert _score_result_by_fields("功能主治", []) == 0

    @pytest.mark.parametrize("title", [None, "", "   "])
    def test_empty_or_whitespace_title_returns_zero(self, title):
        """空 / None / 全空白的 title 不应被任何 field 误判命中（避免空串包含假阳性）。"""
        from webrun import _score_result_by_fields

        assert _score_result_by_fields(title, ["性状", "功能主治"]) == 0

    def test_skips_empty_fields_in_list(self):
        """target_fields 中的空字符串应被跳过，不影响其它合法字段的命中。"""
        from webrun import _score_result_by_fields

        assert _score_result_by_fields("性状", ["", "性状", None]) == 1


# -----------------------------------------------------------------------------
# update_config
# -----------------------------------------------------------------------------
class TestUpdateConfig:
    """配置 tab 写回 config 后必须失效 ES + OpenAI 客户端缓存。"""

    def test_writes_back_to_config(self):
        """update_config 应直接写回 webrun 模块持有的 config 单例。
        手工 snapshot/restore 而非借用 restore_config fixture——后者的 config 对象
        在 test_config.py 的 importlib.reload 之后已与 webrun.config 不再是同一实例。"""
        import webrun

        keys = ("ES_HOST", "ES_PORT", "ES_USER", "ES_PASSWORD",
                "ES_INDEX", "VECTOR_DB_PATH", "ES_SCHEME")
        snapshot = {k: getattr(webrun.config, k) for k in keys}
        try:
            webrun.update_config(
                es_host="newhost",
                es_port="9300",
                es_user="newuser",
                es_pass="newpass",
                es_index="new_idx",
                vector_db="/tmp/new.npz",
                es_scheme="https",
            )

            assert webrun.config.ES_HOST == "newhost"
            assert webrun.config.ES_PORT == 9300, "ES_PORT 必须被转 int"
            assert isinstance(webrun.config.ES_PORT, int)
            assert webrun.config.ES_USER == "newuser"
            assert webrun.config.ES_PASSWORD == "newpass"
            assert webrun.config.ES_INDEX == "new_idx"
            assert webrun.config.VECTOR_DB_PATH == "/tmp/new.npz"
            assert webrun.config.ES_SCHEME == "https"
        finally:
            for k, v in snapshot.items():
                setattr(webrun.config, k, v)

    def test_invalidates_es_and_openai_caches(self):
        import embed
        import webrun

        # 先在两个 cache 里塞点东西
        webrun._es_client = MagicMock(name="StaleES")
        embed._openai_client = MagicMock(name="StaleOpenAI")

        webrun.update_config("h", "1234", "u", "p", "i", "/tmp/x.npz", "http")

        assert webrun._es_client is None, "update_config 必须清 ES 客户端缓存"
        assert embed._openai_client is None, "update_config 必须清 OpenAI 客户端缓存"

    def test_returns_friendly_message(self):
        import webrun

        msg = webrun.update_config("h", "1234", "u", "p", "i", "/tmp/x.npz", "http")
        assert isinstance(msg, str) and "配置" in msg


# -----------------------------------------------------------------------------
# UploadDoc.store_in_elasticsearch
# -----------------------------------------------------------------------------
class TestUploadDocStoreInElasticsearch:
    """ES 客户端不可用时 store_in_elasticsearch 必须优雅降级，不抛异常。"""

    def test_graceful_when_es_not_ready(self, tmp_path):
        """get_es_client 返回 None 时，方法应直接 return，不调用 .index。"""
        import webrun

        uploader = webrun.UploadDoc(file_input=str(tmp_path / "fake.docx"))
        with patch.object(webrun, "get_es_client", return_value=None):
            # 不应抛异常
            uploader.store_in_elasticsearch({"当归": ["【性状】白色"]})

    def test_indexes_each_chapter_when_es_ready(self, tmp_path):
        """ES 就绪时应对每个 (title, content) 调一次 .index()。"""
        import webrun

        fake_es = MagicMock(name="FakeES")
        uploader = webrun.UploadDoc(file_input=str(tmp_path / "fake.docx"))
        uploader.es_index = "test-index"

        content_dict = {
            "当归": ["【性状】白色粉末"],
            "黄连": ["【功能主治】清热"],
        }
        with patch.object(webrun, "get_es_client", return_value=fake_es):
            uploader.store_in_elasticsearch(content_dict)

        assert fake_es.index.call_count == 2
        # 第一次调用的 id 应为字典中第一个 title
        first_call = fake_es.index.call_args_list[0]
        assert first_call.kwargs.get("index") == "test-index"
        assert first_call.kwargs.get("id") in content_dict

    def test_es_index_exception_does_not_propagate(self, tmp_path):
        """单条 .index() 抛 ConnectionError / TransportError 时其它条目应继续处理。"""
        import webrun
        from elasticsearch import exceptions

        fake_es = MagicMock(name="FakeES")
        # 第一次抛 ConnectionError，第二次正常
        fake_es.index.side_effect = [exceptions.ConnectionError("boom"), None]

        uploader = webrun.UploadDoc(file_input=str(tmp_path / "fake.docx"))
        uploader.es_index = "test-index"
        content_dict = {"a": ["x"], "b": ["y"]}
        with patch.object(webrun, "get_es_client", return_value=fake_es):
            # 不应向上抛
            uploader.store_in_elasticsearch(content_dict)

        assert fake_es.index.call_count == 2


# -----------------------------------------------------------------------------
# get_es_client / clear_es_cache 的懒加载语义
# -----------------------------------------------------------------------------
class TestEsClientLifecycle:
    """与 OpenAI 客户端类似：首次连接 → 缓存 → clear 后重建。"""

    def test_first_call_invokes_connect(self):
        import webrun

        fake_es = MagicMock(name="FakeES")
        with patch.object(webrun, "connect_elasticsearch", return_value=fake_es) as mock_connect:
            client = webrun.get_es_client()
            assert client is fake_es
            mock_connect.assert_called_once()

    def test_second_call_uses_cache(self):
        import webrun

        fake_es = MagicMock(name="FakeES")
        with patch.object(webrun, "connect_elasticsearch", return_value=fake_es) as mock_connect:
            webrun.get_es_client()
            webrun.get_es_client()
            mock_connect.assert_called_once(), "二次调用必须命中缓存，不再连接"

    def test_clear_then_reconnect(self):
        import webrun

        fake_es1 = MagicMock(name="FakeES1")
        fake_es2 = MagicMock(name="FakeES2")
        with patch.object(webrun, "connect_elasticsearch", side_effect=[fake_es1, fake_es2]):
            assert webrun.get_es_client() is fake_es1
            webrun.clear_es_cache()
            assert webrun._es_client is None
            assert webrun.get_es_client() is fake_es2
