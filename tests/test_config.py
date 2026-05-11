"""
pkg/config.py 测试：
- ENABLE_INTENT_ROUTING 的布尔解析（最具回归价值的一项）；
- 其它 ES / OpenAI 类配置项能从环境变量被正确读入。
重点：Config 在 class 定义时读取 env，因此修改 env 后必须 importlib.reload 才能生效。
"""
from __future__ import annotations

import importlib

import pytest


# 将这里集中维护，便于在多个用例中复用
TRUE_VALUES = ["1", "true", "TRUE", "True", "yes", "YES", "on", "ON", "  true  "]
FALSE_VALUES = ["0", "false", "FALSE", "no", "off", "garbage", ""]


@pytest.mark.parametrize("env_value", TRUE_VALUES)
def test_enable_intent_routing_true(monkeypatch, env_value):
    """ENABLE_INTENT_ROUTING 在常见 truthy 字面量下应解析为 True。"""
    monkeypatch.setenv("ENABLE_INTENT_ROUTING", env_value)
    import config as config_module

    importlib.reload(config_module)
    assert config_module.config.ENABLE_INTENT_ROUTING is True, (
        f"ENABLE_INTENT_ROUTING={env_value!r} 期望 True，得到 {config_module.config.ENABLE_INTENT_ROUTING}"
    )


@pytest.mark.parametrize("env_value", FALSE_VALUES)
def test_enable_intent_routing_false(monkeypatch, env_value):
    """ENABLE_INTENT_ROUTING 在 falsy / 非法字面量下应解析为 False（含空字符串显式禁用场景）。"""
    monkeypatch.setenv("ENABLE_INTENT_ROUTING", env_value)
    import config as config_module

    importlib.reload(config_module)
    assert config_module.config.ENABLE_INTENT_ROUTING is False, (
        f"ENABLE_INTENT_ROUTING={env_value!r} 期望 False，得到 {config_module.config.ENABLE_INTENT_ROUTING}"
    )


def test_enable_intent_routing_default_when_unset(monkeypatch):
    """env 未设置时回退默认值 '1'，最终结果应为 True（即默认开启意图路由）。"""
    monkeypatch.delenv("ENABLE_INTENT_ROUTING", raising=False)
    import config as config_module

    importlib.reload(config_module)
    assert config_module.config.ENABLE_INTENT_ROUTING is True


def test_es_config_from_env(monkeypatch):
    """ES_HOST / ES_PORT / ES_USER / ES_PASSWORD / ES_INDEX / ES_SCHEME 都按 env 读取。"""
    monkeypatch.setenv("ES_HOST", "192.168.1.10")
    monkeypatch.setenv("ES_PORT", "9300")
    monkeypatch.setenv("ES_USER", "admin")
    monkeypatch.setenv("ES_PASSWORD", "secret")
    monkeypatch.setenv("ES_INDEX", "custom_index")
    monkeypatch.setenv("ES_SCHEME", "https")
    import config as config_module

    importlib.reload(config_module)
    cfg = config_module.config
    assert cfg.ES_HOST == "192.168.1.10"
    assert cfg.ES_PORT == 9300, "ES_PORT 必须被转成 int"
    assert isinstance(cfg.ES_PORT, int)
    assert cfg.ES_USER == "admin"
    assert cfg.ES_PASSWORD == "secret"
    assert cfg.ES_INDEX == "custom_index"
    assert cfg.ES_SCHEME == "https"


def test_es_defaults_when_unset(monkeypatch):
    """所有 ES_* 都未设置时应回落到 hard-coded 默认值。"""
    for name in ("ES_HOST", "ES_PORT", "ES_USER", "ES_PASSWORD", "ES_INDEX", "ES_SCHEME"):
        monkeypatch.delenv(name, raising=False)
    import config as config_module

    importlib.reload(config_module)
    cfg = config_module.config
    assert cfg.ES_HOST == "127.0.0.1"
    assert cfg.ES_PORT == 9200
    assert cfg.ES_USER == "elastic"
    assert cfg.ES_PASSWORD == "changeme"
    assert cfg.ES_INDEX == "zhyd"
    assert cfg.ES_SCHEME == "http"


def test_openai_config_from_env(monkeypatch):
    """OpenAI 凭证 / base_url / model 都按 env 读取。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-foo")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.foo.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    import config as config_module

    importlib.reload(config_module)
    cfg = config_module.config
    assert cfg.OPENAI_API_KEY == "sk-foo"
    assert cfg.OPENAI_BASE_URL == "https://api.foo.com/v1"
    assert cfg.LLM_MODEL == "gpt-4o-mini"


def test_openai_defaults_when_unset(monkeypatch):
    """OPENAI_API_KEY 未设置应为 None；base_url 与 model 回落到默认值（DeepSeek）。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    import config as config_module

    importlib.reload(config_module)
    cfg = config_module.config
    assert cfg.OPENAI_API_KEY is None
    assert cfg.OPENAI_BASE_URL == "https://api.deepseek.com"
    assert cfg.LLM_MODEL == "deepseek-chat"


def test_vector_db_path_default_uses_base_dir(monkeypatch):
    """未显式设置 VECTOR_DB_PATH 时应拼接到 BASE_DIR/embeddings2.npz。"""
    monkeypatch.delenv("VECTOR_DB_PATH", raising=False)
    import config as config_module

    importlib.reload(config_module)
    cfg = config_module.config
    assert cfg.VECTOR_DB_PATH.endswith("embeddings2.npz")
    assert cfg.BASE_DIR in cfg.VECTOR_DB_PATH


def test_vector_db_path_from_env(monkeypatch):
    """显式设置 VECTOR_DB_PATH 时直接采用，不再拼接 BASE_DIR。"""
    monkeypatch.setenv("VECTOR_DB_PATH", "/tmp/my.npz")
    import config as config_module

    importlib.reload(config_module)
    assert config_module.config.VECTOR_DB_PATH == "/tmp/my.npz"
