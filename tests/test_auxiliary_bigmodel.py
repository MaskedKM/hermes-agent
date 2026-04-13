"""Test that auxiliary client correctly maps bigmodel.cn endpoints."""

from agent.auxiliary_client import _to_openai_base_url


class TestBigModelOpenAIURL:
    """Z.AI / Zhipu bigmodel.cn should map /anthropic → /paas/v4."""

    def test_bigmodel_anthropic_to_paas_v4(self):
        result = _to_openai_base_url("https://open.bigmodel.cn/api/anthropic")
        assert result == "https://open.bigmodel.cn/api/paas/v4"

    def test_bigmodel_anthropic_with_trailing_slash(self):
        result = _to_openai_base_url("https://open.bigmodel.cn/api/anthropic/")
        assert result == "https://open.bigmodel.cn/api/paas/v4"

    def test_non_bigmodel_still_goes_to_v1(self):
        result = _to_openai_base_url("https://api.minimax.chat/anthropic")
        assert result == "https://api.minimax.chat/v1"

    def test_non_anthropic_url_unchanged(self):
        result = _to_openai_base_url("https://open.bigmodel.cn/api/paas/v4")
        assert result == "https://open.bigmodel.cn/api/paas/v4"

    def test_empty_and_none(self):
        assert _to_openai_base_url("") == ""
        assert _to_openai_base_url(None) == ""
