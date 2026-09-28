import pytest

from governance import llm
from governance.llm import AnthropicClient, GeminiClient, LLMError, OpenAIClient, client_from_env, extract_json


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure! Here it is: {"a": 1} Hope that helps.', {"a": 1}),
        ("[1, 2]", [1, 2]),
    ],
)
def test_extract_json(raw, expected):
    assert extract_json(raw) == expected


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("no json here")


@pytest.mark.parametrize("value", ["", "none", "NONE"])
def test_offline_when_no_provider(monkeypatch, value):
    monkeypatch.setenv("GOV_LLM_PROVIDER", value)
    assert client_from_env() is None


def test_unknown_provider(monkeypatch):
    monkeypatch.setenv("GOV_LLM_PROVIDER", "mystery")
    with pytest.raises(LLMError, match="Unknown"):
        client_from_env()


def test_missing_key(monkeypatch):
    monkeypatch.setenv("GOV_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        client_from_env()


def test_model_override(monkeypatch):
    monkeypatch.setenv("GOV_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("GOV_LLM_MODEL", "my-model")
    c = client_from_env()
    assert isinstance(c, OpenAIClient) and c.model == "my-model"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


@pytest.mark.parametrize(
    "client,payload,check",
    [
        (
            AnthropicClient("k"),
            {"content": [{"type": "text", "text": "hi"}]},
            lambda url, kw: url.endswith("/v1/messages")
            and kw["headers"]["x-api-key"] == "k"
            and kw["json"]["system"] == "S",
        ),
        (
            OpenAIClient("k"),
            {"choices": [{"message": {"content": "hi"}}]},
            lambda url, kw: url.endswith("/chat/completions") and kw["json"]["messages"][0]["role"] == "system",
        ),
        (
            GeminiClient("k"),
            {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]},
            lambda url, kw: ":generateContent" in url and kw["params"]["key"] == "k",
        ),
    ],
    ids=["anthropic", "openai", "gemini"],
)
def test_provider_request_and_response_shapes(monkeypatch, client, payload, check):
    seen = {}

    def fake_post(url, **kw):
        seen["url"], seen["kw"] = url, kw
        return FakeResponse(payload)

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    assert client.complete("S", "P") == "hi"
    assert check(seen["url"], seen["kw"])
    assert seen["kw"]["json"].get("temperature", seen["kw"]["json"].get("generationConfig", {}).get("temperature")) == 0
