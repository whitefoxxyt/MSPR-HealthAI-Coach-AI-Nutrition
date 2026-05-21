from __future__ import annotations

import json
import re
from typing import Any

import httpx
import pytest
import respx

from app.services.llm_provider import MistralProvider, OllamaProvider, get_provider


def _mistral_response(content: str) -> dict[str, Any]:
    """Forme une reponse Mistral /v1/chat/completions."""
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "model": "mistral-small-latest",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _ollama_response(content: str | dict[str, Any]) -> dict[str, Any]:
    body = content if isinstance(content, str) else json.dumps(content)
    return {"response": body, "done": True}


# OllamaProvider : payload sur POST /api/generate (non-regression).


@pytest.mark.asyncio
async def test_ollama_provider_posts_model_prompt_stream_to_generate(
    mock_ollama: respx.MockRouter,
) -> None:
    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_response("hello world")
    )
    # num_predict=None : payload identique a llm_client._call_ollama_generate
    # historique (texte libre, sans options).
    provider = OllamaProvider(base_url="http://ollama:11434", num_predict=None)

    result = await provider.generate("dis bonjour", schema=None)

    assert result == "hello world"
    assert route.called
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "gemma3:4b"
    assert body["prompt"] == "dis bonjour"
    assert body["stream"] is False
    assert "format" not in body
    assert "options" not in body


@pytest.mark.asyncio
async def test_ollama_provider_with_schema_sends_format_and_num_predict(
    mock_ollama: respx.MockRouter,
) -> None:
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_response({"x": 1})
    )
    # Defaults : num_predict=2048 (config decrim_retry_orchestrator historique).
    provider = OllamaProvider(base_url="http://ollama:11434")

    result = await provider.generate("genere un x", schema=schema)

    assert json.loads(result) == {"x": 1}
    body = json.loads(route.calls.last.request.content)
    assert body["format"] == schema
    assert body["options"] == {"num_predict": 2048}


@pytest.mark.asyncio
async def test_ollama_provider_propagates_http_error(
    mock_ollama: respx.MockRouter,
) -> None:
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        500, json={"error": "boom"}
    )
    provider = OllamaProvider(base_url="http://ollama:11434")

    with pytest.raises(httpx.HTTPStatusError):
        await provider.generate("anything", schema=None)


# MistralProvider : POST /v1/chat/completions, Bearer auth, response_format strict.


@pytest.mark.asyncio
async def test_mistral_provider_with_schema_uses_response_format_strict() -> None:
    schema = {
        "$defs": {
            "Macros": {
                "type": "object",
                "properties": {"calories": {"type": "integer"}},
                "required": ["calories"],
            }
        },
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "macros": {"$ref": "#/$defs/Macros"},
        },
        "required": ["name", "macros"],
    }
    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://api.mistral.ai/v1/chat/completions").respond(
            200, json=_mistral_response('{"name":"plat","macros":{"calories":500}}')
        )
        provider = MistralProvider(
            api_key="sk-test", base_url="https://api.mistral.ai/v1"
        )

        result = await provider.generate("genere un plat", schema=schema)

        assert json.loads(result) == {"name": "plat", "macros": {"calories": 500}}
        body = json.loads(route.calls.last.request.content)
        rf = body["response_format"]
        assert rf["type"] == "json_schema"
        js = rf["json_schema"]
        assert js["strict"] is True
        # Le schema a ete sanitize : $defs resolu, additionalProperties false.
        sent = js["schema"]
        assert "$defs" not in sent
        assert sent["additionalProperties"] is False
        macros_inline = sent["properties"]["macros"]
        assert "$ref" not in macros_inline
        assert macros_inline["properties"]["calories"] == {"type": "integer"}


@pytest.mark.asyncio
async def test_mistral_provider_posts_to_chat_completions_with_bearer() -> None:
    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://api.mistral.ai/v1/chat/completions").respond(
            200, json=_mistral_response("plop")
        )
        provider = MistralProvider(
            api_key="sk-test", base_url="https://api.mistral.ai/v1"
        )

        result = await provider.generate("dis plop", schema=None)

        assert result == "plop"
        assert route.called
        request = route.calls.last.request
        assert request.headers["Authorization"] == "Bearer sk-test"
        body = json.loads(request.content)
        assert body["model"] == "mistral-small-latest"
        assert body["messages"] == [{"role": "user", "content": "dis plop"}]
        # Mode texte libre : pas de contrainte response_format.
        assert "response_format" not in body


@pytest.mark.asyncio
async def test_mistral_provider_propagates_429_rate_limit() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.post("https://api.mistral.ai/v1/chat/completions").respond(
            429, json={"error": "rate_limited"}
        )
        provider = MistralProvider(
            api_key="sk-test", base_url="https://api.mistral.ai/v1"
        )

        with pytest.raises(httpx.HTTPStatusError) as exc:
            await provider.generate("anything", schema=None)
        assert exc.value.response.status_code == 429


@pytest.mark.asyncio
async def test_mistral_provider_propagates_5xx() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.post("https://api.mistral.ai/v1/chat/completions").respond(
            500, json={"error": "internal"}
        )
        provider = MistralProvider(
            api_key="sk-test", base_url="https://api.mistral.ai/v1"
        )

        with pytest.raises(httpx.HTTPStatusError) as exc:
            await provider.generate("anything", schema=None)
        assert exc.value.response.status_code == 500


# get_provider : factory lit settings.llm_backend.


def test_get_provider_explicit_ollama_returns_ollama_provider() -> None:
    provider = get_provider("ollama")
    assert isinstance(provider, OllamaProvider)


def test_get_provider_explicit_mistral_returns_mistral_provider() -> None:
    provider = get_provider("mistral")
    assert isinstance(provider, MistralProvider)


def test_get_provider_unknown_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        get_provider("anthropic")


def test_get_provider_default_reads_settings_llm_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import config

    monkeypatch.setattr(config.settings, "llm_backend", "ollama")
    assert isinstance(get_provider(), OllamaProvider)

    monkeypatch.setattr(config.settings, "llm_backend", "mistral")
    assert isinstance(get_provider(), MistralProvider)
