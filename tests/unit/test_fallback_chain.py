from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.services.fallback_chain import FallbackChain
from app.services.llm_provider import LLMProvider


class _FakeProvider(LLMProvider):
    """Provider deterministe pour les tests FallbackChain.

    Soit une reponse (str) a renvoyer, soit une Exception a lever. Compte les
    appels pour verifier l'ordre d'invocation primary -> secondary.
    """

    def __init__(self, response: str | Exception) -> None:
        self._response = response
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def generate(self, prompt: str, schema: dict[str, Any] | None) -> str:
        self.calls.append((prompt, schema))
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


# T1 : tracer bullet. Primary OK -> (content, primary_name), secondaire pas touche.


@pytest.mark.asyncio
async def test_primary_success_returns_primary_backend_name() -> None:
    primary = _FakeProvider('{"plan": "ok"}')
    secondary = _FakeProvider("ne doit pas etre appele")
    chain = FallbackChain(providers={"mistral": primary, "ollama": secondary})

    content, used = await chain.generate(
        "prompt", schema=None, primary_backend="mistral"
    )

    assert content == '{"plan": "ok"}'
    assert used == "mistral"
    assert len(primary.calls) == 1
    assert primary.calls[0] == ("prompt", None)
    assert secondary.calls == []


# T2 : primary leve HTTPError -> secondaire tente -> (content, secondary_name).


@pytest.mark.asyncio
async def test_primary_http_error_falls_back_to_secondary() -> None:
    primary_error = httpx.HTTPStatusError(
        "boom",
        request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
        response=httpx.Response(500),
    )
    primary = _FakeProvider(primary_error)
    secondary = _FakeProvider('{"plan": "from-ollama"}')
    chain = FallbackChain(providers={"mistral": primary, "ollama": secondary})

    content, used = await chain.generate(
        "prompt", schema={"type": "object"}, primary_backend="mistral"
    )

    assert content == '{"plan": "from-ollama"}'
    assert used == "ollama"
    assert len(primary.calls) == 1
    assert len(secondary.calls) == 1
    assert secondary.calls[0] == ("prompt", {"type": "object"})


# T3 : primary fail + secondary fail -> exception du secondaire remontee tel quel.


@pytest.mark.asyncio
async def test_both_providers_fail_raises_secondary_error() -> None:
    primary_error = httpx.ReadTimeout(
        "primary timeout",
        request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
    )
    secondary_error = httpx.HTTPStatusError(
        "ollama 500",
        request=httpx.Request("POST", "http://ollama:11434/api/generate"),
        response=httpx.Response(500),
    )
    primary = _FakeProvider(primary_error)
    secondary = _FakeProvider(secondary_error)
    chain = FallbackChain(providers={"mistral": primary, "ollama": secondary})

    with pytest.raises(httpx.HTTPStatusError) as exc:
        await chain.generate("prompt", schema=None, primary_backend="mistral")

    assert exc.value is secondary_error
    assert len(primary.calls) == 1
    assert len(secondary.calls) == 1


# T4 : sequence symetrique. Si primary=ollama et ollama down -> bascule mistral.


@pytest.mark.asyncio
async def test_ollama_primary_falls_back_to_mistral() -> None:
    ollama_error = httpx.ConnectError(
        "ollama unreachable",
        request=httpx.Request("POST", "http://ollama:11434/api/generate"),
    )
    mistral = _FakeProvider('{"plan":"from-mistral"}')
    ollama = _FakeProvider(ollama_error)
    chain = FallbackChain(providers={"mistral": mistral, "ollama": ollama})

    content, used = await chain.generate(
        "prompt", schema=None, primary_backend="ollama"
    )

    assert content == '{"plan":"from-mistral"}'
    assert used == "mistral"
    assert len(ollama.calls) == 1
    assert len(mistral.calls) == 1
