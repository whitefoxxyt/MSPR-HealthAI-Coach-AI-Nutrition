from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.config import settings
from app.services.schema_sanitizer import sanitize_for_mistral_strict

_OLLAMA_DEFAULT_MODEL = "gemma3:4b"
_OLLAMA_DEFAULT_TIMEOUT_S = 180.0
_OLLAMA_DEFAULT_NUM_PREDICT = 2048

_MISTRAL_DEFAULT_MODEL = "mistral-small-latest"
_MISTRAL_DEFAULT_BASE_URL = "https://api.mistral.ai/v1"
_MISTRAL_DEFAULT_TIMEOUT_S = 60.0


class LLMProvider(ABC):
    """Interface unique d'inference LLM : `await provider.generate(prompt, schema)`.

    schema=None : sortie texte libre.
    schema=dict : sortie JSON contrainte par le schema (mode strict cote Mistral,
    parametre `format` cote Ollama).
    """

    @abstractmethod
    async def generate(self, prompt: str, schema: dict[str, Any] | None) -> str:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        base_url: str,
        model: str = _OLLAMA_DEFAULT_MODEL,
        timeout: float = _OLLAMA_DEFAULT_TIMEOUT_S,
        num_predict: int | None = _OLLAMA_DEFAULT_NUM_PREDICT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._num_predict = num_predict

    async def generate(self, prompt: str, schema: dict[str, Any] | None) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if schema is not None:
            payload["format"] = schema
        if self._num_predict is not None:
            payload["options"] = {"num_predict": self._num_predict}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data.get("response", "")


class MistralProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = _MISTRAL_DEFAULT_MODEL,
        base_url: str = _MISTRAL_DEFAULT_BASE_URL,
        timeout: float = _MISTRAL_DEFAULT_TIMEOUT_S,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def generate(self, prompt: str, schema: dict[str, Any] | None) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": sanitize_for_mistral_strict(schema),
                    "strict": True,
                },
            }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]


def get_provider(name: str | None = None) -> LLMProvider:
    """Factory : retourne un provider configure depuis `app.config.settings`.

    name=None -> lit settings.llm_backend.
    """
    backend = (name or settings.llm_backend).lower()
    if backend == "ollama":
        return OllamaProvider(base_url=settings.ollama_host)
    if backend == "mistral":
        return MistralProvider(
            api_key=settings.mistral_api_key,
            model=settings.mistral_model,
            base_url=settings.mistral_base_url,
        )
    raise ValueError(f"Backend LLM inconnu : {backend!r}")
