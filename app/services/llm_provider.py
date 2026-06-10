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
        num_ctx: int | None = None,
        temperature: float | None = None,
        num_gpu: int | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._num_predict = num_predict
        self._num_ctx = num_ctx
        self._temperature = temperature
        self._num_gpu = num_gpu

    async def generate(self, prompt: str, schema: dict[str, Any] | None) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if schema is not None:
            payload["format"] = schema
        options: dict[str, Any] = {}
        if self._num_predict is not None:
            options["num_predict"] = self._num_predict
        if self._num_ctx is not None:
            options["num_ctx"] = self._num_ctx
        if self._temperature is not None:
            options["temperature"] = self._temperature
        if self._num_gpu is not None:
            options["num_gpu"] = self._num_gpu
        if options:
            payload["options"] = options
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
        # Defensif : mirror du pattern Ollama (`data.get("response", "")`). Une
        # reponse 200 mais malformee renvoie "", ce qui declenche les retries
        # cote llm_client (JSONDecodeError ou ValidationError) plutot qu'un
        # KeyError non capture.
        choices = data.get("choices") or [{}]
        return choices[0].get("message", {}).get("content", "")


def get_provider(name: str | None = None) -> LLMProvider:
    """Factory : retourne un provider configure depuis `app.config.settings`.

    name=None -> lit settings.default_llm.
    """
    backend = (name or settings.default_llm).lower()
    if backend == "ollama":
        return OllamaProvider(
            base_url=settings.ollama_host,
            model=settings.ollama_model,
            num_ctx=settings.ollama_num_ctx,
            temperature=settings.ollama_temperature,
            num_gpu=settings.ollama_num_gpu,
        )
    if backend == "mistral":
        if not settings.mistral_api_key:
            raise ValueError("MISTRAL_API_KEY manquante alors que LLM_BACKEND=mistral.")
        return MistralProvider(
            api_key=settings.mistral_api_key,
            model=settings.mistral_model,
            base_url=settings.mistral_base_url,
        )
    raise ValueError(f"Backend LLM inconnu : {backend!r}")
