from __future__ import annotations

from typing import Any

import httpx

from app.config import settings
from app.services.llm_provider import LLMProvider, MistralProvider, OllamaProvider


class FallbackChain:
    """Sequence primaire puis secondaire d'appels LLMProvider.

    Couvre l'exigence PDF MSPR2 III.3 sur les mecanismes de fallback. Si le
    backend prioritaire echoue (HTTP, timeout, 429, 5xx), le chain bascule
    automatiquement sur l'autre provider connu. Si les deux echouent, la
    derniere exception remonte au caller (qui declenchera typiquement le
    static_fallback applicatif).
    """

    def __init__(self, providers: dict[str, LLMProvider]) -> None:
        self._providers = providers

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
        primary_backend: str,
    ) -> tuple[str, str]:
        if primary_backend not in self._providers:
            # Misconfiguration typique : DEFAULT_LLM=mistral mais MISTRAL_API_KEY
            # vide -> build_default_chain n'a ajoute qu'Ollama. CLAUDE.md promet
            # une bascule automatique vers le provider restant ; on respecte ce
            # contrat plutot que de remonter une erreur opaque.
            if not self._providers:
                raise ValueError(
                    "Aucun provider LLM disponible (verifie OLLAMA_HOST / MISTRAL_API_KEY)."
                )
            fallback = next(iter(self._providers))
            content = await self._providers[fallback].generate(prompt, schema)
            return content, fallback
        try:
            content = await self._providers[primary_backend].generate(prompt, schema)
            return content, primary_backend
        except httpx.HTTPError:
            secondary = self._pick_secondary(primary_backend)
            if secondary is None:
                raise
            content = await self._providers[secondary].generate(prompt, schema)
            return content, secondary

    def _pick_secondary(self, primary_backend: str) -> str | None:
        for name in self._providers:
            if name != primary_backend:
                return name
        return None


def build_default_chain() -> FallbackChain:
    """Construit un chain depuis `app.config.settings`.

    Ollama est toujours disponible (host configurable, pas de cle requise).
    Mistral est ajoute uniquement si `MISTRAL_API_KEY` est defini, sinon le
    chain reste mono-provider Ollama (deploiement "Mistral non configure").
    """
    providers: dict[str, LLMProvider] = {
        "ollama": OllamaProvider(base_url=settings.ollama_host),
    }
    if settings.mistral_api_key:
        providers["mistral"] = MistralProvider(
            api_key=settings.mistral_api_key,
            model=settings.mistral_model,
            base_url=settings.mistral_base_url,
        )
    return FallbackChain(providers=providers)


def format_fallback_warning(primary_backend: str, used_backend: str) -> str:
    """Message standard injecte dans compliance_warnings lorsqu'un fallback fire."""
    labels = {"mistral": "Mistral", "ollama": "Ollama"}
    primary = labels.get(primary_backend, primary_backend.capitalize())
    used = labels.get(used_backend, used_backend.capitalize())
    return f"{primary} indisponible, plan genere via {used}"
