from __future__ import annotations

import os

import httpx
import pytest


@pytest.mark.slow
def test_ollama_tags_endpoint_responds() -> None:
    """Smoke test : Ollama est joignable et expose /api/tags.

    Ne mock pas : exige un serveur Ollama reel. Skip si OLLAMA_HOST n'est pas defini
    ou si la connexion echoue (le service n'est pas tourne en local).
    """
    host = os.environ.get("OLLAMA_HOST")
    if not host:
        pytest.skip("OLLAMA_HOST not set, skipping real Ollama smoke test.")

    try:
        response = httpx.get(f"{host}/api/tags", timeout=5.0)
    except httpx.RequestError as exc:
        pytest.skip(f"Ollama unreachable at {host}: {exc}")

    assert response.status_code == 200
    data = response.json()
    assert "models" in data
