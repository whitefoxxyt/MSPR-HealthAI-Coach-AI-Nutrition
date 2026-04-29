from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app

VALID_TAGS = {"Analyse", "Plans", "Profil", "Historique", "Sante"}

# (path, method, expected_tag, expected_error_codes)
EXPECTED_OPERATIONS: list[tuple[str, str, str, set[str]]] = [
    ("/health", "get", "Sante", set()),
    ("/api/v1/analyze-meal", "post", "Analyse", {"401", "413", "415", "422"}),
    ("/api/v1/meal-analyses/me", "get", "Historique", {"401"}),
    ("/api/v1/meal-plans/me", "get", "Historique", {"401"}),
    ("/api/v1/generate-meal-plan", "post", "Plans", {"401", "422", "429"}),
    ("/api/v1/nutrition-goals/me", "get", "Profil", {"401", "404"}),
    ("/api/v1/nutrition-goals/me", "put", "Profil", {"401", "422"}),
]


@pytest.fixture(scope="module")
def openapi_schema() -> dict[str, Any]:
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


@pytest.mark.parametrize("path", ["/docs", "/redoc"])
def test_swagger_and_redoc_render(path: str) -> None:
    response = TestClient(app).get(path)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert b"<html" in response.content.lower()


def test_app_has_global_metadata(openapi_schema: dict[str, Any]) -> None:
    info = openapi_schema["info"]
    assert info["title"]
    assert info.get("version")
    assert info.get("description")
    assert len(info["description"]) >= 80, "description globale trop courte"
    assert info.get("contact"), "info.contact manquant"


def test_app_declares_all_business_tags(openapi_schema: dict[str, Any]) -> None:
    declared = {tag["name"] for tag in openapi_schema.get("tags", [])}
    missing = VALID_TAGS - declared
    assert not missing, f"tags non declares dans tags_metadata : {missing}"
    for tag in openapi_schema.get("tags", []):
        if tag["name"] in VALID_TAGS:
            assert tag.get("description"), f"tag {tag['name']} sans description"


@pytest.mark.parametrize(
    "path,method,expected_tag,expected_errors",
    EXPECTED_OPERATIONS,
    ids=[f"{m.upper()} {p}" for p, m, _, _ in EXPECTED_OPERATIONS],
)
def test_operation_is_richly_documented(
    openapi_schema: dict[str, Any],
    path: str,
    method: str,
    expected_tag: str,
    expected_errors: set[str],
) -> None:
    paths = openapi_schema["paths"]
    assert path in paths, f"path absent du schema OpenAPI : {path}"
    assert method in paths[path], f"methode {method} absente sur {path}"
    op = paths[path][method]

    assert op.get("summary"), f"{method.upper()} {path} : summary manquant"
    description = op.get("description") or ""
    assert (
        len(description) >= 80
    ), f"{method.upper()} {path} : description trop courte ({len(description)} chars)"

    tags = op.get("tags") or []
    assert tags, f"{method.upper()} {path} : aucun tag"
    assert (
        tags[0] in VALID_TAGS
    ), f"{method.upper()} {path} : tag {tags[0]} hors whitelist"
    assert (
        tags[0] == expected_tag
    ), f"{method.upper()} {path} : tag {tags[0]} != {expected_tag} attendu"

    responses = op.get("responses") or {}
    success_codes = {code for code in responses if code.startswith("2")}
    assert success_codes, f"{method.upper()} {path} : aucune reponse 2xx documentee"

    missing_errors = expected_errors - set(responses)
    assert (
        not missing_errors
    ), f"{method.upper()} {path} : codes d'erreur manquants {missing_errors}"
