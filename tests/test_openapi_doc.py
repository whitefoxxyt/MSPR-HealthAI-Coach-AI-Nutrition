from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENAPI_SNAPSHOT_PATH = REPO_ROOT / "docs" / "openapi.json"

VALID_TAGS = {"Analyse", "Plans", "Profil", "Historique", "Sante"}

# AC issue 56 : tous les endpoints metier (prefixe /api/v1/) exposent au moins
# {200, 400, 401, 403, 422, 503} dans leur documentation OpenAPI.
# Ces codes peuvent ne pas etre tous reachables (ex. 403 sans entitlements
# specifique), ils sont documentes pour annoncer aux clients l'eventail des
# reponses possibles cote service.
AC_BUSINESS_ERROR_CODES = {"400", "401", "403", "422", "503"}

# Codes specifiques en sus du baseline AC, par operation. /health est hors AC
# (endpoint sante unauth, ne touche pas Ollama, renvoie 200 avec status).
EXTRA_ERROR_CODES: list[tuple[str, str, str, set[str]]] = [
    ("/health", "get", "Sante", set()),
    ("/api/v1/analyze-meal", "post", "Analyse", {"413", "415"}),
    ("/api/v1/meal-analyses/me", "get", "Historique", set()),
    ("/api/v1/meal-plans/me", "get", "Historique", set()),
    ("/api/v1/generate-meal-plan", "post", "Plans", {"429"}),
    ("/api/v1/nutrition-goals/me", "get", "Profil", {"404"}),
    ("/api/v1/nutrition-goals/me", "put", "Profil", set()),
    ("/api/v1/me/macros", "get", "Profil", set()),
]


def _expected_error_codes(path: str, extra: set[str]) -> set[str]:
    """AC baseline pour les endpoints metier, vide pour /health."""
    if path.startswith("/api/v1/"):
        return AC_BUSINESS_ERROR_CODES | extra
    return extra


EXPECTED_OPERATIONS: list[tuple[str, str, str, set[str]]] = [
    (path, method, tag, _expected_error_codes(path, extra))
    for path, method, tag, extra in EXTRA_ERROR_CODES
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


# AC issue 56 : chaque endpoint metier doit fournir un exemple de reponse 200
# pour faciliter la decouverte API depuis Swagger / Redoc.
ENDPOINTS_WITH_RESPONSE_EXAMPLE: list[tuple[str, str]] = [
    ("/api/v1/analyze-meal", "post"),
    ("/api/v1/meal-analyses/me", "get"),
    ("/api/v1/meal-plans/me", "get"),
    ("/api/v1/generate-meal-plan", "post"),
    ("/api/v1/nutrition-goals/me", "get"),
    ("/api/v1/nutrition-goals/me", "put"),
    ("/api/v1/me/macros", "get"),
]


@pytest.mark.parametrize(
    "path,method",
    ENDPOINTS_WITH_RESPONSE_EXAMPLE,
    ids=[f"{m.upper()} {p}" for p, m in ENDPOINTS_WITH_RESPONSE_EXAMPLE],
)
def test_endpoint_has_200_example(
    openapi_schema: dict[str, Any], path: str, method: str
) -> None:
    op = openapi_schema["paths"][path][method]
    success = op["responses"].get("200") or {}
    content = (success.get("content") or {}).get("application/json") or {}
    assert (
        content.get("example") or content.get("examples")
    ), f"{method.upper()} {path} : pas d'example dans la reponse 200"


# AC issue 56 : les endpoints qui acceptent un body doivent fournir au moins un
# exemple de requete (Swagger l'affiche dans le "Try it out").
ENDPOINTS_WITH_REQUEST_EXAMPLE: list[tuple[str, str, str]] = [
    ("/api/v1/generate-meal-plan", "post", "application/json"),
    ("/api/v1/nutrition-goals/me", "put", "application/json"),
]


@pytest.mark.parametrize(
    "path,method,media_type",
    ENDPOINTS_WITH_REQUEST_EXAMPLE,
    ids=[f"{m.upper()} {p}" for p, m, _ in ENDPOINTS_WITH_REQUEST_EXAMPLE],
)
def test_endpoint_has_request_example(
    openapi_schema: dict[str, Any], path: str, method: str, media_type: str
) -> None:
    op = openapi_schema["paths"][path][method]
    request_body = op.get("requestBody") or {}
    content = (request_body.get("content") or {}).get(media_type) or {}
    assert (
        content.get("example") or content.get("examples")
    ), f"{method.upper()} {path} : pas d'example dans le requestBody {media_type}"


def test_openapi_snapshot_is_up_to_date(openapi_schema: dict[str, Any]) -> None:
    """Le snapshot `docs/openapi.json` doit refleter le schema courant.

    Drift detection : si le schema FastAPI change (nouveau endpoint, code
    d'erreur, example...), regenerer via `python scripts/export_openapi.py`
    et committer le diff.
    """
    assert OPENAPI_SNAPSHOT_PATH.exists(), (
        f"snapshot OpenAPI manquant : {OPENAPI_SNAPSHOT_PATH} "
        "(regenerer avec `python scripts/export_openapi.py`)"
    )
    snapshot = json.loads(OPENAPI_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert snapshot == openapi_schema, (
        "snapshot OpenAPI obsolete : regenerer avec "
        "`python scripts/export_openapi.py` et committer le diff."
    )
