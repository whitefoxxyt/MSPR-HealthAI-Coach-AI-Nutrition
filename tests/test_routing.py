from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_stays_at_root():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}


def test_analyze_meal_no_prefix_returns_404():
    response = client.post(
        "/analyze-meal",
        files={"photo": ("x.txt", b"hello", "text/plain")},
        headers={"Authorization": "Bearer fake-token"},
    )
    assert response.status_code == 404


def test_analyze_meal_v1_returns_415_for_bad_content_type():
    response = client.post(
        "/api/v1/analyze-meal",
        files={"photo": ("x.txt", b"hello", "text/plain")},
        headers={"Authorization": "Bearer fake-token"},
    )
    assert response.status_code == 415


def test_openapi_lists_versioned_route():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/analyze-meal" in paths
    assert "/api/v1/nutrition-goals/me" in paths
    assert "/api/v1/generate-meal-plan" in paths
    assert "/health" in paths
    assert "/analyze-meal" not in paths
    assert "/nutrition-goals/me" not in paths
    assert "/generate-meal-plan" not in paths
