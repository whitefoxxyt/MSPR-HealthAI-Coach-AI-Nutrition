from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
import respx
from cachetools import TTLCache

from app.config import settings
from app.services import entitlements_client
from app.services.entitlements_client import Entitlements, get_entitlements

AUTH_URL = "http://mspr-healthai-auth-test:3000"
ENDPOINT = f"{AUTH_URL}/api/entitlements/me"


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "auth_api_url", AUTH_URL)
    entitlements_client._cache.clear()
    entitlements_client._stale.clear()


async def test_returns_tier_free_when_auth_responds_with_free() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(ENDPOINT).respond(
            200,
            json={"tier": "free", "expires_at": None, "features": []},
        )

        result = await get_entitlements(user_id="u-1", jwt="jwt-token")

    assert isinstance(result, Entitlements)
    assert result.tier == "free"
    assert result.expires_at is None
    assert result.features == ()


async def test_second_call_with_same_user_id_hits_cache() -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.get(ENDPOINT).respond(
            200,
            json={"tier": "premium", "expires_at": None, "features": []},
        )

        first = await get_entitlements(user_id="u-cache", jwt="jwt-token")
        second = await get_entitlements(user_id="u-cache", jwt="jwt-token")

    assert route.call_count == 1
    assert first.tier == "premium"
    assert second.tier == "premium"


async def test_cache_expires_after_ttl_and_refetches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Horloge pilotee : permet de simuler le passage de plus de 60s sans dormir.
    clock = {"now": 1000.0}
    monkeypatch.setattr(
        entitlements_client,
        "_cache",
        TTLCache(maxsize=10000, ttl=60.0, timer=lambda: clock["now"]),
    )

    with respx.mock(assert_all_called=True) as router:
        route = router.get(ENDPOINT).respond(
            200,
            json={"tier": "free", "expires_at": None, "features": []},
        )

        await get_entitlements(user_id="u-ttl", jwt="jwt-token")
        clock["now"] += 61.0
        await get_entitlements(user_id="u-ttl", jwt="jwt-token")

    assert route.call_count == 2


async def test_timeout_without_prior_cache_degrades_to_free() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(ENDPOINT).mock(side_effect=httpx.ConnectTimeout("auth down"))

        result = await get_entitlements(user_id="u-timeout", jwt="jwt-token")

    assert result.tier == "free"
    assert result.expires_at is None
    assert result.features == ()


async def test_timeout_returns_stale_cache_when_available() -> None:
    # Premier appel : MSPR-AUTH repond, on remplit le cache et le stale.
    with respx.mock(assert_all_called=True) as router:
        router.get(ENDPOINT).respond(
            200,
            json={"tier": "premium_plus", "expires_at": None, "features": ["a"]},
        )
        await get_entitlements(user_id="u-stale", jwt="jwt-token")

    # Le TTL expire (purge du cache vif), mais _stale conserve la valeur.
    entitlements_client._cache.clear()

    # Second appel : MSPR-AUTH timeout, on doit retomber sur la valeur stale.
    with respx.mock(assert_all_called=True) as router:
        router.get(ENDPOINT).mock(side_effect=httpx.ReadTimeout("auth slow"))

        result = await get_entitlements(user_id="u-stale", jwt="jwt-token")

    assert result.tier == "premium_plus"
    assert result.features == ("a",)


async def test_unknown_tier_value_degrades_to_free() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(ENDPOINT).respond(
            200,
            json={"tier": "diamond", "expires_at": None, "features": []},
        )

        result = await get_entitlements(user_id="u-bad-tier", jwt="jwt-token")

    assert result.tier == "free"


async def test_invalid_json_payload_degrades_to_free() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(ENDPOINT).respond(200, text="<html>not json</html>")

        result = await get_entitlements(user_id="u-html", jwt="jwt-token")

    assert result.tier == "free"


async def test_missing_tier_field_degrades_to_free() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(ENDPOINT).respond(200, json={"expires_at": None, "features": []})

        result = await get_entitlements(user_id="u-no-tier", jwt="jwt-token")

    assert result.tier == "free"


async def test_401_from_auth_degrades_to_free_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(ENDPOINT).respond(401, json={"error": "unauthorized"})

        with caplog.at_level("WARNING", logger="app.services.entitlements_client"):
            result = await get_entitlements(user_id="alice", jwt="bad-jwt")

    assert result.tier == "free"
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        "401" in r.getMessage() for r in warnings
    ), f"aucun warning ne mentionne 401: {[r.getMessage() for r in warnings]}"


async def test_500_from_auth_degrades_without_caching() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(ENDPOINT).respond(500, json={"error": "boom"})

        result = await get_entitlements(user_id="u-500", jwt="jwt-token")

    assert result.tier == "free"
    # Le 5xx ne doit pas etre persiste : un appel ulterieur doit retenter MSPR-AUTH.
    assert "u-500" not in entitlements_client._cache


async def test_returns_tier_premium_with_features_and_expiry() -> None:
    expires_iso = "2026-12-31T23:59:59+00:00"
    with respx.mock(assert_all_called=True) as router:
        router.get(ENDPOINT).respond(
            200,
            json={
                "tier": "premium",
                "expires_at": expires_iso,
                "features": ["meal_plans_unlimited", "no_cache"],
            },
        )

        result = await get_entitlements(user_id="u-2", jwt="jwt-token")

    assert result.tier == "premium"
    assert result.expires_at == datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    assert result.features == ("meal_plans_unlimited", "no_cache")
