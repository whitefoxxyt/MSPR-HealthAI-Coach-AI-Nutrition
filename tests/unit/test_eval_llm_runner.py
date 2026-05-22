from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.models.schemas import PlanInputs
from app.services.llm_provider import LLMProvider
from scripts.eval.llm_runner import (
    _call_for_eval,
    _instrument_chain_calls,
    _resolve_backend,
    _run_key,
    run_llm_eval,
)


class _FakeProvider(LLMProvider):
    """Provider scripte : retourne les reponses en sequence, compte les appels."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def generate(self, prompt: str, schema: dict[str, Any] | None) -> str:
        self.calls.append((prompt, schema))
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _valid_plan_json() -> str:
    return json.dumps(
        {
            "fallback": False,
            "days": [
                {
                    "day": 1,
                    "meals": [
                        {
                            "name": "salade",
                            "macros": {
                                "calories": 400,
                                "protein_g": 20.0,
                                "carbs_g": 30.0,
                                "fat_g": 10.0,
                            },
                            "ingredients": ["laitue", "poulet"],
                            "est_budget_eur": 5.0,
                            "prep_time_min": 10,
                        }
                    ],
                }
            ],
        }
    )


def _sample_inputs() -> PlanInputs:
    return PlanInputs(
        user_id="1",
        objective="weight_loss",
        duration_days=1,
        diet_type="omnivore",
        allergies=[],
    )


def test_resolve_backend_defaults_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    assert _resolve_backend() == "ollama"


def test_resolve_backend_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "Mistral")
    assert _resolve_backend() == "mistral"


def test_run_key_ollama_aliased_to_gemma() -> None:
    assert _run_key("ollama", 20) == "gemma_n20"


def test_run_key_mistral_keeps_name() -> None:
    assert _run_key("mistral", 100) == "mistral_n100"


async def test_call_for_eval_delegates_to_provider() -> None:
    provider = _FakeProvider([_valid_plan_json()])

    outcome, plan = await _call_for_eval(provider, _sample_inputs())

    assert len(provider.calls) == 1
    assert outcome.json_valid_first_try is True
    assert outcome.used_fallback is False
    assert outcome.latency_ms >= 0
    assert plan is not None
    assert plan.days[0].meals[0].name == "salade"


class _StubChain:
    """Chain stub : compatible API FallbackChain.generate."""

    def __init__(self, response: str = "{}") -> None:
        self._response = response

    async def generate(
        self, prompt: str, schema: dict[str, Any] | None, primary_backend: str
    ) -> tuple[str, str]:
        return self._response, primary_backend


async def test_instrument_chain_calls_counts_each_chain_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import decrim_retry_orchestrator as orch

    monkeypatch.setattr(orch, "build_default_chain", lambda: _StubChain())

    async with _instrument_chain_calls() as counter:
        chain = orch.build_default_chain()
        await chain.generate("p1", None, "ollama")
        await chain.generate("p2", None, "ollama")

    assert counter["calls"] == 2


async def test_run_llm_eval_returns_payload_keyed_by_run_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.eval import llm_runner

    provider = _FakeProvider([_valid_plan_json()] * 10)
    monkeypatch.setattr(llm_runner, "get_provider", lambda name: provider)
    monkeypatch.setenv("LLM_BACKEND", "ollama")

    pipeline_calls: list[tuple[int, int, str]] = []

    async def fake_pipeline_eval(
        n: int, seed: int, primary_backend: str
    ) -> dict[str, Any]:
        pipeline_calls.append((n, seed, primary_backend))
        return {"n_generations": n, "constraint_satisfaction": 0.0}

    monkeypatch.setattr(llm_runner, "_run_pipeline_eval", fake_pipeline_eval)

    payload = await run_llm_eval(
        n_generations=1,
        n_constraint_plans=2,
        hitl_csv=tmp_path / "noop.csv",
        seed=42,
        output_dir=tmp_path,
    )

    assert "gemma_n2" in payload
    run = payload["gemma_n2"]
    assert run["backend"] == "ollama"
    assert "naive" in run
    assert "pipeline" in run
    assert pipeline_calls == [(2, 42, "ollama")]
