from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval_metrics import _persist, build_parser


def test_cli_parses_classifier_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["classifier", "--n-food101", "100", "--seed", "42", "--output-dir", "docs/"]
    )

    assert args.command == "classifier"
    assert args.n_food101 == 100
    assert args.seed == 42
    assert str(args.output_dir).rstrip("/").endswith("docs")


def test_cli_parses_llm_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["llm", "--n-generations", "100", "--seed", "7", "--output-dir", "docs/"]
    )

    assert args.command == "llm"
    assert args.n_generations == 100
    assert args.seed == 7


def test_cli_requires_subcommand() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_persist_merges_llm_section_to_accumulate_runs(tmp_path: Path) -> None:
    """Slice 5 (#75) : 3 runs comparatifs s'empilent sous llm.<run_key>."""
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"llm": {"gemma_n20": {"backend": "ollama"}}}),
        encoding="utf-8",
    )

    _persist({"mistral_n20": {"backend": "mistral"}}, tmp_path, "llm")

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(data["llm"].keys()) == {"gemma_n20", "mistral_n20"}


def test_persist_overwrites_classifier_section(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"classifier": {"food101": {"top1": 0.7}}}),
        encoding="utf-8",
    )

    _persist({"food101": {"top1": 0.9}}, tmp_path, "classifier")

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert data["classifier"] == {"food101": {"top1": 0.9}}


def _run_with_latencies(backend: str, latencies: list[float]) -> dict:
    return {
        "backend": backend,
        "naive": {
            "n_generations": len(latencies),
            "json_validity_rate": 1.0,
            "fallback_rate": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_max_ms": 0.0,
            "constraint_satisfaction": 0.0,
            "constraint_n_plans": 0,
            "by_constraint": {"allergies": 0.0, "budget": 0.0, "diet": 0.0},
            "latencies_ms_raw": latencies,
        },
        "pipeline": {
            "n_generations": len(latencies),
            "constraint_satisfaction": 0.0,
            "partial_compliance": 0.0,
            "static_fallback": 0.0,
            "abandoned_503": 0.0,
            "by_constraint": {"allergies": 0.0, "budget": 0.0, "diet": 0.0},
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "retry_count_distribution": {"0": 1},
        },
    }


def test_persist_generates_latency_comparison_png_for_multi_backend_runs(
    tmp_path: Path,
) -> None:
    """Slice 5 (#75) : apres merge, le PNG comparatif Gemma vs Mistral existe."""
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "llm": {
                    "gemma_n20": _run_with_latencies(
                        "ollama", [100_000.0, 120_000.0, 90_000.0]
                    ),
                }
            }
        ),
        encoding="utf-8",
    )

    _persist(
        {"mistral_n20": _run_with_latencies("mistral", [3000.0, 4000.0, 2500.0])},
        tmp_path,
        "llm",
    )

    png = tmp_path / "llm_latency_distribution.png"
    assert png.exists()
    assert png.stat().st_size > 1000
