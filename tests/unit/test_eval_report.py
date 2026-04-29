from __future__ import annotations

import json
from pathlib import Path

from scripts.eval.report import dump_metrics_json, render_metrics_md


def test_dump_metrics_json_writes_pretty(tmp_path: Path) -> None:
    out = tmp_path / "metrics.json"
    payload = {"classifier": {"top1_accuracy": 0.71}, "llm": {"json_validity": 0.94}}

    dump_metrics_json(payload, out)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == payload
    # Pretty : indentation visible, donc plus d'une ligne.
    assert "\n" in out.read_text(encoding="utf-8")


def test_render_metrics_md_includes_required_sections(tmp_path: Path) -> None:
    payload = {
        "classifier": {
            "food101": {
                "top1_accuracy": 0.71,
                "top5_accuracy": 0.92,
                "n_samples": 1000,
                "confusion_matrix_png": "docs/confusion_matrix_food101.png",
                "per_class": {
                    "pizza": {
                        "precision": 0.8,
                        "recall": 0.7,
                        "f1": 0.74,
                        "support": 10,
                    },
                },
            },
            "terrain": {
                "top1_accuracy": 0.45,
                "top5_accuracy": 0.7,
                "n_samples": 50,
            },
        },
        "llm": {
            "json_validity_rate": 0.94,
            "fallback_rate": 0.04,
            "latency": {"p50_ms": 800.0, "p95_ms": 1500.0, "max_ms": 3000.0},
            "constraint_satisfaction_rate": 0.87,
            "hitl": {
                "n_ratings": 20,
                "mean_nutrition": 4.1,
                "mean_originalite": 3.2,
                "mean_coherence": 4.0,
            },
            "latency_distribution_png": "docs/llm_latency.png",
        },
    }
    out = tmp_path / "metrics.md"

    render_metrics_md(payload, out)

    content = out.read_text(encoding="utf-8")
    # Sections obligatoires demandees par les acceptance criteria
    assert "## Classifier" in content
    assert "Food-101" in content
    assert "Terrain" in content
    assert "## LLM" in content
    assert "Latence" in content
    assert "JSON" in content
    assert "Fallback" in content
    assert "## Discussion" in content
    # Chiffres bruts
    assert "0.71" in content or "71" in content  # accuracy
    assert "1500" in content or "1.5" in content  # p95
    # Embed des PNGs
    assert "docs/confusion_matrix_food101.png" in content
    assert "docs/llm_latency.png" in content
