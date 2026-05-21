from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
            "naive": {
                "n_generations": 30,
                "json_validity_rate": 0.94,
                "fallback_rate": 0.04,
                "latency_p50_ms": 800.0,
                "latency_p95_ms": 1500.0,
                "latency_max_ms": 3000.0,
                "constraint_satisfaction": 0.87,
                "by_constraint": {
                    "allergies": 0.5,
                    "budget": 0.6,
                    "diet": 0.4,
                },
                "hitl": {
                    "n_ratings": 20,
                    "mean_nutrition": 4.1,
                    "mean_originalite": 3.2,
                    "mean_coherence": 4.0,
                },
                "latency_distribution_png": "docs/llm_latency.png",
            },
            "pipeline": {
                "n_generations": 30,
                "constraint_satisfaction": 0.85,
                "partial_compliance": 0.10,
                "static_fallback": 0.03,
                "abandoned_503": 0.02,
                "by_constraint": {
                    "allergies": 1.0,
                    "budget": 0.85,
                    "diet": 1.0,
                },
                "latency_p50_ms": 1200.0,
                "latency_p95_ms": 2500.0,
                "retry_count_distribution": {"0": 25, "1": 3, "2": 2},
            },
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
    assert "1500" in content or "1.5" in content  # p95 naive
    # Slice 7 : niveau pipeline + comparaison naive vs pipeline
    assert "naive" in content.lower()
    assert "pipeline" in content.lower()
    assert "Comparaison" in content
    # Embed des PNGs
    assert "docs/confusion_matrix_food101.png" in content
    assert "docs/llm_latency.png" in content


def _run_payload(
    *,
    allergies: float,
    diet: float,
    p50: float,
    p95: float,
    json_valid: float,
    compliance_full: float,
    retries: dict[str, int],
) -> dict[str, Any]:
    return {
        "naive": {
            "n_generations": 20,
            "json_validity_rate": json_valid,
            "fallback_rate": 0.0,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "latency_max_ms": p95,
            "constraint_satisfaction": 0.0,
            "by_constraint": {"allergies": allergies, "budget": 0.5, "diet": diet},
        },
        "pipeline": {
            "n_generations": 20,
            "constraint_satisfaction": compliance_full,
            "partial_compliance": 0.0,
            "static_fallback": 0.0,
            "abandoned_503": 0.0,
            "by_constraint": {"allergies": allergies, "budget": 0.5, "diet": diet},
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "retry_count_distribution": retries,
        },
    }


def test_render_metrics_md_shows_gemma_vs_mistral_comparison(tmp_path: Path) -> None:
    """Slice 5 (#75) : tableau side-by-side + bloc bonus N=100."""
    payload: dict[str, Any] = {
        "llm": {
            "gemma_n20": {
                "backend": "ollama",
                **_run_payload(
                    allergies=0.8,
                    diet=0.7,
                    p50=120000.0,
                    p95=250000.0,
                    json_valid=0.85,
                    compliance_full=0.25,
                    retries={"0": 5, "3": 15},
                ),
            },
            "mistral_n20": {
                "backend": "mistral",
                **_run_payload(
                    allergies=1.0,
                    diet=0.95,
                    p50=3500.0,
                    p95=6800.0,
                    json_valid=1.0,
                    compliance_full=0.55,
                    retries={"0": 18, "1": 2},
                ),
            },
            "mistral_n100": {
                "backend": "mistral",
                **_run_payload(
                    allergies=0.98,
                    diet=0.94,
                    p50=3600.0,
                    p95=7200.0,
                    json_valid=0.99,
                    compliance_full=0.52,
                    retries={"0": 90, "1": 10},
                ),
            },
        }
    }
    out = tmp_path / "metrics.md"

    from scripts.eval.report import render_metrics_md

    render_metrics_md(payload, out)

    content = out.read_text(encoding="utf-8")
    # Section comparative principale
    assert "Comparaison Gemma3:4b local vs Mistral Small managed" in content
    # Tableau avec entete cible
    assert "| Metrique | Gemma3:4b local | Mistral Small managed |" in content
    # Les 5 metriques + retry count
    for label in (
        "compliance_status=full",
        "allergy compliance",
        "diet compliance",
        "JSON validity",
        "latence p50",
        "latence p95",
        "retry count moyen",
    ):
        assert label in content, f"Metrique attendue absente du MD : {label}"
    # Bloc bonus N=100
    assert "Bonus Mistral N=100" in content
    # Quelques valeurs concretes pour eviter un format vide
    assert "0.25" in content  # compliance full gemma
    assert "0.55" in content  # compliance full mistral


def test_render_metrics_md_hides_comparison_when_all_terrain_unknown(
    tmp_path: Path,
) -> None:
    # n_samples > 0 mais unknown_rate == 1.0 -> truths vide cote runner,
    # top1 == 0.0 ; afficher la comparaison serait trompeur.
    payload = {
        "classifier": {
            "food101": {
                "top1_accuracy": 0.71,
                "top5_accuracy": 0.92,
                "n_samples": 1000,
            },
            "terrain": {
                "n_samples": 5,
                "top1_accuracy": 0.0,
                "top5_accuracy": 0.0,
                "unknown_rate": 1.0,
            },
        }
    }
    out = tmp_path / "metrics.md"

    render_metrics_md(payload, out)

    content = out.read_text(encoding="utf-8")
    assert "Comparaison Food-101 vs terrain" not in content


def test_render_metrics_md_includes_terrain_unknown_rate(tmp_path: Path) -> None:
    payload = {
        "classifier": {
            "food101": {
                "top1_accuracy": 0.71,
                "top5_accuracy": 0.92,
                "n_samples": 1000,
            },
            "terrain": {
                "n_samples": 50,
                "n_classified": 45,
                "top1_accuracy": 0.45,
                "top5_accuracy": 0.7,
                "unknown_rate": 0.1,
            },
        }
    }
    out = tmp_path / "metrics.md"

    render_metrics_md(payload, out)

    content = out.read_text(encoding="utf-8")
    assert "unknown_rate" in content.lower() or "hors-distribution" in content.lower()
    assert "0.10" in content or "10.0" in content or "10 %" in content


def test_render_metrics_md_distinguishes_n_samples_from_n_classified(
    tmp_path: Path,
) -> None:
    # Garde-fou : le report doit afficher les DEUX comptes pour eviter
    # qu'un lecteur lise "top-1 1.0 sur 50 echantillons" alors qu'un
    # sous-ensemble seulement a ete classifie (unknown + images manquantes).
    payload = {
        "classifier": {
            "food101": {
                "top1_accuracy": 0.71,
                "top5_accuracy": 0.92,
                "n_samples": 1000,
            },
            "terrain": {
                "n_samples": 50,
                "n_classified": 30,
                "top1_accuracy": 0.45,
                "top5_accuracy": 0.7,
                "unknown_rate": 0.4,
            },
        }
    }
    out = tmp_path / "metrics.md"

    render_metrics_md(payload, out)

    content = out.read_text(encoding="utf-8")
    assert "50" in content  # n_samples
    assert "30" in content  # n_classified
    assert "classifies" in content.lower()
