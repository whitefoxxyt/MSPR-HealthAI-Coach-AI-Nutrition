from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dump_metrics_json(payload: dict[str, Any], out_path: Path) -> None:
    """Ecrit le payload brut en JSON indente, pour tracage versionable."""
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def render_metrics_md(payload: dict[str, Any], out_path: Path) -> None:
    """Rend le payload en Markdown avec sections classifier + LLM + discussion."""
    parts: list[str] = []
    parts.append("# Metriques IA : MSPR-AI-Nutrition")
    parts.append("")
    parts.append("Genere par `scripts/eval_metrics.py`. Ne pas editer a la main.")
    parts.append("")

    parts.extend(_render_classifier_section(payload.get("classifier") or {}))
    parts.extend(_render_llm_section(payload.get("llm") or {}))
    parts.extend(_render_discussion_section(payload))

    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _render_classifier_section(data: dict[str, Any]) -> list[str]:
    if not data:
        return []
    out: list[str] = ["## Classifier (HuggingFace nateraw/food)", ""]

    food101 = data.get("food101")
    if food101:
        out.append("### Food-101 (test split, sous-echantillon)")
        out.append("")
        out.append(f"- N samples : {food101.get('n_samples', 0)}")
        out.append(f"- Top-1 accuracy : {food101.get('top1_accuracy', 0.0):.4f}")
        out.append(f"- Top-5 accuracy : {food101.get('top5_accuracy', 0.0):.4f}")
        cm_png = food101.get("confusion_matrix_png")
        if cm_png:
            out.append("")
            out.append(f"![Matrice de confusion Food-101]({cm_png})")
        per_class = food101.get("per_class") or {}
        if per_class:
            out.append("")
            out.append("Top classes (precision / rappel / F1 / support) :")
            out.append("")
            out.append("| Classe | Precision | Rappel | F1 | Support |")
            out.append("|---|---|---|---|---|")
            for cls, m in sorted(per_class.items()):
                out.append(
                    f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | "
                    f"{m['f1']:.3f} | {m['support']} |"
                )
        out.append("")

    terrain = data.get("terrain")
    if terrain:
        out.append("### Terrain (photos telephone, eval HITL)")
        out.append("")
        out.append(f"- N samples : {terrain.get('n_samples', 0)}")
        out.append(f"- Top-1 accuracy : {terrain.get('top1_accuracy', 0.0):.4f}")
        out.append(f"- Top-5 accuracy : {terrain.get('top5_accuracy', 0.0):.4f}")
        out.append(
            f"- Unknown_rate (plats hors-distribution Food-101) : "
            f"{terrain.get('unknown_rate', 0.0):.4f}"
        )
        out.append("")

    if (
        food101
        and terrain
        and terrain.get("n_samples", 0) > 0
        and terrain.get("unknown_rate", 0.0) < 1.0
    ):
        out.append("### Comparaison Food-101 vs terrain")
        out.append("")
        delta_top1 = food101.get("top1_accuracy", 0.0) - terrain.get(
            "top1_accuracy", 0.0
        )
        out.append(
            f"Ecart top-1 (academique - terrain) : {delta_top1:+.4f}. "
            "Un ecart positif confirme le biais de domaine du dataset."
        )
        out.append("")

    return out


def _render_llm_section(data: dict[str, Any]) -> list[str]:
    if not data:
        return []
    out: list[str] = ["## LLM (Ollama gemma3:4b)", ""]

    out.append(
        f"- Taux de validite JSON (1er essai) : {data.get('json_validity_rate', 0.0):.4f}"
    )
    out.append(f"- Taux d'invocation Fallback : {data.get('fallback_rate', 0.0):.4f}")
    latency = data.get("latency") or {}
    out.append(
        f"- Latence : p50 {latency.get('p50_ms', 0.0):.0f} ms, "
        f"p95 {latency.get('p95_ms', 0.0):.0f} ms, "
        f"max {latency.get('max_ms', 0.0):.0f} ms"
    )
    out.append(
        f"- Respect simultanee allergies + budget + regime : "
        f"{data.get('constraint_satisfaction_rate', 0.0):.4f}"
    )
    out.append("")

    hitl = data.get("hitl") or {}
    if hitl:
        out.append("### Evaluation qualitative humaine (HITL, 1-5)")
        out.append("")
        out.append(f"- N ratings : {hitl.get('n_ratings', 0)}")
        out.append(f"- Pertinence nutrition : {hitl.get('mean_nutrition', 0.0):.2f}")
        out.append(f"- Originalite : {hitl.get('mean_originalite', 0.0):.2f}")
        out.append(f"- Coherence : {hitl.get('mean_coherence', 0.0):.2f}")
        out.append("")

    latency_png = data.get("latency_distribution_png")
    if latency_png:
        out.append(f"![Distribution latence LLM]({latency_png})")
        out.append("")

    return out


def _render_discussion_section(payload: dict[str, Any]) -> list[str]:
    out = ["## Discussion", ""]
    out.append(
        "- **Limitations dataset Food-101** : 101 classes academiques, photos cadrees, "
        "fond neutre. Tres different des photos prises au telephone (eclairage, "
        "angle, plat composite)."
    )
    out.append(
        "- **Biais du modele** : fine-tune sur Food-101 -> classes hors-distribution "
        "(ex : plats francais traditionnels, repas ethniques specifiques) sont "
        "systematiquement misclassifies vers la classe la plus proche visuellement."
    )
    out.append(
        "- **Cas d'echec frequents** : plats mixtes (assiette avec plusieurs aliments), "
        "decoupes inhabituelles, photos en faible luminosite, gros plans non cadres."
    )
    out.append(
        "- **LLM** : la latence p95 sur CPU reste contraignante ; le fallback statique "
        "garantit une UX correcte hors disponibilite Ollama. Les violations de "
        "contraintes proviennent souvent du regime alimentaire (vegan/sans gluten "
        "moins bien respectes que les allergies)."
    )
    out.append("")
    return out
