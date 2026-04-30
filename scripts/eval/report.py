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
        out.append(f"- N samples (CSV) : {terrain.get('n_samples', 0)}")
        out.append(
            f"- N classifies (hors `unknown` et images manquantes) : "
            f"{terrain.get('n_classified', 0)}"
        )
        out.append(
            f"- Top-1 accuracy (sur les classifies) : "
            f"{terrain.get('top1_accuracy', 0.0):.4f}"
        )
        out.append(
            f"- Top-5 accuracy (sur les classifies) : "
            f"{terrain.get('top5_accuracy', 0.0):.4f}"
        )
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

    naive = data.get("naive") or {}
    pipeline = data.get("pipeline") or {}

    # Retro-compatibilite : si la cle "naive" n'existe pas, on suppose le format
    # pre-slice 7 (champs a plat directement sous "llm").
    if not naive and not pipeline:
        naive = data

    if naive:
        out.append("### Niveau naive (LLM nu, sans DeCRIM-light)")
        out.append("")
        out.append(
            f"- Taux de validite JSON (1er essai) : "
            f"{naive.get('json_validity_rate', 0.0):.4f}"
        )
        out.append(
            f"- Taux d'invocation Fallback : {naive.get('fallback_rate', 0.0):.4f}"
        )
        out.append(
            f"- Latence : p50 {naive.get('latency_p50_ms', 0.0):.0f} ms, "
            f"p95 {naive.get('latency_p95_ms', 0.0):.0f} ms, "
            f"max {naive.get('latency_max_ms', 0.0):.0f} ms"
        )
        out.append(
            f"- Respect simultanee allergies + budget + regime : "
            f"{naive.get('constraint_satisfaction', 0.0):.4f}"
        )
        by_c = naive.get("by_constraint") or {}
        if by_c:
            out.append(
                f"- Par contrainte : allergies {by_c.get('allergies', 0.0):.4f}, "
                f"budget {by_c.get('budget', 0.0):.4f}, "
                f"regime {by_c.get('diet', 0.0):.4f}"
            )
        out.append("")

        hitl = naive.get("hitl") or {}
        if hitl:
            out.append("#### Evaluation qualitative humaine (HITL, 1-5)")
            out.append("")
            out.append(f"- N ratings : {hitl.get('n_ratings', 0)}")
            out.append(
                f"- Pertinence nutrition : {hitl.get('mean_nutrition', 0.0):.2f}"
            )
            out.append(f"- Originalite : {hitl.get('mean_originalite', 0.0):.2f}")
            out.append(f"- Coherence : {hitl.get('mean_coherence', 0.0):.2f}")
            out.append("")

        latency_png = naive.get("latency_distribution_png")
        if latency_png:
            out.append(f"![Distribution latence LLM]({latency_png})")
            out.append("")

    if pipeline:
        out.append("### Niveau pipeline (DeCRIM-light + cache bypass)")
        out.append("")
        out.append(
            f"- N generations : {pipeline.get('n_generations', 0)}"
        )
        out.append(
            f"- compliance_status full : "
            f"{pipeline.get('constraint_satisfaction', 0.0):.4f}"
        )
        out.append(
            f"- compliance_status partial_budget : "
            f"{pipeline.get('partial_compliance', 0.0):.4f}"
        )
        out.append(
            f"- compliance_status static_fallback : "
            f"{pipeline.get('static_fallback', 0.0):.4f}"
        )
        out.append(
            f"- abandoned_503 (contraintes infaisables) : "
            f"{pipeline.get('abandoned_503', 0.0):.4f}"
        )
        by_c = pipeline.get("by_constraint") or {}
        if by_c:
            out.append(
                f"- Par contrainte : allergies {by_c.get('allergies', 0.0):.4f}, "
                f"budget {by_c.get('budget', 0.0):.4f}, "
                f"regime {by_c.get('diet', 0.0):.4f}"
            )
        out.append(
            f"- Latence : p50 {pipeline.get('latency_p50_ms', 0.0):.0f} ms, "
            f"p95 {pipeline.get('latency_p95_ms', 0.0):.0f} ms"
        )
        rcd = pipeline.get("retry_count_distribution") or {}
        if rcd:
            buckets = ", ".join(f"{k} retry: {v}" for k, v in sorted(rcd.items()))
            out.append(f"- Distribution retries : {buckets}")
        out.append("")

    if naive and pipeline:
        out.extend(_render_naive_vs_pipeline_comparison(naive, pipeline))

    return out


def _render_naive_vs_pipeline_comparison(
    naive: dict[str, Any], pipeline: dict[str, Any]
) -> list[str]:
    """Section comparative explicite entre naive et pipeline (slice 7).

    Quantifie l'apport du retry DeCRIM-light : delta de constraint_satisfaction,
    repartition des relachements, surcout en latence.
    """
    out = ["### Comparaison naive vs pipeline", ""]

    delta_full = pipeline.get("constraint_satisfaction", 0.0) - naive.get(
        "constraint_satisfaction", 0.0
    )
    out.append(
        f"- Gain compliance_status full vs naive : {delta_full:+.4f}. "
        "Un gain positif quantifie l'apport du retry cible (allergie/regime "
        "partiels, budget complet) et du fallback hierarchique."
    )

    naive_by = naive.get("by_constraint") or {}
    pipe_by = pipeline.get("by_constraint") or {}
    if naive_by and pipe_by:
        for key in ("allergies", "budget", "diet"):
            out.append(
                f"- Delta {key} : "
                f"naive {naive_by.get(key, 0.0):.4f} -> "
                f"pipeline {pipe_by.get(key, 0.0):.4f} "
                f"({pipe_by.get(key, 0.0) - naive_by.get(key, 0.0):+.4f})"
            )

    naive_p95 = naive.get("latency_p95_ms", 0.0)
    pipe_p95 = pipeline.get("latency_p95_ms", 0.0)
    if naive_p95 and pipe_p95:
        ratio = pipe_p95 / naive_p95 if naive_p95 else 0.0
        out.append(
            f"- Surcout latence p95 : naive {naive_p95:.0f} ms -> "
            f"pipeline {pipe_p95:.0f} ms (ratio x{ratio:.2f}). "
            "Le pipeline paie le prix des retries internes pour reduire les "
            "violations critiques (allergies/regime)."
        )
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
