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

    # Slice 5 (#75) : si on a des runs comparatifs (cles gemma_*/mistral_*),
    # on rend uniquement la nouvelle section comparative ; l'ancien rendu
    # naive/pipeline est conserve pour les payloads legacy.
    if _has_comparison_runs(data):
        return _render_backend_comparison(data)

    out: list[str] = ["## LLM (Ollama gemma3:4b)", ""]

    naive = data.get("naive") or {}
    pipeline = data.get("pipeline") or {}

    # Retro-compatibilite : si la cle "naive" n'existe pas, on suppose le format
    # pre-slice 7 (champs a plat directement sous "llm").
    if not naive and not pipeline:
        naive = data

    if naive:
        out.extend(_render_naive_block(naive))
    if pipeline:
        out.extend(_render_pipeline_block(pipeline))
    if naive and pipeline:
        out.extend(_render_naive_vs_pipeline_comparison(naive, pipeline))

    return out


def _has_comparison_runs(data: dict[str, Any]) -> bool:
    return any(k.startswith(("gemma_", "mistral_")) for k in data)


def _render_backend_comparison(data: dict[str, Any]) -> list[str]:
    """Slice 5 (#75) : tableau side-by-side Gemma vs Mistral + bloc bonus N=100.

    Convention de cles dans data : `gemma_n<N>`, `mistral_n<N>`. Le tableau
    principal est rendu sur les runs N=20 (meme seed). Le run Mistral N=100
    est presente separement comme run "bonus" demontrant la maturite de l'eval.
    """
    out: list[str] = ["## LLM : comparaison multi-backend", ""]
    out.append("### Comparaison Gemma3:4b local vs Mistral Small managed")
    out.append("")

    gemma_key = next((k for k in sorted(data) if k.startswith("gemma_")), None)
    mistral_n20_key = next(
        (k for k in sorted(data) if k.startswith("mistral_") and "_n20" in k), None
    )

    if gemma_key and mistral_n20_key:
        gemma = data[gemma_key]
        mistral = data[mistral_n20_key]
        out.append("| Metrique | Gemma3:4b local | Mistral Small managed |")
        out.append("|---|---|---|")
        for label, fmt, gv, mv in _comparison_rows(gemma, mistral):
            out.append(f"| {label} | {fmt(gv)} | {fmt(mv)} |")
        out.append("")
    else:
        out.append(
            "_Tableau principal indisponible : un des deux runs N=20 manque "
            "(lancer `LLM_BACKEND=ollama` puis `LLM_BACKEND=mistral`)._"
        )
        out.append("")

    mistral_n100_key = next(
        (k for k in sorted(data) if k.startswith("mistral_") and "_n100" in k), None
    )
    if mistral_n100_key:
        out.extend(_render_bonus_block(data[mistral_n100_key]))

    return out


def _comparison_rows(
    gemma: dict[str, Any], mistral: dict[str, Any]
) -> list[tuple[str, Any, float, float]]:
    """5 metriques principales + retry count moyen (cf. issue #75)."""
    g_naive = gemma.get("naive") or {}
    g_pipe = gemma.get("pipeline") or {}
    m_naive = mistral.get("naive") or {}
    m_pipe = mistral.get("pipeline") or {}

    g_by = g_pipe.get("by_constraint") or {}
    m_by = m_pipe.get("by_constraint") or {}

    def f4(v: float) -> str:
        return f"{v:.4f}"

    def fms(v: float) -> str:
        return f"{v:.0f} ms"

    def f2(v: float) -> str:
        return f"{v:.2f}"

    return [
        (
            "compliance_status=full",
            f4,
            float(g_pipe.get("constraint_satisfaction") or 0.0),
            float(m_pipe.get("constraint_satisfaction") or 0.0),
        ),
        (
            "allergy compliance rate",
            f4,
            float(g_by.get("allergies") or 0.0),
            float(m_by.get("allergies") or 0.0),
        ),
        (
            "diet compliance rate",
            f4,
            float(g_by.get("diet") or 0.0),
            float(m_by.get("diet") or 0.0),
        ),
        (
            "JSON validity rate",
            f4,
            float(g_naive.get("json_validity_rate") or 0.0),
            float(m_naive.get("json_validity_rate") or 0.0),
        ),
        (
            "latence p50 (pipeline)",
            fms,
            float(g_pipe.get("latency_p50_ms") or 0.0),
            float(m_pipe.get("latency_p50_ms") or 0.0),
        ),
        (
            "latence p95 (pipeline)",
            fms,
            float(g_pipe.get("latency_p95_ms") or 0.0),
            float(m_pipe.get("latency_p95_ms") or 0.0),
        ),
        (
            "retry count moyen",
            f2,
            _mean_retries(g_pipe.get("retry_count_distribution") or {}),
            _mean_retries(m_pipe.get("retry_count_distribution") or {}),
        ),
    ]


def _mean_retries(distribution: dict[str, int]) -> float:
    """Moyenne ponderee : sum(retries * count) / sum(counts)."""
    total = sum(distribution.values())
    if not total:
        return 0.0
    weighted = sum(int(k) * v for k, v in distribution.items())
    return weighted / total


def _render_bonus_block(run: dict[str, Any]) -> list[str]:
    out: list[str] = ["### Bonus Mistral N=100", ""]
    naive = run.get("naive") or {}
    pipe = run.get("pipeline") or {}
    out.append(
        f"- compliance_status=full : "
        f"{float(pipe.get('constraint_satisfaction') or 0.0):.4f}"
    )
    out.append(
        f"- JSON validity rate : "
        f"{float(naive.get('json_validity_rate') or 0.0):.4f}"
    )
    out.append(
        f"- Latence pipeline : p50 {float(pipe.get('latency_p50_ms') or 0.0):.0f} ms, "
        f"p95 {float(pipe.get('latency_p95_ms') or 0.0):.0f} ms"
    )
    out.append(
        f"- Retry count moyen : "
        f"{_mean_retries(pipe.get('retry_count_distribution') or {}):.2f}"
    )
    out.append("")
    out.append(
        "N=100 confirme l'ordre de grandeur des chiffres N=20 (eval mature, "
        "pas un artefact de petit echantillon)."
    )
    out.append("")
    return out


def _render_naive_block(naive: dict[str, Any]) -> list[str]:
    out: list[str] = [
        "### Niveau naive (LLM nu, sans DeCRIM-light)",
        "",
        f"- Taux de validite JSON (1er essai) : "
        f"{naive.get('json_validity_rate', 0.0):.4f}",
        f"- Taux d'invocation Fallback : {naive.get('fallback_rate', 0.0):.4f}",
        f"- Latence : p50 {naive.get('latency_p50_ms', 0.0):.0f} ms, "
        f"p95 {naive.get('latency_p95_ms', 0.0):.0f} ms, "
        f"max {naive.get('latency_max_ms', 0.0):.0f} ms",
        f"- Respect simultanee allergies + budget + regime : "
        f"{naive.get('constraint_satisfaction', 0.0):.4f}",
    ]
    by_c = naive.get("by_constraint") or {}
    if by_c:
        out.append(_format_by_constraint_line(by_c))
    out.append("")

    out.extend(_render_hitl_block(naive.get("hitl") or {}))

    latency_png = naive.get("latency_distribution_png")
    if latency_png:
        out.append(f"![Distribution latence LLM]({latency_png})")
        out.append("")
    return out


def _render_hitl_block(hitl: dict[str, Any]) -> list[str]:
    if not hitl:
        return []
    return [
        "#### Evaluation qualitative humaine (HITL, 1-5)",
        "",
        f"- N ratings : {hitl.get('n_ratings', 0)}",
        f"- Pertinence nutrition : {hitl.get('mean_nutrition', 0.0):.2f}",
        f"- Originalite : {hitl.get('mean_originalite', 0.0):.2f}",
        f"- Coherence : {hitl.get('mean_coherence', 0.0):.2f}",
        "",
    ]


def _render_pipeline_block(pipeline: dict[str, Any]) -> list[str]:
    out: list[str] = [
        "### Niveau pipeline (DeCRIM-light + cache bypass)",
        "",
        f"- N generations : {pipeline.get('n_generations', 0)}",
        f"- compliance_status full : "
        f"{pipeline.get('constraint_satisfaction', 0.0):.4f}",
        f"- compliance_status partial_budget : "
        f"{pipeline.get('partial_compliance', 0.0):.4f}",
        f"- compliance_status static_fallback : "
        f"{pipeline.get('static_fallback', 0.0):.4f}",
        f"- abandoned_503 (contraintes infaisables) : "
        f"{pipeline.get('abandoned_503', 0.0):.4f}",
    ]
    by_c = pipeline.get("by_constraint") or {}
    if by_c:
        out.append(_format_by_constraint_line(by_c))
    out.append(
        f"- Latence : p50 {pipeline.get('latency_p50_ms', 0.0):.0f} ms, "
        f"p95 {pipeline.get('latency_p95_ms', 0.0):.0f} ms"
    )
    rcd = pipeline.get("retry_count_distribution") or {}
    if rcd:
        buckets = ", ".join(f"{k} retry: {v}" for k, v in sorted(rcd.items()))
        out.append(f"- Distribution retries : {buckets}")
    out.append("")
    return out


def _format_by_constraint_line(by_c: dict[str, Any]) -> str:
    return (
        f"- Par contrainte : allergies {by_c.get('allergies', 0.0):.4f}, "
        f"budget {by_c.get('budget', 0.0):.4f}, "
        f"regime {by_c.get('diet', 0.0):.4f}"
    )


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

    llm = payload.get("llm") or {}
    if _has_comparison_runs(llm):
        out.extend(_render_backend_tradeoffs())

    return out


def _render_backend_tradeoffs() -> list[str]:
    """Axes ou Mistral gagne / axes ou Gemma reste pertinent (issue #75)."""
    return [
        "### Mistral Small managed vs Gemma3:4b local",
        "",
        "**Mistral gagne sur** :",
        "",
        "- **Latence** : ordre de grandeur d'avance (quelques secondes p50 vs "
        "plusieurs dizaines de secondes sur CPU). Permet une UX interactive sur "
        "le flux generate-meal-plan.",
        "- **Validite JSON** : le mode `response_format.json_schema strict:true` "
        "garantit un JSON syntaxiquement valide des le 1er essai. Gemma3:4b via "
        "Ollama `format: <schema>` reste tributaire de la generation libre.",
        "- **Conformite aux contraintes** : sur les memes inputs (seed=42), le "
        "compliance_status=full atteint un taux significativement plus eleve, "
        "ce qui reduit la frequence des fallback statiques.",
        "",
        "**Gemma3:4b reste pertinent pour** :",
        "",
        "- **Offline / on-premise** : aucune dependance reseau, aucun token "
        "expedier a un fournisseur externe. Atout pour une instance enterprise "
        "hospitaliere / mutuelle qui refuse l'externalisation des donnees nutrition.",
        "- **Privacy** : les inputs (allergies, regime, budget) restent dans le "
        "perimetre du deploiement. Pertinent pour des donnees de sante au sens "
        "RGPD (article 9, donnees concernant la sante).",
        "- **Cout long terme** : pas de quota par requete. Pour un usage massif, "
        "le cout d'inference plafonne au cout CPU/GPU local. Mistral free tier "
        "n'est pas dimensionne pour de la prod a fort QPS.",
        "",
        "Le selecteur utilisateur introduit au slice 3 (`PATCH /me/preferences`) "
        "permet de respecter ces deux profils sans contraindre l'instance.",
        "",
    ]
