"""
Calcule les metriques IA exigees par le PDF (precision, rappel, F1, latence,
respect des contraintes) et produit docs/metrics.json + docs/metrics.md.

Sous-commandes :
  classifier  -> evalue HuggingFace nateraw/food sur Food-101 et terrain
  llm         -> evalue le LLM sur N generations (backend selectionne par
                 LLM_BACKEND : ollama gemma3:4b local, ou mistral managed)

Reproductibilite : seed fixe (--seed). Les chiffres restent stables a +/- 5%
d'une execution a l'autre tant que :
  - le modele HuggingFace ne change pas (nateraw/food, version pin)
  - le modele LLM ne change pas (gemma3:4b cote Ollama, mistral-small managed)
  - le sampling utilise le meme seed.

Usage :
  python scripts/eval_metrics.py classifier --n-food101 1000 --seed 42
  LLM_BACKEND=mistral python scripts/eval_metrics.py llm --n-generations 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("eval_metrics")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Eval des metriques IA (classifier + LLM)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_clf = sub.add_parser("classifier", help="Eval HuggingFace nateraw/food")
    p_clf.add_argument("--n-food101", type=int, default=1000)
    p_clf.add_argument(
        "--terrain-dir",
        type=Path,
        default=Path("data/eval_terrain"),
        help="Dossier contenant photos + labels.csv (annotation manuelle)",
    )
    p_clf.add_argument("--seed", type=int, default=42)
    p_clf.add_argument("--output-dir", type=Path, default=Path("docs"))

    p_llm = sub.add_parser(
        "llm", help="Eval LLM (backend selectionne par LLM_BACKEND : ollama | mistral)"
    )
    p_llm.add_argument("--n-generations", type=int, default=100)
    p_llm.add_argument(
        "--n-constraint-plans",
        type=int,
        default=30,
        help="Plans evalues sur le respect simultanee allergies+budget+regime",
    )
    p_llm.add_argument(
        "--hitl-csv",
        type=Path,
        default=Path("docs/llm_hitl_ratings.csv"),
    )
    p_llm.add_argument("--seed", type=int, default=42)
    p_llm.add_argument("--output-dir", type=Path, default=Path("docs"))

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "classifier":
        return _run_classifier_eval(args)
    if args.command == "llm":
        return _run_llm_eval(args)
    return 2


def _run_classifier_eval(args: argparse.Namespace) -> int:
    # Imports tardifs : transformers / datasets / matplotlib sont lourds et
    # absents en environnement de test.
    from scripts.eval.classifier_runner import run_classifier_eval

    payload = run_classifier_eval(
        n_food101=args.n_food101,
        terrain_dir=args.terrain_dir,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    _persist(payload, args.output_dir, section="classifier")
    return 0


def _run_llm_eval(args: argparse.Namespace) -> int:
    from scripts.eval.llm_runner import run_llm_eval

    payload = asyncio.run(
        run_llm_eval(
            n_generations=args.n_generations,
            n_constraint_plans=args.n_constraint_plans,
            hitl_csv=args.hitl_csv,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    )
    _persist(payload, args.output_dir, section="llm")
    return 0


def _persist(section_payload: dict[str, Any], output_dir: Path, section: str) -> None:
    """Merge la section calculee dans docs/metrics.json puis re-rend metrics.md.

    Pour la section `llm` (slice 5 #75) on merge plutot qu'on ecrase : les
    runs Gemma / Mistral N=20 / Mistral N=100 s'empilent sous des cles dediees
    (`gemma_n20`, `mistral_n20`, `mistral_n100`) au fil des executions
    successives, sans perdre les precedents. Et apres merge, on rend le PNG
    comparatif `llm_latency_distribution.png` agregeant les latences brutes de
    tous les runs presents.
    """
    from scripts.eval.report import dump_metrics_json, render_metrics_md

    metrics_json = output_dir / "metrics.json"
    metrics_md = output_dir / "metrics.md"

    full: dict[str, Any] = {}
    if metrics_json.exists():
        full = json.loads(metrics_json.read_text(encoding="utf-8"))
    if section == "llm":
        existing = full.get(section) or {}
        full[section] = {**existing, **section_payload}
    else:
        full[section] = section_payload
    full["generated_at"] = int(time.time())

    dump_metrics_json(full, metrics_json)
    if section == "llm":
        _maybe_render_latency_comparison_png(full.get("llm") or {}, output_dir)
    render_metrics_md(full, metrics_md)
    logger.info("metriques exportees : %s + %s", metrics_json, metrics_md)


def _maybe_render_latency_comparison_png(
    llm_section: dict[str, Any], output_dir: Path
) -> None:
    """Genere le PNG boxplot multi-backend si au moins un run apporte des
    latences brutes (`naive.latencies_ms_raw`)."""
    from scripts.eval.plotting import save_latency_comparison_png

    series: dict[str, list[float]] = {}
    for run_key, run in llm_section.items():
        if not isinstance(run, dict):
            continue
        latencies = ((run.get("naive") or {}).get("latencies_ms_raw")) or []
        if latencies:
            series[run_key] = [float(v) for v in latencies]

    if not series:
        return
    save_latency_comparison_png(series, output_dir / "llm_latency_distribution.png")


if __name__ == "__main__":
    sys.exit(main())
