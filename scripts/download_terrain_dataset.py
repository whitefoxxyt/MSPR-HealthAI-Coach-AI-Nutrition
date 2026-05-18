"""Telecharge un dataset HF alternatif a Food-101 pour mesurer la generalisation
du classifier hors-studio (eval terrain) sans necessiter de collecte manuelle
de photos telephone.

Choix : Matthijs/snacks (HF, CC-BY 4.0). 20 classes de Google Open Images,
photos amateures (vs. studio Food-101). 3 classes ont un mapping direct vers
Food-101 :

  doughnut  -> donuts
  hot dog   -> hot_dog
  ice cream -> ice_cream

Les 17 autres classes sont labellisees `unknown` : elles servent a mesurer
`unknown_rate` (taux de plats hors-distribution Food-101).

Le dataset est telecharge depuis HuggingFace, decompresse, et un sous-echantillon
~50 images est copie dans `data/eval_terrain/images/` avec un `labels.csv`
genere automatiquement.

Usage :
    python scripts/download_terrain_dataset.py
"""

from __future__ import annotations

import csv
import logging
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

_DATASET_URL = (
    "https://huggingface.co/datasets/Matthijs/snacks/resolve/main/images.zip"
)
_SEED = 42

# Mapping snacks -> Food-101. `None` signifie qu'on label `unknown` (plat
# hors-distribution Food-101) : la prediction du classifier sera ignoree des
# metriques top-1/top-5, mais l'image compte dans `unknown_rate`.
_LABEL_MAPPING: dict[str, str | None] = {
    "doughnut": "donuts",
    "hot dog": "hot_dog",
    "ice cream": "ice_cream",
    "apple": None,
    "banana": None,
    "cake": None,
    "candy": None,
    "carrot": None,
    "cookie": None,
    "grape": None,
    "juice": None,
    "muffin": None,
    "orange": None,
    "pineapple": None,
    "popcorn": None,
    "pretzel": None,
    "salad": None,
    "strawberry": None,
    "waffle": None,
    "watermelon": None,
}

# Cibles par classe : 5 images pour les classes mappees (top-1/top-5),
# 2 images pour chaque classe `unknown` (unknown_rate). Total = 15 + 17*2 = 49.
_TARGETS = {label: 5 if mapped else 2 for label, mapped in _LABEL_MAPPING.items()}


def _download_zip(url: str, dest: Path) -> None:
    if dest.exists():
        logger.info("zip deja telecharge : %s (%d bytes)", dest, dest.stat().st_size)
        return
    logger.info("telechargement de %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
    logger.info("telecharge : %d bytes", dest.stat().st_size)


def _extract_subset(
    zip_path: Path,
    output_dir: Path,
    targets: dict[str, int],
    seed: int,
) -> list[tuple[str, str]]:
    """Extrait un sous-echantillon equilibre du zip. Retourne [(filename, snack_label)]."""
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        # Indexe les .jpg par classe : "data/test/apple/0.jpg" -> "apple".
        # Skip les stubs macOS AppleDouble (`__MACOSX/.../._*.jpg`, 297 bytes
        # chacun) qui co-existent avec les vraies images dans le zip.
        by_class: dict[str, list[str]] = {label: [] for label in targets}
        for name in zf.namelist():
            if not name.endswith(".jpg"):
                continue
            if name.startswith("__MACOSX/") or "/._" in name:
                continue
            parts = name.split("/")
            if len(parts) < 4:
                continue
            class_name = parts[-2].replace("_", " ")
            if class_name in by_class:
                by_class[class_name].append(name)

        selected: list[tuple[str, str]] = []
        for snack_label, n_target in targets.items():
            candidates = by_class.get(snack_label, [])
            if not candidates:
                logger.warning("aucune image pour la classe %s", snack_label)
                continue
            sample = rng.sample(candidates, min(n_target, len(candidates)))
            for idx, internal_name in enumerate(sample):
                # Nomme les fichiers sortants de facon stable (snack_label normalise + idx).
                safe_label = snack_label.replace(" ", "_")
                out_name = f"IMG_{safe_label}_{idx:02d}.jpg"
                target = output_dir / out_name
                with zf.open(internal_name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                selected.append((out_name, snack_label))

    return selected


def _write_labels_csv(
    selected: list[tuple[str, str]],
    csv_path: Path,
) -> None:
    """Genere labels.csv : filename,label_food101 avec mapping snacks -> food101."""
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "label_food101"])
        for filename, snack_label in selected:
            mapped = _LABEL_MAPPING.get(snack_label)
            label = mapped if mapped else "unknown"
            writer.writerow([filename, label])
    logger.info("labels.csv ecrit : %s (%d lignes)", csv_path, len(selected))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parent.parent
    terrain_dir = repo_root / "data" / "eval_terrain"
    images_dir = terrain_dir / "images"
    labels_csv = terrain_dir / "labels.csv"
    cache_dir = repo_root / ".cache"
    zip_path = cache_dir / "snacks_images.zip"

    _download_zip(_DATASET_URL, zip_path)

    # Nettoie les images precedentes pour eviter les melanges si on rejoue le script.
    for old in images_dir.glob("IMG_*.jpg"):
        old.unlink()

    selected = _extract_subset(
        zip_path=zip_path,
        output_dir=images_dir,
        targets=_TARGETS,
        seed=_SEED,
    )
    _write_labels_csv(selected, labels_csv)

    logger.info(
        "Termine. %d images extraites dans %s, labels dans %s.",
        len(selected),
        images_dir,
        labels_csv,
    )
    logger.info(
        "Prochaine etape : python scripts/eval_metrics.py classifier "
        "--n-food101 0 --terrain-dir data/eval_terrain (dans le container)."
    )


if __name__ == "__main__":
    main()
