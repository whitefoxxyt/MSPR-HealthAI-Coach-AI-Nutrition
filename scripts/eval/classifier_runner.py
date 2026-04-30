"""
Runner classifier : telecharge un sous-ensemble Food-101, classe les images,
calcule les metriques + matrice de confusion, et eventuellement evalue les
photos terrain annotees dans data/eval_terrain/.

Necessite : transformers, datasets, torch (CPU), matplotlib, Pillow.
"""

from __future__ import annotations

import logging
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.eval.classifier_metrics import (
    confusion_matrix,
    per_class_metrics,
    top_k_accuracy,
)
from scripts.eval.plotting import save_confusion_matrix_png
from scripts.eval.terrain import UNKNOWN_LABEL, load_terrain_labels

logger = logging.getLogger(__name__)

_MODEL_ID = "nateraw/food"

_EMPTY_TERRAIN_PAYLOAD: dict[str, Any] = {
    "n_samples": 0,
    "n_classified": 0,
    "top1_accuracy": 0.0,
    "top5_accuracy": 0.0,
    "unknown_rate": 0.0,
}


def run_classifier_eval(
    n_food101: int,
    terrain_dir: Path,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Renvoie un payload {food101: ..., terrain: ...} pour metrics.json.

    `seed` est utilise pour shuffler le split Food-101 (sinon le streaming
    sert les classes par ordre alphabetique et un petit n_samples ne couvre
    que les premieres classes).
    """
    classifier = _load_classifier()
    payload: dict[str, Any] = {}

    payload["food101"] = _eval_food101(classifier, n_food101, seed, output_dir)
    payload["terrain"] = _eval_terrain(classifier, terrain_dir)
    return payload


def _load_classifier() -> Any:
    from transformers import pipeline

    return pipeline("image-classification", model=_MODEL_ID)


def _eval_food101(
    classifier: Any, n_samples: int, seed: int, output_dir: Path
) -> dict[str, Any]:
    """Echantillonne n_samples du test split Food-101 et calcule les metriques.

    Shuffle avec `seed` (par defaut passe par run_classifier_eval) puis prend
    les n_samples premiers, sinon le streaming sert les classes par ordre
    alphabetique (250 images par classe) et un petit n_samples ne couvre que
    les premieres classes du dataset.
    """
    from datasets import Image as HFImage
    from datasets import load_dataset

    logger.info("food101 : load_dataset (split=validation, streaming, shuffled seed=%d)", seed)
    ds = load_dataset("food101", split="validation", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    ds = ds.cast_column("image", HFImage(decode=False))

    samples: list[tuple[Image.Image, str]] = []
    label_feature = ds.features["label"]

    for sample in ds:
        try:
            raw: bytes = sample["image"]["bytes"]
            img = Image.open(BytesIO(raw)).convert("RGB")
        except (OSError, ValueError, KeyError):
            continue
        truth = label_feature.int2str(sample["label"])
        samples.append((img, truth))
        if len(samples) >= n_samples:
            break

    logger.info("food101 : %d images echantillonnees", len(samples))

    top1_preds: list[str] = []
    top5_preds: list[list[str]] = []
    truths: list[str] = []
    for img, truth in samples:
        results = classifier(img, top_k=5)
        top5 = [r["label"].lower().replace(" ", "_") for r in results]
        top1_preds.append(top5[0])
        top5_preds.append(top5)
        truths.append(truth)

    top1_acc = top_k_accuracy([[p] for p in top1_preds], truths, k=1)
    top5_acc = top_k_accuracy(top5_preds, truths, k=5)
    metrics = per_class_metrics(top1_preds, truths)

    # Matrice de confusion : top 20 classes les plus frequentes pour rester lisible
    top_classes = [c for c, _ in Counter(truths).most_common(20)]
    cm = confusion_matrix(top1_preds, truths, top_classes)
    cm_path = output_dir / "confusion_matrix_food101.png"
    save_confusion_matrix_png(cm, top_classes, cm_path, title="Food-101 (top 20)")

    return {
        "n_samples": len(samples),
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "per_class": metrics,
        "confusion_matrix_png": str(cm_path),
        "confusion_matrix_labels": top_classes,
        "confusion_matrix": cm,
    }


def _eval_terrain(classifier: Any, terrain_dir: Path) -> dict[str, Any]:
    labels_csv = terrain_dir / "labels.csv"
    if not labels_csv.exists():
        logger.warning(
            "terrain : %s introuvable, on saute l'eval terrain. "
            "Annoter 50 photos dans %s pour activer cette metrique.",
            labels_csv,
            terrain_dir,
        )
        return dict(_EMPTY_TERRAIN_PAYLOAD)

    samples = load_terrain_labels(labels_csv)
    if not samples:
        logger.warning("terrain : %s est vide", labels_csv)
        return dict(_EMPTY_TERRAIN_PAYLOAD)

    images_dir = terrain_dir / "images"
    n_samples = len(samples)
    n_unknown = 0
    top1_preds: list[str] = []
    top5_preds: list[list[str]] = []
    truths: list[str] = []
    for sample in samples:
        if sample.label == UNKNOWN_LABEL:
            # Plat hors-distribution Food-101 : compte dans unknown_rate
            # mais n'est pas envoye au classifier.
            n_unknown += 1
            continue
        img_path = images_dir / sample.filename
        if not img_path.exists():
            logger.warning("terrain : image %s manquante, ignoree", img_path)
            continue
        img = Image.open(img_path).convert("RGB")
        results = classifier(img, top_k=5)
        top5 = [r["label"].lower().replace(" ", "_") for r in results]
        top1_preds.append(top5[0])
        top5_preds.append(top5)
        truths.append(sample.label)

    unknown_rate = n_unknown / n_samples if n_samples else 0.0
    # n_classified : combien d'images ont ete reellement passees au classifier.
    # Distinct de n_samples (= taille du CSV) : exclut les `unknown` et les
    # images manquantes. Sans ce champ, le report affiche "100 % sur 2
    # echantillons" alors qu'un seul a ete classe.
    n_classified = len(truths)
    if not truths:
        return {
            "n_samples": n_samples,
            "n_classified": 0,
            "top1_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "unknown_rate": unknown_rate,
        }

    top1_acc = top_k_accuracy([[p] for p in top1_preds], truths, k=1)
    top5_acc = top_k_accuracy(top5_preds, truths, k=5)
    return {
        "n_samples": n_samples,
        "n_classified": n_classified,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "unknown_rate": unknown_rate,
    }
