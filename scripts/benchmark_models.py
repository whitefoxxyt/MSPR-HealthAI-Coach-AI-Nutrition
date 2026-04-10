"""
Benchmark des modèles HuggingFace pour la classification d'aliments.

Modèles comparés :
  - nateraw/food          (ViT-base, 101 classes Food-101)
  - Kaludi/food-category-classification-v2.0  (12 catégories larges)

Usage :
  python scripts/benchmark_models.py --images-dir data/benchmark_images/
  python scripts/benchmark_models.py --images-dir data/benchmark_images/ --output docs/results.json
"""

import argparse
import json
import time
from pathlib import Path

from PIL import Image
from transformers import pipeline


MODELS = {
    "nateraw/food": {
        "task": "image-classification",
        "label": "nateraw/food (ViT-base, 101 classes)",
    },
    "Kaludi/food-category-classification-v2.0": {
        "task": "image-classification",
        "label": "Kaludi/food-category-classification-v2.0 (12 classes)",
    },
}

GROUND_TRUTH = {
    "pizza.jpg": "pizza",
    "sushi.jpg": "sushi",
    "hamburger.jpg": "hamburger",
    "apple_pie.jpg": "apple_pie",
    "french_fries.jpg": "french_fries",
    "omelette.jpg": "omelette",
    "caesar_salad.jpg": "caesar_salad",
    "spaghetti_bolognese.jpg": "spaghetti_bolognese",
    "grilled_salmon.jpg": "grilled_salmon",
    "chocolate_cake.jpg": "chocolate_cake",
}


def load_images(images_dir: Path) -> list[tuple[str, Image.Image]]:
    supported = {".jpg", ".jpeg", ".png", ".webp"}
    images = []
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() in supported:
            images.append((path.name, Image.open(path).convert("RGB")))
    if not images:
        raise FileNotFoundError(f"Aucune image trouvée dans {images_dir}")
    return images


def benchmark_model(model_id: str, images: list[tuple[str, Image.Image]]) -> dict:
    print(f"\n--- Chargement du modèle : {model_id} ---")
    clf = pipeline(MODELS[model_id]["task"], model=model_id)

    latencies: list[float] = []
    top1_correct = 0
    predictions: list[dict] = []

    for filename, img in images:
        t0 = time.perf_counter()
        results = clf(img, top_k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        top1_label = results[0]["label"].lower().replace(" ", "_")
        ground = GROUND_TRUTH.get(filename, "").lower()
        correct = ground and ground in top1_label

        if correct:
            top1_correct += 1

        predictions.append({
            "image": filename,
            "ground_truth": ground or "unknown",
            "top1_pred": results[0]["label"],
            "top1_score": round(results[0]["score"], 4),
            "top5": [{"label": r["label"], "score": round(r["score"], 4)} for r in results],
            "latency_ms": round(elapsed_ms, 1),
            "correct": correct,
        })
        print(f"  {filename}: {results[0]['label']} ({results[0]['score']:.2%}) | {elapsed_ms:.0f}ms")

    labeled = [p for p in predictions if p["ground_truth"] != "unknown"]
    top1_acc = top1_correct / len(labeled) if labeled else None

    return {
        "model_id": model_id,
        "label": MODELS[model_id]["label"],
        "n_images": len(images),
        "n_labeled": len(labeled),
        "top1_accuracy": round(top1_acc, 4) if top1_acc is not None else None,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark food classification models")
    parser.add_argument("--images-dir", type=Path, required=True, help="Dossier contenant les images de test")
    parser.add_argument("--output", type=Path, default=None, help="Fichier JSON de sortie")
    args = parser.parse_args()

    images = load_images(args.images_dir)
    print(f"{len(images)} image(s) chargée(s) depuis {args.images_dir}")

    results = {}
    for model_id in MODELS:
        results[model_id] = benchmark_model(model_id, images)

    print("\n=== Résumé ===")
    for model_id, r in results.items():
        acc = f"{r['top1_accuracy']:.1%}" if r["top1_accuracy"] is not None else "N/A"
        print(f"  {model_id}")
        print(f"    Top-1 accuracy : {acc}")
        print(f"    Latence moy.   : {r['avg_latency_ms']} ms")

    if args.output:
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nRésultats exportés dans {args.output}")


if __name__ == "__main__":
    main()
