"""
Télécharge des images de test depuis le dataset Food-101 (HuggingFace).
Les images sont sauvegardées dans data/benchmark_images/ avec le nom
de la classe comme nom de fichier (ex: pizza.jpg).

Usage :
  python scripts/download_test_images.py
  python scripts/download_test_images.py --output-dir data/benchmark_images/ --n 15
"""

import argparse
from pathlib import Path

from datasets import load_dataset


CLASSES = [
    "pizza",
    "sushi",
    "hamburger",
    "apple_pie",
    "french_fries",
    "omelette",
    "caesar_salad",
    "spaghetti_bolognese",
    "grilled_salmon",
    "chocolate_cake",
    "bibimbap",
    "pad_thai",
    "tacos",
    "bruschetta",
    "tiramisu",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Food-101 test images")
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmark_images"))
    parser.add_argument("--n", type=int, default=15, help="Nombre d'images à télécharger")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Chargement du dataset Food-101 (streaming)...")
    ds = load_dataset("food101", split="validation", streaming=True)

    targets = set(CLASSES[: args.n])
    saved: dict[str, bool] = {}

    for sample in ds:
        label: str = sample["label"]
        class_name: str = ds.features["label"].int2str(label)

        if class_name not in targets or class_name in saved:
            continue

        out_path = args.output_dir / f"{class_name}.jpg"
        try:
            img = sample["image"]
            # Reconstruire depuis les bytes bruts pour contourner les EXIF corrompus
            from io import BytesIO
            buf = BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            from PIL import Image
            clean = Image.open(buf).convert("RGB")
            clean.save(out_path, "JPEG")
        except Exception as e:
            print(f"  {class_name} ignoré (image corrompue : {e}), passage au suivant...")
            continue

        saved[class_name] = True
        print(f"  {class_name}.jpg sauvegardé")

        if len(saved) >= args.n:
            break

    print(f"\n{len(saved)} image(s) sauvegardée(s) dans {args.output_dir}")
    missing = targets - set(saved.keys())
    if missing:
        print(f"Classes non trouvées dans le dataset : {missing}")


if __name__ == "__main__":
    main()
