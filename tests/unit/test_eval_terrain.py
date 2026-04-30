from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from scripts.eval.classifier_runner import _eval_terrain
from scripts.eval.terrain import TerrainSample, load_terrain_labels


def _stub_classifier(*results_per_call: list[dict[str, Any]]):
    """Renvoie un fake classifier qui retourne `results_per_call[i]` au i-eme appel."""
    calls = iter(results_per_call)

    def _call(image: Any, top_k: int = 5) -> list[dict[str, Any]]:
        return next(calls)[:top_k]

    return _call


def _write_image(path: Path) -> None:
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(path, format="JPEG")


def test_eval_terrain_returns_zeros_dict_when_csv_missing(tmp_path: Path) -> None:
    classifier = _stub_classifier()  # ne sera jamais appele

    payload = _eval_terrain(classifier, tmp_path)

    assert payload == {
        "n_samples": 0,
        "n_classified": 0,
        "top1_accuracy": 0.0,
        "top5_accuracy": 0.0,
        "unknown_rate": 0.0,
    }


def test_eval_terrain_returns_zeros_dict_when_csv_has_header_only(
    tmp_path: Path,
) -> None:
    (tmp_path / "labels.csv").write_text("filename,label_food101\n", encoding="utf-8")
    classifier = _stub_classifier()

    payload = _eval_terrain(classifier, tmp_path)

    assert payload == {
        "n_samples": 0,
        "n_classified": 0,
        "top1_accuracy": 0.0,
        "top5_accuracy": 0.0,
        "unknown_rate": 0.0,
    }


def test_eval_terrain_top1_and_top5_hit_on_normal_sample(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_image(images_dir / "img_001.jpg")
    (tmp_path / "labels.csv").write_text(
        "filename,label_food101\nimg_001.jpg,pizza\n", encoding="utf-8"
    )
    classifier = _stub_classifier(
        [
            {"label": "pizza", "score": 0.9},
            {"label": "lasagna", "score": 0.05},
            {"label": "spaghetti_bolognese", "score": 0.02},
            {"label": "ravioli", "score": 0.02},
            {"label": "carbonara", "score": 0.01},
        ]
    )

    payload = _eval_terrain(classifier, tmp_path)

    assert payload == {
        "n_samples": 1,
        "n_classified": 1,
        "top1_accuracy": 1.0,
        "top5_accuracy": 1.0,
        "unknown_rate": 0.0,
    }


def test_eval_terrain_top1_miss_top5_hit(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_image(images_dir / "img_001.jpg")
    (tmp_path / "labels.csv").write_text(
        "filename,label_food101\nimg_001.jpg,pizza\n", encoding="utf-8"
    )
    classifier = _stub_classifier(
        [
            {"label": "lasagna", "score": 0.6},
            {"label": "pizza", "score": 0.3},
            {"label": "spaghetti_bolognese", "score": 0.05},
            {"label": "ravioli", "score": 0.03},
            {"label": "carbonara", "score": 0.02},
        ]
    )

    payload = _eval_terrain(classifier, tmp_path)

    assert payload["n_samples"] == 1
    assert payload["n_classified"] == 1
    assert payload["top1_accuracy"] == pytest.approx(0.0)
    assert payload["top5_accuracy"] == pytest.approx(1.0)
    assert payload["unknown_rate"] == pytest.approx(0.0)


def test_eval_terrain_unknown_label_only_contributes_to_unknown_rate(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_image(images_dir / "img_001.jpg")
    _write_image(images_dir / "img_002.jpg")
    (tmp_path / "labels.csv").write_text(
        "filename,label_food101\nimg_001.jpg,pizza\nimg_002.jpg,unknown\n",
        encoding="utf-8",
    )
    classifier = _stub_classifier(
        [
            {"label": "pizza", "score": 0.9},
            {"label": "lasagna", "score": 0.05},
            {"label": "spaghetti_bolognese", "score": 0.02},
            {"label": "ravioli", "score": 0.02},
            {"label": "carbonara", "score": 0.01},
        ],
        [
            {"label": "tiramisu", "score": 0.4},
            {"label": "creme_brulee", "score": 0.3},
            {"label": "panna_cotta", "score": 0.15},
            {"label": "chocolate_mousse", "score": 0.1},
            {"label": "ice_cream", "score": 0.05},
        ],
    )

    payload = _eval_terrain(classifier, tmp_path)

    assert payload == {
        "n_samples": 2,
        "n_classified": 1,
        "top1_accuracy": 1.0,
        "top5_accuracy": 1.0,
        "unknown_rate": 0.5,
    }


def test_eval_terrain_skips_missing_image_files(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_image(images_dir / "img_001.jpg")
    # img_002.jpg est referencee dans le CSV mais absente du dossier images/
    (tmp_path / "labels.csv").write_text(
        "filename,label_food101\nimg_001.jpg,pizza\nimg_002.jpg,sushi\n",
        encoding="utf-8",
    )
    classifier = _stub_classifier(
        [
            {"label": "pizza", "score": 0.9},
            {"label": "lasagna", "score": 0.05},
            {"label": "spaghetti_bolognese", "score": 0.02},
            {"label": "ravioli", "score": 0.02},
            {"label": "carbonara", "score": 0.01},
        ]
    )

    payload = _eval_terrain(classifier, tmp_path)

    # n_samples reflete le CSV (2), n_classified l'execution effective (1).
    # La distinction evite que le report annonce "1.0 sur 2 echantillons".
    assert payload["n_samples"] == 2
    assert payload["n_classified"] == 1
    assert payload["top1_accuracy"] == pytest.approx(1.0)
    assert payload["top5_accuracy"] == pytest.approx(1.0)
    assert payload["unknown_rate"] == pytest.approx(0.0)


def test_eval_terrain_only_unknown_samples_yields_zero_accuracy(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_image(images_dir / "img_001.jpg")
    (tmp_path / "labels.csv").write_text(
        "filename,label_food101\nimg_001.jpg,unknown\n", encoding="utf-8"
    )
    classifier = _stub_classifier(
        [
            {"label": "pizza", "score": 0.9},
            {"label": "lasagna", "score": 0.05},
            {"label": "spaghetti_bolognese", "score": 0.02},
            {"label": "ravioli", "score": 0.02},
            {"label": "carbonara", "score": 0.01},
        ]
    )

    payload = _eval_terrain(classifier, tmp_path)

    assert payload == {
        "n_samples": 1,
        "n_classified": 0,
        "top1_accuracy": 0.0,
        "top5_accuracy": 0.0,
        "unknown_rate": 1.0,
    }


def test_load_terrain_labels_reads_csv(tmp_path: Path) -> None:
    csv = tmp_path / "labels.csv"
    csv.write_text("filename,label_food101\nimg_001.jpg,pizza\nimg_002.jpg,sushi\n")

    samples = load_terrain_labels(csv)

    assert samples == [
        TerrainSample(filename="img_001.jpg", label="pizza"),
        TerrainSample(filename="img_002.jpg", label="sushi"),
    ]


def test_load_terrain_labels_skips_blank_rows(tmp_path: Path) -> None:
    csv = tmp_path / "labels.csv"
    csv.write_text("filename,label_food101\nimg_001.jpg,pizza\n,\nimg_002.jpg,sushi\n")

    samples = load_terrain_labels(csv)

    assert [s.filename for s in samples] == ["img_001.jpg", "img_002.jpg"]


def test_load_terrain_labels_rejects_missing_columns(tmp_path: Path) -> None:
    csv = tmp_path / "bad.csv"
    csv.write_text("filename\nimg_001.jpg\n")

    with pytest.raises(ValueError, match="filename.*label"):
        load_terrain_labels(csv)
