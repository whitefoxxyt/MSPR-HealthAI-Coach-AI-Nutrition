from __future__ import annotations

from pathlib import Path

import pytest

from scripts.eval.terrain import TerrainSample, load_terrain_labels


def test_load_terrain_labels_reads_csv(tmp_path: Path) -> None:
    csv = tmp_path / "labels.csv"
    csv.write_text("filename,label\nimg_001.jpg,pizza\nimg_002.jpg,sushi\n")

    samples = load_terrain_labels(csv)

    assert samples == [
        TerrainSample(filename="img_001.jpg", label="pizza"),
        TerrainSample(filename="img_002.jpg", label="sushi"),
    ]


def test_load_terrain_labels_skips_blank_rows(tmp_path: Path) -> None:
    csv = tmp_path / "labels.csv"
    csv.write_text("filename,label\nimg_001.jpg,pizza\n,\nimg_002.jpg,sushi\n")

    samples = load_terrain_labels(csv)

    assert [s.filename for s in samples] == ["img_001.jpg", "img_002.jpg"]


def test_load_terrain_labels_rejects_missing_columns(tmp_path: Path) -> None:
    csv = tmp_path / "bad.csv"
    csv.write_text("filename\nimg_001.jpg\n")

    with pytest.raises(ValueError, match="filename.*label"):
        load_terrain_labels(csv)
