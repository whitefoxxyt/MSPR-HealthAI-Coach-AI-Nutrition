from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TerrainSample:
    filename: str
    label: str


def load_terrain_labels(csv_path: Path) -> list[TerrainSample]:
    """Lit data/eval_terrain/labels.csv -> liste de samples annotes."""
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or set(reader.fieldnames) < {"filename", "label"}:
            raise ValueError(
                f"CSV {csv_path} doit avoir les colonnes filename,label "
                f"(trouve : {reader.fieldnames})"
            )
        samples: list[TerrainSample] = []
        for row in reader:
            filename = (row.get("filename") or "").strip()
            label = (row.get("label") or "").strip()
            if not filename or not label:
                continue
            samples.append(TerrainSample(filename=filename, label=label))
    return samples
