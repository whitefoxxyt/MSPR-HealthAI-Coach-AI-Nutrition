"""Dump le schema OpenAPI courant dans `docs/openapi.json`.

Livrable #6 MSPR2 : snapshot versionne du contrat API. Le test
`tests/test_openapi_doc.py::test_openapi_snapshot_is_up_to_date` echoue tant que
ce fichier n'est pas regenere apres une modification du code des routers.

Usage :
    python scripts/export_openapi.py

Le fichier est ecrit en JSON UTF-8 indente 2 espaces, sans ASCII escaping
(pour preserver les accents francais des descriptions).
"""

from __future__ import annotations

import json
from pathlib import Path

from app.main import app

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "docs" / "openapi.json"


def main() -> None:
    schema = app.openapi()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"Schema OpenAPI ecrit dans {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
