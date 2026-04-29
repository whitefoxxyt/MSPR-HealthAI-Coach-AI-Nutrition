from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "fallback_plans.json"


@lru_cache(maxsize=1)
def _load_all() -> dict[str, dict]:
    with _DATA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def load_fallback_plan(health_goal: str, diet_type: str) -> dict | None:
    key = f"{health_goal}_{diet_type}"
    return _load_all().get(key)
