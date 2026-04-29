from __future__ import annotations

import sys
import types

import sqlalchemy


def _stub_transformers_pipeline(*_args, **_kwargs):
    raise RuntimeError("transformers.pipeline stub : non utilisable en tests")


def _stub_pil_image_open(*_args, **_kwargs):
    raise RuntimeError("PIL.Image.open stub : non utilisable en tests")


def _install_stubs() -> None:
    """Stubs charges avant l'import de app.main pour eviter les deps lourdes
    (transformers, torch, PIL, psycopg2) dans les tests d'integration de routing."""
    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")
        transformers.pipeline = _stub_transformers_pipeline
        sys.modules["transformers"] = transformers

    if "PIL" not in sys.modules:
        pil = types.ModuleType("PIL")
        pil_image = types.ModuleType("PIL.Image")
        pil_image.open = _stub_pil_image_open
        pil.Image = pil_image
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = pil_image


_install_stubs()


# Force un engine SQLite en memoire pour les tests : evite la dep psycopg2
# qui n'a pas de wheel pour Python 3.14 (en local). On force l'import de
# app.main sous patch puis on restaure create_engine pour eviter la fuite
# globale du monkeypatch sur le reste du process de test.
_real_create_engine = sqlalchemy.create_engine


def _fake_create_engine(_url, *_args, **_kwargs):
    return _real_create_engine("sqlite:///:memory:")


sqlalchemy.create_engine = _fake_create_engine
import app.main  # noqa: E402, F401  -- declenche la creation de l'engine sous patch
sqlalchemy.create_engine = _real_create_engine
