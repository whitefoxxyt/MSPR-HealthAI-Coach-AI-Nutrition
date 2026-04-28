from __future__ import annotations

import sys
import types

import sqlalchemy

# Stubs charges avant l'import de app.main pour eviter les deps lourdes
# (transformers, torch, PIL, psycopg2) dans les tests d'integration de routing.
def _install_stubs() -> None:
    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")
        transformers.pipeline = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("transformers.pipeline stub : non utilisable en tests")
        )
        sys.modules["transformers"] = transformers

    if "PIL" not in sys.modules:
        pil = types.ModuleType("PIL")
        pil_image = types.ModuleType("PIL.Image")
        pil_image.open = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("PIL.Image.open stub : non utilisable en tests")
        )
        pil.Image = pil_image
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = pil_image


_install_stubs()


# Force un engine SQLite en memoire pour les tests : evite la dep psycopg2
# qui n'a pas de wheel pour Python 3.14.
_real_create_engine = sqlalchemy.create_engine


def _fake_create_engine(url, *args, **kwargs):
    kwargs.pop("pool_pre_ping", None)
    return _real_create_engine("sqlite:///:memory:")


sqlalchemy.create_engine = _fake_create_engine
