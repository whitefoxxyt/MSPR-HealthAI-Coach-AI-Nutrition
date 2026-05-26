from __future__ import annotations

import base64
import io
import logging

from PIL import Image

_LOGGER = logging.getLogger(__name__)

_MAX_SIDE_PX = 512
_JPEG_QUALITY = 75


def to_data_url(image_bytes: bytes) -> str | None:
    """Genere un data URL JPEG base64 d'au plus 512x512 a partir des bytes
    originaux. Retourne None si l'image ne peut pas etre decodee, pour ne
    pas faire echouer le pipeline d'analyse a cause d'un thumbnail.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.thumbnail((_MAX_SIDE_PX, _MAX_SIDE_PX))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        _LOGGER.exception("Generation de thumbnail echouee")
        return None
