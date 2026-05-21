import io
import base64
from PIL import Image

from ingestion.config import MAX_IMAGE_SIZE, JPEG_QUALITY


def pil_to_base64(image: Image.Image) -> str | None:
    """Resize a PIL image and return it as a base64-encoded JPEG string.

    The image is:
      - converted to RGB if needed (drops alpha channel)
      - downscaled so the longest side is at most ``MAX_IMAGE_SIZE`` pixels
      - encoded as JPEG with ``JPEG_QUALITY``

    Args:
        image: A PIL ``Image.Image`` object.  Pass ``None`` to get ``None`` back.

    Returns:
        Base64-encoded JPEG string, or ``None`` if the input is ``None``
        or if conversion fails.
    """
    if image is None:
        return None

    try:
        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size
        scale = MAX_IMAGE_SIZE / max(w, h)
        if scale < 1.0:  # only downscale, never upscale
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"[WARN] pil_to_base64 failed: {e}")
        return None


def base64_to_pil(b64_str: str) -> Image.Image | None:
    """Decode a base64 JPEG string back into a PIL image.

    Useful for debugging / spot-checking stored payloads.

    Args:
        b64_str: Base64-encoded JPEG string produced by :func:`pil_to_base64`.

    Returns:
        A PIL ``Image.Image`` object, or ``None`` if decoding fails.
    """
    if not b64_str:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64_str)))
    except Exception as e:
        print(f"[WARN] base64_to_pil failed: {e}")
        return None