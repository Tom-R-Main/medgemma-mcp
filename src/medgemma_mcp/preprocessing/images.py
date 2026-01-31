"""Image loading utilities for medical images.

Handles loading from file paths and base64 strings,
with automatic format detection for DICOM, JPEG, PNG, etc.
"""

import base64
import io
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# File extensions recognized as DICOM
DICOM_EXTENSIONS = {".dcm", ".dicom", ".dic"}

# File extensions recognized as standard images
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_image(source: str) -> Image.Image:
    """Load a medical image from a file path or base64 string.

    Automatically detects:
    - File paths (checked for existence)
    - DICOM files (by extension, delegates to dicom module)
    - Standard image formats (JPEG, PNG, etc.)
    - Base64-encoded image data

    Args:
        source: File path or base64-encoded image data.

    Returns:
        PIL Image in RGB mode.

    Raises:
        ValueError: If the source cannot be loaded as an image.
    """
    # Try as file path first (guard against very long strings that can't be paths)
    if len(source) < 4096:
        path = Path(source)
        try:
            if path.exists() and path.is_file():
                return _load_from_path(path)
        except OSError:
            pass  # Path too long or invalid — fall through to base64

    # Try as base64
    return _load_from_base64(source)


def _load_from_path(path: Path) -> Image.Image:
    """Load image from a file path."""
    suffix = path.suffix.lower()

    if suffix in DICOM_EXTENSIONS:
        from medgemma_mcp.preprocessing.dicom import dicom_to_pil

        return dicom_to_pil(path)

    if suffix in IMAGE_EXTENSIONS or suffix == "":
        # PIL handles format detection internally
        img = Image.open(path)
        return img.convert("RGB")

    # Try PIL anyway — it may recognize the format
    try:
        img = Image.open(path)
        return img.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot load image from {path}: unsupported format '{suffix}'") from exc


def _load_from_base64(data: str) -> Image.Image:
    """Decode a base64 string to a PIL Image."""
    # Strip optional data URI prefix (e.g., "data:image/png;base64,...")
    if "," in data and data.index(",") < 100:
        data = data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(data)
    except Exception as exc:
        raise ValueError("Source is not a valid file path or base64-encoded image data") from exc

    try:
        img = Image.open(io.BytesIO(image_bytes))
        return img.convert("RGB")
    except Exception as exc:
        raise ValueError("Decoded base64 data is not a valid image") from exc
