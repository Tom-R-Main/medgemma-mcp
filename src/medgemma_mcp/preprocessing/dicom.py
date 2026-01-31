"""DICOM to PIL Image conversion.

Handles common medical imaging DICOM files:
- Applies rescale slope/intercept for Hounsfield units
- Applies basic windowing for display
- Converts grayscale to RGB (required by MedGemma's SigLIP encoder)
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Default soft-tissue window for general viewing
DEFAULT_WINDOW_CENTER = 40
DEFAULT_WINDOW_WIDTH = 400


def dicom_to_pil(
    path: str | Path,
    window_center: float | None = None,
    window_width: float | None = None,
) -> Image.Image:
    """Convert a DICOM file to a PIL RGB Image.

    Applies rescale slope/intercept and optional windowing.
    For chest X-rays, windowing is typically not needed (the full
    dynamic range is informative). For CT, windowing selects the
    tissue of interest.

    Args:
        path: Path to the DICOM file.
        window_center: Window center (Hounsfield units). If None, uses
            the value from the DICOM header or auto-scales.
        window_width: Window width (Hounsfield units). If None, uses
            the value from the DICOM header or auto-scales.

    Returns:
        PIL Image in RGB mode, normalized to [0, 255].

    Raises:
        ValueError: If the DICOM file cannot be read or has no pixel data.
    """
    import pydicom

    try:
        ds = pydicom.dcmread(str(path))
    except Exception as exc:
        raise ValueError(f"Cannot read DICOM file: {path}") from exc

    if not hasattr(ds, "pixel_array"):
        raise ValueError(f"DICOM file has no pixel data: {path}")

    pixels = ds.pixel_array.astype(np.float64)

    # Apply rescale slope and intercept (converts to Hounsfield units for CT)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept

    # Determine windowing parameters
    wc, ww = _resolve_window(ds, window_center, window_width)

    if wc is not None and ww is not None:
        # Apply windowing
        lower = wc - ww / 2
        upper = wc + ww / 2
        pixels = np.clip(pixels, lower, upper)
        pixels = (pixels - lower) / (upper - lower)
    else:
        # Auto-scale to full range
        pmin, pmax = float(pixels.min()), float(pixels.max())
        if pmax > pmin:
            pixels = (pixels - pmin) / (pmax - pmin)
        else:
            pixels = np.zeros_like(pixels)

    # Convert to uint8
    pixels_u8 = (pixels * 255).astype(np.uint8)

    # Handle PhotometricInterpretation
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        # Invert: 0 = white, max = black
        pixels_u8 = 255 - pixels_u8

    # Convert grayscale to RGB (MedGemma expects RGB input)
    img = Image.fromarray(pixels_u8, mode="L")
    return img.convert("RGB")


def _resolve_window(
    ds: "pydicom.Dataset",
    user_center: float | None,
    user_width: float | None,
) -> tuple[float | None, float | None]:
    """Resolve window center/width from user args or DICOM header."""
    if user_center is not None and user_width is not None:
        return user_center, user_width

    # Try DICOM header
    header_center = getattr(ds, "WindowCenter", None)
    header_width = getattr(ds, "WindowWidth", None)

    if header_center is not None and header_width is not None:
        # These can be lists (multiple windows) — take the first
        if isinstance(header_center, (list, pydicom.multival.MultiValue)):
            header_center = header_center[0]
        if isinstance(header_width, (list, pydicom.multival.MultiValue)):
            header_width = header_width[0]
        return float(header_center), float(header_width)

    # No windowing info available — will auto-scale
    return None, None


# Import pydicom at module level for type checking in _resolve_window
try:
    import pydicom
except ImportError:
    pydicom = None  # type: ignore[assignment]
