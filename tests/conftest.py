"""Shared test fixtures."""

import io
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """Create a minimal RGB test image."""
    return Image.new("RGB", (256, 256), color=(128, 128, 128))


@pytest.fixture
def sample_grayscale_image() -> Image.Image:
    """Create a minimal grayscale test image."""
    return Image.new("L", (256, 256), color=128)


@pytest.fixture
def sample_image_path(tmp_path: Path, sample_rgb_image: Image.Image) -> Path:
    """Save a test image to a temporary file and return its path."""
    path = tmp_path / "test_image.png"
    sample_rgb_image.save(path)
    return path


@pytest.fixture
def sample_image_base64(sample_rgb_image: Image.Image) -> str:
    """Create a base64-encoded test image string."""
    import base64

    buf = io.BytesIO()
    sample_rgb_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
