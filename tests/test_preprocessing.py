"""Tests for image preprocessing utilities."""

from pathlib import Path

import pytest
from PIL import Image

from medgemma_mcp.preprocessing.images import load_image


def test_load_from_file_path(sample_image_path: Path):
    """Load an image from a file path."""
    img = load_image(str(sample_image_path))
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_load_from_base64(sample_image_base64: str):
    """Load an image from a base64 string."""
    img = load_image(sample_image_base64)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_load_from_base64_with_data_uri(sample_image_base64: str):
    """Load an image from a base64 data URI."""
    data_uri = f"data:image/png;base64,{sample_image_base64}"
    img = load_image(data_uri)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_load_grayscale_converts_to_rgb(tmp_path: Path):
    """Grayscale images are automatically converted to RGB."""
    gray_img = Image.new("L", (128, 128), color=200)
    path = tmp_path / "gray.png"
    gray_img.save(path)

    img = load_image(str(path))
    assert img.mode == "RGB"


def test_load_invalid_path_invalid_base64():
    """Invalid source raises ValueError."""
    with pytest.raises(ValueError):
        load_image("not_a_file_and_not_base64!!!")


def test_load_nonexistent_file_tries_base64():
    """Non-existent path falls through to base64 attempt."""
    with pytest.raises(ValueError):
        load_image("/nonexistent/path/to/image.png")


def test_load_jpeg(tmp_path: Path):
    """Load a JPEG image."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    path = tmp_path / "test.jpg"
    img.save(path, format="JPEG")

    loaded = load_image(str(path))
    assert loaded.mode == "RGB"
    assert loaded.size == (100, 100)
