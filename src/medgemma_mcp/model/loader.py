"""Lifespan-managed model loading for MedGemma.

Loads the model once at server startup and keeps it in memory for all tool calls.
Uses the MCP SDK lifespan pattern with a typed dataclass context.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "google/medgemma-4b-it"


@dataclass
class MedGemmaContext:
    """Typed lifespan context holding the loaded model and processor.

    Accessed in tools via: ctx.request_context.lifespan_context
    """

    model: AutoModelForImageTextToText
    processor: AutoProcessor
    device: str
    model_id: str


def _detect_device() -> tuple[str, torch.dtype]:
    """Detect the best available device and appropriate dtype.

    MPS (Apple Silicon) requires float32 — float16 causes numerical
    instability where MedGemma generates only pad tokens.
    """
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def _load_model(model_id: str) -> tuple[AutoModelForImageTextToText, AutoProcessor, str]:
    """Load MedGemma model and processor. Blocking call run in thread pool."""
    device, dtype = _detect_device()

    logger.info("Loading %s on %s with %s", model_id, device, dtype)

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    # For non-CUDA devices, move model manually
    if device != "cuda":
        model = model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    logger.info("Model loaded successfully")
    return model, processor, device


@asynccontextmanager
async def medgemma_lifespan(server: "FastMCP") -> AsyncIterator[MedGemmaContext]:
    """Load MedGemma at server startup, clean up on shutdown.

    This follows the MCP SDK lifespan pattern. The yielded MedGemmaContext
    is available in tools via ctx.request_context.lifespan_context.
    """
    model_id = DEFAULT_MODEL_ID

    # Load model in thread pool to avoid blocking the event loop
    model, processor, device = await anyio.to_thread.run_sync(lambda: _load_model(model_id))

    try:
        yield MedGemmaContext(
            model=model,
            processor=processor,
            device=device,
            model_id=model_id,
        )
    finally:
        logger.info("Shutting down, releasing model")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
