"""Inference wrapper for MedGemma.

Provides a clean interface for running inference with MedGemma,
handling message formatting, tokenization, and generation.
"""

import logging

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)


def run_image_inference(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    image: Image.Image,
    prompt: str,
    system_prompt: str | None = None,
    max_new_tokens: int = 1024,
) -> str:
    """Run multimodal inference with a single image.

    Blocking call — should be run in a thread pool from async context.

    Args:
        model: Loaded MedGemma model.
        processor: Loaded MedGemma processor.
        image: PIL Image (any mode, processor handles conversion).
        prompt: User prompt text.
        system_prompt: Optional system instructions.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated text response.
    """
    messages = _build_multimodal_messages(prompt, image, system_prompt)
    return _generate(model, processor, messages, max_new_tokens)


def run_text_inference(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    prompt: str,
    system_prompt: str | None = None,
    max_new_tokens: int = 1024,
) -> str:
    """Run text-only inference (no image).

    Blocking call — should be run in a thread pool from async context.

    Args:
        model: Loaded MedGemma model.
        processor: Loaded MedGemma processor.
        prompt: User prompt text.
        system_prompt: Optional system instructions.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated text response.
    """
    messages = _build_text_messages(prompt, system_prompt)
    return _generate(model, processor, messages, max_new_tokens)


def _build_multimodal_messages(
    prompt: str,
    image: Image.Image,
    system_prompt: str | None = None,
) -> list[dict]:
    """Build the chat messages list for multimodal input.

    MedGemma supports system role per the official model card.
    Images are passed as PIL objects in the content list.
    """
    messages: list[dict] = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        }
    )

    return messages


def _build_text_messages(
    prompt: str,
    system_prompt: str | None = None,
) -> list[dict]:
    """Build the chat messages list for text-only input."""
    messages: list[dict] = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    )

    return messages


def _generate(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    messages: list[dict],
    max_new_tokens: int,
) -> str:
    """Run generation and decode output.

    Uses greedy decoding (do_sample=False) for deterministic medical outputs.
    """
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the generated tokens (skip the input prompt)
    generated_ids = output_ids[0][input_len:]
    return processor.decode(generated_ids, skip_special_tokens=True)
