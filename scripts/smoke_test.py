#!/usr/bin/env python3
"""End-to-end smoke test for medgemma-mcp.

Validates the full pipeline: model loading, text inference, and
optionally image inference. Run this after installation to verify
everything works on your hardware.

Usage:
    python scripts/smoke_test.py                    # text-only test
    python scripts/smoke_test.py --image /path.jpg  # image test too

Requirements:
    - HuggingFace access to google/medgemma-4b-it
    - Sufficient memory (~8GB for the model)
    - Run `huggingface-cli login` first if you haven't accepted the license
"""

import argparse
import sys
import time


def check_dependencies() -> bool:
    """Verify required packages are importable."""
    print("Checking dependencies...")
    missing = []
    for pkg in ["torch", "transformers", "PIL", "mcp"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"  MISSING: {', '.join(missing)}")
        print("  Run: pip install -e .")
        return False

    print("  All dependencies available.")
    return True


def check_device() -> str:
    """Detect and report compute device."""
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  Device: {device} ({name}, {mem:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print(f"  Device: {device} (Apple Silicon GPU)")
    else:
        device = "cpu"
        print(f"  Device: {device} (will be slow)")

    return device


def test_model_loading() -> tuple:
    """Load MedGemma model and processor."""
    from medgemma_mcp.model.loader import DEFAULT_MODEL_ID, _detect_device, _load_model

    print(f"\nLoading model: {DEFAULT_MODEL_ID}")
    device, dtype = _detect_device()
    print(f"  Device: {device}, dtype: {dtype}")

    start = time.time()
    model, processor, device = _load_model(DEFAULT_MODEL_ID)
    elapsed = time.time() - start
    print(f"  Model loaded in {elapsed:.1f}s")

    return model, processor


def test_text_inference(model, processor) -> None:
    """Run a simple text inference."""
    from medgemma_mcp.model.inference import run_text_inference
    from medgemma_mcp.prompts.templates import Modality, build_text_prompt, get_system_prompt

    print("\nRunning text inference...")
    question = "What are the most common causes of chest pain in adults?"
    system_prompt = get_system_prompt(Modality.TEXT_ONLY)
    user_prompt = build_text_prompt(question)

    start = time.time()
    result = run_text_inference(
        model=model,
        processor=processor,
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_new_tokens=256,
    )
    elapsed = time.time() - start

    print(f"  Generated {len(result)} chars in {elapsed:.1f}s")
    print(f"  First 200 chars: {result[:200]}...")

    # Basic sanity checks
    assert len(result) > 50, "Output too short — model may not be generating correctly"
    print("  Text inference: PASS")


def test_image_inference(model, processor, image_path: str) -> None:
    """Run image inference on a provided image."""
    from medgemma_mcp.model.inference import run_image_inference
    from medgemma_mcp.preprocessing.images import load_image
    from medgemma_mcp.prompts.templates import Modality, build_image_prompt, get_system_prompt

    print(f"\nRunning image inference on: {image_path}")

    image = load_image(image_path)
    print(f"  Image loaded: {image.size} {image.mode}")

    system_prompt = get_system_prompt(Modality.CHEST_XRAY)
    user_prompt = build_image_prompt(Modality.CHEST_XRAY, "Analyze this image for any abnormalities.")

    start = time.time()
    result = run_image_inference(
        model=model,
        processor=processor,
        image=image,
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_new_tokens=512,
    )
    elapsed = time.time() - start

    print(f"  Generated {len(result)} chars in {elapsed:.1f}s")
    print(f"  First 200 chars: {result[:200]}...")

    assert len(result) > 50, "Output too short"
    print("  Image inference: PASS")


def test_fhir_parsing() -> None:
    """Test FHIR bundle parsing (no model needed)."""
    from medgemma_mcp.preprocessing.fhir import fhir_bundle_to_summary

    print("\nTesting FHIR parsing...")
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "name": [{"given": ["Test"], "family": "Patient"}],
                    "gender": "male",
                    "birthDate": "1960-01-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "code": {"text": "Hypertension"},
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                }
            },
        ],
    }

    summary = fhir_bundle_to_summary(bundle)
    assert "Test Patient" in summary
    assert "Hypertension" in summary
    print(f"  Summary: {summary[:100]}...")
    print("  FHIR parsing: PASS")


def test_confidence_extraction() -> None:
    """Test confidence extraction (no model needed)."""
    from medgemma_mcp.safety.confidence import extract_confidence

    print("\nTesting confidence extraction...")
    assert extract_confidence("Confidence: 0.85") == 0.85
    assert extract_confidence("This clearly shows a fracture") == 0.85
    assert extract_confidence("Cannot exclude artifact") == 0.45
    assert extract_confidence("Normal findings") == 0.60
    print("  Confidence extraction: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="MedGemma MCP smoke test")
    parser.add_argument("--image", help="Path to a test image (optional)")
    parser.add_argument("--skip-model", action="store_true", help="Skip model loading (test parsing only)")
    args = parser.parse_args()

    print("=" * 60)
    print("MedGemma MCP Smoke Test")
    print("=" * 60)

    if not check_dependencies():
        sys.exit(1)

    device = check_device()

    # Tests that don't require model loading
    test_fhir_parsing()
    test_confidence_extraction()

    if args.skip_model:
        print("\n--skip-model: Skipping model-dependent tests.")
        print("\nAll non-model tests PASSED.")
        return

    # Model-dependent tests
    try:
        model, processor = test_model_loading()
    except Exception as exc:
        print(f"\n  Model loading FAILED: {exc}")
        print("\n  Common fixes:")
        print("    - Run: huggingface-cli login")
        print("    - Accept license at: https://huggingface.co/google/medgemma-4b-it")
        print(f"    - Ensure sufficient memory (need ~8GB, device: {device})")
        sys.exit(1)

    test_text_inference(model, processor)

    if args.image:
        test_image_inference(model, processor, args.image)
    else:
        print("\n  Skipping image test (no --image provided)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
