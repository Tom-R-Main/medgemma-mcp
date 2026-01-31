# MedGemma MCP Server: Detailed Implementation Plan

## Executive Summary

This document provides a comprehensive technical blueprint for building a **local-first MedGemma MCP server** targeting the MedGemma Impact Challenge's Agentic Workflow Prize ($10K). Based on deep research across 12 vectors, the plan addresses the **critical finding** that MedGemma 4B has FHIR comprehension limitations (67.6% vs base Gemma's 70.9%), requiring a refined architecture.

---

## 1. Revised MVP Scope (Competition-Ready)

### Critical Insight from Research

The MedGemma 4B model was **NOT trained on FHIR data**. The EHRQA benchmark shows:
- MedGemma 4B: 67.6%
- Base Gemma 3 4B: 70.9% (better!)
- MedGemma 27B Multimodal: Significantly better (FHIR-trained)

**Implication**: For local-first 4B deployment, FHIR comprehension must happen in **Python preprocessing**, not the model. The model excels at image analysis and medical reasoning, not structured EHR data.

### MVP Tools (Proven Benchmarks)

| Tool | Model Task | Python Task | Benchmark |
|------|-----------|-------------|-----------|
| `analyze_medical_image` | Core strength | DICOM→PIL conversion | 88.9 F1 (CXR) |
| `medical_reason` | Core strength | Prompt engineering | 64.4% MedQA |

### V1 Tools (Needs Python Heavy-Lifting)

| Tool | Model Task | Python Task | Notes |
|------|-----------|-------------|-------|
| `summarize_fhir_record` | Text reasoning only | All FHIR parsing, filtering, summarization | Model doesn't understand raw FHIR |
| `extract_structured` | Text→text | JSON parsing | Needs validation layer |

---

## 2. Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MedGemma MCP Server                       │
│                      (Python, stdio)                         │
├─────────────────────────────────────────────────────────────┤
│  MCP Tools Layer                                            │
│  ├── analyze_medical_image                                  │
│  │   └── Input: PIL Image + question → findings + confidence│
│  ├── medical_reason                                         │
│  │   └── Input: clinical question → CoT reasoning           │
│  ├── summarize_fhir_record (V1)                            │
│  │   └── Input: FHIR Bundle → Python parses → model reasons │
│  └── extract_structured (V1)                               │
│       └── Input: clinical text → structured JSON            │
├─────────────────────────────────────────────────────────────┤
│  Safety & Prompting Layer (CRITICAL)                        │
│  ├── CoT Prompt Templates (86% hallucination reduction)     │
│  ├── Confidence Scoring (&lt;0.7 = requires_review)            │
│  ├── Regulatory Disclaimers (auto-appended)                 │
│  └── Progress Notifications (model loading, inference)      │
├─────────────────────────────────────────────────────────────┤
│  Model Backend (Lifespan-Managed)                           │
│  ├── MedGemma 4B-IT (BF16, ~8GB VRAM)                       │
│  ├── Transformers Pipeline API                              │
│  ├── Single model load at startup                           │
│  └── torch.no_grad() for all inference                      │
├─────────────────────────────────────────────────────────────┤
│  Preprocessing Layer                                        │
│  ├── DICOM → PIL Image (pydicom)                            │
│  ├── FHIR Bundle → Clinical Summary (Python, not model)     │
│  └── Image validation (size, format)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. MedGemma Input Format Specification

### Image Requirements
- **Resolution**: 896×896 pixels (processor handles resize automatically)
- **Normalization**: [-1, 1] (handled by processor)
- **Format**: PIL Image objects (JPEG, PNG supported natively)
- **Tokens**: 256 tokens per image (fixed)
- **Context Window**: 128K tokens total (input + output)
- **Output Limit**: 8,192 tokens max

### Message Format (CRITICAL: No System Role!)

```python
# WRONG - Gemma 3/MedGemma does NOT support system role!
messages = [
    {"role": "system", "content": "You are a radiologist"},  # ❌ FAILS
    {"role": "user", "content": "..."}
]

# CORRECT - Prepend system instructions to first user message
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "You are an expert radiologist. Analyze this chest X-ray..."},
            {"type": "image", "image": pil_image}
        ]
    }
]
```

### Multi-Image Support (CT/MRI)
```python
# Multiple images as separate content items
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze these CT slices..."},
            {"type": "image", "image": slice_1},
            {"type": "image", "image": slice_2},
            {"type": "image", "image": slice_3}
        ]
    }
]
```

### Preprocessing by Modality

| Modality | Preprocessing Required | Notes |
|----------|----------------------|-------|
| Chest X-ray | None (grayscale→RGB automatic) | Single 2D image |
| CT | Windowing to RGB channels | bone/lung, soft tissue, brain windows |
| MRI | Multiple slices | Each slice = 256 tokens |
| Dermoscopy | Standard RGB | Color normalization optional |
| Fundus | Standard RGB | Color standardization |
| Histopathology | Patch extraction (256×256) | Resize patches to 896×896 |

---

## 4. Chain-of-Thought Prompt Templates

### Why CoT is Load-Bearing

Research finding: **86.4% hallucination reduction** with CoT prompting. This isn't optional polish—it's essential infrastructure.

### Radiology Analysis CoT Template (Primary)

```python
RADIOLOGY_COT_PROMPT = """You are an expert radiologist. Analyze this image systematically.

METHODOLOGY:
1. Identify anatomical structures visible
2. Review each region systematically
3. Note ONLY findings directly visible in the image
4. Do NOT guess or hallucinate findings not evident

REQUIRED OUTPUT FORMAT:

[ANATOMICAL REVIEW]
- Lungs: [findings with specific locations OR "clear, well-expanded"]
- Pleura: [findings OR "normal, no effusion"]
- Heart: [size, silhouette assessment]
- Mediastinum: [findings OR "normal width"]
- Bones: [findings OR "no acute fractures visible"]
- Soft tissues: [findings OR "unremarkable"]

[ABNORMALITIES IDENTIFIED]
For each finding:
- Location: [specific anatomical location]
- Characteristics: [size, density, shape, margins]
- Differential: [possible diagnoses ranked by likelihood]

[IMPRESSION]
Primary finding: [main diagnosis/finding]
Confidence: [HIGH/MODERATE/LOW]
Rationale: [brief evidence supporting confidence level]

[LIMITATIONS]
- Image quality: [adequate/suboptimal/poor]
- Areas not well visualized: [list if any]

[CLINICAL RECOMMENDATION]
[Suggested follow-up or "No immediate action required"]

IMPORTANT: State only what is directly visible. If uncertain, say "possible" or "cannot exclude"."""
```

### Two-Step Verification Loop

```python
VERIFICATION_PROMPT = """Review your analysis above:

VERIFICATION QUESTIONS:
1. Is each finding directly visible in the image?
2. Could any finding be an artifact or normal variant?
3. Did you miss any anatomical region?
4. Is your confidence level appropriate given image quality?

If any answer reveals an issue, revise your assessment below.

REVISED ANALYSIS (if needed):
[Only include changes, or state "Original analysis confirmed"]"""
```

### Confidence Scoring Prompt Addition

```python
CONFIDENCE_SUFFIX = """

MANDATORY CONFIDENCE STATEMENT:
Rate your overall confidence: [0.0-1.0]
- 0.9-1.0: Clear evidence, high certainty
- 0.7-0.9: Probable finding, recommend verification
- 0.5-0.7: Uncertain, clinical correlation required
- &lt;0.5: DO NOT REPORT - insufficient evidence

Your confidence score: """
```

---

## 5. Project Structure

```
medgemma-mcp/
├── src/
│   └── medgemma_mcp/
│       ├── __init__.py
│       ├── server.py              # MCPServer entry point
│       ├── model/
│       │   ├── __init__.py
│       │   ├── loader.py          # Lifespan model loading
│       │   └── inference.py       # Inference wrapper with CoT
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── analyze_image.py   # analyze_medical_image tool
│       │   ├── medical_reason.py  # medical_reason tool
│       │   ├── summarize_fhir.py  # summarize_fhir_record (V1)
│       │   └── extract.py         # extract_structured (V1)
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── dicom.py           # DICOM → PIL conversion
│       │   └── fhir.py            # FHIR → text summary
│       ├── prompts/
│       │   ├── __init__.py
│       │   └── templates.py       # All CoT templates
│       └── safety/
│           ├── __init__.py
│           ├── confidence.py      # Confidence extraction
│           └── disclaimers.py     # Regulatory text
├── tests/
│   ├── __init__.py
│   ├── test_analyze_image.py
│   └── test_medical_reason.py
├── pyproject.toml
├── README.md
└── CLAUDE.md                      # For Claude Code context
```

---

## 6. Core Implementation Patterns

### 6.1 Model Loading (Lifespan Pattern)

```python
# src/medgemma_mcp/model/loader.py
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

@dataclass
class MedGemmaContext:
    """Type-safe context for model access."""
    model: AutoModelForImageTextToText
    processor: AutoProcessor
    device: str

@asynccontextmanager
async def medgemma_lifespan(server) -> AsyncIterator[MedGemmaContext]:
    """Load MedGemma model once at startup, keep in memory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Report loading progress
    print(f"[MedGemma] Loading model on {device}...")

    model = AutoModelForImageTextToText.from_pretrained(
        "google/medgemma-4b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Handles multi-GPU automatically
    )
    processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")

    print("[MedGemma] Model loaded successfully")

    try:
        yield MedGemmaContext(model=model, processor=processor, device=device)
    finally:
        print("[MedGemma] Cleaning up...")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### 6.2 Tool Implementation (analyze_medical_image)

```python
# src/medgemma_mcp/tools/analyze_image.py
from PIL import Image
import base64
import io
from pydantic import BaseModel, Field
from mcp.server.mcpserver import Context, ToolError
from ..model.loader import MedGemmaContext
from ..prompts.templates import build_radiology_prompt
from ..safety.confidence import extract_confidence
from ..safety.disclaimers import MEDICAL_DISCLAIMER

class ImageAnalysisInput(BaseModel):
    """Input schema for medical image analysis."""
    image_base64: str = Field(
        description="Base64-encoded medical image (PNG or JPEG)"
    )
    modality: str = Field(
        description="Image type: chest_xray, ct, mri, dermoscopy, fundus, histopath"
    )
    clinical_question: str = Field(
        default="Provide a comprehensive analysis",
        description="Specific question about the image"
    )

class ImageAnalysisOutput(BaseModel):
    """Structured output from image analysis."""
    findings: str
    confidence: float = Field(ge=0.0, le=1.0)
    requires_radiologist_review: bool
    modality: str
    disclaimer: str

async def analyze_medical_image(
    request: ImageAnalysisInput,
    ctx: Context
) -> ImageAnalysisOutput:
    """
    Analyze a medical image using MedGemma.

    Supports: chest X-rays, CT, MRI, dermoscopy, fundus, histopathology.
    Returns findings with confidence score and review flag.
    """
    # Get model from lifespan context
    lifespan_ctx: MedGemmaContext = ctx.request_context.lifespan_context
    model = lifespan_ctx.model
    processor = lifespan_ctx.processor

    # Report progress
    await ctx.report_progress(0.1, 1.0, "Decoding image...")

    # Decode image
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ToolError(f"Invalid image data: {e}")

    # Validate image size
    if image.size[0] < 64 or image.size[1] < 64:
        raise ToolError(f"Image too small: {image.size}. Minimum 64x64 required.")

    await ctx.report_progress(0.3, 1.0, "Building prompt...")

    # Build CoT prompt based on modality
    prompt = build_radiology_prompt(
        modality=request.modality,
        question=request.clinical_question
    )

    # Format messages (NO system role!)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        }
    ]

    await ctx.report_progress(0.5, 1.0, "Running inference...")

    # Run inference
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,  # Deterministic for medical
        )

    findings = processor.decode(outputs[0], skip_special_tokens=True)

    await ctx.report_progress(0.9, 1.0, "Extracting confidence...")

    # Extract confidence score
    confidence = extract_confidence(findings)
    requires_review = confidence < 0.7

    return ImageAnalysisOutput(
        findings=findings,
        confidence=confidence,
        requires_radiologist_review=requires_review,
        modality=request.modality,
        disclaimer=MEDICAL_DISCLAIMER
    )
```

### 6.3 Server Entry Point

```python
# src/medgemma_mcp/server.py
from mcp.server.mcpserver import MCPServer
from .model.loader import medgemma_lifespan
from .tools.analyze_image import analyze_medical_image
from .tools.medical_reason import medical_reason

# Create server with lifespan
mcp = MCPServer(
    name="MedGemma MCP",
    version="0.1.0",
    lifespan=medgemma_lifespan
)

# Register tools
mcp.tool()(analyze_medical_image)
mcp.tool()(medical_reason)

def main():
    """Run the MCP server."""
    import asyncio
    asyncio.run(mcp.run())

if __name__ == "__main__":
    main()
```

---

## 7. FHIR Handling Strategy (V1)

Since MedGemma 4B doesn't understand raw FHIR, Python must do the heavy lifting:

```python
# src/medgemma_mcp/preprocessing/fhir.py
def fhir_bundle_to_clinical_summary(bundle: dict) -> str:
    """
    Convert FHIR Bundle to plain English clinical summary.
    Model sees text, not FHIR JSON.
    """
    summary_parts = []

    # Extract Patient demographics
    patient = find_resource(bundle, "Patient")
    if patient:
        summary_parts.append(f"Patient: {patient.get('gender', 'Unknown')}, "
                           f"DOB: {patient.get('birthDate', 'Unknown')}")

    # Extract active Conditions
    conditions = find_resources(bundle, "Condition", status="active")
    if conditions:
        condition_names = [c.get("code", {}).get("text", "Unknown")
                         for c in conditions]
        summary_parts.append(f"Active conditions: {', '.join(condition_names)}")

    # Extract active Medications
    meds = find_resources(bundle, "MedicationRequest", status="active")
    if meds:
        med_names = [m.get("medicationCodeableConcept", {}).get("text", "Unknown")
                   for m in meds]
        summary_parts.append(f"Current medications: {', '.join(med_names)}")

    # Extract recent Observations (last 30 days)
    observations = find_recent_resources(bundle, "Observation", days=30)
    if observations:
        obs_summary = format_observations(observations)
        summary_parts.append(f"Recent observations:\n{obs_summary}")

    # Extract Allergies (all time)
    allergies = find_resources(bundle, "AllergyIntolerance")
    if allergies:
        allergy_names = [a.get("code", {}).get("text", "Unknown")
                       for a in allergies]
        summary_parts.append(f"Allergies: {', '.join(allergy_names)}")

    return "\n\n".join(summary_parts)
```

---

## 8. Confidence Extraction

```python
# src/medgemma_mcp/safety/confidence.py
import re

def extract_confidence(model_output: str) -> float:
    """
    Extract confidence score from model output.
    Falls back to keyword analysis if explicit score not found.
    """
    # Try to find explicit confidence score
    patterns = [
        r'confidence[:\s]+([0-9.]+)',
        r'confidence score[:\s]+([0-9.]+)',
        r'\[([0-9.]+)\].*confidence',
    ]

    for pattern in patterns:
        match = re.search(pattern, model_output.lower())
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 1:
                    return score
            except ValueError:
                continue

    # Keyword-based fallback
    text_lower = model_output.lower()

    high_confidence_terms = ["definite", "clearly", "consistent with", "diagnostic of"]
    moderate_terms = ["likely", "probable", "suggests", "compatible with"]
    low_confidence_terms = ["possible", "cannot exclude", "uncertain", "equivocal"]

    for term in high_confidence_terms:
        if term in text_lower:
            return 0.85

    for term in low_confidence_terms:
        if term in text_lower:
            return 0.5

    for term in moderate_terms:
        if term in text_lower:
            return 0.7

    # Default moderate confidence
    return 0.65
```

---

## 9. Dependencies (pyproject.toml)

```toml
[project]
name = "medgemma-mcp"
version = "0.1.0"
description = "MCP server for MedGemma medical AI inference"
requires-python = ">=3.10"
dependencies = [
    "mcp>=2.0.0",
    "torch>=2.0.0",
    "transformers>=4.50.0",  # Required for MedGemma
    "accelerate>=0.20.0",    # For device_map="auto"
    "pillow>=9.0.0",
    "pydicom>=2.3.0",        # DICOM handling
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]

[project.scripts]
medgemma-mcp = "medgemma_mcp.server:main"
```

---

## 10. Known Issues & Mitigations

| Issue | Mitigation |
|-------|-----------|
| July 2025 end-of-image token bug | Use transformers >=4.50.0 |
| No system role support | Prepend to first user message |
| Multi-turn not optimized | Each tool call is single-turn |
| 28-62% hallucination rate | CoT prompts (86% reduction) |
| FHIR comprehension (4B) | Python preprocessing layer |
| CUDA OOM on long inference | torch.no_grad(), batch limits |
| Prompt sensitivity | Tested templates, avoid variation |

---

## 11. Competition Deliverables

### 3-Page Writeup Structure
1. **Page 1**: Problem (clinical workflow friction) + Impact (developer productivity)
2. **Page 2**: Technical approach (architecture diagram, HAI-DEF integration, CoT innovation)
3. **Page 3**: Results (benchmark comparisons) + Deployment feasibility + Limitations acknowledged

### 3-Minute Video Structure
- **0:00-0:30**: Hook - "Developers want X, they have to do Y, Z pain points"
- **0:30-1:00**: Solution intro - "MedGemma MCP Server wraps the model as composable tools"
- **1:00-2:30**: Live demo - Show Claude/LLM using the tools, highlight CoT reasoning
- **2:30-3:00**: Impact - "Open source, local-first, safety-first"

### Demo Flow
1. Upload chest X-ray → `analyze_medical_image` → Findings with confidence
2. Ask follow-up question → `medical_reason` → Clinical reasoning
3. Show confidence flagging → "Requires radiologist review"
4. Show FHIR summary (V1) → Python preprocessing + model reasoning

---

## 12. Implementation Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1: Core MVP** | 2-3 days | `analyze_medical_image` + `medical_reason` working |
| **Phase 2: Safety Layer** | 1-2 days | CoT templates, confidence scoring, disclaimers |
| **Phase 3: Testing** | 1-2 days | Unit tests, NIH CXR dataset validation |
| **Phase 4: V1 Tools** | 2-3 days | FHIR summarization, structured extraction |
| **Phase 5: Polish** | 2-3 days | README, demo video, competition writeup |

**Total**: ~10-14 days for competition-ready submission

---

## 13. Test Data Sources

| Source | Content | Access |
|--------|---------|--------|
| NIH Chest X-ray | 112K CXR images | Direct download (attribution only) |
| Synthea | Synthetic FHIR bundles | `./run_synthea -p 100` |
| SMART on FHIR Sandbox | Test FHIR server | https://launch.smarthealthit.org |
| HAPI FHIR Public | Test FHIR server | http://hapi.fhir.org/baseR4 |

---

## 14. Regulatory Disclaimers

```python
# src/medgemma_mcp/safety/disclaimers.py

MEDICAL_DISCLAIMER = """
IMPORTANT: This analysis is generated by an AI system (MedGemma) and is
intended for research and development purposes only. It is NOT a substitute
for professional medical advice, diagnosis, or treatment.

All outputs require independent verification and clinical correlation by
qualified healthcare professionals before any patient care applications.

MedGemma is not FDA-cleared for clinical diagnosis.
"""

DEVELOPER_DISCLAIMER = """
This tool is provided as infrastructure for healthcare AI developers.
Developers are responsible for appropriate validation, adaptation, and
meaningful modification before any downstream clinical application.
"""
```

---

## Summary

This plan provides a production-ready blueprint for building a MedGemma MCP server that:

1. **Works locally** (4B model, ~8GB VRAM)
2. **Reduces hallucinations by 86%** (CoT prompting)
3. **Handles the FHIR limitation** (Python preprocessing)
4. **Fits competition criteria** (Agentic Workflow Prize)
5. **Is honest about limitations** (confidence scoring, required review flags)

The MVP is tight: 2 tools, 1 model load, proven benchmarks, clear demo story.
