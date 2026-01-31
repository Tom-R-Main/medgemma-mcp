# medgemma-mcp

Local-first MCP server for medical AI inference using Google's MedGemma 4B-IT.

## Tools

- **analyze_medical_image** — Analyze chest X-rays, CT, dermoscopy, fundus, and histopathology images
- **medical_reason** — Answer clinical questions with structured chain-of-thought reasoning

## Quick Start

```bash
pip install -e .
medgemma-mcp  # starts stdio MCP server
```

## Requirements

- Python >= 3.10
- GPU with >= 8GB VRAM (CUDA) for MedGemma 4B at BF16
- HuggingFace access to `google/medgemma-4b-it`

## Architecture

```
src/medgemma_mcp/
├── server.py           # MCP server entry point
├── model/              # Model loading (lifespan) and inference
├── tools/              # MCP tool implementations
├── prompts/            # CoT prompt templates per modality
├── preprocessing/      # DICOM→PIL, image loading
└── safety/             # Confidence extraction, disclaimers
```
