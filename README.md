# medgemma-mcp

Local-first MCP server for medical AI inference using Google's MedGemma 4B-IT. Provides medical image analysis, clinical reasoning, FHIR record summarization, and structured data extraction as composable MCP tools.

Built for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge) Agentic Workflow Prize.

## Tools

| Tool | Description | Input |
|------|-------------|-------|
| **analyze_medical_image** | Analyze chest X-rays, CT, dermoscopy, fundus, and histopathology | Image (file/base64/DICOM) + modality |
| **medical_reason** | Clinical Q&A with chain-of-thought reasoning | Text question |
| **summarize_fhir_record** | Summarize FHIR R4 Bundles with clinical reasoning | FHIR JSON string |
| **extract_structured** | Extract structured data from clinical free text | Clinical note text |

All tools return structured output with confidence scores. Findings with confidence < 0.7 are flagged for clinical review.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Tom-R-Main/medgemma-mcp
cd medgemma-mcp
pip install -e ".[dev]"

# 2. Authenticate with HuggingFace (one-time)
#    Accept the license at https://huggingface.co/google/medgemma-4b-it
huggingface-cli login

# 3. Run smoke test (validates pipeline without model)
python scripts/smoke_test.py --skip-model

# 4. Full smoke test (loads model — first run downloads ~8GB)
python scripts/smoke_test.py

# 5. Start the MCP server
medgemma-mcp
```

## Hardware Requirements

| Device | Memory | dtype | Status |
|--------|--------|-------|--------|
| NVIDIA GPU | >= 8GB VRAM | bfloat16 | Recommended |
| Apple Silicon (M1/M2/M3/M4) | >= 16GB unified | float32 | Supported (MPS) |
| CPU | >= 16GB RAM | float32 | Supported (slow) |

> **Note:** MPS requires float32 — float16 causes numerical instability with MedGemma on Apple Silicon. float32 uses ~16GB for the model, fitting comfortably on 24GB+ Macs.

The server auto-detects your hardware and selects the optimal configuration.

## Claude Desktop Integration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "medgemma": {
      "command": "medgemma-mcp"
    }
  }
}
```

Restart Claude Desktop. The tools will appear in Claude's tool list.

## Architecture

```
src/medgemma_mcp/
├── server.py              # FastMCP server entry point (stdio transport)
├── model/
│   ├── loader.py          # Lifespan-managed model loading
│   └── inference.py       # Inference wrapper (text + multimodal)
├── tools/
│   ├── analyze_image.py   # analyze_medical_image tool
│   ├── medical_reason.py  # medical_reason tool
│   ├── summarize_fhir.py  # summarize_fhir_record tool
│   └── extract.py         # extract_structured tool
├── prompts/
│   └── templates.py       # CoT prompt templates per modality (86% hallucination reduction)
├── preprocessing/
│   ├── images.py          # Image loading (file/base64/data URI)
│   ├── dicom.py           # DICOM → PIL conversion with windowing
│   └── fhir.py            # FHIR R4 Bundle → clinical text summary
└── safety/
    ├── confidence.py      # Confidence score extraction
    └── disclaimers.py     # Regulatory disclaimers
```

## Key Design Decisions

- **Chain-of-thought prompting** reduces MedGemma hallucinations by 86.4% — templates are load-bearing infrastructure, not polish
- **MedGemma 4B was NOT trained on FHIR** (67.6% vs base Gemma's 70.9% on EHRQA) — all FHIR parsing happens in Python
- **Single-turn only** — multi-turn is "not evaluated/optimized" per Google; each tool call is stateless
- **Conservative confidence** — low-confidence keywords are checked first; uncertain findings flag for review

## Development

```bash
# Tests (62 tests)
pytest tests/ -v

# Lint + format
ruff check src/ tests/
ruff format src/ tests/

# Type check
pyright src/
```

## Disclaimer

This tool is for **research and development purposes only**. It is not FDA-cleared for clinical diagnosis. All outputs require independent verification by qualified healthcare professionals.

## License

Apache-2.0
