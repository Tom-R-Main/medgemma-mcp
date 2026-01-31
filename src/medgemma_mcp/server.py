"""MedGemma MCP Server entry point.

Local-first MCP server providing medical image analysis and clinical reasoning
via Google's MedGemma 4B-IT model. Communicates over stdio transport.

Usage:
    medgemma-mcp          # via installed entry point
    python -m medgemma_mcp.server   # direct execution
"""

import logging

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from medgemma_mcp.model.loader import medgemma_lifespan
from medgemma_mcp.tools.analyze_image import analyze_medical_image
from medgemma_mcp.tools.extract import extract_structured
from medgemma_mcp.tools.medical_reason import medical_reason
from medgemma_mcp.tools.summarize_fhir import summarize_fhir_record

logger = logging.getLogger(__name__)

# Create server with lifespan-managed model loading
mcp = FastMCP(
    name="medgemma-mcp",
    instructions=(
        "This server provides two tools: 'analyze_medical_image' for interpreting "
        "medical images, and 'medical_reason' for answering clinical questions. "
        "All outputs include confidence scores and require clinical review when "
        "confidence is below 0.7. Outputs are for research purposes only."
    ),
    lifespan=medgemma_lifespan,
)

# Register tools with annotations
mcp.add_tool(
    analyze_medical_image,
    name="analyze_medical_image",
    title="Analyze Medical Image",
    description=(
        "Analyze a medical image using MedGemma AI with chain-of-thought reasoning. "
        "Supports chest X-rays (88.9 F1), dermoscopy, fundus, histopathology, and CT. "
        "Returns structured findings with confidence score and review flag. "
        "Accepts file paths (JPEG, PNG, DICOM) or base64-encoded image data."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)

mcp.add_tool(
    medical_reason,
    name="medical_reason",
    title="Medical Reasoning",
    description=(
        "Answer a clinical question with structured chain-of-thought medical reasoning. "
        "Covers differential diagnosis, treatment considerations, and evidence-based analysis. "
        "Returns step-by-step reasoning with confidence score. "
        "For image analysis, use analyze_medical_image instead."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)

mcp.add_tool(
    summarize_fhir_record,
    name="summarize_fhir_record",
    title="Summarize FHIR Record",
    description=(
        "Summarize a FHIR R4 Bundle with clinical reasoning. "
        "Parses Patient, Condition, MedicationRequest, Observation, AllergyIntolerance, "
        "DiagnosticReport, and Procedure resources. Python handles all FHIR parsing; "
        "MedGemma reasons about the extracted clinical data."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)

mcp.add_tool(
    extract_structured,
    name="extract_structured",
    title="Extract Structured Data",
    description=(
        "Extract structured medical data from free-text clinical notes. "
        "Identifies and categorizes conditions, medications, allergies, procedures, "
        "vitals, lab results, symptoms, and family history. "
        "Returns both raw reasoning and parsed structured output."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)


def main() -> None:
    """Run the MedGemma MCP server on stdio transport."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    mcp.run()


if __name__ == "__main__":
    main()
