"""MCP tool: summarize_fhir_record

Summarizes a FHIR R4 Bundle into clinical insights using MedGemma.
Python handles all FHIR parsing; the model receives plain-text summaries only.
"""

import json
import logging

import anyio
from mcp.server.fastmcp import Context
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field

from medgemma_mcp.model.inference import run_text_inference
from medgemma_mcp.model.loader import MedGemmaContext
from medgemma_mcp.preprocessing.fhir import fhir_bundle_to_summary
from medgemma_mcp.prompts.templates import Modality, build_fhir_summary_prompt, get_system_prompt
from medgemma_mcp.safety.confidence import extract_confidence
from medgemma_mcp.safety.disclaimers import MEDICAL_DISCLAIMER

logger = logging.getLogger(__name__)


class FHIRSummaryResult(BaseModel):
    """Structured output from FHIR record summarization."""

    clinical_summary: str = Field(description="Plain-text clinical summary extracted from the FHIR Bundle")
    analysis: str = Field(description="MedGemma's clinical analysis and reasoning")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score")
    requires_review: bool = Field(description="True if confidence < 0.7 — needs clinical review")
    resource_counts: dict[str, int] = Field(description="Count of each FHIR resource type found")
    disclaimer: str = Field(description="Required medical disclaimer")


async def summarize_fhir_record(
    fhir_bundle: str,
    clinical_question: str = "Provide a comprehensive clinical summary with key concerns.",
    ctx: Context[ServerSession, MedGemmaContext] = None,  # type: ignore[assignment]
) -> FHIRSummaryResult:
    """Summarize a FHIR R4 Bundle with clinical reasoning.

    Parses the FHIR Bundle in Python (the model does not understand raw FHIR),
    extracts clinical data, and uses MedGemma for medical reasoning about the
    patient's record. Handles Patient, Condition, MedicationRequest, Observation,
    AllergyIntolerance, DiagnosticReport, and Procedure resources.

    Args:
        fhir_bundle: FHIR R4 Bundle as a JSON string.
        clinical_question: Specific question about the patient record.
    """
    # Parse the FHIR JSON
    await ctx.report_progress(0.1, 1.0, "Parsing FHIR Bundle...")
    try:
        bundle = json.loads(fhir_bundle)
    except json.JSONDecodeError as exc:
        raise ToolError(f"Invalid JSON in fhir_bundle: {exc}") from exc

    # Convert FHIR to clinical text
    try:
        clinical_summary = fhir_bundle_to_summary(bundle)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc

    # Count resources for metadata
    resource_counts: dict[str, int] = {}
    for entry in bundle.get("entry", []):
        rt = entry.get("resource", {}).get("resourceType", "Unknown")
        resource_counts[rt] = resource_counts.get(rt, 0) + 1

    # Get model from lifespan context
    model_ctx: MedGemmaContext = ctx.request_context.lifespan_context

    # Build prompt
    await ctx.report_progress(0.3, 1.0, "Building clinical reasoning prompt...")
    system_prompt = get_system_prompt(Modality.TEXT_ONLY)
    user_prompt = build_fhir_summary_prompt(clinical_summary, clinical_question)

    # Run inference
    await ctx.report_progress(0.4, 1.0, "Running MedGemma reasoning...")
    analysis = await anyio.to_thread.run_sync(
        lambda: run_text_inference(
            model=model_ctx.model,
            processor=model_ctx.processor,
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_new_tokens=1024,
        )
    )

    # Extract confidence
    await ctx.report_progress(0.9, 1.0, "Extracting confidence score...")
    confidence = extract_confidence(analysis)
    requires_review = confidence < 0.7

    if requires_review:
        await ctx.warning(f"Low confidence ({confidence:.2f}) — clinical review recommended")

    return FHIRSummaryResult(
        clinical_summary=clinical_summary,
        analysis=analysis,
        confidence=confidence,
        requires_review=requires_review,
        resource_counts=resource_counts,
        disclaimer=MEDICAL_DISCLAIMER,
    )
