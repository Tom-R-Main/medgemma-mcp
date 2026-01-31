"""MCP tool: extract_structured

Extracts structured medical data from free-text clinical notes using MedGemma.
Returns categorized findings (conditions, medications, allergies, vitals, etc.)
with confidence scoring.
"""

import logging
import re

import anyio
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field

from medgemma_mcp.model.inference import run_text_inference
from medgemma_mcp.model.loader import MedGemmaContext
from medgemma_mcp.prompts.templates import Modality, build_extraction_prompt, get_system_prompt
from medgemma_mcp.safety.confidence import extract_confidence
from medgemma_mcp.safety.disclaimers import MEDICAL_DISCLAIMER

logger = logging.getLogger(__name__)


class ExtractedData(BaseModel):
    """Structured clinical data extracted from free text."""

    conditions: list[str] = Field(default_factory=list, description="Conditions and diagnoses")
    medications: list[str] = Field(default_factory=list, description="Medications with dosage")
    allergies: list[str] = Field(default_factory=list, description="Allergies and intolerances")
    procedures: list[str] = Field(default_factory=list, description="Procedures mentioned")
    vitals: list[str] = Field(default_factory=list, description="Vital signs with values")
    lab_results: list[str] = Field(default_factory=list, description="Lab results with values")
    symptoms: list[str] = Field(default_factory=list, description="Symptoms reported")
    family_history: list[str] = Field(default_factory=list, description="Family history items")


class ExtractionResult(BaseModel):
    """Structured output from clinical text extraction."""

    raw_reasoning: str = Field(description="Full MedGemma reasoning and extraction output")
    extracted: ExtractedData = Field(description="Parsed structured data")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score")
    requires_review: bool = Field(description="True if confidence < 0.7 — needs clinical review")
    disclaimer: str = Field(description="Required medical disclaimer")


async def extract_structured(
    clinical_text: str,
    clinical_question: str = "Extract all structured medical information from this text.",
    ctx: Context[ServerSession, MedGemmaContext] = None,  # type: ignore[assignment]
) -> ExtractionResult:
    """Extract structured medical data from clinical free text.

    Uses MedGemma to identify and categorize medical entities from unstructured
    clinical notes, discharge summaries, or reports. Extracts conditions,
    medications, allergies, procedures, vitals, lab results, symptoms, and
    family history.

    Args:
        clinical_text: Free-text clinical note, report, or discharge summary.
        clinical_question: Optional specific extraction focus.
    """
    if not clinical_text.strip():
        from mcp.server.fastmcp.exceptions import ToolError

        raise ToolError("clinical_text cannot be empty")

    # Get model from lifespan context
    model_ctx: MedGemmaContext = ctx.request_context.lifespan_context

    # Build prompt
    await ctx.report_progress(0.1, 1.0, "Preparing extraction prompt...")
    system_prompt = get_system_prompt(Modality.TEXT_ONLY)
    user_prompt = build_extraction_prompt(clinical_text, clinical_question)

    # Run inference
    await ctx.report_progress(0.3, 1.0, "Running MedGemma extraction...")
    raw_output = await anyio.to_thread.run_sync(
        lambda: run_text_inference(
            model=model_ctx.model,
            processor=model_ctx.processor,
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_new_tokens=1024,
        )
    )

    # Parse structured data from model output
    await ctx.report_progress(0.8, 1.0, "Parsing extracted data...")
    extracted = _parse_extraction_output(raw_output)

    # Extract confidence
    await ctx.report_progress(0.9, 1.0, "Extracting confidence score...")
    confidence = extract_confidence(raw_output)
    requires_review = confidence < 0.7

    if requires_review:
        await ctx.warning(f"Low confidence ({confidence:.2f}) — clinical review recommended")

    return ExtractionResult(
        raw_reasoning=raw_output,
        extracted=extracted,
        confidence=confidence,
        requires_review=requires_review,
        disclaimer=MEDICAL_DISCLAIMER,
    )


def _parse_extraction_output(text: str) -> ExtractedData:
    """Parse model output into structured ExtractedData.

    Looks for labeled sections (CONDITIONS:, MEDICATIONS:, etc.) and
    extracts list items from each.
    """
    categories = {
        "conditions": _extract_section(text, "CONDITIONS"),
        "medications": _extract_section(text, "MEDICATIONS"),
        "allergies": _extract_section(text, "ALLERGIES"),
        "procedures": _extract_section(text, "PROCEDURES"),
        "vitals": _extract_section(text, "VITALS"),
        "lab_results": _extract_section(text, "LAB_RESULTS"),
        "symptoms": _extract_section(text, "SYMPTOMS"),
        "family_history": _extract_section(text, "FAMILY_HISTORY"),
    }

    return ExtractedData(**categories)


def _extract_section(text: str, section_name: str) -> list[str]:
    """Extract items from a labeled section in the model output.

    Handles formats like:
        CONDITIONS: [item1, item2]
        CONDITIONS:
        - item1
        - item2
        CONDITIONS: item1, item2
    """
    # Pattern: SECTION_NAME: followed by content until next section or end
    next_sections = (
        "CONDITIONS|MEDICATIONS|ALLERGIES|PROCEDURES|VITALS|LAB_RESULTS|SYMPTOMS|FAMILY_HISTORY|STEP|CONFIDENCE"
    )
    pattern = rf"(?:^|\n)\s*{section_name}\s*:\s*(.*?)(?=\n\s*(?:{next_sections})\s*:|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

    if not match:
        return []

    content = match.group(1).strip()

    # Skip "none" / "none documented" / "n/a" / etc.
    none_pattern = (
        r"^(?:none|n/?a|not (?:applicable|mentioned|available|documented)"
        r"|no .+ (?:mentioned|documented|found))\.?$"
    )
    if re.match(none_pattern, content, re.IGNORECASE):
        return []

    # Try bracket list format: [item1, item2]
    bracket_match = re.match(r"\[(.*)\]", content, re.DOTALL)
    if bracket_match:
        inner = bracket_match.group(1)
        items = [item.strip().strip("\"'") for item in inner.split(",")]
        return [item for item in items if item and not _is_none_value(item)]

    # Try bullet/dash list format
    bullet_items = re.findall(r"[-*]\s*(.+)", content)
    if bullet_items:
        return [item.strip() for item in bullet_items if item.strip() and not _is_none_value(item.strip())]

    # Try numbered list format
    numbered_items = re.findall(r"\d+[.)]\s*(.+)", content)
    if numbered_items:
        return [item.strip() for item in numbered_items if item.strip() and not _is_none_value(item.strip())]

    # Comma-separated single line
    if "," in content and "\n" not in content:
        items = [item.strip() for item in content.split(",")]
        return [item for item in items if item and not _is_none_value(item)]

    # Single item
    if content and not _is_none_value(content):
        return [content]

    return []


def _is_none_value(text: str) -> bool:
    """Check if a string represents a 'none' value."""
    return bool(
        re.match(r"^(?:none|n/?a|not (?:applicable|mentioned|available|documented))\.?$", text.strip(), re.IGNORECASE)
    )
