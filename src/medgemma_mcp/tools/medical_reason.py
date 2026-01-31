"""MCP tool: medical_reason

Answers clinical questions with structured chain-of-thought reasoning
using MedGemma's text-only capabilities (64.4% MedQA accuracy).
"""

import logging

import anyio
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field

from medgemma_mcp.model.inference import run_text_inference
from medgemma_mcp.model.loader import MedGemmaContext
from medgemma_mcp.prompts.templates import Modality, build_text_prompt, get_system_prompt
from medgemma_mcp.safety.confidence import extract_confidence
from medgemma_mcp.safety.disclaimers import MEDICAL_DISCLAIMER

logger = logging.getLogger(__name__)


class MedicalReasoningResult(BaseModel):
    """Structured output from medical text reasoning."""

    reasoning: str = Field(description="Step-by-step clinical reasoning with CoT methodology")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score")
    requires_review: bool = Field(description="True if confidence < 0.7 — needs clinical review")
    disclaimer: str = Field(description="Required medical disclaimer")


async def medical_reason(
    clinical_question: str,
    clinical_context: str = "",
    ctx: Context[ServerSession, MedGemmaContext] = None,  # type: ignore[assignment]
) -> MedicalReasoningResult:
    """Answer a clinical question with structured medical reasoning.

    Uses chain-of-thought prompting for systematic analysis including
    differential diagnosis, treatment considerations, and confidence
    assessment. No image input — for image analysis use analyze_medical_image.

    Args:
        clinical_question: The medical question to answer.
        clinical_context: Optional additional clinical context (patient history,
            lab results, etc.) to include in the reasoning.
    """
    # Get model from lifespan context
    model_ctx: MedGemmaContext = ctx.request_context.lifespan_context

    # Build prompt
    await ctx.report_progress(0.1, 1.0, "Preparing clinical reasoning...")
    system_prompt = get_system_prompt(Modality.TEXT_ONLY)

    # If clinical context provided, prepend it to the question
    full_question = clinical_question
    if clinical_context:
        full_question = f"Clinical context:\n{clinical_context}\n\nQuestion:\n{clinical_question}"

    user_prompt = build_text_prompt(full_question)

    # Run inference (blocking — run in thread pool)
    await ctx.report_progress(0.3, 1.0, "Running MedGemma reasoning...")
    reasoning = await anyio.to_thread.run_sync(
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
    confidence = extract_confidence(reasoning)
    requires_review = confidence < 0.7

    if requires_review:
        await ctx.warning(f"Low confidence ({confidence:.2f}) — clinical review recommended")

    return MedicalReasoningResult(
        reasoning=reasoning,
        confidence=confidence,
        requires_review=requires_review,
        disclaimer=MEDICAL_DISCLAIMER,
    )
