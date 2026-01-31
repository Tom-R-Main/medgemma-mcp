"""Confidence score extraction from MedGemma output.

Extracts the model's self-reported confidence and falls back to
keyword-based heuristics when an explicit score isn't found.
"""

import re


def extract_confidence(model_output: str) -> float:
    """Extract confidence score from model output.

    Tries explicit numeric scores first, then falls back to
    keyword-based analysis of hedging language.

    Args:
        model_output: Raw text output from MedGemma.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    # Try explicit numeric patterns first
    score = _extract_explicit_score(model_output)
    if score is not None:
        return score

    # Fall back to keyword analysis
    return _keyword_confidence(model_output)


def _extract_explicit_score(text: str) -> float | None:
    """Try to find an explicit confidence score in the text.

    Looks for patterns like:
        Confidence: 0.85
        Score: 0.7
        confidence score: 0.92
        [0.85] confidence
    """
    patterns = [
        r"(?:confidence|score)\s*[:=]\s*\[?\s*([01]\.?\d*)\s*\]?",
        r"\[([01]\.?\d*)\]\s*(?:confidence)?",
        r"(?:confidence|score)\s+(?:is\s+)?([01]\.?\d*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue

    return None


def _keyword_confidence(text: str) -> float:
    """Estimate confidence from hedging language in the output.

    Returns a conservative estimate based on the strongest
    qualifier found in the text.
    """
    text_lower = text.lower()

    # Check from highest to lowest confidence
    high_confidence = [
        "definite",
        "definitively",
        "clearly shows",
        "diagnostic of",
        "consistent with",
        "classic presentation",
        "pathognomonic",
    ]
    moderate_confidence = [
        "likely",
        "probable",
        "suggest",
        "compatible with",
        "favors",
        "suspicious for",
    ]
    low_confidence = [
        "possible",
        "cannot exclude",
        "uncertain",
        "equivocal",
        "indeterminate",
        "nonspecific",
        "limited study",
        "suboptimal",
    ]

    for term in low_confidence:
        if term in text_lower:
            return 0.45

    for term in high_confidence:
        if term in text_lower:
            return 0.85

    for term in moderate_confidence:
        if term in text_lower:
            return 0.70

    # Default: moderate-low when no qualifiers found
    return 0.60
