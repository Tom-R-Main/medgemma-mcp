"""Tests for confidence score extraction."""

from medgemma_mcp.safety.confidence import extract_confidence


def test_explicit_score_colon_format():
    """Extract confidence from 'Confidence: 0.85' pattern."""
    text = "Some analysis...\nConfidence: 0.85\nMore text"
    assert extract_confidence(text) == 0.85


def test_explicit_score_equals_format():
    """Extract confidence from 'Score = 0.7' pattern."""
    text = "Analysis complete.\nScore = 0.70"
    assert extract_confidence(text) == 0.70


def test_explicit_score_bracket_format():
    """Extract confidence from '[0.92]' pattern."""
    text = "Findings: clear lungs. [0.92] confidence"
    assert extract_confidence(text) == 0.92


def test_explicit_score_boundary_zero():
    """Handle confidence of exactly 0.0."""
    text = "Confidence: 0.0"
    assert extract_confidence(text) == 0.0


def test_explicit_score_boundary_one():
    """Handle confidence of exactly 1.0."""
    text = "Score: 1.0"
    assert extract_confidence(text) == 1.0


def test_explicit_score_out_of_range_ignored():
    """Scores > 1.0 should be ignored, falling back to keywords."""
    text = "Confidence: 2.5\nThis is likely a finding."
    result = extract_confidence(text)
    # Should fall back to keyword analysis ("likely" -> 0.70)
    assert result == 0.70


def test_keyword_high_confidence():
    """High-confidence keywords return ~0.85."""
    assert extract_confidence("This is clearly shows a pneumothorax") == 0.85
    assert extract_confidence("Findings diagnostic of fracture") == 0.85
    assert extract_confidence("Pattern consistent with pneumonia") == 0.85


def test_keyword_moderate_confidence():
    """Moderate-confidence keywords return ~0.70."""
    assert extract_confidence("This likely represents an effusion") == 0.70
    assert extract_confidence("Findings suggest cardiomegaly") == 0.70


def test_keyword_low_confidence():
    """Low-confidence keywords return ~0.45."""
    assert extract_confidence("Cannot exclude a small nodule") == 0.45
    assert extract_confidence("Findings are equivocal") == 0.45
    assert extract_confidence("Uncertain significance") == 0.45


def test_keyword_low_takes_precedence():
    """Low-confidence terms should be checked first (conservative)."""
    text = "Clearly visible finding, but cannot exclude artifact"
    assert extract_confidence(text) == 0.45


def test_no_qualifiers_default():
    """No recognized qualifiers returns default 0.60."""
    assert extract_confidence("Normal chest X-ray. No acute findings.") == 0.60


def test_empty_string():
    """Empty string returns default."""
    assert extract_confidence("") == 0.60


def test_case_insensitive():
    """Confidence extraction is case-insensitive."""
    assert extract_confidence("CONFIDENCE: 0.8") == 0.8
    assert extract_confidence("confidence: 0.8") == 0.8
