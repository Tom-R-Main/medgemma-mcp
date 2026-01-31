"""Tests for structured extraction output parsing."""

from medgemma_mcp.tools.extract import _extract_section, _parse_extraction_output


def test_extract_section_bullet_list():
    """Extract items from dash-prefixed list."""
    text = """CONDITIONS:
- Type 2 Diabetes
- Hypertension
- Chronic kidney disease

MEDICATIONS:
- Metformin 500mg"""
    items = _extract_section(text, "CONDITIONS")
    assert items == ["Type 2 Diabetes", "Hypertension", "Chronic kidney disease"]


def test_extract_section_bracket_list():
    """Extract items from bracket-delimited list."""
    text = "MEDICATIONS: [Metformin 500mg, Lisinopril 10mg, Atorvastatin 20mg]\nALLERGIES: none"
    items = _extract_section(text, "MEDICATIONS")
    assert items == ["Metformin 500mg", "Lisinopril 10mg", "Atorvastatin 20mg"]


def test_extract_section_comma_separated():
    """Extract items from comma-separated single line."""
    text = "SYMPTOMS: chest pain, shortness of breath, fatigue\nLAB_RESULTS: none"
    items = _extract_section(text, "SYMPTOMS")
    assert items == ["chest pain", "shortness of breath", "fatigue"]


def test_extract_section_numbered_list():
    """Extract items from numbered list."""
    text = """CONDITIONS:
1. Hypertension
2. Type 2 Diabetes Mellitus
3. Hyperlipidemia

MEDICATIONS: none"""
    items = _extract_section(text, "CONDITIONS")
    assert items == ["Hypertension", "Type 2 Diabetes Mellitus", "Hyperlipidemia"]


def test_extract_section_none_value():
    """Section with 'none' returns empty list."""
    text = "ALLERGIES: none\nPROCEDURES: N/A"
    assert _extract_section(text, "ALLERGIES") == []
    assert _extract_section(text, "PROCEDURES") == []


def test_extract_section_none_documented():
    """Section with 'none documented' returns empty list."""
    text = "FAMILY_HISTORY: not mentioned\nSYMPTOMS: none"
    assert _extract_section(text, "FAMILY_HISTORY") == []


def test_extract_section_missing():
    """Missing section returns empty list."""
    text = "CONDITIONS: Diabetes\nMEDICATIONS: Metformin"
    assert _extract_section(text, "ALLERGIES") == []


def test_extract_section_single_item():
    """Section with a single item (no list format)."""
    text = "ALLERGIES: Penicillin\nPROCEDURES: none"
    items = _extract_section(text, "ALLERGIES")
    assert items == ["Penicillin"]


def test_parse_extraction_output_full():
    """Parse a complete model output into ExtractedData."""
    model_output = """STEP 1: IDENTIFY ENTITIES
The text mentions several medical entities.

STEP 2: CATEGORIZE FINDINGS

CONDITIONS:
- Type 2 Diabetes Mellitus
- Essential hypertension

MEDICATIONS:
- Metformin 500mg twice daily
- Lisinopril 10mg daily

ALLERGIES: Penicillin

PROCEDURES: none

VITALS:
- BP: 142/88 mmHg
- HR: 78 bpm

LAB_RESULTS:
- HbA1c: 7.2%
- Creatinine: 1.1 mg/dL

SYMPTOMS: none

FAMILY_HISTORY:
- Father: MI at age 55

STEP 3: CLINICAL SIGNIFICANCE
The elevated HbA1c suggests suboptimal glycemic control.

CONFIDENCE: 0.85"""

    result = _parse_extraction_output(model_output)

    assert result.conditions == ["Type 2 Diabetes Mellitus", "Essential hypertension"]
    assert result.medications == ["Metformin 500mg twice daily", "Lisinopril 10mg daily"]
    assert result.allergies == ["Penicillin"]
    assert result.procedures == []
    assert len(result.vitals) == 2
    assert len(result.lab_results) == 2
    assert result.symptoms == []
    assert result.family_history == ["Father: MI at age 55"]


def test_parse_extraction_output_all_none():
    """When all sections are 'none', all lists are empty."""
    model_output = """CONDITIONS: none
MEDICATIONS: none
ALLERGIES: N/A
PROCEDURES: none
VITALS: none
LAB_RESULTS: none
SYMPTOMS: none
FAMILY_HISTORY: not mentioned"""

    result = _parse_extraction_output(model_output)
    assert result.conditions == []
    assert result.medications == []
    assert result.allergies == []
    assert result.procedures == []
    assert result.vitals == []
    assert result.lab_results == []
    assert result.symptoms == []
    assert result.family_history == []


def test_parse_extraction_output_partial():
    """Partial output still extracts available sections."""
    model_output = """CONDITIONS:
- Pneumonia

MEDICATIONS:
- Amoxicillin 500mg

CONFIDENCE: 0.72"""

    result = _parse_extraction_output(model_output)
    assert result.conditions == ["Pneumonia"]
    assert result.medications == ["Amoxicillin 500mg"]
    assert result.allergies == []
