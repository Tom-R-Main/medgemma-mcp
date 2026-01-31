"""Tests for FHIR Bundle to clinical summary conversion."""

import pytest

from medgemma_mcp.preprocessing.fhir import fhir_bundle_to_summary

# ---------------------------------------------------------------------------
# Minimal FHIR Bundle fixtures
# ---------------------------------------------------------------------------


def _make_bundle(*resources: dict) -> dict:
    """Create a minimal FHIR Bundle wrapping the given resources."""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": r} for r in resources],
    }


PATIENT_RESOURCE = {
    "resourceType": "Patient",
    "name": [{"given": ["Jane"], "family": "Doe"}],
    "gender": "female",
    "birthDate": "1958-03-15",
}

CONDITION_ACTIVE = {
    "resourceType": "Condition",
    "code": {"text": "Type 2 Diabetes Mellitus"},
    "clinicalStatus": {"coding": [{"code": "active"}]},
    "onsetDateTime": "2019-06-01",
}

CONDITION_RESOLVED = {
    "resourceType": "Condition",
    "code": {"text": "Acute bronchitis"},
    "clinicalStatus": {"coding": [{"code": "resolved"}]},
}

MEDICATION_RESOURCE = {
    "resourceType": "MedicationRequest",
    "status": "active",
    "medicationCodeableConcept": {"text": "Metformin 500mg"},
    "dosageInstruction": [{"text": "Take 1 tablet twice daily with meals"}],
}

ALLERGY_RESOURCE = {
    "resourceType": "AllergyIntolerance",
    "code": {"text": "Penicillin"},
    "category": ["medication"],
    "reaction": [{"manifestation": [{"text": "Hives"}, {"text": "Anaphylaxis"}]}],
}

OBSERVATION_VITAL = {
    "resourceType": "Observation",
    "code": {
        "coding": [{"code": "8480-6", "display": "Systolic blood pressure"}],
        "text": "Systolic blood pressure",
    },
    "valueQuantity": {"value": 142, "unit": "mmHg"},
    "effectiveDateTime": "2024-11-15T10:00:00Z",
}

OBSERVATION_LAB = {
    "resourceType": "Observation",
    "code": {"coding": [{"code": "4548-4", "display": "HbA1c"}], "text": "HbA1c"},
    "valueQuantity": {"value": 7.2, "unit": "%"},
    "effectiveDateTime": "2024-11-10T08:00:00Z",
}

OBSERVATION_BP_PANEL = {
    "resourceType": "Observation",
    "code": {
        "coding": [{"code": "85354-9", "display": "Blood pressure panel"}],
        "text": "Blood pressure panel",
    },
    "component": [
        {
            "code": {"coding": [{"code": "8480-6"}], "text": "Systolic"},
            "valueQuantity": {"value": 138, "unit": "mmHg"},
        },
        {
            "code": {"coding": [{"code": "8462-4"}], "text": "Diastolic"},
            "valueQuantity": {"value": 88, "unit": "mmHg"},
        },
    ],
    "effectiveDateTime": "2024-11-20T09:30:00Z",
}

REPORT_RESOURCE = {
    "resourceType": "DiagnosticReport",
    "code": {"text": "Chest X-ray"},
    "status": "final",
    "effectiveDateTime": "2024-11-12T14:00:00Z",
    "conclusion": "No acute cardiopulmonary abnormality.",
}

PROCEDURE_RESOURCE = {
    "resourceType": "Procedure",
    "code": {"text": "Coronary artery bypass grafting"},
    "status": "completed",
    "performedDateTime": "2022-05-10T08:00:00Z",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_bundle():
    """Empty bundle returns informative message."""
    bundle = {"resourceType": "Bundle", "entry": []}
    result = fhir_bundle_to_summary(bundle)
    assert "Empty FHIR Bundle" in result


def test_not_a_bundle():
    """Non-Bundle resourceType raises ValueError."""
    with pytest.raises(ValueError, match="Expected resourceType 'Bundle'"):
        fhir_bundle_to_summary({"resourceType": "Patient"})


def test_not_a_dict():
    """Non-dict input raises ValueError."""
    with pytest.raises(ValueError, match="Expected a FHIR Bundle"):
        fhir_bundle_to_summary("not a dict")  # type: ignore[arg-type]


def test_patient_demographics():
    """Patient resource is summarized with name, gender, DOB."""
    bundle = _make_bundle(PATIENT_RESOURCE)
    result = fhir_bundle_to_summary(bundle)
    assert "PATIENT DEMOGRAPHICS" in result
    assert "Jane Doe" in result
    assert "female" in result
    assert "1958-03-15" in result


def test_active_and_resolved_conditions():
    """Active and resolved conditions are separated."""
    bundle = _make_bundle(CONDITION_ACTIVE, CONDITION_RESOLVED)
    result = fhir_bundle_to_summary(bundle)
    assert "ACTIVE CONDITIONS" in result
    assert "Type 2 Diabetes Mellitus" in result
    assert "RESOLVED CONDITIONS" in result
    assert "Acute bronchitis" in result


def test_condition_onset_date():
    """Condition onset date is included."""
    bundle = _make_bundle(CONDITION_ACTIVE)
    result = fhir_bundle_to_summary(bundle)
    assert "2019-06-01" in result


def test_medications():
    """Medications include name and dosage instructions."""
    bundle = _make_bundle(MEDICATION_RESOURCE)
    result = fhir_bundle_to_summary(bundle)
    assert "CURRENT MEDICATIONS" in result
    assert "Metformin 500mg" in result
    assert "twice daily" in result


def test_allergies_with_reactions():
    """Allergies include substance, category, and reactions."""
    bundle = _make_bundle(ALLERGY_RESOURCE)
    result = fhir_bundle_to_summary(bundle)
    assert "ALLERGIES" in result
    assert "Penicillin" in result
    assert "medication" in result
    assert "Hives" in result
    assert "Anaphylaxis" in result


def test_vital_signs():
    """Vital sign observations are classified correctly."""
    bundle = _make_bundle(OBSERVATION_VITAL)
    result = fhir_bundle_to_summary(bundle)
    assert "VITAL SIGNS" in result
    assert "Systolic blood pressure" in result
    assert "142" in result
    assert "mmHg" in result


def test_lab_results():
    """Lab observations are classified correctly."""
    bundle = _make_bundle(OBSERVATION_LAB)
    result = fhir_bundle_to_summary(bundle)
    assert "LABORATORY RESULTS" in result
    assert "HbA1c" in result
    assert "7.2" in result


def test_bp_panel_components():
    """Blood pressure panel components are extracted."""
    bundle = _make_bundle(OBSERVATION_BP_PANEL)
    result = fhir_bundle_to_summary(bundle)
    assert "VITAL SIGNS" in result
    assert "138" in result
    assert "88" in result


def test_diagnostic_reports():
    """DiagnosticReports include name, date, status, conclusion."""
    bundle = _make_bundle(REPORT_RESOURCE)
    result = fhir_bundle_to_summary(bundle)
    assert "DIAGNOSTIC REPORTS" in result
    assert "Chest X-ray" in result
    assert "2024-11-12" in result
    assert "No acute cardiopulmonary abnormality" in result


def test_procedures():
    """Procedures include name and date."""
    bundle = _make_bundle(PROCEDURE_RESOURCE)
    result = fhir_bundle_to_summary(bundle)
    assert "PROCEDURES" in result
    assert "Coronary artery bypass grafting" in result
    assert "2022-05-10" in result


def test_full_bundle():
    """Full bundle with all resource types produces comprehensive summary."""
    bundle = _make_bundle(
        PATIENT_RESOURCE,
        CONDITION_ACTIVE,
        CONDITION_RESOLVED,
        MEDICATION_RESOURCE,
        ALLERGY_RESOURCE,
        OBSERVATION_VITAL,
        OBSERVATION_LAB,
        REPORT_RESOURCE,
        PROCEDURE_RESOURCE,
    )
    result = fhir_bundle_to_summary(bundle)
    assert "PATIENT DEMOGRAPHICS" in result
    assert "ACTIVE CONDITIONS" in result
    assert "CURRENT MEDICATIONS" in result
    assert "ALLERGIES" in result
    assert "VITAL SIGNS" in result
    assert "LABORATORY RESULTS" in result
    assert "DIAGNOSTIC REPORTS" in result
    assert "PROCEDURES" in result


def test_missing_optional_fields():
    """Resources with minimal fields don't crash."""
    minimal_condition = {
        "resourceType": "Condition",
        "code": {"coding": [{"code": "E11", "display": "Type 2 diabetes"}]},
    }
    minimal_med = {
        "resourceType": "MedicationRequest",
        "status": "active",
        "medicationReference": {"display": "Insulin glargine"},
    }
    bundle = _make_bundle(minimal_condition, minimal_med)
    result = fhir_bundle_to_summary(bundle)
    assert "Type 2 diabetes" in result
    assert "Insulin glargine" in result


def test_codeable_concept_fallback_to_coding_display():
    """When .text is missing, falls back to coding[0].display."""
    condition = {
        "resourceType": "Condition",
        "code": {"coding": [{"system": "http://snomed.info/sct", "code": "73211009", "display": "Diabetes mellitus"}]},
        "clinicalStatus": {"coding": [{"code": "active"}]},
    }
    bundle = _make_bundle(condition)
    result = fhir_bundle_to_summary(bundle)
    assert "Diabetes mellitus" in result


def test_observation_value_string():
    """Observation with valueString is handled."""
    obs = {
        "resourceType": "Observation",
        "code": {"text": "Clinical note"},
        "valueString": "Patient reports improvement",
    }
    bundle = _make_bundle(obs)
    result = fhir_bundle_to_summary(bundle)
    assert "Patient reports improvement" in result


def test_observation_value_codeable_concept():
    """Observation with valueCodeableConcept is handled."""
    obs = {
        "resourceType": "Observation",
        "code": {"text": "Tobacco use"},
        "valueCodeableConcept": {"text": "Current smoker"},
    }
    bundle = _make_bundle(obs)
    result = fhir_bundle_to_summary(bundle)
    assert "Current smoker" in result


def test_bundle_no_entry_key():
    """Bundle with no 'entry' key returns empty message."""
    bundle = {"resourceType": "Bundle"}
    result = fhir_bundle_to_summary(bundle)
    assert "Empty FHIR Bundle" in result
