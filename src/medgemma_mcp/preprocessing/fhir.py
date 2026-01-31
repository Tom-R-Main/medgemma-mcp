"""FHIR Bundle to clinical text summary.

MedGemma 4B was NOT trained on FHIR data (scores 67.6% vs base Gemma's
70.9% on EHRQA). All FHIR comprehension must happen here in Python —
the model only receives plain-text clinical summaries.

Supports FHIR R4 Bundle resources: Patient, Condition, MedicationRequest,
Observation, AllergyIntolerance, DiagnosticReport, Procedure.
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def fhir_bundle_to_summary(bundle: dict) -> str:
    """Convert a FHIR R4 Bundle to a plain-text clinical summary.

    Args:
        bundle: Parsed FHIR R4 Bundle JSON (as a Python dict).

    Returns:
        Multi-section clinical summary suitable for MedGemma reasoning.

    Raises:
        ValueError: If the input is not a valid FHIR Bundle.
    """
    if not isinstance(bundle, dict):
        raise ValueError("Expected a FHIR Bundle (dict)")

    resource_type = bundle.get("resourceType", "")
    if resource_type != "Bundle":
        raise ValueError(f"Expected resourceType 'Bundle', got '{resource_type}'")

    entries = bundle.get("entry", [])
    if not entries:
        return "Empty FHIR Bundle — no clinical data available."

    sections: list[str] = []

    # Patient demographics
    patients = _extract_resources(entries, "Patient")
    if patients:
        sections.append(_summarize_patient(patients[0]))

    # Active conditions
    conditions = _extract_resources(entries, "Condition")
    if conditions:
        sections.append(_summarize_conditions(conditions))

    # Current medications
    medications = _extract_resources(entries, "MedicationRequest")
    if not medications:
        medications = _extract_resources(entries, "MedicationStatement")
    if medications:
        sections.append(_summarize_medications(medications))

    # Allergies
    allergies = _extract_resources(entries, "AllergyIntolerance")
    if allergies:
        sections.append(_summarize_allergies(allergies))

    # Observations (labs, vitals)
    observations = _extract_resources(entries, "Observation")
    if observations:
        sections.append(_summarize_observations(observations))

    # Diagnostic reports
    reports = _extract_resources(entries, "DiagnosticReport")
    if reports:
        sections.append(_summarize_reports(reports))

    # Procedures
    procedures = _extract_resources(entries, "Procedure")
    if procedures:
        sections.append(_summarize_procedures(procedures))

    if not sections:
        return "FHIR Bundle contained entries but no recognized clinical resources."

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Resource extraction
# ---------------------------------------------------------------------------


def _extract_resources(entries: list[dict], resource_type: str) -> list[dict]:
    """Pull all resources of a given type from Bundle entries."""
    results = []
    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == resource_type:
            results.append(resource)
    return results


# ---------------------------------------------------------------------------
# Per-resource summarizers
# ---------------------------------------------------------------------------


def _summarize_patient(patient: dict) -> str:
    """Format Patient demographics into readable text."""
    parts = ["PATIENT DEMOGRAPHICS"]

    name = _extract_name(patient)
    if name:
        parts.append(f"Name: {name}")

    gender = patient.get("gender", "unknown")
    parts.append(f"Gender: {gender}")

    birth_date = patient.get("birthDate")
    if birth_date:
        age = _calculate_age(birth_date)
        parts.append(f"Date of birth: {birth_date} (age {age})" if age else f"Date of birth: {birth_date}")

    # Marital status
    marital = _codeable_concept_text(patient.get("maritalStatus"))
    if marital:
        parts.append(f"Marital status: {marital}")

    return "\n".join(parts)


def _summarize_conditions(conditions: list[dict]) -> str:
    """Format Conditions into a readable list."""
    active = []
    resolved = []

    for cond in conditions:
        text = _codeable_concept_text(cond.get("code"))
        if not text:
            text = "Unknown condition"

        status = cond.get("clinicalStatus", {})
        status_code = ""
        if isinstance(status, dict):
            codings = status.get("coding", [])
            status_code = codings[0].get("code", "") if codings else ""

        onset = cond.get("onsetDateTime", "")
        suffix = f" (onset: {onset[:10]})" if onset else ""

        entry = f"- {text}{suffix}"
        if status_code in ("resolved", "inactive", "remission"):
            resolved.append(entry)
        else:
            active.append(entry)

    sections = []
    if active:
        sections.append("ACTIVE CONDITIONS\n" + "\n".join(active))
    if resolved:
        sections.append("RESOLVED CONDITIONS\n" + "\n".join(resolved))

    return "\n\n".join(sections) if sections else "CONDITIONS\nNone documented."


def _summarize_medications(medications: list[dict]) -> str:
    """Format MedicationRequests/Statements into a readable list."""
    lines = ["CURRENT MEDICATIONS"]
    for med in medications:
        name = _codeable_concept_text(med.get("medicationCodeableConcept"))
        if not name:
            # Try medicationReference
            ref = med.get("medicationReference", {})
            name = ref.get("display", "Unknown medication")

        status = med.get("status", "")
        dosage_text = ""
        dosage_instructions = med.get("dosageInstruction", [])
        if dosage_instructions:
            dosage_text = dosage_instructions[0].get("text", "")

        entry = f"- {name}"
        if dosage_text:
            entry += f" — {dosage_text}"
        if status and status != "active":
            entry += f" [{status}]"
        lines.append(entry)

    return "\n".join(lines)


def _summarize_allergies(allergies: list[dict]) -> str:
    """Format AllergyIntolerances into a readable list."""
    lines = ["ALLERGIES AND INTOLERANCES"]
    for allergy in allergies:
        substance = _codeable_concept_text(allergy.get("code"))
        if not substance:
            substance = "Unknown substance"

        category = allergy.get("category", [])
        cat_str = ", ".join(category) if category else ""

        reactions = allergy.get("reaction", [])
        reaction_texts = []
        for rxn in reactions:
            for manifestation in rxn.get("manifestation", []):
                txt = _codeable_concept_text(manifestation)
                if txt:
                    reaction_texts.append(txt)

        entry = f"- {substance}"
        if cat_str:
            entry += f" ({cat_str})"
        if reaction_texts:
            entry += f": {', '.join(reaction_texts)}"
        lines.append(entry)

    return "\n".join(lines)


def _summarize_observations(observations: list[dict]) -> str:
    """Format Observations (labs, vitals) into a readable list."""
    vitals = []
    labs = []

    vital_codes = {
        "8310-5",  # Body temperature
        "8867-4",  # Heart rate
        "9279-1",  # Respiratory rate
        "8480-6",  # Systolic BP
        "8462-4",  # Diastolic BP
        "85354-9",  # Blood pressure panel
        "29463-7",  # Body weight
        "8302-2",  # Body height
        "2708-6",  # SpO2
        "39156-5",  # BMI
    }

    for obs in observations:
        text = _codeable_concept_text(obs.get("code"))
        if not text:
            text = "Unknown observation"

        value = _format_observation_value(obs)
        date = (obs.get("effectiveDateTime") or obs.get("issued") or "")[:10]

        entry = f"- {text}: {value}"
        if date:
            entry += f" ({date})"

        # Classify as vital or lab
        codes = {c.get("code", "") for c in obs.get("code", {}).get("coding", [])}
        if codes & vital_codes:
            vitals.append(entry)
        else:
            labs.append(entry)

    sections = []
    if vitals:
        sections.append("VITAL SIGNS\n" + "\n".join(vitals))
    if labs:
        sections.append("LABORATORY RESULTS\n" + "\n".join(labs))

    return "\n\n".join(sections) if sections else "OBSERVATIONS\nNone documented."


def _summarize_reports(reports: list[dict]) -> str:
    """Format DiagnosticReports into readable text."""
    lines = ["DIAGNOSTIC REPORTS"]
    for report in reports:
        name = _codeable_concept_text(report.get("code"))
        if not name:
            name = "Unknown report"

        date = (report.get("effectiveDateTime") or report.get("issued") or "")[:10]
        conclusion = report.get("conclusion", "")
        status = report.get("status", "")

        entry = f"- {name}"
        if date:
            entry += f" ({date})"
        if status:
            entry += f" [{status}]"
        if conclusion:
            entry += f"\n  {conclusion}"
        lines.append(entry)

    return "\n".join(lines)


def _summarize_procedures(procedures: list[dict]) -> str:
    """Format Procedures into a readable list."""
    lines = ["PROCEDURES"]
    for proc in procedures:
        text = _codeable_concept_text(proc.get("code"))
        if not text:
            text = "Unknown procedure"

        date = (proc.get("performedDateTime") or "")[:10]
        status = proc.get("status", "")

        entry = f"- {text}"
        if date:
            entry += f" ({date})"
        if status and status != "completed":
            entry += f" [{status}]"
        lines.append(entry)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _codeable_concept_text(concept: dict | None) -> str:
    """Extract display text from a FHIR CodeableConcept."""
    if not concept or not isinstance(concept, dict):
        return ""
    # Prefer .text, then .coding[0].display
    text = concept.get("text", "")
    if text:
        return text
    codings = concept.get("coding", [])
    if codings and isinstance(codings, list):
        return codings[0].get("display", codings[0].get("code", ""))
    return ""


def _extract_name(patient: dict) -> str:
    """Extract a human-readable name from a Patient resource."""
    names = patient.get("name", [])
    if not names:
        return ""
    name = names[0]
    given = " ".join(name.get("given", []))
    family = name.get("family", "")
    return f"{given} {family}".strip()


def _calculate_age(birth_date: str) -> int | None:
    """Calculate age from a birth date string (YYYY-MM-DD)."""
    try:
        dob = datetime.strptime(birth_date[:10], "%Y-%m-%d")
        today = datetime.now()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age
    except (ValueError, IndexError):
        return None


def _format_observation_value(obs: dict) -> str:
    """Format an Observation's value into readable text."""
    # valueQuantity
    vq = obs.get("valueQuantity")
    if vq:
        val = vq.get("value", "")
        unit = vq.get("unit", vq.get("code", ""))
        return f"{val} {unit}".strip()

    # valueCodeableConcept
    vcc = obs.get("valueCodeableConcept")
    if vcc:
        return _codeable_concept_text(vcc) or "present"

    # valueString
    vs = obs.get("valueString")
    if vs:
        return vs

    # component (e.g., blood pressure panel)
    components = obs.get("component", [])
    if components:
        parts = []
        for comp in components:
            comp_name = _codeable_concept_text(comp.get("code"))
            comp_val = _format_observation_value(comp)
            if comp_name and comp_val:
                parts.append(f"{comp_name}: {comp_val}")
        return "; ".join(parts) if parts else "no value"

    return "no value"
