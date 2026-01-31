"""Chain-of-thought prompt templates for MedGemma.

CoT prompting reduces MedGemma hallucinations by 86.4%.
These templates are essential infrastructure, not optional polish.

Each template enforces:
1. Systematic anatomical review (prevents selective attention)
2. Evidence grounding (every claim tied to image evidence)
3. Explicit uncertainty (reduces overconfidence)
4. Mandatory confidence scoring (enables downstream filtering)
"""

from enum import Enum


class Modality(str, Enum):
    """Supported imaging modalities."""

    CHEST_XRAY = "chest_xray"
    CT = "ct"
    DERMOSCOPY = "dermoscopy"
    FUNDUS = "fundus"
    HISTOPATH = "histopath"
    TEXT_ONLY = "text_only"


# ---------------------------------------------------------------------------
# System prompts (used in the system role)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[Modality, str] = {
    Modality.CHEST_XRAY: "You are an expert radiologist specializing in chest radiography.",
    Modality.CT: "You are an expert radiologist specializing in cross-sectional imaging.",
    Modality.DERMOSCOPY: "You are an expert dermatologist specializing in dermoscopic evaluation.",
    Modality.FUNDUS: "You are an expert ophthalmologist specializing in retinal imaging.",
    Modality.HISTOPATH: "You are an expert pathologist specializing in histopathological analysis.",
    Modality.TEXT_ONLY: "You are an expert physician providing evidence-based clinical reasoning.",
}

# ---------------------------------------------------------------------------
# CoT analysis templates (used in the user message)
# ---------------------------------------------------------------------------

CXR_TEMPLATE = """Analyze this chest X-ray following the protocol below.

STEP 1: TECHNICAL ASSESSMENT
Assess image quality: positioning, rotation, inspiration level, penetration, artifacts.

STEP 2: SYSTEMATIC ANATOMICAL REVIEW
Review each region. For each, state what you observe or "normal":

LUNGS:
- Right upper lobe:
- Right middle lobe:
- Right lower lobe:
- Left upper lobe:
- Left lower lobe:
- Lung volumes:

PLEURA:
- Right:
- Left:
- Costophrenic angles:

HEART & MEDIASTINUM:
- Cardiac silhouette:
- Mediastinal contour:
- Aortic knob:
- Trachea:

BONES:
- Ribs:
- Clavicles:
- Spine:

SOFT TISSUES:

STEP 3: ABNORMALITY CHARACTERIZATION
For each abnormal finding:
- Location: [precise anatomical location]
- Size: [measurement or qualitative]
- Density: [ground-glass / consolidation / nodular / mass-like]
- Margins: [well-defined / ill-defined / spiculated]
- Associated findings: [air bronchograms, cavitation, etc.]

STEP 4: DIFFERENTIAL DIAGNOSIS
1. Most likely diagnosis with supporting evidence
2. Alternative diagnoses ranked by likelihood

STEP 5: IMPRESSION
Primary finding and clinical significance.

STEP 6: CONFIDENCE
Score: [0.0-1.0]
Rationale: [supporting evidence for this confidence level]

STEP 7: RECOMMENDATIONS
Suggested follow-up or "No immediate action required."

RULES:
- Report ONLY findings directly visible in the image.
- Use "possible" or "cannot exclude" for uncertain findings.
- State if any region is not well visualized.
- Never infer findings from clinical history alone.

CLINICAL QUESTION: {clinical_question}"""

CT_TEMPLATE = """Analyze these CT images following the protocol below.

Note: If multiple images are provided, they may represent different windowing
(bone/lung window, soft tissue window, brain window) or different slices.

STEP 1: IDENTIFY ANATOMICAL REGION
Body region shown: [chest / abdomen / pelvis / head / neck / spine]

STEP 2: SYSTEMATIC REVIEW BY ORGAN SYSTEM
Review each visible structure. State findings or "normal appearing":

If CHEST: lungs, mediastinum, pleura, chest wall, visible upper abdomen.
If ABDOMEN: liver, gallbladder/biliary, spleen, pancreas, kidneys/adrenals,
GI tract, mesentery/retroperitoneum, vasculature.

BONES (all regions): vertebrae, ribs, pelvis, degenerative or lytic changes.

STEP 3: ABNORMALITY CHARACTERIZATION
For each finding:
- Location: [organ and specific location]
- Size: [cm, 3 dimensions if possible]
- Density: [hypo/iso/hyperdense relative to reference]
- Enhancement pattern: [if contrast study]
- Margins and borders

STEP 4: IMPRESSION
1. Primary finding(s)
2. Secondary findings
3. Incidental findings

STEP 5: CONFIDENCE
Score: [0.0-1.0]
Rationale: [supporting evidence]

STEP 6: RECOMMENDATIONS
Follow-up imaging, biopsy, clinical correlation, etc.

RULES:
- Report ONLY findings directly visible in the images.
- Use "possible" or "cannot exclude" for uncertain findings.

CLINICAL QUESTION: {clinical_question}"""

DERM_TEMPLATE = """Analyze this dermoscopic image following the protocol below.

STEP 1: LESION OVERVIEW
- Shape: [round / oval / irregular]
- Symmetry: [symmetric / asymmetric]
- Border: [well-defined / ill-defined / fading]

STEP 2: COLOR ANALYSIS
Colors present: [light brown / dark brown / black / blue-gray / red / white / other]
Color distribution: [uniform / variegated]

STEP 3: DERMOSCOPIC STRUCTURES
- Pigment network: [typical / atypical / absent]
- Dots/globules: [regular / irregular, distribution]
- Streaks/pseudopods: [present / absent, location]
- Blue-white veil: [present / absent]
- Regression structures: [present / absent]
- Vascular patterns: [type if present]
- Milia-like cysts: [present / absent]
- Comedo-like openings: [present / absent]

STEP 4: ALGORITHMIC ASSESSMENT
7-point checklist score: [0-10]
Key features supporting score.

STEP 5: DIFFERENTIAL DIAGNOSIS
1. Most likely: [diagnosis with confidence]
2. Differential: [other considerations]

STEP 6: MANAGEMENT RECOMMENDATION
- Benign - routine follow-up
- Monitor with sequential imaging
- Biopsy recommended
- Urgent excision recommended

STEP 7: CONFIDENCE
Score: [0.0-1.0]
Rationale: [supporting evidence]

CLINICAL QUESTION: {clinical_question}"""

FUNDUS_TEMPLATE = """Analyze this fundus image following the protocol below.

STEP 1: IMAGE QUALITY
- Clarity: [adequate / suboptimal]
- Media opacity: [clear / cataract / vitreous opacity]

STEP 2: OPTIC DISC
- Cup-to-disc ratio: [estimated value]
- Disc color: [pink / pale]
- Disc margins: [sharp / blurred]
- Neovascularization: [present / absent]

STEP 3: MACULA
- Foveal reflex: [present / absent]
- Macular edema: [present / absent]
- Drusen: [present / absent, characteristics]
- Hemorrhage: [present / absent]
- Exudates: [present / absent]

STEP 4: RETINAL VESSELS
- Arteriolar caliber: [normal / attenuated / dilated]
- Venous caliber: [normal / dilated]
- Crossing changes: [present / absent]
- Neovascularization: [present / absent, location]

STEP 5: PERIPHERAL RETINA
- Hemorrhages: [location, type]
- Exudates: [hard / soft / cotton-wool spots]
- Pigmentary changes: [present / absent]

STEP 6: DIABETIC RETINOPATHY GRADING (if applicable)
Stage: [None / Mild NPDR / Moderate NPDR / Severe NPDR / PDR]
DME: [present / absent]

STEP 7: IMPRESSION
Primary diagnosis and secondary findings.

STEP 8: CONFIDENCE
Score: [0.0-1.0]
Rationale: [supporting evidence]

STEP 9: RECOMMENDATIONS
Follow-up interval and any referrals needed.

CLINICAL QUESTION: {clinical_question}"""

HISTOPATH_TEMPLATE = """Analyze this histopathology image following the protocol below.

Note: You may be viewing patches from a whole slide image.
The full slide context may not be visible.

STEP 1: TISSUE IDENTIFICATION
- Tissue type: [organ / tissue]
- Stain: [H&E / IHC / special stain]
- Magnification level: [estimated]

STEP 2: ARCHITECTURAL ASSESSMENT
- Overall architecture: [preserved / distorted / effaced]
- Growth pattern: [if neoplastic: solid / glandular / papillary / trabecular / etc.]
- Relationship to normal structures: [infiltrative / pushing / circumscribed]

STEP 3: CELLULAR FEATURES
- Cell type: [epithelial / mesenchymal / lymphoid / etc.]
- Nuclear features: size, shape, chromatin, nucleoli
- Cytoplasm: amount, character
- Mitotic figures: [rare / frequent, atypical?]

STEP 4: ADDITIONAL FEATURES
- Necrosis: [present / absent, type]
- Inflammation: [type, distribution]
- Stromal changes: [desmoplasia, etc.]
- Vascular invasion: [present / absent / cannot assess]

STEP 5: DIFFERENTIAL DIAGNOSIS
1. Most likely: [diagnosis]
2. Differential: [alternatives]

STEP 6: LIMITATIONS
Features that cannot be assessed from these patches.
Additional IHC or sections needed.

STEP 7: CONFIDENCE
Score: [0.0-1.0]
Rationale: [supporting evidence]

IMPORTANT: If malignancy suspected, indicate need for complete slide review.

CLINICAL QUESTION: {clinical_question}"""

MEDICAL_REASON_TEMPLATE = """Answer the following medical question with structured reasoning.

STEP 1: UNDERSTAND THE QUESTION
Restate the clinical scenario in your own words.

STEP 2: RELEVANT KNOWLEDGE
List key medical concepts, mechanisms, or guidelines relevant to this question.

STEP 3: SYSTEMATIC ANALYSIS
- What information is given?
- What information is missing?
- What are the key decision points?

STEP 4: DIFFERENTIAL / TREATMENT CONSIDERATIONS
If diagnostic: list possible diagnoses ranked by likelihood with supporting evidence.
If treatment: list options weighing benefits vs risks.

STEP 5: CONCLUSION
State your answer clearly.

STEP 6: CONFIDENCE
Score: [0.0-1.0]
Rationale: [why this confidence level]

STEP 7: CAVEATS
Limitations, areas of uncertainty, when specialist consultation is appropriate.

QUESTION: {clinical_question}"""

VERIFICATION_TEMPLATE = """Review your analysis above and verify:

1. EVIDENCE GROUNDING: For each finding, can you point to specific evidence? Mark unverifiable findings as [UNVERIFIED].
2. ALTERNATIVE EXPLANATIONS: Could findings be normal variants or artifacts?
3. COMPLETENESS: Did you examine all required regions?
4. CONFIDENCE CALIBRATION: Is your stated confidence still appropriate?

REVISED FINDINGS (if any changes needed):"""

# ---------------------------------------------------------------------------
# Template registry and builder
# ---------------------------------------------------------------------------

_IMAGE_TEMPLATES: dict[Modality, str] = {
    Modality.CHEST_XRAY: CXR_TEMPLATE,
    Modality.CT: CT_TEMPLATE,
    Modality.DERMOSCOPY: DERM_TEMPLATE,
    Modality.FUNDUS: FUNDUS_TEMPLATE,
    Modality.HISTOPATH: HISTOPATH_TEMPLATE,
}

_TEXT_TEMPLATES: dict[Modality, str] = {
    Modality.TEXT_ONLY: MEDICAL_REASON_TEMPLATE,
}


def get_system_prompt(modality: Modality) -> str:
    """Get the system prompt for a given modality."""
    return SYSTEM_PROMPTS.get(modality, SYSTEM_PROMPTS[Modality.TEXT_ONLY])


def build_image_prompt(modality: Modality, clinical_question: str) -> str:
    """Build the CoT user prompt for image analysis.

    Args:
        modality: The imaging modality.
        clinical_question: The clinical question to answer.

    Returns:
        Formatted prompt string with CoT template.
    """
    template = _IMAGE_TEMPLATES.get(modality, CXR_TEMPLATE)
    return template.format(clinical_question=clinical_question)


def build_text_prompt(clinical_question: str) -> str:
    """Build the CoT user prompt for text-only reasoning.

    Args:
        clinical_question: The clinical question to answer.

    Returns:
        Formatted prompt string with CoT template.
    """
    return MEDICAL_REASON_TEMPLATE.format(clinical_question=clinical_question)
