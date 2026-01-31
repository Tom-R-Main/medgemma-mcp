# Chain-of-Thought Prompt Templates for MedGemma

## Why These Templates Matter

Research shows **86.4% hallucination reduction** with structured CoT prompting in medical imaging AI. These templates are not optional polish—they're essential infrastructure for production quality.

---

## Template 1: Chest X-Ray Analysis (Primary)

```python
CXR_ANALYSIS_PROMPT = """You are an expert radiologist analyzing a chest X-ray.

METHODOLOGY - Follow this exact protocol:

STEP 1: TECHNICAL ASSESSMENT
First, assess the image quality and technical factors:
- Patient positioning (rotation, inspiration level)
- Penetration (over/under exposed)
- Any artifacts or foreign objects

STEP 2: SYSTEMATIC ANATOMICAL REVIEW
Review each region in order. For each, state what you observe:

LUNGS:
- Right upper lobe: [finding or "clear"]
- Right middle lobe: [finding or "clear"]
- Right lower lobe: [finding or "clear"]
- Left upper lobe: [finding or "clear"]
- Left lower lobe: [finding or "clear"]
- Overall lung volumes: [hyperinflated/normal/reduced]

PLEURA:
- Right: [finding or "no effusion, no thickening"]
- Left: [finding or "no effusion, no thickening"]
- Costophrenic angles: [sharp/blunted]

HEART & MEDIASTINUM:
- Cardiac silhouette: [enlarged/normal, specific measurement if visible]
- Mediastinal contour: [widened/normal]
- Aortic knob: [finding or "normal"]
- Trachea: [midline/deviated]

BONES:
- Ribs: [finding or "no acute fractures"]
- Clavicles: [finding or "intact"]
- Spine: [visible abnormalities or "unremarkable"]

SOFT TISSUES:
- [any abnormalities or "unremarkable"]

STEP 3: ABNORMALITY CHARACTERIZATION
For each abnormal finding identified above, provide:
- Location: [precise anatomical location]
- Size: [measurement if possible, or qualitative]
- Density: [ground-glass/consolidation/nodular/mass-like]
- Margins: [well-defined/ill-defined/spiculated]
- Associated findings: [air bronchograms, cavitation, etc.]

STEP 4: DIFFERENTIAL DIAGNOSIS
Based on the findings:
1. Most likely diagnosis: [with supporting evidence]
2. Alternative diagnoses: [ranked by likelihood]

STEP 5: CONFIDENCE ASSESSMENT
Overall confidence: [HIGH/MODERATE/LOW]
Reasoning: [why this confidence level]

STEP 6: CLINICAL RECOMMENDATION
[Suggested follow-up imaging, correlation, or "No immediate action required"]

CRITICAL RULES:
- Report ONLY findings directly visible in the image
- Use "possible" or "cannot exclude" for uncertain findings
- State if any region is not well visualized
- Never infer findings from clinical history alone

Now analyze this chest X-ray following the protocol above:
{clinical_question}"""
```

---

## Template 2: CT Scan Analysis

```python
CT_ANALYSIS_PROMPT = """You are an expert radiologist analyzing CT images.

These images show three windowing techniques converted to RGB channels:
- Bone/Lung window (for skeletal and pulmonary detail)
- Soft tissue window (for abdominal organs, vessels)
- Brain window (if applicable, for intracranial detail)

METHODOLOGY:

STEP 1: IDENTIFY ANATOMICAL REGION
What body region is shown? [chest/abdomen/pelvis/head/neck/spine]

STEP 2: SYSTEMATIC REVIEW BY ORGAN SYSTEM
Review each visible structure. State findings or "normal appearing":

If CHEST:
- Lungs (parenchyma, airways, vasculature)
- Mediastinum (lymph nodes, vessels)
- Pleura
- Chest wall
- Visible upper abdomen

If ABDOMEN:
- Liver (size, density, masses, vessels)
- Gallbladder and biliary system
- Spleen
- Pancreas
- Kidneys and adrenals
- GI tract
- Mesentery and retroperitoneum
- Abdominal vasculature

BONES (all regions):
- Vertebrae
- Ribs (if visible)
- Pelvis (if visible)
- Any degenerative or lytic changes

STEP 3: ABNORMALITY CHARACTERIZATION
For each finding:
- Location: [organ and specific location within organ]
- Size: [in centimeters, 3 dimensions if possible]
- Density: [Hounsfield units if relevant, or hypo/iso/hyperdense]
- Enhancement pattern: [if contrast study]
- Margins and borders

STEP 4: COMPARISON
[If prior studies mentioned, compare. Otherwise state "No comparison available"]

STEP 5: IMPRESSION
1. Primary finding(s)
2. Secondary findings
3. Incidental findings

STEP 6: CONFIDENCE
Score: [0.0-1.0]
Rationale: [supporting evidence]

STEP 7: RECOMMENDATIONS
[Follow-up imaging, biopsy, clinical correlation, etc.]

Now analyze these CT images:
{clinical_question}"""
```

---

## Template 3: Dermoscopy Analysis

```python
DERM_ANALYSIS_PROMPT = """You are an expert dermatologist analyzing a dermoscopic image.

METHODOLOGY:

STEP 1: LESION OVERVIEW
- Location: [anatomical site if known]
- Size: [estimated or measured]
- Shape: [round/oval/irregular]
- Symmetry: [symmetric/asymmetric]
- Border: [well-defined/ill-defined/fading]

STEP 2: COLOR ANALYSIS
Colors present (check all that apply):
- [ ] Light brown
- [ ] Dark brown
- [ ] Black
- [ ] Blue-gray
- [ ] Red
- [ ] White
- [ ] Other: ____

Color distribution: [uniform/variegated]

STEP 3: DERMOSCOPIC STRUCTURES
Identify presence of:
- Pigment network: [typical/atypical/absent]
- Dots/globules: [regular/irregular, distribution]
- Streaks/pseudopods: [present/absent, location]
- Blue-white veil: [present/absent]
- Regression structures: [present/absent]
- Vascular patterns: [type if present]
- Milia-like cysts: [present/absent]
- Comedo-like openings: [present/absent]

STEP 4: DERMOSCOPIC ALGORITHM APPLICATION
Apply appropriate algorithm:
- 7-point checklist score: [0-10]
- ABCD score: [if calculable]
- Menzies method: [benign/malignant features]

STEP 5: DIFFERENTIAL DIAGNOSIS
1. Most likely: [diagnosis with confidence]
2. Differential: [other considerations]

STEP 6: MANAGEMENT RECOMMENDATION
- [ ] Benign - routine follow-up
- [ ] Monitor with sequential imaging
- [ ] Biopsy recommended
- [ ] Urgent excision recommended

CONFIDENCE: [HIGH/MODERATE/LOW]

Now analyze this dermoscopic image:
{clinical_question}"""
```

---

## Template 4: Fundus/Ophthalmology Analysis

```python
FUNDUS_ANALYSIS_PROMPT = """You are an expert ophthalmologist analyzing a fundus image.

METHODOLOGY:

STEP 1: IMAGE QUALITY
- Clarity: [adequate/suboptimal]
- Field of view: [standard/wide-field]
- Media opacity: [clear/cataract/vitreous opacity]

STEP 2: OPTIC DISC
- Cup-to-disc ratio: [estimated value]
- Disc color: [pink/pale]
- Disc margins: [sharp/blurred]
- Neovascularization: [present/absent]
- Disc hemorrhages: [present/absent]

STEP 3: MACULA
- Foveal reflex: [present/absent]
- Macular edema: [present/absent]
- Drusen: [present/absent, characteristics]
- Pigmentary changes: [present/absent]
- Hemorrhage: [present/absent]
- Exudates: [present/absent]

STEP 4: RETINAL VESSELS
- Arteriolar caliber: [normal/attenuated/dilated]
- Venous caliber: [normal/dilated]
- A/V ratio: [estimated]
- Crossing changes: [present/absent]
- Neovascularization: [present/absent, location]

STEP 5: PERIPHERAL RETINA
- Hemorrhages: [location, type]
- Exudates: [hard/soft/cotton-wool spots]
- Pigmentary changes: [present/absent]
- Lattice degeneration: [present/absent]
- Holes/tears: [present/absent]

STEP 6: DIABETIC RETINOPATHY GRADING (if applicable)
Stage: [None/Mild NPDR/Moderate NPDR/Severe NPDR/PDR]
DME: [present/absent]

STEP 7: IMPRESSION
Primary diagnosis:
Secondary findings:

STEP 8: RECOMMENDATIONS
- [ ] Routine follow-up
- [ ] More frequent monitoring
- [ ] Referral to retina specialist
- [ ] Treatment indicated

CONFIDENCE: [0.0-1.0]

Now analyze this fundus image:
{clinical_question}"""
```

---

## Template 5: Histopathology Analysis

```python
HISTOPATH_ANALYSIS_PROMPT = """You are an expert pathologist analyzing histopathology images.

Note: You are viewing patches from a whole slide image. Consider that the full slide
context may not be visible.

METHODOLOGY:

STEP 1: TISSUE IDENTIFICATION
- Tissue type: [organ/tissue]
- Stain: [H&E/IHC/special stain]
- Magnification level: [estimated]

STEP 2: ARCHITECTURAL ASSESSMENT
- Overall architecture: [preserved/distorted/effaced]
- Growth pattern: [if neoplastic: solid/glandular/papillary/trabecular/etc.]
- Relationship to normal structures: [infiltrative/pushing/circumscribed]

STEP 3: CELLULAR FEATURES
- Cell type: [epithelial/mesenchymal/lymphoid/etc.]
- Cell size: [small/intermediate/large]
- Nuclear features:
  - Size: [normal/enlarged]
  - Shape: [regular/pleomorphic]
  - Chromatin: [fine/coarse/vesicular]
  - Nucleoli: [inconspicuous/prominent]
- Cytoplasm: [amount, character]
- Mitotic figures: [rare/frequent, atypical?]

STEP 4: ADDITIONAL FEATURES
- Necrosis: [present/absent, type]
- Inflammation: [type, distribution]
- Stromal changes: [desmoplasia, etc.]
- Vascular/lymphatic invasion: [present/absent/cannot assess]
- Perineural invasion: [present/absent/cannot assess]

STEP 5: DIFFERENTIAL DIAGNOSIS
Considering the findings:
1. Most likely: [diagnosis]
2. Differential: [alternatives]

STEP 6: LIMITATIONS
- [List any features that cannot be assessed from these patches]
- [Note if additional IHC or sections needed]

STEP 7: GRADE/STAGE (if applicable)
[Histologic grade, relevant staging features visible]

CONFIDENCE: [0.0-1.0]

IMPORTANT: If malignancy suspected, indicate need for complete slide review
and appropriate IHC workup.

Now analyze these histopathology patches:
{clinical_question}"""
```

---

## Template 6: Medical Text Reasoning (No Image)

```python
MEDICAL_REASON_PROMPT = """You are an expert physician providing clinical reasoning.

TASK: Answer the following medical question with detailed reasoning.

METHODOLOGY:

STEP 1: UNDERSTAND THE QUESTION
Restate the clinical scenario in your own words.

STEP 2: RELEVANT KNOWLEDGE
List the key medical concepts, mechanisms, or guidelines relevant to this question.

STEP 3: SYSTEMATIC ANALYSIS
Work through the problem step by step:
- What information is given?
- What information is missing?
- What assumptions must we make?
- What are the key decision points?

STEP 4: DIFFERENTIAL CONSIDERATIONS
If this is a diagnostic question:
- List possible diagnoses
- Rank by likelihood
- Explain supporting/refuting evidence for each

If this is a treatment question:
- List treatment options
- Weigh benefits vs risks
- Consider patient factors

STEP 5: CONCLUSION
State your answer clearly.

STEP 6: CONFIDENCE
Rate your confidence: [HIGH/MODERATE/LOW]
Explain: [Why this level of confidence?]

STEP 7: CAVEATS
Note any limitations, areas of uncertainty, or when specialist consultation would be appropriate.

QUESTION:
{clinical_question}

Please provide your analysis following the methodology above."""
```

---

## Template 7: Self-Verification Loop

Use this after initial analysis to catch hallucinations:

```python
VERIFICATION_PROMPT = """VERIFICATION CHECK

Review your analysis above and answer these questions:

1. EVIDENCE GROUNDING
For each finding you reported:
- Can you point to a specific area in the image that shows this?
- If you cannot, mark it as [UNVERIFIED]

2. ALTERNATIVE EXPLANATIONS
For your primary diagnosis:
- What else could explain these findings?
- Have you considered normal variants?
- Have you considered artifacts?

3. COMPLETENESS CHECK
- Did you examine all required anatomical regions?
- Are there any regions poorly visualized that you didn't mention?

4. CONFIDENCE CALIBRATION
Given your verification:
- Is your stated confidence level still appropriate?
- Should any findings be downgraded to "possible"?

5. REVISION
Based on this verification:
[ ] Original analysis stands - no changes needed
[ ] Revisions required (list below)

REVISED FINDINGS (if any):
"""
```

---

## Template 8: Confidence Extraction Helper

```python
CONFIDENCE_SUFFIX = """

MANDATORY CONFIDENCE STATEMENT

Based on your analysis, provide a numerical confidence score:

Score: [Enter a value between 0.0 and 1.0]

Interpretation guide:
- 0.90-1.00: Highly confident - clear evidence, classic presentation
- 0.70-0.89: Moderately confident - findings present but not definitive
- 0.50-0.69: Uncertain - subtle findings, multiple interpretations possible
- Below 0.50: Do not report - insufficient evidence for conclusion

Your score and brief rationale:"""
```

---

## Python Implementation

```python
# src/medgemma_mcp/prompts/templates.py

from enum import Enum

class Modality(str, Enum):
    CHEST_XRAY = "chest_xray"
    CT = "ct"
    MRI = "mri"
    DERMOSCOPY = "dermoscopy"
    FUNDUS = "fundus"
    HISTOPATH = "histopath"
    TEXT_ONLY = "text_only"

# Store all templates
TEMPLATES = {
    Modality.CHEST_XRAY: CXR_ANALYSIS_PROMPT,
    Modality.CT: CT_ANALYSIS_PROMPT,
    Modality.DERMOSCOPY: DERM_ANALYSIS_PROMPT,
    Modality.FUNDUS: FUNDUS_ANALYSIS_PROMPT,
    Modality.HISTOPATH: HISTOPATH_ANALYSIS_PROMPT,
    Modality.TEXT_ONLY: MEDICAL_REASON_PROMPT,
}

def build_radiology_prompt(modality: str, question: str) -> str:
    """Build the appropriate CoT prompt for the given modality."""
    template = TEMPLATES.get(Modality(modality), CXR_ANALYSIS_PROMPT)
    return template.format(clinical_question=question)

def add_verification_step(analysis: str) -> str:
    """Add verification prompt to existing analysis."""
    return f"{analysis}\n\n---\n\n{VERIFICATION_PROMPT}"

def add_confidence_suffix(prompt: str) -> str:
    """Ensure confidence scoring is requested."""
    return f"{prompt}\n{CONFIDENCE_SUFFIX}"
```

---

## Usage Example

```python
from medgemma_mcp.prompts.templates import build_radiology_prompt, Modality

# For chest X-ray analysis
prompt = build_radiology_prompt(
    modality=Modality.CHEST_XRAY,
    question="Evaluate for pneumonia or other acute findings"
)

# For text-only medical reasoning
prompt = build_radiology_prompt(
    modality=Modality.TEXT_ONLY,
    question="A 65-year-old patient presents with sudden onset chest pain..."
)
```

---

## Key Research Findings Supporting These Templates

1. **Two-Step Processing**: Organizing information before analyzing achieves 60.6% accuracy vs 56.5% baseline (p=0.042)

2. **Systematic Anatomical Review**: Prevents selective attention hallucinations

3. **Evidence Grounding**: Every claim tied to image location reduces fabrication

4. **Explicit Uncertainty**: Stating "possible" or "cannot exclude" reduces overconfidence

5. **Verification Loop**: Self-checking catches 86.4% of hallucinations

6. **Confidence Calibration**: Enables downstream filtering of uncertain outputs

7. **Negative Findings**: Stating what IS normal proves complete examination
