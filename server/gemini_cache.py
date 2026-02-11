"""
Gemini Context Caching — stores the Neomotion technical document, research
protocol, and doctor corrections as cached context so every biomarker
analysis reuses the same pre-processed tokens.

Corrections flow:
  1. Doctor validates a prediction via /validate
  2. add_correction() appends it to corrections.json and rebuilds the cache
  3. Next generate_gemini_report() call uses the updated cache automatically
"""

import json
import os
from google.genai import Client
import google.genai.types as types

# Module-level state
_cache_name: str | None = None
_client: Client | None = None

CORRECTIONS_FILE = os.path.join(os.path.dirname(__file__), "analysis_logs", "corrections.json")

# ---------------------------------------------------------------------------
# Static document content (from the two PDFs)
# ---------------------------------------------------------------------------

TECHNICAL_DOCUMENT = """
NEOMOTION TECHNICAL DOCUMENT — Automated video-based movement analysis of
high-risk newborns using on-device AI to support bedside clinical decisions.

1. Introduction
This document presents the technical description of the system developed to
analyze movements of high-risk newborns and the reasoning that guided its
design. The goal is to demonstrate to the Ethics Committee of the University
of Calgary that each architectural decision was made to maximize security,
local control, data minimization, and adherence to ethical principles in
research involving vulnerable populations.

2. How the system was conceived
The starting point was the need to quantify spontaneous movements in newborns
without increasing the ethical risk inherent in video capture. Three central
premises: (1) Everything must occur locally — no sensitive data should leave
the environment under the direct control of the research team. (2) Collect
the minimum necessary — use video only as technical input, extract kinematic
variables, and discard any information not strictly required. (3) Analysis
tool, not diagnostic tool — designed to generate objective measures of
movement, not to replace clinical judgment or issue therapeutic
recommendations.

3. Data flow: from video to metric
Two authorized input paths: (1) Direct capture via device camera. (2) Manual
upload through the web interface. No fields for name, medical record number,
date of birth, or any personal identifier. Upon receiving the video, the API
triggers an isolated Python subprocess that:
  1. Reads the file directly from the local /videos folder.
  2. Uses pose-estimation to detect skeletal points (landmarks) — geometric
     coordinates, not facial recognition or advanced biometrics.
  3. Computes parameters: mean displacement, amplitude, temporal variation,
     sample entropy, jerk. These constitute the scientific raw material —
     numerical series representing motor behavior without automated clinical
     interpretation.
  4. Produces two outputs: a video with skeleton overlay and graphs generated
     with matplotlib summarizing the movement pattern over time.
Both files stored only in local /videos and /output folders. Each video
generates a closed analysis cycle with no resident processes, no cloud
transmission, and no automatic synchronization with external services.

4. Why processing is local
The decision to keep the workflow local is ethical. By avoiding communication
with external servers or cloud services, the system reduces: (1) risk of data
leakage; (2) complexity of complying with data-protection laws across
jurisdictions; (3) need to rely on third parties for storage or processing.
No built-in routines for remote backup, synchronization with cloud folders,
or automatic upload to any external provider.

5. Data minimization and deletion
Designed according to the principle of data minimization: (1) What the
algorithm sees is converted into body-point coordinates; (2) What the
researcher analyzes are graphs and numerical series; (3) The video is not
enriched with identifying information. The study protocol defines retention
duration and secure deletion procedures.

6. Intentional system limitations
By design, the system: (1) does not formulate diagnoses; (2) does not
generate clinical reports; (3) does not issue automatic alerts about health
status. Its function is strictly to provide objective, standardized, and
reproducible measurement of movement. Interpretation remains the sole
responsibility of the clinical and scientific team.

Furthermore, the de-identified data generated will be used to develop and
train an AI-based decision-support system to provide bedside healthcare
professionals with an automated indication of the infant's state. Any such
models will be trained and validated under appropriate research protocols,
with outputs presented strictly as decision-support information, with final
clinical judgment remaining the responsibility of the attending professional.
"""

RESEARCH_PROTOCOL = """
RESEARCH PROTOCOL — Machine learning and visual computing for diagnosis of
clinical encephalopathy.

Background:
Hypoxic Ischemic Encephalopathy (HIE) is a devastating condition that can
lead to mortality or multiple lifelong morbidities, including brain injury,
epilepsy, cerebral palsy, and learning disabilities. HIE is caused by
decreased blood and oxygen supply to the neonatal brain prior to delivery.
HIE is categorised as mild, moderate, and severe based on the modified Sarnat
scoring. The only evidence-based treatment for term and near term neonates
with HIE is Therapeutic Hypothermia (TH) — lowering core body temperature to
33-34 degrees centigrade for 72 hours. TH decreases mortality and long-term
neurodevelopmental disability in neonates with moderate to severe HIE. No
evidence supports cooling neonates with mild HIE.

TH is protective only when started within 6 hours of birth. Accurate
neurological examination in encephalopathic and sick neonates is vital.
The neonatal neurological examination requires specialized training.
Misinterpretation can affect outcome if TH is deferred in moderate/severe
HIE or initiated in mild HIE. Variability in neurological status in the
initial hours of birth is a challenge.

Modified Sarnat Encephalopathy Scale (6 components):
- Level of consciousness: Moderate = Lethargic; Severe = Stupor or coma
- Spontaneous activity: Moderate = Decreased; Severe = No activity
- Posture: Moderate = Distal flexion, complete extension; Severe = Decerebrate
- Tone: Moderate = Hypotonia (focal or general); Severe = Flaccid
- Primitive reflexes (Suck): Moderate = Weak; Severe = Absent
- Primitive reflexes (Moro): Moderate = Incomplete; Severe = Absent
- Autonomic system (Pupils): Moderate = Constricted; Severe = Deviated/dilated/nonreactive
- Autonomic system (Heart rate): Moderate = Bradycardia; Severe = Variable
- Autonomic system (Respiration): Moderate = Periodic breathing; Severe = Apnoea

Chalak et al. showed that level of consciousness, spontaneous movements,
autonomic nervous system, and posture have highest specificity. Tone and
primitive reflexes have high sensitivity but poor specificity for long-term
disability.

Study in three stages:
Stage 1: Create a library of normal movement patterns in term neonates and
train the ML algorithm to identify infants without encephalopathy.
(Sept 2025 — Aug 2026, n=250 healthy neonates, Foothills Medical Centre)

Stage 2: Create a library of abnormal movement patterns in term neonates
with diagnosis of mild, moderate or severe encephalopathy.

Stage 3: Train the algorithm to identify infants qualifying for therapeutic
hypothermia with moderate/severe encephalopathy.
(Stages 2-3: Sept 2026 — Aug 2029, n=90 neonates with perinatal depression,
FMC and Alberta Children's Hospital)

Study question: Is a machine learning algorithm able to detect neonates with
moderate to severe encephalopathy?

Objectives:
- Train ML algorithm to identify level of consciousness, spontaneous
  movements, and posture of term neonates in the initial 6 hours of life.
- Validate the algorithm in detecting neonatal encephalopathy with expert
  clinical assessment.
- Associate findings with neurodevelopmental outcomes at 18-21 months.

Primary outcome: Validation of the ML algorithm in detecting moderate to
severe encephalopathy in neonates with HIE.

Secondary outcomes:
1. Development of a library of normal movement patterns.
2. Correlating with MRI findings in neonates with HIE.
3. Correlating with continuous video EEG findings.
4. Correlating with long-term neurodevelopmental outcomes.

Neurodevelopmental assessment: Bayley Scales of Infant Development, 3rd
Edition (BSID-III) at 21 months. Cerebral palsy classified GMFCS 1-5.

NDI Classification:
- Normal: BSID-III score >= 85 in all domains
- Mild-moderate NDI: Any domain 70-84, or CP with GMFCS 1-2
- Severe NDI: Any domain <70, CP with GMFCS 3-5, hearing aid/cochlear
  implant, bilateral visual impairment, or autism spectrum disorder

Future directions: Once validated, the ML algorithm will be packaged into a
mobile software application for global implementation.
"""


# ---------------------------------------------------------------------------
# Corrections persistence
# ---------------------------------------------------------------------------

def _load_corrections() -> list[dict]:
    """Load corrections from disk."""
    if not os.path.exists(CORRECTIONS_FILE):
        return []
    try:
        with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_corrections(corrections: list[dict]) -> None:
    """Write corrections to disk."""
    os.makedirs(os.path.dirname(CORRECTIONS_FILE), exist_ok=True)
    with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(corrections, f, indent=2)


def _format_corrections_text(corrections: list[dict]) -> str:
    """Format corrections into a text block for the system instruction."""
    if not corrections:
        return ""
    lines = [
        "\n=== DOCTOR CORRECTIONS ===",
        "The following are real cases where a doctor corrected the AI's "
        "classification. Learn from these corrections to improve accuracy.\n",
    ]
    for i, c in enumerate(corrections, 1):
        lines.append(f"Correction {i}:")
        lines.append(f"  Biomarkers: {json.dumps(c['biomarkers'])}")
        lines.append(f"  AI predicted: {c['ai_classification']}")
        lines.append(f"  Doctor's correct diagnosis: {c['doctor_classification']}")
        if c.get("doctor_notes"):
            lines.append(f"  Doctor's notes: {c['doctor_notes']}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Client + Cache management
# ---------------------------------------------------------------------------

def get_client() -> Client | None:
    """Return a cached Client instance, or create one."""
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        return None
    _client = Client(api_key=api_key)
    return _client


def _build_system_instruction() -> str:
    """Assemble the full system instruction from documents + corrections."""
    corrections_text = _format_corrections_text(_load_corrections())
    return (
        "You are an expert Neonatal Neurologist working on the "
        "Neomotion research project at the University of Calgary. "
        "You have deep knowledge of the project's technical "
        "architecture and research protocol provided below. Use "
        "this context when analyzing infant motion biomarkers.\n\n"
        "=== TECHNICAL DOCUMENT ===\n"
        f"{TECHNICAL_DOCUMENT}\n\n"
        "=== RESEARCH PROTOCOL ===\n"
        f"{RESEARCH_PROTOCOL}"
        f"{corrections_text}"
    )


def _create_cache(client: Client) -> str | None:
    """Create a new Gemini cached-content object. Returns cache name."""
    try:
        cache = client.caches.create(
            model="gemini-3-pro-preview",
            config=types.CreateCachedContentConfig(
                display_name="neomotion-clinical-context",
                system_instruction=_build_system_instruction(),
                ttl="3600s",
            ),
        )
        print(f"gemini_cache: Created cache '{cache.name}'")
        return cache.name
    except Exception as e:
        print(f"gemini_cache: Failed to create cache: {e}")
        return None


def init_cache() -> str | None:
    """
    Create (or reuse) a Gemini cached-content object containing the static
    document context + any existing corrections. Returns cache name.
    """
    global _cache_name
    if _cache_name is not None:
        return _cache_name

    client = get_client()
    if client is None:
        print("gemini_cache: No API key found, skipping cache init.")
        return None

    _cache_name = _create_cache(client)
    return _cache_name


def rebuild_cache() -> str | None:
    """
    Delete the current cache and create a new one with updated corrections.
    Called after a doctor submits a correction via /validate.
    """
    global _cache_name
    client = get_client()
    if client is None:
        return None

    # Delete the old cache
    if _cache_name is not None:
        try:
            client.caches.delete(name=_cache_name)
            print(f"gemini_cache: Deleted old cache '{_cache_name}'")
        except Exception as e:
            print(f"gemini_cache: Failed to delete old cache: {e}")
        _cache_name = None

    # Create new cache with updated corrections
    _cache_name = _create_cache(client)
    return _cache_name


def add_correction(
    biomarkers: dict,
    ai_classification: str,
    doctor_classification: str,
    doctor_notes: str | None = None,
) -> str | None:
    """
    Store a doctor's correction and rebuild the Gemini cache.
    Returns the new cache name, or None on failure.
    """
    corrections = _load_corrections()
    corrections.append({
        "biomarkers": biomarkers,
        "ai_classification": ai_classification,
        "doctor_classification": doctor_classification,
        "doctor_notes": doctor_notes,
    })
    _save_corrections(corrections)
    print(f"gemini_cache: Stored correction #{len(corrections)} — rebuilding cache")
    return rebuild_cache()


def get_cache_name() -> str | None:
    """Return the current cache name, initializing if needed."""
    if _cache_name is None:
        return init_cache()
    return _cache_name
