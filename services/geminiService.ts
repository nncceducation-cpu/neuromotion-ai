import { GoogleGenAI } from "@google/genai";
import { AnalysisReport, ExpertCorrection, MotionConfig } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// Helper to check if current data matches a training example
const findMatchingExample = (current: any, examples: { inputs: any, groundTruth: ExpertCorrection }[]) => {
    return examples.find(ex => {
        const t = ex.inputs;
        // Check fingerprint (Entropy, Fluency, Complexity) 
        const entropyMatch = Math.abs(current.entropy - t.entropy) < 0.2; 
        const fluencyMatch = Math.abs(current.fluency - t.fluency) < 2.0; 
        const complexityMatch = Math.abs(current.complexity - t.complexity) < 0.2; 
        
        // Check Posture similarity if available (Strict check on categorical tone)
        const toneMatch = current.posture?.tone_label === t.posture?.tone_label;

        return entropyMatch && fluencyMatch && complexityMatch && toneMatch;
    });
};

export const generateGMAReport = async (biomarkers: any, trainingExamples: { inputs: any, groundTruth: ExpertCorrection }[] = []): Promise<AnalysisReport> => {
  // Use gemini-3-pro-preview for complex reasoning
  const model = "gemini-3-pro-preview";
  
  const normalizedInput = {
    entropy: biomarkers.peak_sample_entropy,
    fluency: biomarkers.average_jerk,
    complexity: biomarkers.peak_fractal_dimension,
    variabilityIndex: biomarkers.periodicity_index,
    csRiskScore: biomarkers.autocorrelation_risk_score,
    posture: biomarkers.posture,
    seizure: biomarkers.seizure,
    avg_kinetic_energy: biomarkers.avg_kinetic_energy,
    avg_root_stress: biomarkers.avg_root_stress
  };
  
  // 1. DETERMINISTIC CHECK
  const matchedExample = findMatchingExample(normalizedInput, trainingExamples);

  let trainingContext = "";
  let overrideInstruction = "";

  if (matchedExample) {
     overrideInstruction = `
     !!! CRITICAL INSTRUCTION: KNOWN VIDEO PATTERN RECOGNIZED !!!
     SYSTEM ALERT: "Memory Recall Active".
     The input biomarkers match a specific video signature that was previously analyzed and corrected by a human expert.
     YOUR PREVIOUS (WRONG) DIAGNOSIS: [Irrelevant]
     HUMAN EXPERT DIAGNOSIS: "${matchedExample.groundTruth.correctClassification}"
     EXPERT NOTES: "${matchedExample.groundTruth.notes}"
     MANDATORY ACTION:
     1. IGNORE your internal probability thresholds.
     2. You MUST classify this case as "${matchedExample.groundTruth.correctClassification}".
     3. In your "clinicalAnalysis", start with: "MEMORY RECALL: This analysis is based on a matched pattern from the Expert Knowledge Base."
     `;
  } else if (trainingExamples.length > 0) {
    trainingContext = `
    --- 🧠 EXPERT KNOWLEDGE BASE (FEW-SHOT LEARNING) ---
    ${trainingExamples.map((ex, i) => `
    [CASE ${i+1}]
    - INPUTS: Entropy=${ex.inputs.entropy.toFixed(2)}, Jerk=${ex.inputs.fluency.toFixed(2)}, Tone=${ex.inputs.posture?.tone_label}
    - CORRECT DIAGNOSIS: "${ex.groundTruth.correctClassification}"
    - REASONING: "${ex.groundTruth.notes}"
    `).join('\n')}
    `;
  }

  // --- NEW SYSTEM PROMPT BASED ON PYTHON NEONATAL ASSESSMENT TOOL ---
  const prompt = `
    You are an expert Neonatal Neurologist and Computer Vision Specialist.
    Your task is to analyze the provided BIOMARKER DATA (extracted from video) and generate a structured clinical assessment.

    ### DIAGNOSTIC CRITERIA (MODIFIED SARNAT & ILAE):

    1. **NEONATAL ENCEPHALOPATHY (Modified Sarnat Score)**:
       - **Normal:** Alert, flexed posture, normal tone, smooth movements. 
         - Markers: High Entropy, Normal Tone, Activity > 4.0, Low Frog Leg Score.
         - **AROUSAL:** High Arousal Index (>0.5) and High State Transition Probability (>0.1). Indicates baby wakes up/responds with complexity spike.
         - **POSITIVE CONSCIOUSNESS SIGNS:** High Crying Index (>0.3) OR High Eye Openness Index (>0.4). PRESENCE of these strongly suggests NORMAL or SARNAT I consciousness level (unless Seizure is present).
       - **Sarnat Stage I (Mild):** Hyperalert, stare, JITTERINESS, hyper-reflexia. 
         - Markers: HIGH Jerk (>8.0), HIGH Spontaneous Activity (>8.0), High Frequency (>5Hz), NO Eye Deviation.
       - **Sarnat Stage II (Moderate):** Lethargic/obtunded, HYPOTONIC (Frog-leg/Extended).
         - Markers: LOW Spontaneous Activity (<4.0), HIGH Frog Leg Score (>0.6), Low Entropy (<0.4). Seizures common.
         - **LETHARGY:** "Sticky States" (Low State Transition Probability < 0.05). Low Arousal Index (<0.3) means even when movement bursts occur, they lack complexity.
         - *Absence of Crying or Eye Opening*.
       - **Sarnat Stage III (Severe):** Comatose/stuporous, FLACCID tone.
         - Markers: ZERO Activity (~0), ZERO Entropy, High Frog Leg/Flatness.
         - **COMA:** ZERO Transitions (0.0). Flatline Arousal.
         - *Absence of Crying or Eye Opening*.

    2. **SEIZURE CLASSIFICATION RULES (STRICT - ILAE 2021)**:
       You must ONLY diagnose "Seizures" if the calculated biophysical type matches one of the three allowed categories.
       Look at \`seizure.calculated_type\` in the input JSON.

       - **IF calculated_type == 'None'**: 
         - You MUST NOT classify as "Seizures". 
         - **Reasoning:** 
           - If Frequency > 5Hz: It is TREMOR / JITTERINESS (Sarnat I).
           - If Frequency < 1.5Hz: It is NORMAL WRITHING / BREATHING.
           - If Crying Index > 0.4: Babies DO NOT cry during seizures (Veto).
       
       - **IF calculated_type == 'Clonic'**:
         - DIAGNOSIS: "Seizures" (Type: Clonic).
         - **Reasoning:** Rhythmic jerking strictly in the seizure band (1.5 - 5 Hz) with high regularity.
       
       - **IF calculated_type == 'Tonic'**:
         - DIAGNOSIS: "Seizures" (Type: Tonic).
         - **Reasoning:** EITHER Sustained total body stiffness (High Stiffness Score) OR Sustained Tonic Eye Deviation. Presence of one is sufficient.
       
       - **IF calculated_type == 'Myoclonic'**:
         - DIAGNOSIS: "Seizures" (Type: Myoclonic).
         - **Reasoning:** Shock-like, isolated high-jerk events without sustained rhythm.

    3. **DIFFERENTIAL DIAGNOSIS (MIMICS)**:
       - **Jitteriness:** Tremor-like, High Frequency (>5Hz), NO ocular deviation. Biomarkers: High Jerk, Low Eye Deviation.
       - **Benign Neonatal Sleep Myoclonus (BNSM):** Occurs during sleep/hypotonia. 

    ### CONSCIOUSNESS & AROUSAL RULES:
    1. IF (Crying Index > 0.3 OR Eye Openness Index > 0.4) AND (Seizure Detected == False):
       - Classification MUST lean towards **Normal** or **Sarnat Stage I**.
       - These are signs of intact brainstem arousal.
    2. IF (Seizure Detected == True):
       - Ignore Crying/Eye Opening as positive consciousness signs (Eye opening may be staring/deviation).
       - Classify based on Seizure Type.

    ${trainingContext}
    
    ${overrideInstruction}

    CURRENT INPUT BIOMARKERS:
    ${JSON.stringify(normalizedInput, null, 2)}

    OUTPUT FORMAT (JSON ONLY):
    {
      "classification": "Normal" | "Sarnat Stage I" | "Sarnat Stage II" | "Sarnat Stage III" | "Seizures",
      "seizureDetected": boolean,
      "seizureType": "None" | "Clonic" | "Myoclonic" | "Tonic",
      "differentialAlert": "String (e.g., 'Warning: Frequency indicates Tremor (>5Hz) not Seizure.') or null",
      "confidence": Number (0-100),
      "rawData": { ... },
      "clinicalAnalysis": "Detailed explanation citing specific biomarkers observed (e.g., 'Observed Stage II signs: Hypotonia indicated by Frog Leg Score (0.8) and Poverty of Movement (Activity < 2.0)...').",
      "recommendations": ["String", "String"]
    }
  `;

  try {
    const response = await ai.models.generateContent({
      model: model,
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        thinkingConfig: { thinkingBudget: 1024 },
        temperature: 0 // Force deterministic output
      }
    });

    const text = response.text;
    if (!text) throw new Error("No response from AI");
    
    const result = JSON.parse(text) as AnalysisReport;
    result.rawData = normalizedInput;

    if (matchedExample) {
        result.confidence = 100;
        result.expertCorrection = matchedExample.groundTruth; 
    }

    // Fallback defaults if AI omits fields
    if (result.seizureDetected === undefined) result.seizureDetected = result.classification === 'Seizures';
    if (!result.seizureType) result.seizureType = "None";

    return result;
  } catch (error) {
    console.error("Gemini API Error:", error);
    return {
        classification: "Sarnat Stage II",
        confidence: 0,
        seizureDetected: false,
        seizureType: "None",
        rawData: normalizedInput,
        clinicalAnalysis: "AI Service Interrupted.",
        recommendations: ["Check connection"]
    } as AnalysisReport;
  }
};

// --- PARAMETER OPTIMIZATION SERVICE ---

export const refineModelParameters = async (
  currentReport: AnalysisReport | null,
  expertDiagnosis: string,
  annotation: string,
  currentConfig: MotionConfig
): Promise<MotionConfig> => {
   
   const model = "gemini-2.5-flash"; // Fast model for parameter tuning

   const prompt = `
     You are a Senior Computer Vision Engineer optimizing a physics-based motor assessment algorithm for neonates.
     
     ### GOAL
     The current algorithm misclassified a video. You must adjust the signal processing parameters (MotionConfig) so that the physics engine extracts biomarkers that lead to the CORRECT diagnosis ("${expertDiagnosis}").
     
     ### CONTEXT
     - **Current Diagnosis**: ${currentReport?.classification || "Unknown"}
     - **Expert Diagnosis**: "${expertDiagnosis}"
     - **Expert Annotation**: "${annotation}"
     
     ### PHYSICS ENGINE LOGIC & PARAMETERS
     1. **sensitivity** (0.1-1.0): 
        - Controls peak detection threshold. 
        - INCREASE to detect faint/subtle seizures.
        - DECREASE to ignore noise (fix False Positives).
     
     2. **windowSize** (10-60 frames): 
        - Smoothing window for Entropy/Fractals. 
        - INCREASE = More smoothing -> Lower Entropy -> Better for Lethargy/Hypotonia detection.
        - DECREASE = Less smoothing -> Higher Entropy -> Better for detecting Jitteriness/Tremors.
     
     3. **entropyThreshold** (0.05-0.5): 
        - The 'r' radius for Sample Entropy matching. 
        - INCREASE = More matches -> Lower Entropy Score.
        - DECREASE = Strict matching -> Higher Entropy Score.
     
     4. **rhythmicityWeight** (0.0-2.0): 
        - Multiplier for the Seizure Rhythmicity Score.
        - INCREASE if Seizure was missed.
        - DECREASE if Normal/Jitteriness was misclassified as Seizure.
        
     5. **jerkThreshold** (1.0-10.0):
        - Baseline offset for Smoothness/Fluency.
        - INCREASE if 'Jitteriness' (High Jerk) was missed.
        - DECREASE if 'Normal' was misclassified as Jittery.
     
     6. **stiffnessThreshold** (0.1-2.0):
        - Variance divider for stiffness.
        - LOWER = Stricter (Requires absolute rigidity).
        - HIGHER = Lenient (Allows some movement while calling it "stiff").
     
     ### CURRENT CONFIGURATION
     ${JSON.stringify(currentConfig, null, 2)}
     
     ### BIOMARKER SNAPSHOT
     ${currentReport ? JSON.stringify(currentReport.rawData, null, 2) : "N/A"}
     
     ### OPTIMIZATION STRATEGY (Reasoning Required)
     - **Missed Seizure?** Boost 'sensitivity' & 'rhythmicityWeight'.
     - **False Seizure (Jitteriness)?** Reduce 'rhythmicityWeight', Reduce 'windowSize' (to catch high freq), Increase 'jerkThreshold'.
     - **Missed Lethargy (Sarnat II)?** We need LOW Entropy. Increase 'windowSize', Increase 'entropyThreshold'.
     - **False Lethargy?** We need HIGH Entropy. Decrease 'entropyThreshold', Decrease 'windowSize'.
     
     OUTPUT JSON ONLY (The new MotionConfig object).
   `;

   try {
     const response = await ai.models.generateContent({
        model: model,
        contents: prompt,
        config: { 
            responseMimeType: "application/json",
            temperature: 0 // Force deterministic tuning
        }
     });
     
     const text = response.text;
     if (!text) throw new Error("No config generated");
     return JSON.parse(text) as MotionConfig;

   } catch (error) {
      console.error("Parameter tuning failed", error);
      // Fallback: Slight random perturbation to simulate trying something new
      return {
          ...currentConfig,
          sensitivity: Math.min(1.0, currentConfig.sensitivity * 1.05),
          rhythmicityWeight: currentConfig.rhythmicityWeight * 0.95
      };
   }
};