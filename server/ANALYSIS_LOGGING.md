# Analysis Logging & Model Training Pipeline

## Overview

This document explains the improvements made to the Gemini AI integration and how to use the logged data for future model training.

---

## What Changed? (3 Key Improvements)

### ✅ 1. Structured Output Schema

**Before:**
```python
prompt = "Provide JSON output with classification and reasoning."
```

**After:**
```python
prompt = """
**REQUIRED OUTPUT FORMAT (strict JSON schema):**
{
  "classification": "Normal" | "Sarnat Stage I" | "Sarnat Stage II" | "Sarnat Stage III" | "Seizures" | "Uncertain",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<2-3 sentence clinical explanation>",
  "recommendations": "<optional clinical actions>"
}
"""
```

**Why this matters:**
- **Reliability:** Gemini now returns consistent JSON every time (no missing fields)
- **Confidence scores:** You can filter low-confidence predictions for manual review
- **Structured data:** Easier to parse, validate, and store in databases

---

### ✅ 2. Few-Shot Learning

**What is few-shot learning?**
Instead of just describing the task ("classify this"), we show the AI 2-3 examples of input → output pairs. The AI learns the pattern by example.

**Example in the prompt:**
```
Example 1:
Input: {"average_sample_entropy": 0.38, "peak_sample_entropy": 0.52, "average_jerk": 5.2}
Output: {
  "classification": "Normal",
  "confidence": 0.92,
  "reasoning": "All biomarkers within normal ranges..."
}

Example 2:
Input: {"average_sample_entropy": 0.82, "peak_sample_entropy": 1.15, "average_jerk": 9.3}
Output: {
  "classification": "Sarnat Stage II",
  "confidence": 0.78,
  "reasoning": "Elevated entropy indicates dysregulated movements..."
}
```

**Why this works:**
- AI learns what "normal" vs "abnormal" biomarker values look like
- AI learns your preferred reasoning style (citing specific values)
- Significantly improves accuracy (often 20-30% boost in specialized tasks)

**How to improve it further:**
- Replace the hardcoded examples with real cases from your logs
- Add more examples (3-5 is optimal for Gemini)
- Include edge cases (e.g., borderline values, contradictory biomarkers)

---

### ✅ 3. Automatic Logging to JSONL

**What gets logged:**
Every time an analysis runs, we save:
- **Input:** The biomarkers (entropy, jerk, etc.)
- **Output:** Gemini's classification, confidence, and reasoning
- **Ground truth:** `null` initially, filled in later when a doctor validates
- **Metadata:** Source (video/frontend), frame count, timestamp

**Where it's stored:**
`server/analysis_logs/gemini_predictions.jsonl`

**JSONL format example:**
```json
{"timestamp": "2026-02-08T15:30:00.123456+00:00", "biomarkers": {"average_sample_entropy": 0.72, "peak_sample_entropy": 0.95, "average_jerk": 7.8}, "gemini_prediction": {"classification": "Sarnat Stage II", "confidence": 0.81, "reasoning": "..."}, "ground_truth": null, "metadata": {"source": "video_upload", "frames_processed": 147}}
{"timestamp": "2026-02-08T15:45:12.987654+00:00", "biomarkers": {"average_sample_entropy": 0.42, "peak_sample_entropy": 0.58, "average_jerk": 5.3}, "gemini_prediction": {"classification": "Normal", "confidence": 0.93, "reasoning": "..."}, "ground_truth": "Normal", "metadata": {"source": "frontend_mediapipe", "frames_processed": 203}}
```

**Why JSONL?**
- Each line is a separate JSON object (no commas between records)
- Easy to append new records without reading the entire file
- Standard format for machine learning datasets (used by OpenAI, Hugging Face, etc.)

---

## How to Use the Logged Data

### Step 1: Collect Ground Truth Labels

After a patient is diagnosed by a neurologist, submit the ground truth via the `/validate` endpoint:

**Example request:**
```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2026-02-08T15:30:00.123456+00:00",
    "ground_truth_classification": "Sarnat Stage II",
    "doctor_notes": "EEG confirmed moderate HIE. Started therapeutic hypothermia."
  }'
```

**What happens:**
- The system finds the log entry with that timestamp
- Updates the `ground_truth` field with the doctor's diagnosis
- Adds `doctor_notes` and `validated_at` timestamp

---

### Step 2: Analyze Model Performance

Once you have 50+ validated cases, analyze Gemini's accuracy:

**Python script to calculate accuracy:**
```python
import json
from collections import Counter

# Load all log entries
with open("server/analysis_logs/gemini_predictions.jsonl", "r") as f:
    logs = [json.loads(line) for line in f]

# Filter to validated cases only
validated = [log for log in logs if log["ground_truth"] is not None]

print(f"Total analyses: {len(logs)}")
print(f"Validated cases: {len(validated)}")

# Calculate accuracy
correct = sum(1 for log in validated if log["gemini_prediction"]["classification"] == log["ground_truth"])
accuracy = correct / len(validated) if validated else 0
print(f"Accuracy: {accuracy:.2%}")

# Confusion matrix (which errors does Gemini make?)
errors = [(log["ground_truth"], log["gemini_prediction"]["classification"])
          for log in validated
          if log["ground_truth"] != log["gemini_prediction"]["classification"]]
print(f"\nMost common errors:")
for (true_label, pred_label), count in Counter(errors).most_common(5):
    print(f"  Predicted '{pred_label}' when actually '{true_label}': {count} times")
```

---

### Step 3: Train a Fine-Tuned Model (once you have 500+ cases)

**Why 500+?** Machine learning models need sufficient examples to learn patterns. For medical classification with ~5 classes (Normal, Sarnat I/II/III, Seizures), aim for at least 100 examples per class.

**Simple training pipeline (using scikit-learn):**
```python
import json
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load validated data
with open("server/analysis_logs/gemini_predictions.jsonl", "r") as f:
    logs = [json.loads(line) for line in f if json.loads(line)["ground_truth"]]

# 2. Extract features (X) and labels (y)
X = np.array([[
    log["biomarkers"]["average_sample_entropy"],
    log["biomarkers"]["peak_sample_entropy"],
    log["biomarkers"]["average_jerk"]
] for log in logs])

y = np.array([log["ground_truth"] for log in logs])

# 3. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train a Gradient Boosting classifier
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Save the model
import joblib
joblib.dump(model, "server/models/fine_tuned_classifier.pkl")
print("✓ Model saved to server/models/fine_tuned_classifier.pkl")
```

**Using the trained model in production:**
```python
import joblib

# Load once at startup
model = joblib.load("server/models/fine_tuned_classifier.pkl")

# In your endpoint
biomarker_array = np.array([[
    biomarkers["average_sample_entropy"],
    biomarkers["peak_sample_entropy"],
    biomarkers["average_jerk"]
]])
classification = model.predict(biomarker_array)[0]
confidence = model.predict_proba(biomarker_array).max()

# Still use Gemini for explanations
gemini_explanation = generate_gemini_report(biomarkers)  # But ignore its classification

return {
    "classification": classification,  # From your fine-tuned model (more accurate)
    "confidence": float(confidence),
    "reasoning": gemini_explanation["reasoning"]  # From Gemini (natural language)
}
```

---

## Best Practices

### Data Collection
- ✅ **Start logging from day 1** (even without ground truth)
- ✅ **Add unique identifiers** (patient_id, video_id) to metadata for traceability
- ✅ **Validate edge cases first** (borderline values, ambiguous movements)
- ❌ **Don't delete logs** - even "wrong" predictions are useful for understanding failure modes

### Ground Truth Validation
- ✅ **Get multiple expert opinions** for difficult cases (inter-rater reliability)
- ✅ **Include clinical context** in doctor_notes (EEG results, imaging findings)
- ✅ **Track changes over time** (if diagnosis changes after more tests, log both)

### Model Training
- ✅ **Wait until 500+ validated cases** before training a specialized model
- ✅ **Use cross-validation** to prevent overfitting
- ✅ **Monitor for data drift** (patient population changes over time)
- ✅ **Retrain periodically** (quarterly) as new data accumulates

### Privacy & Compliance
- ✅ **De-identify all logs** (no patient names, MRNs, or PHI)
- ✅ **Encrypt logs at rest** if storing on shared servers
- ✅ **Get IRB approval** if using data for research/publications
- ✅ **HIPAA compliance**: Ensure log storage meets regulatory requirements

---

## Future Enhancements

1. **Add more biomarkers to logs:**
   - Currently only logging 3 features (entropy, peak_entropy, jerk)
   - Physics engine computes 9+ features - log them all for richer models

2. **Build a validation UI:**
   - Web interface for doctors to review cases and submit ground truth
   - Show video playback alongside biomarkers
   - Track validation progress (X/Y cases labeled)

3. **Active learning pipeline:**
   - Automatically flag low-confidence predictions for prioritized review
   - Focus doctor validation efforts on cases where AI is uncertain

4. **A/B testing:**
   - Run Gemini and fine-tuned model in parallel
   - Compare which performs better on new cases
   - Gradual rollout of fine-tuned model once validated

---

## Questions?

- **Q: How many cases until I can train a model?**
  - A: Minimum 500 validated cases, ideally 1000+. Aim for balanced classes (100+ per label).

- **Q: Should I stop using Gemini once I have a fine-tuned model?**
  - A: No! Use both. Fine-tuned model for classification, Gemini for natural language explanations.

- **Q: What if Gemini makes mistakes?**
  - A: That's expected! Log everything, validate with doctors, and those "mistakes" become training data.

- **Q: Can I share logs with other hospitals?**
  - A: Only if fully de-identified and approved by your IRB. Consider federated learning for privacy-preserving collaboration.
