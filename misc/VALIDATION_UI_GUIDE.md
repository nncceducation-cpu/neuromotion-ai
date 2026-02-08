# Doctor Validation UI - User Guide

## Overview

The Doctor Validation UI is a comprehensive interface for neurologists to review AI predictions, visualize pose detection accuracy, and provide ground truth labels for model training.

---

## Features

### ✅ What's New

1. **Skeleton Visualization**
   - Visual display of YOLO26's pose detection on the first video frame
   - All 17 COCO keypoints labeled (nose, shoulders, elbows, wrists, hips, knees, ankles, etc.)
   - Color-coded joints: Red (face), Orange (torso), Cyan (limbs)
   - Verify that pose detection worked correctly before trusting the analysis

2. **Enhanced Logging**
   - First frame skeleton now saved with every analysis
   - Doctor notes field added to logs
   - Ground truth classification stored for model training
   - Timestamps for validation tracking

3. **Doctor Validation Interface**
   - Review pending cases (unvalidated predictions)
   - See AI's classification, confidence, and reasoning
   - View all biomarkers (entropy, jerk, fractal dimension)
   - Submit ground truth diagnosis with optional clinical notes

4. **New Backend Endpoints**
   - `/pending_validations` - Fetch cases needing review
   - `/validate` - Submit doctor's ground truth (enhanced with notes)

---

## How to Use

### Step 1: Access the Validation UI

1. Start the backend server:
   ```bash
   cd server
   uvicorn api:app --reload
   ```

2. Start the frontend:
   ```bash
   npm run dev
   ```

3. From the dashboard, click the **"Doctor Validation"** button (green, with clipboard icon)

### Step 2: Review Cases

**Left Panel: Case List**
- Shows all pending validations (or all cases if you toggle "Show All Cases")
- Each card displays:
  - Timestamp of analysis
  - AI's predicted classification
  - Confidence score
  - Key biomarkers (entropy, jerk, frame count)
  - Validation status (✓ if already validated)

**Right Panel: Detailed View**
- Skeleton visualization (canvas drawing of first frame)
  - Verify YOLO26 detected the pose correctly
  - All 17 keypoints labeled
  - If skeleton looks wrong, the analysis may be unreliable
- AI Prediction section:
  - Classification (Normal, Sarnat I/II/III, Seizures)
  - Confidence percentage
  - Gemini's clinical reasoning
  - Recommendations (if any)
- Biomarkers grid:
  - Average entropy, peak entropy
  - Average jerk
  - Total frames processed

### Step 3: Validate a Prediction

1. Click a case from the list to select it
2. Review the skeleton visualization - does YOLO26 detect the pose correctly?
3. Review the biomarkers and AI's reasoning
4. In the "Doctor Validation" section:
   - **Ground Truth Classification** (required): Select the correct diagnosis
     - Normal
     - Sarnat Stage I / II / III
     - Seizures
     - Uncertain (if more data needed)
   - **Clinical Notes** (optional): Add context
     - E.g., "EEG confirmed moderate HIE, started therapeutic hypothermia"
     - E.g., "MRI showed basal ganglia injury consistent with Sarnat II"
5. Click **"Submit Validation"**

### Step 4: Track Progress

Run the analysis script to see your validation progress:
```bash
cd server
python analyze_logs.py
```

**Sample output:**
```
📁 DATASET SUMMARY
   Total analyses logged: 127
   Validated (with ground truth): 43
   Unvalidated: 84

📊 ACCURACY METRICS
   Correct predictions: 38/43
   Accuracy: 88.4%

❌ COMMON ERRORS (5 total):
   • Predicted 'Normal' when actually 'Sarnat Stage I': 3x
   • Predicted 'Sarnat Stage II' when actually 'Sarnat Stage I': 2x
```

---

## Understanding the Skeleton Visualization

### 17 COCO Keypoints (in order)

| # | Keypoint Name | Body Part | Color |
|---|---------------|-----------|-------|
| 0 | nose | Face | Red |
| 1 | left_eye | Face | Red |
| 2 | right_eye | Face | Red |
| 3 | left_ear | Face | Red |
| 4 | right_ear | Face | Red |
| 5 | left_shoulder | Upper body | Orange |
| 6 | right_shoulder | Upper body | Orange |
| 7 | left_elbow | Arms | Cyan |
| 8 | right_elbow | Arms | Cyan |
| 9 | left_wrist | Arms | Cyan |
| 10 | right_wrist | Arms | Cyan |
| 11 | left_hip | Lower body | Orange |
| 12 | right_hip | Lower body | Orange |
| 13 | left_knee | Legs | Cyan |
| 14 | right_knee | Legs | Cyan |
| 15 | left_ankle | Legs | Cyan |
| 16 | right_ankle | Legs | Cyan |

### What to Look For

✅ **Good Pose Detection:**
- All joints are positioned correctly on the infant's body
- Skeleton follows natural human proportions
- No joints floating in space or wildly misplaced
- Symmetry looks reasonable (left/right sides similar)

❌ **Bad Pose Detection (flag for review):**
- Wrists detected on head or feet
- Knees in wrong location
- Missing critical joints (low visibility)
- Skeleton doesn't resemble human shape
- Multiple people detected (extra skeletons)

**If pose detection is bad, mark the case as "Uncertain" and add a note like:**
> "YOLO26 pose detection failed - wrists misdetected as ankles. Recommend re-analysis or manual review."

---

## API Usage (for integration)

### Fetch Pending Validations

```bash
# Get unvalidated cases only
curl http://localhost:8000/pending_validations

# Get all cases (including validated)
curl http://localhost:8000/pending_validations?include_validated=true

# Limit results
curl http://localhost:8000/pending_validations?limit=20
```

**Response:**
```json
{
  "cases": [
    {
      "timestamp": "2026-02-08T15:30:00.123456+00:00",
      "biomarkers": {
        "average_sample_entropy": 0.72,
        "peak_sample_entropy": 0.95,
        "average_jerk": 7.8,
        "frames_processed": 147
      },
      "gemini_prediction": {
        "classification": "Sarnat Stage II",
        "confidence": 0.81,
        "reasoning": "Elevated entropy...",
        "recommendations": "EEG monitoring recommended"
      },
      "first_frame_skeleton": {
        "joints": { "nose": {"x": 45.2, "y": 12.3, "z": 0, "visibility": 0.98}, ... },
        "keypoint_labels": ["nose", "left_eye", ...],
        "note": "All 17 COCO keypoints - verify YOLO26 detected pose correctly"
      },
      "ground_truth": null,
      "doctor_notes": null,
      "metadata": {"source": "video_upload", "filename": "patient_123.mp4"}
    }
  ],
  "total": 1,
  "showing": "unvalidated_only"
}
```

### Submit Validation

```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2026-02-08T15:30:00.123456+00:00",
    "ground_truth_classification": "Sarnat Stage II",
    "doctor_notes": "EEG confirmed moderate HIE. Started therapeutic hypothermia at 4 hours of life."
  }'
```

**Response:**
```json
{
  "status": "success",
  "message": "Ground truth label 'Sarnat Stage II' saved for analysis at 2026-02-08T15:30:00.123456+00:00"
}
```

---

## Data Flow

```
┌──────────────────┐
│  Video Upload    │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│  YOLO26 Pose Detection   │ ← Extract 17 keypoints from first frame
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Physics Engine          │ ← Compute entropy, jerk, fractal dimension
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Gemini AI Analysis      │ ← Classify: Normal / Sarnat I/II/III / Seizures
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  LOG TO FILE (gemini_predictions.jsonl)      │
│  - Biomarkers                                │
│  - Gemini prediction                         │
│  - First frame skeleton (NEW!)               │
│  - ground_truth: null (to be filled)         │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Doctor Validation UI    │ ← Review + Submit Ground Truth
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  /validate Endpoint      │ ← Update log with doctor's diagnosis + notes
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Labeled Dataset         │ ← Ready for model training!
│  (500+ cases = ready)    │
└──────────────────────────┘
```

---

## Best Practices

### For Doctors

1. **Verify Skeleton First**
   - Always check the skeleton visualization before trusting biomarkers
   - If pose detection is bad, the entire analysis is unreliable
   - Mark bad detections as "Uncertain" with notes

2. **Prioritize Low-Confidence Cases**
   - Focus validation efforts on cases where AI is uncertain (<70% confidence)
   - These are the most valuable for improving the model

3. **Add Clinical Context**
   - Include EEG findings, imaging results, clinical course
   - Example: "Initial Sarnat II, progressed to Sarnat III by day 2"
   - This context helps future clinicians understand edge cases

4. **Consistent Labeling**
   - Use the same diagnostic criteria across all cases
   - If unsure, select "Uncertain" rather than guessing
   - Document your reasoning in notes

### For System Administrators

1. **Regular Backups**
   - Backup `server/analysis_logs/gemini_predictions.jsonl` daily
   - This is your labeled dataset - don't lose it!

2. **Monitor Validation Progress**
   - Run `python analyze_logs.py` weekly
   - Track accuracy trends over time
   - Goal: 500+ validated cases before training a specialized model

3. **Privacy Compliance**
   - Ensure logs don't contain PHI (patient names, MRNs, etc.)
   - Use anonymized identifiers in metadata
   - Follow HIPAA/GDPR guidelines for data storage

---

## Troubleshooting

### "No pending validations" but I know there are unvalidated cases

**Solution:** Check that the backend server is running and logs exist:
```bash
ls server/analysis_logs/
# Should see: gemini_predictions.jsonl
```

### Skeleton canvas is blank

**Possible causes:**
1. First frame skeleton wasn't logged (old analysis before this feature)
2. YOLO26 failed to detect any keypoints (all visibility < 0.5)
3. Canvas rendering error (check browser console)

**Solution:** Re-run the analysis to generate new logs with skeleton data.

### Validation submission fails with 404

**Error:** `No analysis found with timestamp X`

**Cause:** The timestamp doesn't match any log entry (check for typos or timezone issues)

**Solution:** Copy the exact timestamp from the UI (ISO 8601 format with timezone)

### TypeScript errors in IDE

The IDE may show warnings like `'ValidationView' is declared but never read` - these are false positives. The component is used in JSX. To fix:
- Restart TypeScript server in VSCode: `Ctrl+Shift+P` → "TypeScript: Restart TS Server"
- Or just ignore - the app will run fine

---

## Future Enhancements

### Planned Features

1. **Batch Validation**
   - Validate multiple cases at once
   - Bulk operations for common diagnoses

2. **Video Playback**
   - Show full video alongside skeleton
   - Scrub through timeline to verify motion patterns

3. **Inter-Rater Reliability**
   - Multiple doctors validate same case
   - Calculate agreement scores (Cohen's kappa)

4. **Active Learning**
   - AI flags uncertain cases for prioritized review
   - Focus doctor time on most valuable validations

5. **Export Validated Dataset**
   - One-click export to CSV/JSON for model training
   - Split train/test sets automatically

---

## Summary

The Doctor Validation UI transforms your Neuromotion AI system from a static analyzer into a **self-improving learning system**:

1. **Every analysis is logged** with skeleton visualization
2. **Doctors review and validate** predictions
3. **Labeled dataset grows** over time
4. **Accuracy improves** as you collect more ground truth
5. **Eventually train a specialized model** (once you have 500+ cases)

**Next Steps:**
1. Run some analyses (video upload or live mode)
2. Access the Validation UI from dashboard
3. Review 5-10 cases to test the workflow
4. Check progress with `python analyze_logs.py`
5. Aim for 50 validated cases in the first month

---

## Questions?

- **How many cases until I can train my own model?** 500+ validated cases minimum, ideally 1000+
- **Can I edit a validation after submitting?** Currently no - submit another validation with the same timestamp to overwrite
- **What if two doctors disagree?** Log both validations and use majority vote or senior clinician's judgment
- **Can I export data?** Yes, `gemini_predictions.jsonl` is standard JSONL format - import into Python/pandas easily

Happy validating! 🏥
