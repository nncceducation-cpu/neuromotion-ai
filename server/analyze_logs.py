#!/usr/bin/env python3
"""
Utility script to analyze Gemini prediction logs and calculate performance metrics.

Usage:
    python analyze_logs.py

This script:
1. Reads all logged predictions from analysis_logs/gemini_predictions.jsonl
2. Calculates accuracy for validated cases (those with ground truth)
3. Shows confusion matrix (which errors does Gemini make?)
4. Displays confidence distribution
5. Identifies cases that need validation
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def load_logs(log_file: str = "analysis_logs/gemini_predictions.jsonl"):
    """Load all log entries from JSONL file."""
    log_path = Path(__file__).parent / log_file

    if not log_path.exists():
        print(f"❌ No log file found at {log_path}")
        print("   Run some analyses first to generate logs.")
        return []

    logs = []
    with open(log_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping malformed line {i}: {e}")

    return logs


def analyze_accuracy(logs: list):
    """Calculate accuracy metrics for validated cases."""
    validated = [log for log in logs if log.get("ground_truth")]

    if not validated:
        print("\n📊 ACCURACY METRICS")
        print("   No validated cases yet. Use the /validate endpoint to add ground truth labels.")
        return

    # Calculate overall accuracy
    correct = sum(1 for log in validated
                  if log["gemini_prediction"].get("classification") == log["ground_truth"])
    accuracy = correct / len(validated)

    print("\n📊 ACCURACY METRICS")
    print(f"   Total validated cases: {len(validated)}")
    print(f"   Correct predictions: {correct}")
    print(f"   Accuracy: {accuracy:.1%}")

    # Confusion matrix
    errors = []
    for log in validated:
        pred = log["gemini_prediction"].get("classification", "Unknown")
        true = log["ground_truth"]
        if pred != true:
            errors.append((true, pred))

    if errors:
        print(f"\n❌ COMMON ERRORS ({len(errors)} total):")
        for (true_label, pred_label), count in Counter(errors).most_common(10):
            print(f"   • Predicted '{pred_label}' when actually '{true_label}': {count}x")
    else:
        print("\n✅ No prediction errors! (Either 100% accurate or need more validation)")

    # Per-class accuracy
    print("\n📈 PER-CLASS BREAKDOWN:")
    class_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for log in validated:
        true_label = log["ground_truth"]
        pred_label = log["gemini_prediction"].get("classification", "Unknown")
        class_stats[true_label]["total"] += 1
        if pred_label == true_label:
            class_stats[true_label]["correct"] += 1

    for label, stats in sorted(class_stats.items()):
        acc = stats["correct"] / stats["total"]
        print(f"   {label:20s}: {stats['correct']:3d}/{stats['total']:3d} ({acc:.1%})")


def analyze_confidence(logs: list):
    """Analyze confidence score distribution."""
    confidences = [log["gemini_prediction"].get("confidence", 0) for log in logs]

    if not confidences:
        return

    avg_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)
    max_conf = max(confidences)

    # Group by confidence ranges (0-100 integer scale)
    ranges = {
        "Very Low (0-20)":  sum(1 for c in confidences if c < 20),
        "Low (20-50)":      sum(1 for c in confidences if 20 <= c < 50),
        "Medium (50-85)":   sum(1 for c in confidences if 50 <= c < 85),
        "High (85-100)":    sum(1 for c in confidences if c >= 85)
    }

    print("\n🎯 CONFIDENCE DISTRIBUTION")
    print(f"   Average confidence: {avg_conf:.0f}%")
    print(f"   Range: {min_conf:.0f}% to {max_conf:.0f}%")
    print("\n   Breakdown:")
    for range_name, count in ranges.items():
        pct = count / len(confidences) * 100
        bar = "█" * int(pct / 2)  # Simple text bar chart
        print(f"   {range_name:20s}: {count:3d} ({pct:5.1f}%) {bar}")


def identify_review_candidates(logs: list):
    """Find cases that need validation or review."""
    unvalidated = [log for log in logs if not log.get("ground_truth")]
    low_confidence = [log for log in logs
                      if log["gemini_prediction"].get("confidence", 1.0) < 0.7]

    print("\n🔍 REVIEW CANDIDATES")
    print(f"   Unvalidated cases: {len(unvalidated)}")
    print(f"   Low confidence predictions (<0.7): {len(low_confidence)}")

    if low_confidence:
        print("\n   📌 Top 5 low-confidence cases to prioritize:")
        sorted_by_conf = sorted(low_confidence,
                                key=lambda x: x["gemini_prediction"].get("confidence", 0))[:5]
        for log in sorted_by_conf:
            ts = log["timestamp"]
            conf = log["gemini_prediction"].get("confidence", 0)
            classification = log["gemini_prediction"].get("classification", "Unknown")
            print(f"   • {ts}: {classification} (conf={conf:.2f})")
            print(f"     Validate with: curl -X POST http://localhost:8000/validate \\")
            print(f"       -d '{{\n         \"timestamp\": \"{ts}\",\n         \"ground_truth_classification\": \"<TRUE_LABEL>\"\n       }}'")


def analyze_biomarker_stats(logs: list):
    """Show biomarker value distributions."""
    def safe_get(log, key):
        return log.get("biomarkers", {}).get(key, 0)

    biomarker_keys = [
        ("average_sample_entropy", "Sample Entropy"),
        ("average_jerk", "Jerk (Fluency)"),
        ("average_fractal_dimension", "Fractal Dimension"),
        ("average_kinetic_energy", "Kinetic Energy"),
        ("average_root_stress", "Root Stress"),
        ("bilateral_symmetry_index", "Bilateral Symmetry"),
        ("average_lower_limb_ke", "Lower Limb KE"),
        ("angular_jerk_index", "Angular Jerk"),
        ("head_stability_index", "Head Stability"),
        ("average_com_velocity", "CoM Velocity"),
        ("elbow_rom", "Elbow ROM"),
        ("knee_rom", "Knee ROM"),
    ]

    print("\n📐 BIOMARKER STATISTICS")
    for key, label in biomarker_keys:
        values = [safe_get(log, key) for log in logs]
        values = [v for v in values if v != 0]
        if values:
            mean = sum(values) / len(values)
            print(f"   {label}:")
            print(f"     Mean: {mean:.3f}  Range: {min(values):.3f} to {max(values):.3f}  (n={len(values)})")
        else:
            print(f"   {label}: no data")


def main():
    print("=" * 70)
    print("  NEUROMOTION AI - LOG ANALYSIS REPORT")
    print("=" * 70)

    logs = load_logs()

    if not logs:
        return

    print(f"\n📁 DATASET SUMMARY")
    print(f"   Total analyses logged: {len(logs)}")
    validated_count = sum(1 for log in logs if log.get("ground_truth"))
    print(f"   Validated (with ground truth): {validated_count}")
    print(f"   Unvalidated: {len(logs) - validated_count}")

    # Calculate date range
    timestamps = [log["timestamp"] for log in logs]
    print(f"   Date range: {min(timestamps)} to {max(timestamps)}")

    # Run analyses
    analyze_accuracy(logs)
    analyze_confidence(logs)
    analyze_biomarker_stats(logs)
    identify_review_candidates(logs)

    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    if validated_count == 0:
        print("   ⚠️  No validated cases yet! Start by:")
        print("      1. Review some predictions manually")
        print("      2. Submit ground truth via /validate endpoint")
        print("      3. Build a dataset for future model training")
    elif validated_count < 50:
        print(f"   📈 {validated_count} validated cases - keep going!")
        print("      Goal: 50+ for initial accuracy assessment, 500+ for model training")
    elif validated_count < 500:
        print(f"   🎯 {validated_count} validated cases - good progress!")
        print("      Goal: 500+ validated cases before training a fine-tuned model")
        print("      Consider building a validation UI to accelerate labeling")
    else:
        print(f"   🚀 {validated_count} validated cases - ready for model training!")
        print("      See ANALYSIS_LOGGING.md for training instructions")
        print("      Consider experimenting with fine-tuned classifiers (XGBoost, etc.)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
