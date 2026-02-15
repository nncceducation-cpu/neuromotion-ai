"""
Visualize baby motion biomarkers and rhythmicity from stored analysis data.
Reads gemini_predictions.jsonl and produces publication-quality graphs.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

DATA_PATH = Path(__file__).parent / "analysis_logs" / "gemini_predictions.jsonl"
OUTPUT_DIR = Path(__file__).parent / "analysis_logs" / "graphs"


def load_unique_profiles():
    """Load one entry per unique (filename, biomarkers) combination."""
    entries = []
    with open(DATA_PATH) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    seen = {}
    for e in entries:
        fn = e.get("metadata", {}).get("filename", "unknown")
        bio = e["biomarkers"]
        key = (fn, bio.get("average_sample_entropy"), bio.get("average_jerk"))
        if key not in seen:
            seen[key] = e
    return list(seen.values())


def make_label(entry):
    fn = entry["metadata"]["filename"]
    cls = entry["gemini_prediction"]["classification"]
    return f"{fn}\n({cls})"


def plot_biomarker_comparison(profiles, output_dir):
    """Bar chart comparing all biomarkers across videos."""
    labels = [make_label(p) for p in profiles]
    metrics = {
        "Avg Sample Entropy": [p["biomarkers"]["average_sample_entropy"] for p in profiles],
        "Peak Sample Entropy": [p["biomarkers"]["peak_sample_entropy"] for p in profiles],
        "Avg Jerk": [p["biomarkers"]["average_jerk"] for p in profiles],
        "Avg Fractal Dimension": [p["biomarkers"]["average_fractal_dimension"] for p in profiles],
        "Peak Fractal Dimension": [p["biomarkers"]["peak_fractal_dimension"] for p in profiles],
    }

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 14), constrained_layout=True)
    fig.suptitle("Motion Biomarker Comparison Across Videos", fontsize=16, fontweight="bold")

    colors = ["#2ecc71" if "Seizure" not in make_label(p) else "#e74c3c" for p in profiles]

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(values),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.savefig(output_dir / "biomarker_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: biomarker_comparison.png")


def plot_rhythmicity_scatter(profiles, output_dir):
    """Scatter plot: entropy vs fractal dimension — rhythmicity indicators."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for p in profiles:
        bio = p["biomarkers"]
        cls = p["gemini_prediction"]["classification"]
        color = "#e74c3c" if "Seizure" in cls else "#3498db"
        marker = "X" if "Seizure" in cls else "o"
        size = bio["average_kinetic_energy"] / 20  # scale marker by energy

        ax.scatter(bio["average_sample_entropy"], bio["average_fractal_dimension"],
                   s=max(size, 40), c=color, marker=marker, edgecolors="black",
                   linewidths=0.8, zorder=3)
        ax.annotate(p["metadata"]["filename"],
                    (bio["average_sample_entropy"], bio["average_fractal_dimension"]),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)

    # Reference zones
    ax.axvspan(0.3, 0.6, alpha=0.08, color="green", label="Normal entropy range (0.3–0.6)")
    ax.axhline(y=1.5, color="gray", linestyle="--", alpha=0.4, label="Fractal dim = 1.5 (Brownian)")

    ax.set_xlabel("Average Sample Entropy (regularity → complexity)", fontsize=12)
    ax.set_ylabel("Average Fractal Dimension (smoothness → roughness)", fontsize=12)
    ax.set_title("Rhythmicity Profile: Entropy vs Fractal Dimension\n(marker size ∝ kinetic energy)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "rhythmicity_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: rhythmicity_scatter.png")


def plot_motion_intensity(profiles, output_dir):
    """Grouped bar chart: kinetic energy, jerk, and root stress side by side."""
    labels = [p["metadata"]["filename"] for p in profiles]
    ke = [p["biomarkers"]["average_kinetic_energy"] for p in profiles]
    jerk = [p["biomarkers"]["average_jerk"] for p in profiles]
    stress = [p["biomarkers"]["average_root_stress"] for p in profiles]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Kinetic energy on left y-axis (much larger scale)
    bars1 = ax1.bar(x - width, ke, width, label="Kinetic Energy", color="#e67e22", edgecolor="white")
    ax1.set_ylabel("Kinetic Energy", fontsize=11, color="#e67e22")
    ax1.tick_params(axis="y", labelcolor="#e67e22")

    # Jerk and stress on right y-axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x, jerk, width, label="Avg Jerk", color="#9b59b6", edgecolor="white")
    bars3 = ax2.bar(x + width, stress, width, label="Root Stress", color="#1abc9c", edgecolor="white")
    ax2.set_ylabel("Jerk / Root Stress", fontsize=11, color="#555")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_title("Motion Intensity: Energy, Jerk & Muscle Stress", fontsize=14, fontweight="bold")

    # Combined legend
    handles = [bars1, bars2, bars3]
    labels_legend = ["Kinetic Energy", "Avg Jerk", "Root Stress"]
    ax1.legend(handles, labels_legend, loc="upper left", fontsize=9)

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fig.savefig(output_dir / "motion_intensity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: motion_intensity.png")


def plot_skeleton_poses(profiles, output_dir):
    """Draw the first-frame skeleton for each video side by side."""
    BONES = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ]

    n = len(profiles)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    fig.suptitle("First-Frame Skeleton Poses", fontsize=14, fontweight="bold")

    for ax, p in zip(axes, profiles):
        skeleton = p.get("first_frame_skeleton", {})
        joints = skeleton.get("joints", {})
        if not joints:
            ax.text(0.5, 0.5, "No skeleton", ha="center", va="center", transform=ax.transAxes)
            continue

        cls = p["gemini_prediction"]["classification"]
        color = "#e74c3c" if "Seizure" in cls else "#3498db"

        # Plot bones
        for j1, j2 in BONES:
            if j1 in joints and j2 in joints:
                ax.plot([joints[j1]["x"], joints[j2]["x"]],
                        [joints[j1]["y"], joints[j2]["y"]],
                        color=color, linewidth=2, alpha=0.7)

        # Plot joints
        xs = [j["x"] for j in joints.values()]
        ys = [j["y"] for j in joints.values()]
        ax.scatter(xs, ys, c=color, s=30, zorder=5, edgecolors="black", linewidths=0.5)

        ax.set_xlim(0, 100)
        ax.set_ylim(100, 0)  # invert y so head is on top
        ax.set_aspect("equal")
        ax.set_title(f"{p['metadata']['filename']}\n{cls}", fontsize=10)
        ax.set_xlabel("x (normalized)")
        ax.set_ylabel("y (normalized)")

    fig.savefig(output_dir / "skeleton_poses.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: skeleton_poses.png")


def plot_radar_chart(profiles, output_dir):
    """Radar/spider chart comparing normalized biomarkers per video."""
    metric_keys = [
        ("average_sample_entropy", "Entropy"),
        ("average_jerk", "Jerk"),
        ("average_fractal_dimension", "Fractal Dim"),
        ("average_kinetic_energy", "Kinetic Energy"),
        ("average_root_stress", "Root Stress"),
        ("peak_sample_entropy", "Peak Entropy"),
    ]

    # Normalize each metric to 0-1 range
    raw = {}
    for key, _ in metric_keys:
        vals = [p["biomarkers"][key] for p in profiles]
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx != mn else 1
        raw[key] = [(v - mn) / rng for v in vals]

    angles = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(profiles)))

    for i, p in enumerate(profiles):
        values = [raw[key][i] for key, _ in metric_keys]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=colors[i],
                label=p["metadata"]["filename"], markersize=5)
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([name for _, name in metric_keys], fontsize=10)
    ax.set_title("Normalized Motion Profile Radar", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    fig.savefig(output_dir / "radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: radar_chart.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles = load_unique_profiles()
    print(f"Loaded {len(profiles)} unique video profiles\n")
    print("Generating graphs...")

    plot_biomarker_comparison(profiles, OUTPUT_DIR)
    plot_rhythmicity_scatter(profiles, OUTPUT_DIR)
    plot_motion_intensity(profiles, OUTPUT_DIR)
    plot_skeleton_poses(profiles, OUTPUT_DIR)
    plot_radar_chart(profiles, OUTPUT_DIR)

    print(f"\nAll graphs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
