"""
Plotly chart functions for the Neuromotion AI Streamlit UI.
Replaces the Recharts-based Charts.tsx components.
"""

import plotly.graph_objects as go
from typing import List, Dict, Any, Optional

# Neutral color palette (matches original React UI)
COLORS = {
    "primary": "#171717",
    "secondary": "#525252",
    "tertiary": "#737373",
    "muted": "#a3a3a3",
    "dark": "#404040",
    "bg": "#f5f5f5",
    "white": "#ffffff",
    "grid": "#e5e5e5",
}

CHART_PALETTE = ["#171717", "#737373", "#a3a3a3", "#d4d4d4", "#404040"]

_LAYOUT_DEFAULTS = dict(
    margin=dict(l=10, r=10, t=30, b=10),
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Inter, sans-serif", size=11, color="#525252"),
    xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], gridwidth=1, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor=COLORS["grid"], gridwidth=1, zeroline=False),
    hovermode="x unified",
    height=260,
)


def _base_layout(**overrides) -> dict:
    layout = dict(_LAYOUT_DEFAULTS)
    layout.update(overrides)
    return layout


def entropy_chart(data: List[Dict[str, Any]]) -> go.Figure:
    timestamps = [d["timestamp"] for d in data]
    values = [d["entropy"] for d in data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=values, mode="lines", name="Entropy",
        line=dict(color=COLORS["primary"], width=1.5),
        fill="tozeroy", fillcolor="rgba(23,23,23,0.08)",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Sample Entropy", font=dict(size=12)),
        yaxis=dict(range=[0, 1.2], showgrid=True, gridcolor=COLORS["grid"]),
        xaxis=dict(showticklabels=False),
    ))
    return fig


def fluency_chart(data: List[Dict[str, Any]]) -> go.Figure:
    timestamps = [d["timestamp"] for d in data]
    values = [d["fluency_velocity"] for d in data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=values, mode="lines", name="Fluency",
        line=dict(color=COLORS["secondary"], width=1.5),
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Fluency (SAL Proxy)", font=dict(size=12)),
        xaxis=dict(showticklabels=False),
    ))
    return fig


def fractal_chart(data: List[Dict[str, Any]]) -> go.Figure:
    timestamps = [d["timestamp"] for d in data]
    values = [d["fractal_dim"] for d in data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=values, mode="lines", name="Fractal Dim",
        line=dict(color=COLORS["tertiary"], width=1.5, shape="hv"),
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Fractal Dimension", font=dict(size=12)),
        yaxis=dict(range=[1, 2]),
        xaxis=dict(showticklabels=False),
    ))
    return fig


def phase_space_chart(data: List[Dict[str, Any]]) -> go.Figure:
    px_vals = [d["phase_x"] for d in data]
    pv_vals = [d["phase_v"] for d in data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=px_vals, y=pv_vals, mode="markers", name="Limb State",
        marker=dict(color=COLORS["tertiary"], size=4, opacity=0.5),
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Phase Space", font=dict(size=12)),
        xaxis=dict(title="Position"),
        yaxis=dict(title="Velocity"),
    ))
    return fig


def kinetic_energy_chart(data: List[Dict[str, Any]]) -> go.Figure:
    timestamps = [d["timestamp"] for d in data]
    values = [d["kinetic_energy"] for d in data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=values, mode="lines", name="KE",
        line=dict(color=COLORS["dark"], width=1.5),
        fill="tozeroy", fillcolor="rgba(64,64,64,0.08)",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Kinetic Energy", font=dict(size=12)),
        xaxis=dict(showticklabels=False),
    ))
    return fig


def symmetry_chart(data: List[Dict[str, Any]]) -> go.Figure:
    timestamps = [d["timestamp"] for d in data]
    values = [d["bilateral_symmetry"] for d in data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=values, mode="lines", name="Symmetry",
        line=dict(color=COLORS["secondary"], width=1.5),
        fill="tozeroy", fillcolor="rgba(82,82,82,0.08)",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Bilateral Symmetry", font=dict(size=12)),
        yaxis=dict(range=[0, 1]),
        xaxis=dict(showticklabels=False),
    ))
    return fig


def confidence_timeline(data: List[Dict[str, Any]]) -> go.Figure:
    timestamps = [d["timestamp"] for d in data]
    avg_vis = [d.get("avg_visibility", 1.0) for d in data]
    min_vis = [d.get("min_visibility", 1.0) for d in data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=avg_vis, mode="lines", name="Avg Confidence",
        line=dict(color=COLORS["primary"], width=1.5),
        fill="tozeroy", fillcolor="rgba(23,23,23,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=min_vis, mode="lines", name="Min Confidence",
        line=dict(color=COLORS["muted"], width=1, dash="dot"),
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Pose Detection Confidence", font=dict(size=12)),
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(showticklabels=False),
    ))
    return fig


def confidence_gauge(value: float) -> go.Figure:
    color = COLORS["primary"] if value > 70 else COLORS["tertiary"] if value > 40 else COLORS["muted"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix="%", font=dict(size=28, color=COLORS["primary"])),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=0, tickcolor="white"),
            bar=dict(color=color),
            bgcolor=COLORS["bg"],
            borderwidth=0,
            steps=[
                dict(range=[0, 40], color="#f5f5f5"),
                dict(range=[40, 70], color="#e5e5e5"),
                dict(range=[70, 100], color="#d4d4d4"),
            ],
        ),
        title=dict(text="Confidence", font=dict(size=11)),
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=10),
        height=160,
        paper_bgcolor="white",
    )
    return fig


def mini_chart(data: List[Dict[str, Any]], key: str, color: str = "#171717") -> go.Figure:
    if not data:
        return go.Figure()
    timestamps = [d["timestamp"] for d in data]
    values = [d.get(key, 0) for d in data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=values, mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=80,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def comparison_entropy_chart(datasets: List[Dict[str, Any]]) -> go.Figure:
    """Multi-trace entropy time series for comparison view."""
    fig = go.Figure()
    for i, ds in enumerate(datasets):
        data = ds.get("data", [])
        timestamps = [d["timestamp"] for d in data]
        values = [d["entropy"] for d in data]
        fig.add_trace(go.Scatter(
            x=timestamps, y=values, mode="lines",
            name=ds.get("label", f"Dataset {i+1}"),
            line=dict(color=CHART_PALETTE[i % len(CHART_PALETTE)], width=1.5),
        ))
    fig.update_layout(**_base_layout(
        title=dict(text="Entropy Time Series", font=dict(size=12)),
        yaxis=dict(range=[0, 1.2]),
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    ))
    return fig


def comparison_phase_space(datasets: List[Dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    for i, ds in enumerate(datasets):
        data = ds.get("data", [])
        fig.add_trace(go.Scatter(
            x=[d["phase_x"] for d in data],
            y=[d["phase_v"] for d in data],
            mode="markers", name=ds.get("label", f"Dataset {i+1}"),
            marker=dict(color=CHART_PALETTE[i % len(CHART_PALETTE)], size=4, opacity=0.5),
        ))
    fig.update_layout(**_base_layout(
        title=dict(text="Phase Space Overlay", font=dict(size=12)),
        height=300,
        showlegend=True,
    ))
    return fig


_SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
]


def skeleton_chart(joints: Dict[str, Any], title: str = "YOLO26 Pose") -> go.Figure:
    """Render a skeleton as a Plotly scatter plot with bone connections."""
    fig = go.Figure()

    # Draw connections (bones)
    for j1_name, j2_name in _SKELETON_CONNECTIONS:
        j1 = joints.get(j1_name)
        j2 = joints.get(j2_name)
        if not j1 or not j2:
            continue
        if j1.get("visibility", 1.0) < 0.3 or j2.get("visibility", 1.0) < 0.3:
            continue
        fig.add_trace(go.Scatter(
            x=[j1["x"], j2["x"]], y=[j1["y"], j2["y"]],
            mode="lines", line=dict(color="rgba(163,163,163,0.6)", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    # Draw joints
    xs, ys, labels, colors = [], [], [], []
    for name, joint in joints.items():
        if not isinstance(joint, dict) or "x" not in joint:
            continue
        vis = joint.get("visibility", 1.0)
        if vis < 0.3:
            continue
        xs.append(joint["x"])
        ys.append(joint["y"])
        labels.append(name)
        colors.append(COLORS["primary"] if vis > 0.5 else COLORS["muted"])

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=labels,
        textposition="top center", textfont=dict(size=8, color=COLORS["muted"]),
        marker=dict(size=8, color=colors, line=dict(width=1, color="white")),
        showlegend=False,
        hovertemplate="%{text}<br>x=%{x:.1f} y=%{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color=COLORS["primary"])),
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[100, 0], showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        margin=dict(l=5, r=5, t=30, b=5),
        height=350,
    )
    return fig


def comparison_bar_chart(datasets: List[Dict[str, Any]], metric_key: str = "fluency_velocity", title: str = "Mean Velocity") -> go.Figure:
    labels = [ds.get("label", "?") for ds in datasets]
    values = [ds.get("stats", {}).get(metric_key, {}).get("mean", 0) for ds in datasets]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=COLORS["primary"],
        marker_cornerradius=4,
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=12)),
        height=300,
    ))
    return fig
