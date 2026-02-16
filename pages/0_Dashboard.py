"""Dashboard page — report list, stats, CSV export, compare selection."""

import streamlit as st
import sys, os, json, io, csv
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from api import _read_jsonl, _jsonl_entry_to_report, JSONL_FILE, _write_jsonl

st.title(f"Welcome back, {st.session_state.user['name']}")

# Load reports
entries = _read_jsonl()
entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
reports = [_jsonl_entry_to_report(e) for e in entries]

# Correction stats
corrections = [e for e in entries if e.get("expert_correction") or e.get("ground_truth")]
learned_count = len(corrections)

st.caption(f"{len(reports)} analyses completed · {learned_count} expert corrections learned")

# Stats row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Analyses", len(reports))
with col2:
    avg_conf = round(sum(r.get("confidence", 0) for r in reports) / max(1, len(reports)))
    st.metric("Avg Confidence", f"{avg_conf}%")
with col3:
    last_date = datetime.fromisoformat(reports[0]["date"]).strftime("%m/%d/%Y") if reports else "—"
    st.metric("Last Activity", last_date)

st.divider()

# Action buttons
bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
with bcol1:
    if st.button("New Analysis", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Pipeline.py")
with bcol2:
    if st.button("Training", use_container_width=True):
        st.session_state.app_mode = "training"
        st.switch_page("pages/1_Pipeline.py")
with bcol3:
    if st.button("Validation", use_container_width=True):
        st.switch_page("pages/4_Validation.py")
with bcol4:
    if st.button("Compare", use_container_width=True):
        st.switch_page("pages/3_Comparison.py")
with bcol5:
    if st.button("Trends", use_container_width=True):
        st.switch_page("pages/5_Graphs.py")

st.divider()

# Reports list + Knowledge base sidebar
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Recent Assessments")

    if not reports:
        st.info("No assessments yet. Start your first analysis above.")
    else:
        selected_ids = st.session_state.get("selected_report_ids", set())

        for i, report in enumerate(reports):
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([0.5, 3, 2, 1.5])
                with c1:
                    checked = st.checkbox("", value=report["id"] in selected_ids, key=f"sel_{i}", label_visibility="collapsed")
                    if checked:
                        selected_ids.add(report["id"])
                    else:
                        selected_ids.discard(report["id"])
                with c2:
                    st.markdown(f"**{report.get('videoName', 'Unknown')}**")
                    date_str = report.get("date", "")
                    if date_str:
                        try:
                            dt = datetime.fromisoformat(date_str)
                            st.caption(f"{dt.strftime('%m/%d/%Y %H:%M')}")
                        except Exception:
                            st.caption(date_str)
                with c3:
                    cls = report.get("classification", "Normal")
                    conf = report.get("confidence", 0)
                    st.markdown(f"`{cls}` — {conf}%")
                with c4:
                    btn_col1, btn_col2, btn_col3 = st.columns(3)
                    with btn_col1:
                        if st.button("View", key=f"view_{i}"):
                            st.session_state.selected_report = report
                            st.switch_page("pages/2_Report.py")
                    with btn_col2:
                        # CSV export
                        timeline = report.get("timelineData")
                        if timeline:
                            headers = ["timestamp", "entropy", "fluency_velocity", "fluency_jerk",
                                       "fractal_dim", "kinetic_energy", "bilateral_symmetry",
                                       "root_stress", "head_stability", "avg_visibility"]
                            buf = io.StringIO()
                            writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
                            writer.writeheader()
                            for row in timeline:
                                writer.writerow(row)
                            st.download_button("CSV", buf.getvalue(), f"neuromotion_{i}.csv",
                                               "text/csv", key=f"csv_{i}")
                    with btn_col3:
                        if st.button("Del", key=f"del_{i}"):
                            all_entries = _read_jsonl()
                            all_entries = [e for e in all_entries if e.get("id") != report["id"]]
                            _write_jsonl(all_entries)
                            st.rerun()

            st.session_state.selected_report_ids = selected_ids

        # Batch action buttons
        if selected_ids:
            batch1, batch2 = st.columns(2)
            with batch1:
                if st.button(f"Compare {len(selected_ids)} Selected", type="primary", use_container_width=True):
                    st.session_state.reports_to_compare = [
                        r for r in reports if r["id"] in selected_ids
                    ]
                    st.switch_page("pages/3_Comparison.py")
            with batch2:
                # Batch CSV download — combine all selected reports' timeline data
                selected_reports = [r for r in reports if r["id"] in selected_ids and r.get("timelineData")]
                if selected_reports:
                    headers = ["source", "timestamp", "entropy", "fluency_velocity", "fluency_jerk",
                               "fractal_dim", "kinetic_energy", "bilateral_symmetry",
                               "root_stress", "head_stability", "avg_visibility"]
                    buf = io.StringIO()
                    writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
                    writer.writeheader()
                    for r in selected_reports:
                        for row in r["timelineData"]:
                            row_copy = dict(row)
                            row_copy["source"] = r.get("videoName", "Unknown")
                            writer.writerow(row_copy)
                    st.download_button(
                        f"Download All ({len(selected_reports)} reports)",
                        buf.getvalue(), "neuromotion_batch.csv", "text/csv",
                        use_container_width=True,
                    )

with right_col:
    st.subheader("Knowledge Base")
    st.metric("Patterns Learned", learned_count)

    if corrections:
        by_cat = {}
        for e in corrections:
            cat = (e.get("expert_correction") or {}).get("correctClassification") or e.get("ground_truth", "Unknown")
            by_cat[cat] = by_cat.get(cat, 0) + 1
        for cat, count in by_cat.items():
            st.markdown(f"- **{cat}**: {count}")
    else:
        st.caption("No patterns learned yet. Correct AI diagnoses to teach the system.")

    st.divider()
    st.caption("The system uses video fingerprinting to identify previously corrected patterns. "
               "Similar biomarkers will apply your expert rules automatically.")
