# ─────────────────────────────────────────────────────────────────────────────
# streamlit_app.py
# Bank Branch Expansion Simulator
#
# Prerequisite: Run the EDA notebook first to generate the processed CSVs.
# Run with:    streamlit run app/streamlit_app.py
#              (from the project root directory)
# ─────────────────────────────────────────────────────────────────────────────

import io
import math
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# ── Path setup ────────────────────────────────────────────────────────────────
# ROOT = Path(__file__).resolve().parent.parent
# DATA_DIR = ROOT / "data" / "processed"

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Branch Expansion Simulator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.section-hdr { font-size: 17px; font-weight: 700; color: #1B3A5C; margin: 20px 0 8px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Distance helpers — inline, no external src/ imports
# ─────────────────────────────────────────────────────────────────────────────

def haversine_vec(lat1: float, lon1: float,
                  lats2: np.ndarray, lons2: np.ndarray) -> np.ndarray:
    """Vectorised Haversine: great-circle distance (km) from one point to many."""
    R = 6_371.0
    lat1_r  = math.radians(lat1)
    lon1_r  = math.radians(lon1)
    lats2_r = np.radians(lats2)
    lons2_r = np.radians(lons2)
    dlat = lats2_r - lat1_r
    dlon = lons2_r - lon1_r
    a = (np.sin(dlat / 2) ** 2
         + math.cos(lat1_r) * np.cos(lats2_r) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_usd(v: float, d: int = 1) -> str:
    if v >= 1e12: return f"${v/1e12:.{d}f}T"
    if v >= 1e9:  return f"${v/1e9:.{d}f}B"
    if v >= 1e6:  return f"${v/1e6:.{d}f}M"
    if v >= 1e3:  return f"${v/1e3:.{d}f}K"
    return f"${v:.{d}f}"

def fmt_num(v: float) -> str:
    if v >= 1e6: return f"{v/1e6:.1f}M"
    if v >= 1e3: return f"{v/1e3:.1f}K"
    return str(int(v))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — from processed CSVs produced by the EDA notebook
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading processed data …")
def load_data():
    zip_path   = "https://raw.githubusercontent.com/Saurav-Sharma100/Bank-Branch-Streamlit-App/blob/main/Data/Processed/zip_metrics.csv"
    state_path = "https://raw.githubusercontent.com/Saurav-Sharma100/Bank-Branch-Streamlit-App/blob/main/Data/Processed/state_metrics.csv"
    
    

    # missing = [str(p) for p in [zip_path, state_path] if not p.exists()]
    # if missing:
    #     msg = (
    #         "**Required files not found:**\n"
    #         + "\n".join(f"- `{p}`" for p in missing)
    #         + "\n\n**Run the EDA notebook first** to generate these files in `data/processed/`."
    #     )
    #     return None, None, msg

    zip_df   = pd.read_csv(zip_path,   dtype={"ZIP": str})
    state_df = pd.read_csv(state_path)

    # Ensure ZIP is zero-padded 5-digit string
    zip_df["ZIP"] = zip_df["ZIP"].str.zfill(5)

    # Cast numeric columns
    for col in ["TOTAL_DEPOSITS", "BRANCH_COUNT", "LAT", "LON",
                "OPPORTUNITY_SCORE", "CAPTURE_PCT", "EXPECTED_CAPTURE_USD"]:
        if col in zip_df.columns:
            zip_df[col] = pd.to_numeric(zip_df[col], errors="coerce")

    # Drop rows missing coordinates — cannot map or optimise
    before = len(zip_df)
    zip_df = zip_df.dropna(subset=["LAT", "LON"]).reset_index(drop=True)
    dropped = before - len(zip_df)
    if dropped:
        st.warning(f"Dropped {dropped} ZIPs with missing coordinates.", icon="⚠️")

    return zip_df, state_df, None


zip_df, state_df, load_error = load_data()


# ─────────────────────────────────────────────────────────────────────────────
# Greedy optimizer
# ─────────────────────────────────────────────────────────────────────────────

def run_optimizer(
    zip_df: pd.DataFrame,
    budget: float,
    cost_per_branch: float,
    min_distance_km: float,
    states: list,
    min_capture: float,
    max_branches_override: int,
    percentile_floor: float = 0.0,
):
    """
    Greedy branch selection.

    Algorithm:
    1. Filter candidates by state, min capture, percentile floor, positive score.
    2. Sort by OPPORTUNITY_SCORE descending.
    3. Seed: select the top-ranked ZIP unconditionally.
    4. For each subsequent ZIP: add it only if it is >= min_distance_km from
       every already-selected branch (vectorised Haversine check).
    5. Stop when max_branches reached or no valid candidates remain.

    Returns (selected_df, summary_dict).
    """
    if budget <= 0 or cost_per_branch <= 0:
        return pd.DataFrame(), {
            "error": "Budget and cost per branch must both be greater than zero."
        }
    if cost_per_branch > budget:
        return pd.DataFrame(), {
            "error": (f"Cost per branch ({fmt_usd(cost_per_branch)}) "
                      f"exceeds total budget ({fmt_usd(budget)}).")
        }

    max_branches = int(budget // cost_per_branch)
    if max_branches_override and max_branches_override > 0:
        max_branches = min(max_branches, max_branches_override)

    # ── Filter ────────────────────────────────────────────────────────────
    cands = zip_df.copy()
    if states:
        cands = cands[cands["STATE"].isin(states)]
    if min_capture > 0:
        cands = cands[cands["EXPECTED_CAPTURE_USD"] >= min_capture]
    if percentile_floor > 0:
        threshold = cands["OPPORTUNITY_SCORE"].quantile(percentile_floor / 100)
        cands = cands[cands["OPPORTUNITY_SCORE"] >= threshold]

    cands = (cands[cands["OPPORTUNITY_SCORE"] > 0]
             .dropna(subset=["LAT", "LON"])
             .sort_values("OPPORTUNITY_SCORE", ascending=False)
             .reset_index(drop=True))

    if cands.empty:
        return pd.DataFrame(), {
            "error": ("No candidates meet the current filters. "
                      "Try relaxing state, minimum capture, or percentile floor.")
        }

    # ── Greedy loop ────────────────────────────────────────────────────────
    selected   = []
    sel_lats   = np.empty(0, dtype=float)
    sel_lons   = np.empty(0, dtype=float)

    for _, row in cands.iterrows():
        if len(selected) >= max_branches:
            break
        lat = float(row["LAT"])
        lon = float(row["LON"])
        if sel_lats.size > 0:
            dists = haversine_vec(lat, lon, sel_lats, sel_lons)
            if dists.min() < min_distance_km:
                continue
        selected.append(row)
        sel_lats = np.append(sel_lats, lat)
        sel_lons = np.append(sel_lons, lon)

    if not selected:
        return pd.DataFrame(), {
            "error": ("No branches selected. The minimum distance constraint "
                      "may be too strict — try reducing it.")
        }

    sel_df = pd.DataFrame(selected).reset_index(drop=True)

    # ── Distance to nearest selected neighbour ─────────────────────────────
    dist_nearest = []
    for i in range(len(sel_df)):
        others_lat = np.delete(sel_lats, i)
        others_lon = np.delete(sel_lons, i)
        if others_lat.size == 0:
            dist_nearest.append(None)
        else:
            d = haversine_vec(sel_lats[i], sel_lons[i], others_lat, others_lon)
            dist_nearest.append(round(float(d.min()), 2))
    sel_df["DIST_TO_NEAREST_KM"] = dist_nearest

    # ── Summary ────────────────────────────────────────────────────────────
    n           = len(sel_df)
    budget_used = n * cost_per_branch
    total_cap   = float(sel_df["EXPECTED_CAPTURE_USD"].sum())

    summary = {
        "branches_opened":        n,
        "max_branches_possible":  int(budget // cost_per_branch),
        "total_expected_capture": total_cap,
        "avg_capture_per_branch": total_cap / n,
        "avg_opportunity_score":  float(sel_df["OPPORTUNITY_SCORE"].mean()),
        "budget_total":           budget,
        "budget_used":            budget_used,
        "budget_used_pct":        round(budget_used / budget * 100, 1),
        "error":                  None,
    }
    return sel_df, summary


# ─────────────────────────────────────────────────────────────────────────────
# App header
# ─────────────────────────────────────────────────────────────────────────────

st.title("🏦 Bank Branch Expansion Simulator")

if load_error:
    st.error(load_error)
    st.stop()

st.caption(
    f"Dataset loaded: **{len(zip_df):,} ZIPs** · "
    f"**{int(zip_df['BRANCH_COUNT'].sum()):,} branches** · "
    f"**{zip_df['STATE'].nunique()} states** · "
    f"Total deposits: **{fmt_usd(zip_df['TOTAL_DEPOSITS'].sum())}**"
)

all_states = sorted(zip_df["STATE"].dropna().unique().tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Parameters")
    st.divider()

    selected_states = st.multiselect(
        "Target states  (blank = all)",
        options=all_states,
        default=[],
        help="Restrict the candidate pool to specific states.",
    )

    budget = st.number_input(
        "Total budget ($)",
        min_value=500_000, max_value=1_000_000_000,
        value=10_000_000, step=500_000, format="%d",
    )
    cost_per_branch = st.number_input(
        "Cost per branch ($)",
        min_value=100_000, max_value=50_000_000,
        value=2_000_000, step=100_000, format="%d",
    )
    st.caption(f"Budget allows up to **{int(budget // cost_per_branch)} branches**.")

    min_distance = st.slider(
        "Min distance between branches (km)",
        min_value=1.0, max_value=200.0, value=10.0, step=1.0,
    )
    min_capture = st.number_input(
        "Min expected capture / branch ($)",
        min_value=0, max_value=500_000_000,
        value=0, step=1_000_000, format="%d",
        help="Set to 0 to include all positive-score ZIPs.",
    )
    max_override = st.number_input(
        "Override max branches  (0 = use budget)",
        min_value=0, max_value=500, value=0, step=1,
    )

    st.divider()
    run_btn = st.button("🚀 Run Simulation", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_eda, tab_map, tab_results, tab_scenarios = st.tabs([
    "📊 EDA Overview",
    "🗺️ Opportunity Map",
    "🎯 Simulation Results",
    "📐 Scenario Comparison",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab_eda:
    st.markdown('<p class="section-hdr">National Summary</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Deposits",  fmt_usd(zip_df["TOTAL_DEPOSITS"].sum()))
    c2.metric("Total Branches",  f"{int(zip_df['BRANCH_COUNT'].sum()):,}")
    c3.metric("Unique ZIPs",     f"{len(zip_df):,}")
    c4.metric("States Covered",  str(zip_df["STATE"].nunique()))

    st.divider()
    left, right = st.columns(2)

    # Top 20 states by deposits
    with left:
        st.markdown('<p class="section-hdr">Top 20 States by Total Deposits</p>',
                    unsafe_allow_html=True)
        top_st = state_df.nlargest(20, "TOTAL_DEPOSITS").copy()
        top_st["DEP_B"] = top_st["TOTAL_DEPOSITS"] / 1e9
        fig = px.bar(
            top_st, x="DEP_B", y="STATE", orientation="h",
            labels={"DEP_B": "Deposits ($B)", "STATE": "State"},
            color="DEP_B", color_continuous_scale="Blues",
        )
        fig.update_layout(
            coloraxis_showscale=False, showlegend=False,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=0, r=0, t=10, b=0), height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Avg deposit per branch distribution
    with right:
        st.markdown('<p class="section-hdr">Avg Deposit per Branch by ZIP</p>',
                    unsafe_allow_html=True)
        avg_dep = (zip_df["TOTAL_DEPOSITS"] / zip_df["BRANCH_COUNT"]) / 1e6
        p99 = avg_dep.quantile(0.99)
        fig = px.histogram(
            avg_dep[avg_dep < p99], nbins=80,
            labels={"value": "Avg Deposits / Branch ($M)"},
            color_discrete_sequence=["#2563EB"],
        )
        fig.update_layout(showlegend=False, bargap=0.05,
                          margin=dict(l=0, r=0, t=10, b=0), height=420)
        st.plotly_chart(fig, use_container_width=True)

    # Pareto
    st.markdown('<p class="section-hdr">Deposit Concentration — Pareto Curve</p>',
                unsafe_allow_html=True)
    zs = zip_df.sort_values("TOTAL_DEPOSITS", ascending=False).reset_index(drop=True)
    zs["CUM_SHARE"]    = zs["TOTAL_DEPOSITS"].cumsum() / zs["TOTAL_DEPOSITS"].sum() * 100
    zs["ZIP_RANK_PCT"] = (zs.index + 1) / len(zs) * 100
    idx_80 = (zs["CUM_SHARE"] >= 80).idxmax()
    pct_80 = float(zs.loc[idx_80, "ZIP_RANK_PCT"])
    st.caption(f"Top **{pct_80:.1f}%** of ZIPs hold **80%** of all US bank deposits.")
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=zs["ZIP_RANK_PCT"], y=zs["CUM_SHARE"],
        mode="lines", line=dict(color="#2563EB", width=2),
    ))
    fig_p.add_hline(y=80, line_dash="dash", line_color="#EF4444",
                    annotation_text="80% of deposits")
    fig_p.update_layout(
        xaxis_title="ZIP Percentile (sorted by deposits)",
        yaxis_title="Cumulative Share (%)",
        margin=dict(l=0, r=0, t=10, b=0), height=280,
    )
    st.plotly_chart(fig_p, use_container_width=True)

    # Opportunity Score distribution
    st.markdown('<p class="section-hdr">Opportunity Score Distribution</p>',
                unsafe_allow_html=True)
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_o = px.histogram(
            zip_df, x="OPPORTUNITY_SCORE", nbins=80,
            labels={"OPPORTUNITY_SCORE": "Opportunity Score"},
            color_discrete_sequence=["#7C3AED"],
        )
        fig_o.add_vline(x=0, line_dash="dash", line_color="#EF4444",
                        annotation_text="Score = 0")
        fig_o.update_layout(showlegend=False, bargap=0.05,
                             margin=dict(l=0, r=0, t=10, b=0), height=280)
        st.plotly_chart(fig_o, use_container_width=True)
    with col_b:
        pos = int((zip_df["OPPORTUNITY_SCORE"] > 0).sum())
        neg = int((zip_df["OPPORTUNITY_SCORE"] <= 0).sum())
        st.metric("Positive-score ZIPs",  f"{pos:,}")
        st.metric("Negative-score ZIPs",  f"{neg:,}")
        st.metric("Top ZIP score",         f"{zip_df['OPPORTUNITY_SCORE'].max():.1f}")

    # Top 20 ZIPs table
    st.markdown('<p class="section-hdr">Top 20 ZIPs by Opportunity Score</p>',
                unsafe_allow_html=True)
    top20 = zip_df.nlargest(20, "OPPORTUNITY_SCORE")[
        ["ZIP", "STATE", "TOTAL_DEPOSITS", "BRANCH_COUNT",
         "OPPORTUNITY_SCORE", "CAPTURE_PCT", "EXPECTED_CAPTURE_USD"]
    ].copy()
    top20["TOTAL_DEPOSITS"]       = top20["TOTAL_DEPOSITS"].apply(fmt_usd)
    top20["EXPECTED_CAPTURE_USD"] = top20["EXPECTED_CAPTURE_USD"].apply(fmt_usd)
    top20["CAPTURE_PCT"]          = (top20["CAPTURE_PCT"] * 100).round(1).astype(str) + "%"
    top20["OPPORTUNITY_SCORE"]    = top20["OPPORTUNITY_SCORE"].round(2)
    top20.columns = ["ZIP", "State", "Total Deposits", "Branches",
                     "Opp. Score", "Capture %", "Expected Capture"]
    st.dataframe(top20, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Opportunity Map
# ══════════════════════════════════════════════════════════════════════════════

with tab_map:
    st.markdown('<p class="section-hdr">National Opportunity Map</p>',
                unsafe_allow_html=True)
    st.caption(
        "**Size** = total deposits in the ZIP.  "
        "**Colour** = Opportunity Score (green = high, red = low).  "
        "Click any bubble for detail."
    )

    map_states = st.multiselect(
        "Filter map by state", options=all_states, default=[], key="map_filter"
    )
    map_df = zip_df.copy()
    if map_states:
        map_df = map_df[map_df["STATE"].isin(map_states)]

    # Cap at 2,000 ZIPs for browser performance
    map_df = map_df.nlargest(2000, "TOTAL_DEPOSITS").copy()

    ctr_lat = float(map_df["LAT"].mean())
    ctr_lon = float(map_df["LON"].mean())
    m = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=4,
                   tiles="CartoDB positron")

    dep_min = float(map_df["TOTAL_DEPOSITS"].min())
    dep_max = float(map_df["TOTAL_DEPOSITS"].max())
    scr_min = float(map_df["OPPORTUNITY_SCORE"].min())
    scr_rng = max(float(map_df["OPPORTUNITY_SCORE"].max()) - scr_min, 1e-9)

    for _, row in map_df.iterrows():
        norm_s  = (float(row["OPPORTUNITY_SCORE"]) - scr_min) / scr_rng
        colour  = f"#{int(255*(1-norm_s)):02x}{int(200*norm_s):02x}40"
        radius  = 4 + 22 * (float(row["TOTAL_DEPOSITS"]) - dep_min) / max(dep_max - dep_min, 1)
        folium.CircleMarker(
            location=[float(row["LAT"]), float(row["LON"])],
            radius=radius, color=colour, fill=True, fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>ZIP {row['ZIP']} — {row['STATE']}</b><br>"
                f"Deposits: {fmt_usd(row['TOTAL_DEPOSITS'])}<br>"
                f"Branches: {int(row['BRANCH_COUNT'])}<br>"
                f"Opp. Score: {row['OPPORTUNITY_SCORE']:.2f}<br>"
                f"Exp. Capture: {fmt_usd(row['EXPECTED_CAPTURE_USD'])}",
                max_width=230,
            ),
        ).add_to(m)

    st_folium(m, width=None, height=520, returned_objects=[])
    st.caption(f"Showing top {len(map_df):,} ZIPs by deposit volume.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Simulation Results
# ══════════════════════════════════════════════════════════════════════════════

with tab_results:
    if not run_btn and "sim_sel" not in st.session_state:
        st.info("Set your parameters in the sidebar and click **🚀 Run Simulation**.")
        st.stop()

    if run_btn:
        with st.spinner("Running greedy optimizer …"):
            prog = st.progress(0, text="Filtering candidates …")
            sel_df, summary = run_optimizer(
                zip_df=zip_df,
                budget=budget,
                cost_per_branch=cost_per_branch,
                min_distance_km=min_distance,
                states=selected_states if selected_states else None,
                min_capture=float(min_capture),
                max_branches_override=int(max_override),
            )
            prog.progress(100)
            prog.empty()

        st.session_state["sim_sel"]     = sel_df
        st.session_state["sim_summary"] = summary
        st.session_state["sim_states"]  = selected_states

    sel_df  = st.session_state.get("sim_sel", pd.DataFrame())
    summary = st.session_state.get("sim_summary", {})

    if summary.get("error"):
        st.error(f"⚠️  {summary['error']}")
        st.stop()

    if sel_df.empty:
        st.warning("No branches selected. Adjust parameters and re-run.")
        st.stop()

    # ── KPI row ───────────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Portfolio Summary</p>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Branches Opened",        str(summary["branches_opened"]))
    k2.metric("Total Expected Capture", fmt_usd(summary["total_expected_capture"]))
    k3.metric("Budget Utilised",        f"{summary['budget_used_pct']} %")
    k4.metric("Avg Capture / Branch",   fmt_usd(summary["avg_capture_per_branch"]))

    st.divider()

    # ── Results table ──────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Recommended Branch Plan</p>',
                unsafe_allow_html=True)

    disp = sel_df[[
        "ZIP", "STATE", "TOTAL_DEPOSITS", "BRANCH_COUNT",
        "OPPORTUNITY_SCORE", "EXPECTED_CAPTURE_USD", "DIST_TO_NEAREST_KM",
    ]].copy()
    disp["TOTAL_DEPOSITS"]       = disp["TOTAL_DEPOSITS"].apply(fmt_usd)
    disp["EXPECTED_CAPTURE_USD"] = disp["EXPECTED_CAPTURE_USD"].apply(fmt_usd)
    disp["OPPORTUNITY_SCORE"]    = disp["OPPORTUNITY_SCORE"].round(2)
    disp["DIST_TO_NEAREST_KM"]   = disp["DIST_TO_NEAREST_KM"].apply(
        lambda x: f"{x:.1f} km" if x is not None else "—"
    )
    disp.columns = ["ZIP", "State", "Total Deposits", "Existing Branches",
                    "Opp. Score", "Expected Capture", "Dist. to Nearest"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

    buf = io.StringIO()
    sel_df.to_csv(buf, index=False)
    st.download_button(
        "⬇️  Download Branch Plan (CSV)",
        data=buf.getvalue(),
        file_name="branch_expansion_plan.csv",
        mime="text/csv",
    )

    st.divider()

    # ── Results map ────────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Expansion Map</p>', unsafe_allow_html=True)
    st.caption("Blue pins = selected branches.  Grey dots = top candidates considered.")

    sim_states = st.session_state.get("sim_states", [])
    res_map = folium.Map(
        location=[float(sel_df["LAT"].mean()), float(sel_df["LON"].mean())],
        zoom_start=5, tiles="CartoDB positron",
    )

    # Candidate cloud
    cand_show = zip_df.copy()
    if sim_states:
        cand_show = cand_show[cand_show["STATE"].isin(sim_states)]
    for _, row in cand_show.nlargest(1000, "OPPORTUNITY_SCORE").iterrows():
        folium.CircleMarker(
            location=[float(row["LAT"]), float(row["LON"])],
            radius=3, color="#9CA3AF", fill=True, fill_opacity=0.25,
        ).add_to(res_map)

    # Selected branches
    for i, row in sel_df.iterrows():
        folium.Marker(
            location=[float(row["LAT"]), float(row["LON"])],
            popup=folium.Popup(
                f"<b>Branch #{i+1}</b><br>"
                f"ZIP {row['ZIP']} — {row['STATE']}<br>"
                f"Opp. Score: {row['OPPORTUNITY_SCORE']:.2f}<br>"
                f"Expected Capture: {fmt_usd(row['EXPECTED_CAPTURE_USD'])}",
                max_width=210,
            ),
            icon=folium.Icon(color="darkblue", icon="home", prefix="fa"),
            tooltip=f"#{i+1}: ZIP {row['ZIP']} ({row['STATE']})",
        ).add_to(res_map)

    st_folium(res_map, width=None, height=500, returned_objects=[])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Scenario Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab_scenarios:
    st.markdown('<p class="section-hdr">Conservative vs Aggressive</p>',
                unsafe_allow_html=True)
    st.caption(
        "Uses the budget, cost, and state selection from the sidebar.\n\n"
        "**Conservative:** top 25% of ZIPs by score · 2× minimum distance\n\n"
        "**Aggressive:** all positive-score ZIPs · 0.5× minimum distance"
    )

    if st.button("▶️  Run Scenario Comparison", type="primary"):
        states_in = selected_states if selected_states else None
        with st.spinner("Conservative …"):
            cons_df, cons_sum = run_optimizer(
                zip_df=zip_df, budget=budget, cost_per_branch=cost_per_branch,
                min_distance_km=min_distance * 2.0, states=states_in,
                min_capture=float(min_capture), max_branches_override=0,
                percentile_floor=75,
            )
        with st.spinner("Aggressive …"):
            aggr_df, aggr_sum = run_optimizer(
                zip_df=zip_df, budget=budget, cost_per_branch=cost_per_branch,
                min_distance_km=max(min_distance * 0.5, 1.0), states=states_in,
                min_capture=float(min_capture), max_branches_override=0,
                percentile_floor=0,
            )
        st.session_state.update({
            "cons_df": cons_df, "cons_sum": cons_sum,
            "aggr_df": aggr_df, "aggr_sum": aggr_sum,
        })

    if "cons_sum" not in st.session_state:
        st.info("Click **▶️  Run Scenario Comparison** above.")
        st.stop()

    cons_sum = st.session_state["cons_sum"]
    aggr_sum = st.session_state["aggr_sum"]
    cons_df  = st.session_state["cons_df"]
    aggr_df  = st.session_state["aggr_df"]

    def _val(d, k):
        if d.get("error"):
            return "Error"
        v = d.get(k, 0)
        return fmt_usd(v) if isinstance(v, float) else str(v)

    cmp = pd.DataFrame({
        "Metric": [
            "Branches Opened",
            "Total Expected Capture",
            "Avg Capture / Branch",
            "Avg Opportunity Score",
            "Budget Utilised",
        ],
        "Conservative": [
            _val(cons_sum, "branches_opened"),
            _val(cons_sum, "total_expected_capture"),
            _val(cons_sum, "avg_capture_per_branch"),
            f"{cons_sum.get('avg_opportunity_score', 0):.2f}" if not cons_sum.get("error") else "—",
            f"{cons_sum.get('budget_used_pct', 0)} %" if not cons_sum.get("error") else "—",
        ],
        "Aggressive": [
            _val(aggr_sum, "branches_opened"),
            _val(aggr_sum, "total_expected_capture"),
            _val(aggr_sum, "avg_capture_per_branch"),
            f"{aggr_sum.get('avg_opportunity_score', 0):.2f}" if not aggr_sum.get("error") else "—",
            f"{aggr_sum.get('budget_used_pct', 0)} %" if not aggr_sum.get("error") else "—",
        ],
    })
    st.dataframe(cmp, use_container_width=True, hide_index=True)

    if not cons_sum.get("error") and not aggr_sum.get("error"):
        chart_df = pd.DataFrame({
            "Scenario": ["Conservative", "Aggressive"],
            "Expected Capture ($M)": [
                cons_sum.get("total_expected_capture", 0) / 1e6,
                aggr_sum.get("total_expected_capture", 0) / 1e6,
            ],
            "Branches": [
                cons_sum.get("branches_opened", 0),
                aggr_sum.get("branches_opened", 0),
            ],
        })
        fig_cmp = px.bar(
            chart_df, x="Scenario", y="Expected Capture ($M)", color="Scenario",
            text="Branches",
            color_discrete_map={"Conservative": "#6B7280", "Aggressive": "#2563EB"},
        )
        fig_cmp.update_traces(texttemplate="%{text} branches", textposition="outside")
        fig_cmp.update_layout(showlegend=False, height=320, margin=dict(t=30))
        st.plotly_chart(fig_cmp, use_container_width=True)

    d1, d2 = st.columns(2)
    if not cons_df.empty:
        b = io.StringIO(); cons_df.to_csv(b, index=False)
        d1.download_button("⬇️  Conservative Plan", b.getvalue(),
                           "conservative_plan.csv", "text/csv")
    if not aggr_df.empty:
        b = io.StringIO(); aggr_df.to_csv(b, index=False)
        d2.download_button("⬇️  Aggressive Plan", b.getvalue(),
                           "aggressive_plan.csv", "text/csv")
