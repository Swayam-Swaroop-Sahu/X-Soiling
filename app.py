"""
X-SOILING: Physical Tipping Point & Explainable AI Dashboard for Solar PV Operations.

Interactive Streamlit Application delivering:
1. Live PV Telemetry & Soiling Monitoring (SR, Delta_SR, Diurnal Cycles).
2. XAI Studio (SHAP 1D Wind Resuspension Curve & 2D Cementation Cliff Contours).
3. 48h Forecast Forensics & Interactive What-If Scenario Simulator.
4. Smart Cleaning Dispatch Engine & Water/Cost Economics Calculator.
5. Model Governance & Purged Time-Series Cross-Validation Benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Import core physical modules from src
from src.data_loader import PVDataLoader
from src.feature_engineering import FeatureEngineer
from src.operational_engine import OperationalEngine
from src.threshold_miner import DiscoveredTippingPoints, ThresholdMiner
from src.train import ModelTrainer
from src.xai_engine import XAIEngine

# ==============================================================================
# 1. PAGE CONFIGURATION & CUSTOM DESIGN SYSTEM
# ==============================================================================
st.set_page_config(
    page_title="X-Soiling | PV Operations & Explainable AI",
    page_icon="PV",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Executive Theme Palette */
    .stApp {
        background-color: #090d16;
        color: #f1f5f9;
    }

    /* Executive Hero Card */
    .hero-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }

    /* Executive Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }

    .kpi-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 18px 20px;
        transition: border-color 0.2s ease;
    }

    .kpi-card:hover {
        border-color: #38bdf8;
    }

    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
    }

    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }

    .kpi-subtext {
        font-size: 0.80rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Corporate Badges */
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        display: inline-block;
    }
    .badge-high {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        display: inline-block;
    }
    .badge-free {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid #10b981;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        display: inline-block;
    }
    .badge-nominal {
        background-color: rgba(148, 163, 184, 0.12);
        color: #94a3b8;
        border: 1px solid #475569;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        display: inline-block;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0px 0px;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.88rem;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.12) !important;
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    /* Section divider */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: 0.02em;
        margin: 22px 0 14px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #1f2937;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. CACHED DATA & PIPELINE SERVICES
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_and_preprocess_data() -> pd.DataFrame:
    """Loads raw PV soiling dataset and executes mechanistic feature extraction."""
    loader = PVDataLoader(raw_dir="data/raw", processed_dir="data/processed")
    raw = loader.load_data(auto_generate_if_empty=True)
    fe = FeatureEngineer()
    featured = fe.extract_features(raw, compute_targets=True)
    return featured


@st.cache_resource(show_spinner=False)
def get_trained_pipeline(df: pd.DataFrame):
    """Retrieves or trains Champion Model and mines physical tipping points."""
    fe = FeatureEngineer()
    feature_cols = fe.get_feature_names()
    trainer = ModelTrainer(output_dir="models")
    X, y, clean = trainer.prepare_dataset(df, feature_cols, target_col="delta_soiling_ratio")

    # Load or train champion
    model_name, model, metrics = trainer.compare_and_select_best(X, y, clean, "delta_soiling_ratio")

    # Compute SHAP values
    xai = XAIEngine(model, list(X.columns), output_dir="figures")
    shap_vals = xai.compute_shap_values(X, max_samples=400)
    xai.compute_interaction_values(X, max_samples=300)

    # Mine tipping points
    miner = ThresholdMiner(smoothing_window=15)
    wind_v, wind_s = xai.get_1d_dependence_curve("wind_speed")
    rh_v, rh_s = xai.get_1d_dependence_curve("relative_humidity")
    dew_inter_v, pm10_inter_v, inter_dew_pm = xai.extract_2d_interaction("dew_deficit", "pm10")

    tipping_points = miner.mine_all_thresholds(
        wind_speeds=wind_v,
        wind_shap=wind_s,
        dew_deficits=dew_inter_v,
        pm10_vals=pm10_inter_v,
        interaction_dew_pm=inter_dew_pm,
        rh_vals=rh_v,
        rh_shap=rh_s,
    )

    return model_name, model, metrics, xai, tipping_points, X, y


# ==============================================================================
# 3. SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.image("assets/banner.jpg", use_container_width=True)
    st.markdown("### **X-SOILING PLATFORM**")
    st.caption("Explainable AI & Physics Tipping Points in Solar PV Dust Deposition")

    st.markdown("---")
    navigation = st.radio(
        "Navigation",
        [
            "Executive Telemetry & Asset Monitor",
            "Physical Tipping Points & Explainability",
            "Scenario Analysis & Weather Forensics",
            "Operational Dispatch & Asset Optimization",
            "Model Governance & Validation Metrics",
            "Research Manuscript & Technical Suite",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### **Plant Specifications**")
    plant_capacity_mw = st.number_input("Array Capacity (MWp)", min_value=1.0, max_value=500.0, value=25.0, step=5.0)
    wash_cost_per_mw = st.number_input("Cleaning Cost ($/MW)", min_value=50.0, max_value=2000.0, value=320.0, step=20.0)
    water_per_mw_liters = st.number_input("Water Use (L/MW)", min_value=100.0, max_value=20000.0, value=2500.0, step=250.0)

    st.markdown("---")
    st.caption("Engineered for Utility-Scale PV & Desert Climate Fleets.")


# ==============================================================================
# 4. LOAD CORE DATA & TRAIN PIPELINE
# ==============================================================================
with st.spinner("Initializing physics engine & Explainable AI models..."):
    df = load_and_preprocess_data()
    model_name, model, metrics, xai, tp, X, y = get_trained_pipeline(df)
    fe = FeatureEngineer()
    op_engine = OperationalEngine(tipping_points=tp, fe=fe, output_dir="reports")


# ==============================================================================
# 5. HERO BANNER
# ==============================================================================
st.markdown(
    f"""
    <div class="hero-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
            <div>
                <span style="color: #38bdf8; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase;">
                    Explainable AI Solar Operations Engine
                </span>
                <h1 style="color: #f8fafc; font-size: 2.1rem; font-weight: 800; margin: 4px 0 8px 0;">
                    Photovoltaic Soiling & Mechanistic Tipping Points
                </h1>
                <p style="color: #94a3b8; font-size: 0.95rem; margin: 0; max-width: 780px;">
                    Uncovering the exact non-linear boundaries where humidity cements particulate matter onto glass,
                    and identifying the aerodynamic wind resuspension threshold (<code style="color: #38bdf8;">v_crit</code>)
                    for automated zero-cost natural cleaning dispatch.
                </p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; color: #64748b; text-transform: uppercase;">Champion Model</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #38bdf8;">{model_name.upper()}</div>
                <div style="margin-top: 6px;">
                    <span class="badge-free">Active v_crit: {tp.wind_resuspension_threshold_mps:.2f} m/s</span>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# VIEW 1: EXECUTIVE TELEMETRY & ASSET MONITOR
# ==============================================================================
if navigation == "Executive Telemetry & Asset Monitor":
    st.markdown('<div class="section-title">Plant Telemetry & Operational Metrics</div>', unsafe_allow_html=True)

    # Extract latest observation
    latest_row = df.iloc[-1]
    curr_sr = float(latest_row["soiling_ratio"])
    curr_power_clean = float(latest_row["power_clean"])
    curr_power_soiled = float(latest_row["power_soiled"])
    curr_dew_def = float(latest_row["dew_deficit"])
    curr_pm10 = float(latest_row["pm10"])
    curr_pm10_24h = float(latest_row.get("pm10_exposure_24h", curr_pm10 * 24.0))

    # Metric Cards HTML
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Instantaneous Soiling Ratio (SR)</div>
                <div class="kpi-value" style="color: {'#34d399' if curr_sr > 0.90 else '#f87171'};">{curr_sr:.3f}</div>
                <div class="kpi-subtext">{"Nominal Efficiency" if curr_sr > 0.90 else "Degraded Efficiency (Dust)"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        power_loss_pct = (1.0 - curr_sr) * 100.0
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Current Soiling Loss</div>
                <div class="kpi-value" style="color: #fbbf24;">{power_loss_pct:.1f}%</div>
                <div class="kpi-subtext">Clean: {curr_power_clean:.1f} kW | Soiled: {curr_power_soiled:.1f} kW</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Condensation Deficit</div>
                <div class="kpi-value" style="color: {'#f87171' if curr_dew_def < tp.cementation_dew_deficit_celsius else '#38bdf8'};">{curr_dew_def:.1f} °C</div>
                <div class="kpi-subtext">Cementation Limit: &lt; {tp.cementation_dew_deficit_celsius:.1f} °C</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">24h PM10 Cumulative Dose</div>
                <div class="kpi-value" style="color: #c084fc;">{curr_pm10_24h:.0f}</div>
                <div class="kpi-subtext">µg·h/m³ (Coarse dust exposure)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Time series range slider filter
    st.markdown('<div class="section-title">Dynamic Diurnal Performance & Atmospheric Telemetry</div>', unsafe_allow_html=True)
    view_window_days = st.slider("Observation Timeline Horizon (Days)", min_value=7, max_value=120, value=30, step=7)
    df_slice = df.iloc[-int(view_window_days * 24) :].copy()

    # Create Multi-Pane Interactive Plotly Dashboard
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Clean vs Soiled PV Power Generation (kW) & Solar Irradiance (GHI)",
            "Temperature-Corrected Soiling Ratio (SR) with Rainfall Recovery Events",
            "Atmospheric Dynamics: Wind Speed vs Critical Resuspension Boundary (v_crit) & PM10",
        ),
        row_heights=[0.36, 0.32, 0.32],
    )

    # Pane 1: Power & Irradiance
    fig.add_trace(
        go.Scatter(
            x=df_slice["datetime"],
            y=df_slice["power_clean"],
            name="Clean Reference Array",
            line=dict(color="#38bdf8", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_slice["datetime"],
            y=df_slice["power_soiled"],
            name="Soiled Monitored Array",
            line=dict(color="#f43f5e", width=1.5),
            fill="tonexty",
            fillcolor="rgba(244, 63, 94, 0.12)",
        ),
        row=1,
        col=1,
    )

    # Pane 2: Soiling Ratio & Rain
    fig.add_trace(
        go.Scatter(
            x=df_slice["datetime"],
            y=df_slice["soiling_ratio"],
            name="Soiling Ratio (SR)",
            line=dict(color="#10b981", width=2.2),
        ),
        row=2,
        col=1,
    )
    # Rainfall markers
    rain_events = df_slice[df_slice["rainfall"] > 0.5]
    if not rain_events.empty:
        fig.add_trace(
            go.Scatter(
                x=rain_events["datetime"],
                y=rain_events["soiling_ratio"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#60a5fa"),
                name="Rain Washing Event",
            ),
            row=2,
            col=1,
        )

    # Pane 3: Wind & v_crit
    fig.add_trace(
        go.Scatter(
            x=df_slice["datetime"],
            y=df_slice["wind_speed"],
            name="Wind Speed (m/s)",
            line=dict(color="#e2e8f0", width=1.2),
        ),
        row=3,
        col=1,
    )
    # Horizontal v_crit line
    fig.add_trace(
        go.Scatter(
            x=[df_slice["datetime"].iloc[0], df_slice["datetime"].iloc[-1]],
            y=[tp.wind_resuspension_threshold_mps, tp.wind_resuspension_threshold_mps],
            mode="lines",
            name=f"v_crit ({tp.wind_resuspension_threshold_mps:.1f} m/s)",
            line=dict(color="#00e5ff", width=2, dash="dash"),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        height=750,
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Soiling Ratio", row=2, col=1, range=[0.6, 1.05])
    fig.update_yaxes(title_text="Wind (m/s)", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# VIEW 2: PHYSICAL TIPPING POINTS & EXPLAINABILITY
# ==============================================================================
elif navigation == "Physical Tipping Points & Explainability":
    st.markdown('<div class="section-title">Discovered Physical Boundaries & Attribution Studio</div>', unsafe_allow_html=True)

    tab_wind, tab_cement, tab_global = st.tabs([
        "Aerodynamic Resuspension Threshold (v_crit)",
        "Moisture Cementation Cliff (2D Interactions)",
        "Global Feature Attribution Hierarchy",
    ])

    # TAB 1: WIND RESUSPENSION
    with tab_wind:
        st.markdown(
            r"""
            #### **Aerodynamic Dust Resuspension: The Zero-Crossing Transition**
            At low velocities, wind transports airborne particulate matter, accelerating dust deposition.
            However, when aerodynamic shear lift stress ($\tau_w \propto v^2$) overcomes van der Waals and capillary adhesion forces,
            wind switches regime to become a **natural cleaning agent** ($\Delta SR > 0$).
            """
        )

        wind_v, wind_s = xai.get_1d_dependence_curve("wind_speed")
        wind_rh = xai.sample_df.iloc[np.argsort(xai.sample_df["wind_speed"].values)]["relative_humidity"]

        fig_wind = px.scatter(
            x=wind_v,
            y=wind_s,
            color=wind_rh,
            color_continuous_scale="Viridis_r",
            labels={"x": "Wind Speed (m/s)", "y": "SHAP Impact on ΔSR", "color": "Relative Humidity (%)"},
            title="1D SHAP Dependence: Wind Speed vs Delta Soiling Ratio",
        )

        # Add Zero line
        fig_wind.add_hline(y=0.0, line_dash="dash", line_color="rgba(255,255,255,0.6)", annotation_text="Neutral Boundary (ΔSR=0)")

        # Add v_crit line
        fig_wind.add_vline(
            x=tp.wind_resuspension_threshold_mps,
            line_dash="dot",
            line_color="#00e5ff",
            line_width=3,
            annotation_text=f"v_crit = {tp.wind_resuspension_threshold_mps:.2f} m/s",
            annotation_position="top left",
        )

        fig_wind.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            height=500,
        )
        st.plotly_chart(fig_wind, use_container_width=True)

        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
        col_w1.metric("Critical Velocity (v_crit)", f"{tp.wind_resuspension_threshold_mps:.2f} m/s")
        col_w2.metric("95% Bootstrap CI", f"[{tp.wind_v_crit_ci_lower_mps:.2f}, {tp.wind_v_crit_ci_upper_mps:.2f}] m/s")
        col_w3.metric("Chow Break Test", f"p = {tp.wind_structural_break_p_value:.2e}", help="Chow structural breakpoint test across piecewise linear segments.")
        col_w4.metric("Knee / Zero Point", f"{tp.wind_zero_crossing_mps:.2f} m/s")

    # TAB 2: CEMENTATION CLIFF
    with tab_cement:
        st.markdown(
            r"""
            #### **The Cementation Cliff: Non-Linear Moisture × Particulate Synergies**
            When particulate matter is exposed to nocturnal relative humidity or condensation ($Dew\_Deficit \to 0$),
            soluble salts deliquesce and form irreversible capillary meniscus bridges upon drying.
            The 2D SHAP interaction values $\Phi(RH \times PM_{10})$ isolate this non-linear interaction cliff.
            """
        )

        inter_choice = st.selectbox(
            "Select Interaction Feature Pair to Inspect",
            ["relative_humidity x pm10", "dew_deficit x pm2_5"],
            index=0,
        )

        f_x, f_y = inter_choice.split(" x ")
        x_vals, y_vals, inter_vals = xai.extract_2d_interaction(f_x, f_y)

        # Plotly 2D Contour
        fig_contour = go.Figure(
            data=go.Contour(
                z=inter_vals,
                x=x_vals,
                y=y_vals,
                colorscale="Spectral_r",
                contours=dict(coloring="heatmap", showlabels=True),
            )
        )
        fig_contour.update_layout(
            title=f"2D SHAP Interaction Surface: {f_x.upper()} × {f_y.upper()}",
            xaxis_title=f_x.replace("_", " ").title(),
            yaxis_title=f_y.replace("_", " ").title(),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=520,
        )
        st.plotly_chart(fig_contour, use_container_width=True)

        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        col_c1.metric("Critical Dew Deficit", f"{tp.cementation_dew_deficit_celsius:.2f} °C", help="Threshold below which cementation accelerates.")
        col_c2.metric("Dew Deficit 95% CI", f"[{tp.cementation_dew_deficit_ci_lower:.2f}, {tp.cementation_dew_deficit_ci_upper:.2f}] °C")
        col_c3.metric("Critical Relative Humidity", f"{tp.critical_relative_humidity_pct:.1f} %", help="Capillary condensation onset.")
        col_c4.metric("Critical RH 95% CI", f"[{tp.critical_rh_ci_lower:.1f}, {tp.critical_rh_ci_upper:.1f}] %")

    # TAB 3: GLOBAL SHAP
    with tab_global:
        st.markdown("#### **Global Feature Importance Ranking (Mean |SHAP|)**")
        feat_imp = pd.DataFrame(
            {
                "Feature": xai.feature_names,
                "Mean_SHAP": np.mean(np.abs(xai.shap_values), axis=0),
            }
        ).sort_values("Mean_SHAP", ascending=True).tail(15)

        fig_bar = px.bar(
            feat_imp,
            x="Mean_SHAP",
            y="Feature",
            orientation="h",
            color="Mean_SHAP",
            color_continuous_scale="Teal",
            title="Global Predictors of PV Soiling Rate (ΔSR)",
        )
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            height=500,
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ==============================================================================
# VIEW 3: SCENARIO ANALYSIS & WEATHER FORENSICS
# ==============================================================================
elif navigation == "Scenario Analysis & Weather Forensics":
    st.markdown('<div class="section-title">Atmospheric Scenario Simulation & Risk Forensics</div>', unsafe_allow_html=True)
    st.markdown(r"Simulate custom or extreme weather events and observe real-time inference on the soiling rate ($\Delta SR$) and dispatch alerts.")

    # Scenario Presets
    preset = st.selectbox(
        "Load Preset Atmospheric Scenario",
        [
            "Custom Scenario",
            "Acute Dust Storm + Nocturnal Dew Condensation (High Cementation Risk)",
            "High-Wind Desert Front (Dry Self-Cleaning)",
            "Monsoon Rain Cleanse",
            "Dry Calm Gravitational Settling",
        ],
    )

    # Defaults based on preset
    def_temp, def_rh, def_ws, def_rain, def_pm10 = 24.0, 50.0, 3.5, 0.0, 50.0

    if preset == "Acute Dust Storm + Nocturnal Dew Condensation (High Cementation Risk)":
        def_temp, def_rh, def_ws, def_rain, def_pm10 = 15.0, 94.0, 1.8, 0.0, 180.0
    elif preset == "High-Wind Desert Front (Dry Self-Cleaning)":
        def_temp, def_rh, def_ws, def_rain, def_pm10 = 34.0, 28.0, 9.2, 0.0, 45.0
    elif preset == "Monsoon Rain Cleanse":
        def_temp, def_rh, def_ws, def_rain, def_pm10 = 22.0, 88.0, 4.0, 12.5, 20.0
    elif preset == "Dry Calm Gravitational Settling":
        def_temp, def_rh, def_ws, def_rain, def_pm10 = 28.0, 42.0, 2.1, 0.0, 85.0

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        sim_temp = st.slider("Ambient Temperature (°C)", -5.0, 50.0, float(def_temp), 0.5)
        sim_rh = st.slider("Relative Humidity (%)", 5.0, 100.0, float(def_rh), 1.0)
    with col_s2:
        sim_ws = st.slider("Wind Speed (m/s)", 0.0, 25.0, float(def_ws), 0.2)
        sim_pm10 = st.slider("PM10 Particulate Matter (µg/m³)", 5.0, 500.0, float(def_pm10), 5.0)
    with col_s3:
        sim_rain = st.slider("Rainfall (mm)", 0.0, 50.0, float(def_rain), 0.5)
        sim_is_night = st.checkbox("Nighttime Conditions (GHI = 0)", value=True)

    # Compute thermodynamics
    t_dew, dew_def = fe.compute_magnus_dew_point(np.array([sim_temp]), np.array([sim_rh]))
    sim_dew_def = float(dew_def[0])

    # Assess Risk & Decision
    mock_fc = pd.DataFrame([{
        "datetime": "2026-10-28 04:00:00",
        "temp_ambient": sim_temp,
        "relative_humidity": sim_rh,
        "wind_speed": sim_ws,
        "rainfall": sim_rain,
        "pm10": sim_pm10,
        "pm2_5": sim_pm10 * 0.42,
        "dew_deficit": sim_dew_def,
    }])
    decisions, _ = op_engine.evaluate_forecast(mock_fc)
    res = decisions[0]

    st.markdown("---")
    st.markdown("### **Simulation Diagnostic Results**")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Computed Dew Deficit", f"{sim_dew_def:.2f} °C", delta=f"{sim_dew_def - tp.cementation_dew_deficit_celsius:.2f} °C vs limit")
    with col_r2:
        st.metric("Cementation Risk Index", f"{res.cementation_risk_score:.1f} / 100", delta=res.cementation_risk_level)
    with col_r3:
        st.metric("Expected Soiling Shift (ΔSR)", f"{res.expected_delta_soiling_ratio:+.5f}")

    # Action Banner
    badge_class = "badge-critical" if res.cementation_risk_level in ["CRITICAL", "HIGH"] else ("badge-free" if res.free_cleaning_predicted else "badge-nominal")
    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid {'#ef4444' if res.cementation_risk_level in ['CRITICAL', 'HIGH'] else '#10b981'}; padding: 16px 20px; border-radius: 8px; margin-top: 16px;">
            <div style="display: flex; gap: 10px; align-items: center;">
                <span class="{badge_class}">{res.recommended_action}</span>
                <span style="font-weight: 700; color: #f8fafc;">{res.action_rationale}</span>
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px;">
                <b>Operational Impact:</b> {res.cost_water_saving_potential}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# VIEW 4: OPERATIONAL DISPATCH & ASSET OPTIMIZATION
# ==============================================================================
elif navigation == "Operational Dispatch & Asset Optimization":
    st.markdown('<div class="section-title">48-Hour Operational Cleaning Schedule & Economic ROI</div>', unsafe_allow_html=True)

    forecast_df = op_engine.generate_sample_forecast(hours=48)
    decisions, summary = op_engine.evaluate_forecast(forecast_df)
    decisions_df = pd.DataFrame([vars(d) for d in decisions])

    # High level alert metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    alerts_count = summary["operational_highlights"]["total_cementation_alerts_count"]
    free_clean_count = summary["operational_highlights"]["free_cleaning_windows_count"]
    net_sr_delta = summary["operational_highlights"]["net_forecasted_soiling_delta"]

    # Calculate Plant Level Savings
    water_saved_liters = free_clean_count * (water_per_mw_liters * plant_capacity_mw / 24.0)
    cost_saved_usd = (free_clean_count / 24.0) * (wash_cost_per_mw * plant_capacity_mw)

    col_m1.metric("Cementation Alerts", f"{alerts_count} Hours", delta="Pre-emptive wash needed", delta_color="inverse")
    col_m2.metric("Free Wind Cleaning", f"{free_clean_count} Hours", delta="Defer wash window")
    col_m3.metric("Est. Water Conserved", f"{water_saved_liters:,.0f} L")
    col_m4.metric("Est. Operational Savings", f"${cost_saved_usd:,.2f}")

    # Interactive Timeline Chart of 48h Dispatch
    st.markdown("#### **Forecast Horizon Dispatch Matrix**")
    fig_sched = px.bar(
        decisions_df,
        x="timestamp",
        y="cementation_risk_score",
        color="cementation_risk_level",
        color_discrete_map={"LOW": "#64748b", "MODERATE": "#fbbf24", "HIGH": "#f97316", "CRITICAL": "#ef4444"},
        title="48h Forward Cementation Risk Timeline",
        hover_data=["recommended_action", "expected_delta_soiling_ratio"],
    )
    fig_sched.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        height=380,
    )
    st.plotly_chart(fig_sched, use_container_width=True)

    # Detailed Dispatch Schedule Table
    st.markdown("#### **Detailed Hourly Action Registry**")
    st.dataframe(
        decisions_df[
            [
                "timestamp",
                "cementation_risk_level",
                "cementation_risk_score",
                "free_cleaning_predicted",
                "recommended_action",
                "cost_water_saving_potential",
            ]
        ],
        use_container_width=True,
        height=320,
    )

    # Export Buttons
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.download_button(
            "Export Dispatch Plan (JSON)",
            data=json.dumps({"summary": summary, "schedule": [vars(d) for d in decisions]}, indent=2),
            file_name="operational_dispatch_plan.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_e2:
        st.download_button(
            "Export Schedule (CSV)",
            data=decisions_df.to_csv(index=False),
            file_name="operational_dispatch_plan.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ==============================================================================
# VIEW 5: MODEL GOVERNANCE & VALIDATION METRICS
# ==============================================================================
elif navigation == "Model Governance & Validation Metrics":
    st.markdown('<div class="section-title">Model Governance & Anti-Leakage Validation</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Solar PV data exhibits strong diurnal autocorrelation. Standard random k-fold cross-validation
        suffers from severe look-ahead leakage. Our pipeline utilizes a **Purged Time-Series Split**
        with a 24-hour embargo buffer between train and test diurnal windows.
        """
    )

    col_g1, col_g2, col_g3 = st.columns(3)
    col_g1.metric("Selected Champion Architecture", model_name.upper())
    col_g2.metric("Cross-Validated Holdout RMSE", f"{metrics.rmse:.6f}")
    col_g3.metric("Cross-Validated Holdout MAE", f"{metrics.mae:.6f}")

    # Load validation metrics json
    val_json_path = Path("models/delta_soiling_ratio_validation_metrics.json")
    if val_json_path.exists():
        with open(val_json_path, "r") as f:
            val_data = json.load(f)

        st.markdown("#### **LightGBM vs XGBoost Comparative Benchmarks**")
        comp_df = pd.DataFrame(val_data).T
        st.dataframe(comp_df[["model_name", "target", "rmse", "mae", "r2", "train_samples"]], use_container_width=True)

    st.markdown("#### **Physical Boundaries Summary Report (JSON Export)**")
    st.json({
        "wind_resuspension_threshold_mps": tp.wind_resuspension_threshold_mps,
        "wind_v_crit_ci_95": [tp.wind_v_crit_ci_lower_mps, tp.wind_v_crit_ci_upper_mps],
        "wind_structural_break_p_value": tp.wind_structural_break_p_value,
        "critical_relative_humidity_pct": tp.critical_relative_humidity_pct,
        "critical_rh_ci_95": [tp.critical_rh_ci_lower, tp.critical_rh_ci_upper],
        "cementation_dew_deficit_celsius": tp.cementation_dew_deficit_celsius,
        "cementation_dew_deficit_ci_95": [tp.cementation_dew_deficit_ci_lower, tp.cementation_dew_deficit_ci_upper],
        "confidence_score": tp.confidence_score,
        "notes": tp.notes,
    })


# ==============================================================================
# VIEW 6: TECHNICAL MANUSCRIPT & RESEARCH SUITE
# ==============================================================================
elif navigation == "Research Manuscript & Technical Suite":
    st.markdown('<div class="section-title">Research Manuscript & Technical Dossier</div>', unsafe_allow_html=True)
    
    st.markdown(
        r"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 20px; margin-bottom: 24px;">
            <div style="font-size: 0.8rem; color: #38bdf8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Manuscript In Review</div>
            <h2 style="font-size: 1.5rem; font-weight: 700; margin: 6px 0 10px 0; color: #f8fafc;">
                Discovering Non-Linear Physical Tipping Points in Solar Photovoltaic Soiling via Explainable AI: The Cementation Boundary and Aerodynamic Cleaning Threshold
            </h2>
            <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 12px;">
                <strong>Swayam Swaroop Sahu</strong> &bull; <em>School of Electrical Engineering and Renewable Energy Systems</em>
            </div>
            <p style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.6; margin-bottom: 16px;">
                <strong>Abstract:</strong> Solar photovoltaic (PV) performance degradation from airborne dust accumulation is governed by coupled thermodynamic and aerodynamic processes. This study presents a physics-informed Explainable AI (XAI) framework integrating SHAP TreeExplainer attribution and high-order interaction tensors with purged group walk-forward cross-validation on multi-site empirical cohorts. We empirically discover two foundational tipping points: (1) an aerodynamic wind resuspension threshold \(v_{\text{crit}} = 9.22\,\text{m/s}\) [95% CI: 7.16–9.57 m/s, Chow test \(p = 1.12 \times 10^{-3}\)], and (2) a non-linear cementation boundary occurring at dawn dew deficit \(T_{\text{deficit}} \le 2.80\,^\circ\text{C}\) and relative humidity \(\ge 72.0\%\). An 8,760-hour techno-economic simulation on a 50 MWp plant proves our predictive dispatch reduces LCOE by 10.5% and saves 38.5% of cleaning water.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Download compiled paper PDF if available
    pdf_path = Path("paper/main.pdf")
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        col_p1, col_p2 = st.columns([1, 3])
        with col_p1:
            st.download_button(
                "Download Manuscript (PDF)",
                data=pdf_bytes,
                file_name="PV_Soiling_XAI_Tipping_Points.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with col_p2:
            st.caption(f"Compiled PDF size: {len(pdf_bytes):,} bytes | MiKTeX 26.5 / pdfTeX engine with BibTeX citations.")

    tab_figs, tab_tables, tab_src = st.tabs([
        "Publication Figures (High-Resolution & Vector PDF)",
        "Comparative Benchmarks & Statistical Tables",
        "Manuscript Architecture & Source Code",
    ])

    with tab_figs:
        st.markdown("#### **Publication Figures Gallery**")
        fig_cols1, fig_cols2 = st.columns(2)
        with fig_cols1:
            st.image("paper/figures/fig1_global_shap_beeswarm.png", caption="Fig 1: Global SHAP Beeswarm summary of diurnal dust deposition drivers.")
            st.image("paper/figures/fig3_cementation_cliff_2d_interaction.png", caption="Fig 3: 2D SHAP Interaction contour (RH × PM10) showing the cementation cliff.")
        with fig_cols2:
            st.image("paper/figures/fig2_wind_resuspension_threshold.png", caption="Fig 2: 1D Wind Resuspension dependence with 95% bootstrap CI ribbon and Chow breakpoint.")
            st.image("paper/figures/fig4_dew_deficit_pm25_interaction.png", caption="Fig 4: 2D SHAP Interaction contour (Dew Deficit × PM2.5) uncovering nocturnal moisture locking.")

    with tab_tables:
        st.markdown("#### **Publication Data & Validation Tables**")
        st.markdown("##### **Table 2: Comparative Model Performance Benchmarks (Purged Group Walk-Forward CV)**")
        t2_path = Path("paper/tables/table2_model_performance_benchmarks.tex")
        if t2_path.exists():
            st.code(t2_path.read_text(encoding="utf-8"), language="latex")

        st.markdown("##### **Table 3: Statistical Validation of Mined Physical Tipping Points**")
        t3_path = Path("paper/tables/table3_tipping_points_statistical_validation.tex")
        if t3_path.exists():
            st.code(t3_path.read_text(encoding="utf-8"), language="latex")

        st.markdown("##### **Table 4: 8,760-Hour Techno-Economic Simulation Results (50 MWp Plant)**")
        t4_path = Path("paper/tables/table4_techno_economic_roi.tex")
        if t4_path.exists():
            st.code(t4_path.read_text(encoding="utf-8"), language="latex")

    with tab_src:
        st.markdown("#### **LaTeX Project Hierarchy (`paper/`)**")
        st.code(
            """paper/
├── main.tex
├── references.bib
├── sections/
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_physics_and_theory.tex
│   ├── 03_methodology.tex
│   ├── 04_results_and_discussion.tex
│   ├── 05_techno_economic_analysis.tex
│   └── 06_conclusion.tex
├── tables/
│   ├── table1_dataset_characteristics.tex
│   ├── table2_model_performance_benchmarks.tex
│   ├── table3_tipping_points_statistical_validation.tex
│   └── table4_techno_economic_roi.tex
└── figures/
    ├── fig1_global_shap_beeswarm.[pdf|png]
    ├── fig2_wind_resuspension_threshold.[pdf|png]
    ├── fig3_cementation_cliff_2d_interaction.[pdf|png]
    └── fig4_dew_deficit_pm25_interaction.[pdf|png]""",
            language="text",
        )

