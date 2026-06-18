from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import sklearn
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from dataset import DATASET_PATH
from model import train_and_save_model
from utils import MODEL_PATH, classify_profit, currency, ensure_dataset, load_artifact, model_ready, profit_color


APP_PATH = Path(__file__).resolve()


def running_in_streamlit() -> bool:
    return get_script_run_ctx() is not None


def launch_with_streamlit() -> None:
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_PATH)], check=False)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --brand-ink: #16351F;
                --brand-muted: #5B6470;
                --brand-card: rgba(255, 255, 255, 0.94);
                --brand-line: rgba(26, 71, 42, 0.08);
                --brand-soft: #f7faf7;
                --brand-accent: #1f5a46;
                --brand-accent-2: #d4a373;
            }
            .stApp {
                background: linear-gradient(180deg, #f4f7f1 0%, #ffffff 48%, #eef5fb 100%);
            }
            .hero-card, .content-card {
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid rgba(26, 71, 42, 0.08);
                border-radius: 20px;
                padding: 1.5rem 1.4rem;
                box-shadow: 0 14px 34px rgba(16, 24, 40, 0.08);
                margin-bottom: 1rem;
            }
            .hero-title {
                text-align: center;
                color: #16351F;
                font-size: 2.3rem;
                font-weight: 700;
                margin-bottom: 0.3rem;
            }
            .hero-subtitle {
                text-align: center;
                color: #5B6470;
                font-size: 1rem;
                margin-bottom: 0;
            }
            .section-title {
                color: #16351F;
                font-size: 1.15rem;
                font-weight: 700;
                margin-bottom: 0.8rem;
            }
            .result-box {
                border-radius: 22px;
                padding: 1.4rem;
                color: white;
                text-align: center;
                box-shadow: 0 18px 36px rgba(0,0,0,0.12);
            }
            .result-label {
                font-size: 0.95rem;
                opacity: 0.92;
                letter-spacing: 0.02em;
                text-transform: uppercase;
            }
            .result-value {
                font-size: 2.2rem;
                font-weight: 800;
                margin: 0.35rem 0;
            }
            .result-tag {
                font-size: 1rem;
                font-weight: 600;
            }
            .metric-card {
                background: #f7faf7;
                border: 1px solid #e4ece4;
                border-radius: 16px;
                padding: 0.9rem 1rem;
                text-align: center;
            }
            .metric-label {
                color: #64748b;
                font-size: 0.85rem;
                text-transform: uppercase;
            }
            .metric-value {
                color: #16351F;
                font-size: 1.35rem;
                font-weight: 700;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #133b2c 0%, #1f5a46 100%);
            }
            [data-testid="stSidebar"] * {
                color: #f8fafc;
            }
            [data-testid="stSidebar"] div[data-baseweb="select"] > div,
            [data-testid="stSidebar"] div[data-baseweb="input"] > div,
            [data-testid="stSidebar"] div[data-baseweb="select"] input,
            [data-testid="stSidebar"] div[data-baseweb="input"] input,
            [data-testid="stSidebar"] div[data-baseweb="base-input"] > div,
            [data-testid="stSidebar"] div[data-baseweb="base-input"] input,
            [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
            [data-testid="stSidebar"] .stNumberInput input,
            [data-testid="stSidebar"] .stTextInput input,
            [data-testid="stSidebar"] .stDateInput input,
            [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] span,
            [data-testid="stSidebar"] textarea,
            [data-testid="stSidebar"] input[type="number"],
            [data-testid="stSidebar"] input[type="text"] {
                color: #16351F !important;
                background-color: #ffffff !important;
                -webkit-text-fill-color: #16351F !important;
            }
            [data-testid="stSidebar"] div[data-baseweb="select"] svg,
            [data-testid="stSidebar"] div[data-baseweb="input"] svg,
            [data-testid="stSidebar"] div[data-baseweb="base-input"] svg {
                fill: #16351F !important;
            }
            [data-testid="stSidebar"] div[data-baseweb="select"] > div,
            [data-testid="stSidebar"] div[data-baseweb="input"] > div,
            [data-testid="stSidebar"] div[data-baseweb="base-input"] > div {
                border-color: #cdd9cf !important;
            }
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] .stSelectbox label,
            [data-testid="stSidebar"] .stNumberInput label,
            [data-testid="stSidebar"] .stSlider label {
                color: #f8fafc !important;
            }
            [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] * {
                color: inherit;
            }
            div[data-testid="stPlotlyChart"], div[data-testid="stImage"], .stDataFrame {
                background: #ffffff;
                border-radius: 18px;
                padding: 0.5rem;
            }
            .home-shell {
                position: relative;
                overflow: hidden;
                background:
                    radial-gradient(circle at top right, rgba(212, 163, 115, 0.22), transparent 28%),
                    radial-gradient(circle at bottom left, rgba(31, 90, 70, 0.16), transparent 34%),
                    linear-gradient(145deg, rgba(255,255,255,0.97) 0%, rgba(244,247,241,0.98) 100%);
                border: 1px solid rgba(26, 71, 42, 0.09);
                border-radius: 28px;
                padding: 3rem 3rem 2.5rem 3rem;
                box-shadow: 0 24px 56px rgba(16, 24, 40, 0.10);
                margin-top: 1rem;
            }
            .home-badge {
                display: inline-block;
                background: rgba(31, 90, 70, 0.09);
                color: var(--brand-accent);
                border: 1px solid rgba(31, 90, 70, 0.12);
                border-radius: 999px;
                padding: 0.35rem 0.8rem;
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                margin-bottom: 1rem;
            }
            .home-title {
                color: var(--brand-ink);
                font-size: 3rem;
                line-height: 1.08;
                font-weight: 800;
                max-width: 820px;
                margin-bottom: 1rem;
            }
            .home-title span {
                color: var(--brand-accent);
            }
            .home-copy {
                color: var(--brand-muted);
                font-size: 1.06rem;
                line-height: 1.8;
                max-width: 760px;
                margin-bottom: 1.4rem;
            }
            .home-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 1rem;
                margin-top: 1.2rem;
            }
            .home-panel {
                background: rgba(255,255,255,0.82);
                border: 1px solid rgba(26, 71, 42, 0.08);
                border-radius: 20px;
                padding: 1.1rem 1rem;
                backdrop-filter: blur(8px);
            }
            .home-panel-title {
                color: var(--brand-ink);
                font-size: 1rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
            }
            .home-panel-copy {
                color: var(--brand-muted);
                font-size: 0.94rem;
                line-height: 1.65;
                margin: 0;
            }
            .home-login-wrap {
                margin-top: 1.5rem;
                max-width: 220px;
            }
            .home-note {
                color: #6b7280;
                font-size: 0.92rem;
                margin-top: 0.85rem;
            }
            div.stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #16351F 0%, #1f5a46 100%);
                border: none;
                border-radius: 14px;
                min-height: 3rem;
                font-weight: 700;
                padding: 0.6rem 1.2rem;
                box-shadow: 0 14px 28px rgba(22, 53, 31, 0.20);
            }
            div.stButton > button[kind="primary"]:hover {
                background: linear-gradient(135deg, #1b4026 0%, #276b53 100%);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    st.session_state.setdefault("show_dashboard", False)


def render_home_page() -> bool:
    st.markdown(
        """
        <div class="home-shell">
            <div class="home-badge">Machine Learning Project</div>
            <div class="home-title">
                Profit Prediction:
                <span>Machine Learning Regression for Business Profit Estimation</span>
            </div>
            <p class="home-copy">
                Explore a professional business profit estimation workflow built on regression modeling.
                The dashboard helps compare operational inputs, analyze business patterns, and estimate
                profit outcomes through an interactive machine learning interface.
            </p>
            <div class="home-grid">
                <div class="home-panel">
                    <div class="home-panel-title">Business Inputs</div>
                    <p class="home-panel-copy">Work with company, country, year, employee count, and cost variables in one guided prediction flow.</p>
                </div>
                <div class="home-panel">
                    <div class="home-panel-title">Regression Insights</div>
                    <p class="home-panel-copy">View model-driven profit estimation with structured comparisons, feature importance, and correlation analysis.</p>
                </div>
                <div class="home-panel">
                    <div class="home-panel-title">Interactive Dashboard</div>
                    <p class="home-panel-copy">Move from a clean landing experience into the full analytics dashboard with a single click.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    login_col, _ = st.columns([1, 3])
    with login_col:
        if st.button("Start", type="primary", use_container_width=True):
            st.session_state["show_dashboard"] = True
            st.rerun()
    st.markdown(
        "<p class='home-note'>Select Start to open the existing profit prediction dashboard.</p>",
        unsafe_allow_html=True,
    )
    return st.session_state["show_dashboard"]


@st.cache_data(show_spinner=False)
def load_dashboard_artifact():
    ensure_dataset()
    if not model_ready(MODEL_PATH):
        return train_and_save_model()

    try:
        artifact = load_artifact(MODEL_PATH)
    except Exception:
        return train_and_save_model()

    if artifact.get("sklearn_version") != sklearn.__version__:
        return train_and_save_model()

    required_columns = {
        "R&D Spend",
        "Administration Cost",
        "Marketing Spend",
        "Logistics Cost",
        "Employee Count",
        "Country",
        "Business Category",
        "Company Name",
        "Year",
        "Profit",
    }
    if required_columns.difference(artifact["dataset"].columns):
        return train_and_save_model()

    return artifact


def render_header(artifact) -> None:
    metrics_df = artifact["metrics"]
    best_model = artifact["best_model_name"]
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">Profit Prediction: Business Profit Estimation Dashboard</div>
            <p class="hero-subtitle">
                Prediction trained only on the original 50 Startups dataset and using
                <strong>{best_model}</strong> as the best-performing model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_row = st.columns(4)
    top_row[0].markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Dataset Rows</div>
            <div class="metric-value">{len(artifact['dataset'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top_row[1].markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Countries</div>
            <div class="metric-value">{artifact['dataset']['Country'].nunique()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top_row[2].markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Best R2 Score</div>
            <div class="metric-value">{metrics_df.iloc[0]['R2 Score']:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top_row[3].markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Average Profit</div>
            <div class="metric-value">{currency(artifact['dataset']['Profit'].mean())}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_sidebar(artifact):
    st.sidebar.markdown("## Input Parameters")
    st.sidebar.caption("Select company, year, business category, country, and spending values from the dataset.")

    dataset = artifact["dataset"].copy().reset_index(drop=True)

    category_options = sorted(dataset["Business Category"].unique().tolist())
    category = st.sidebar.selectbox("Business Category", category_options)

    category_rows = dataset[dataset["Business Category"] == category].copy()
    company_options = sorted(category_rows["Company Name"].unique().tolist())
    company_name = st.sidebar.selectbox("Company Name", company_options)

    company_rows = category_rows[category_rows["Company Name"] == company_name].copy()
    country_options = sorted(dataset["Country"].unique().tolist())
    country = st.sidebar.selectbox("Country", country_options)

    year_options = sorted(dataset["Year"].astype(int).unique().tolist())
    year = st.sidebar.selectbox("Year", year_options)

    matching_rows = dataset[
        (dataset["Business Category"] == category)
        & (dataset["Company Name"] == company_name)
        & (dataset["Country"] == country)
        & (dataset["Year"].astype(int) == int(year))
    ].copy()
    if matching_rows.empty:
        matching_rows = dataset[
            (dataset["Business Category"] == category)
            & (dataset["Company Name"] == company_name)
        ].copy()
    if matching_rows.empty:
        matching_rows = category_rows.copy()

    default_row = matching_rows.iloc[0]

    col1, col2 = st.sidebar.columns(2)
    rd_spend = col1.number_input("R&D Spend", min_value=0.0, value=float(default_row["R&D Spend"]), step=1000.0)
    administration = col2.number_input(
        "Administration Cost",
        min_value=0.0,
        value=float(default_row["Administration Cost"]),
        step=1000.0,
    )
    marketing = st.sidebar.number_input(
        "Marketing Spend",
        min_value=0.0,
        value=float(default_row["Marketing Spend"]),
        step=1000.0,
    )
    logistics = st.sidebar.number_input(
        "Logistics Cost",
        min_value=0.0,
        value=float(default_row["Logistics Cost"]),
        step=1000.0,
    )
    employees = st.sidebar.number_input(
        "Employee Count",
        min_value=1,
        value=int(default_row["Employee Count"]),
        step=1,
    )

    input_frame = pd.DataFrame(
        [
            {
                "R&D Spend": rd_spend,
                "Administration Cost": administration,
                "Marketing Spend": marketing,
                "Logistics Cost": logistics,
                "Employee Count": employees,
                "Country": country,
                "Business Category": category,
                "Company Name": company_name,
                "Year": int(year),
            }
        ]
    )
    prediction = float(artifact["model"].predict(input_frame)[0])
    selection_summary = f"{company_name} | {country} | {year}"
    return prediction, input_frame, selection_summary


def build_dynamic_chart_frame(dataset: pd.DataFrame, input_frame: pd.DataFrame, prediction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_input = input_frame.iloc[0]
    comparison_frame = dataset.copy()
    distance = (
        (comparison_frame["R&D Spend"] - float(current_input["R&D Spend"])).abs()
        + (comparison_frame["Administration Cost"] - float(current_input["Administration Cost"])).abs()
        + (comparison_frame["Marketing Spend"] - float(current_input["Marketing Spend"])).abs()
        + (comparison_frame["Logistics Cost"] - float(current_input["Logistics Cost"])).abs()
        + (comparison_frame["Employee Count"] - float(current_input["Employee Count"])).abs()
    )
    nearest_rows = comparison_frame.loc[distance.nsmallest(10).index].copy()
    nearest_rows["Source"] = "Dataset"

    input_row = pd.DataFrame(
        [
            {
                "R&D Spend": float(current_input["R&D Spend"]),
                "Administration Cost": float(current_input["Administration Cost"]),
                "Marketing Spend": float(current_input["Marketing Spend"]),
                "Logistics Cost": float(current_input["Logistics Cost"]),
                "Employee Count": int(current_input["Employee Count"]),
                "Country": str(current_input["Country"]),
                "Business Category": str(current_input["Business Category"]),
                "Company Name": str(current_input["Company Name"]),
                "Year": int(current_input["Year"]),
                "Profit": float(prediction),
                "Source": "Current Input",
            }
        ]
    )

    country_rows = comparison_frame[comparison_frame["Country"] == str(current_input["Country"])].copy()
    country_rows["Source"] = "Dataset"
    return pd.concat([nearest_rows, input_row], ignore_index=True), pd.concat([country_rows, input_row], ignore_index=True)


def render_prediction_result(prediction: float, thresholds) -> None:
    status = classify_profit(prediction, thresholds)
    color = profit_color(prediction, thresholds)
    st.markdown(
        "<div class='content-card'><div class='section-title'>Prediction Result</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="result-box" style="background: linear-gradient(135deg, {color} 0%, #16351F 100%);">
            <div class="result-label">Predicted Profit</div>
            <div class="result-value">{currency(prediction)}</div>
            <div class="result-tag">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_model_performance(artifact) -> None:
    metrics_df = artifact["metrics"].copy()
    display_df = metrics_df.copy()
    for column in ["MAE", "RMSE"]:
        display_df[column] = display_df[column].map(currency)
    display_df["R2 Score"] = display_df["R2 Score"].map(lambda value: f"{value:.4f}")

    st.markdown("<div class='content-card'><div class='section-title'>Model Performance</div>", unsafe_allow_html=True)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def style_plotly_figure(fig, title: str) -> None:
    fig.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 20, "color": "#16351F"},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F8FBF8",
        margin=dict(l=20, r=20, t=70, b=20),
        font=dict(family="Segoe UI", color="#29443A", size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#D7E4DA",
        tickfont=dict(color="#4B6357"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E4EEE7",
        zeroline=False,
        linecolor="#D7E4DA",
        tickfont=dict(color="#4B6357"),
    )


def render_visualizations(artifact, input_frame: pd.DataFrame, prediction: float) -> None:
    dataset = artifact["dataset"]
    feature_importance = artifact["feature_importance"].head(12)
    nearest_frame, country_frame = build_dynamic_chart_frame(dataset, input_frame, prediction)
    current_country = str(input_frame.iloc[0]["Country"])
    current_values = pd.DataFrame(
        [
            {"Metric": "R&D Spend", "Amount": float(input_frame.iloc[0]["R&D Spend"])},
            {"Metric": "Administration Cost", "Amount": float(input_frame.iloc[0]["Administration Cost"])},
            {"Metric": "Marketing Spend", "Amount": float(input_frame.iloc[0]["Marketing Spend"])},
            {"Metric": "Employee Count", "Amount": float(input_frame.iloc[0]["Employee Count"])},
            {"Metric": "Predicted Profit", "Amount": float(prediction)},
        ]
    )
    accent_palette = {
        "dataset": "#A44A3F",
        "current": "#1F6E6B",
        "gold": "#D4A373",
        "sage": "#7C9A6D",
        "rose": "#B56576",
        "ink": "#2F3E46",
    }

    st.markdown("<div class='content-card'><div class='section-title'>Advanced Visualizations</div>", unsafe_allow_html=True)
    st.caption("A polished view of how the current selection compares with the surrounding business patterns in the dataset.")

    chart_col1, chart_col2 = st.columns(2)

    country_profit = dataset.groupby("Country", as_index=False)["Profit"].mean().sort_values("Profit", ascending=False)
    country_profit["Type"] = "Dataset Average"
    highlighted_country = pd.DataFrame(
        [
            {
                "Country": current_country,
                "Profit": float(prediction),
                "Type": "Current Prediction",
            }
        ]
    )
    fig_bar = px.bar(
        pd.concat([country_profit, highlighted_country], ignore_index=True),
        x="Country",
        y="Profit",
        color="Type",
        barmode="group",
        title="Current Prediction vs Country Averages",
        color_discrete_map={"Dataset Average": accent_palette["gold"], "Current Prediction": accent_palette["current"]},
    )
    style_plotly_figure(fig_bar, "Current Prediction vs Country Averages")
    fig_bar.update_layout(xaxis_tickangle=-35, bargap=0.28)
    chart_col1.plotly_chart(fig_bar, use_container_width=True)

    fig_hist = px.histogram(
        country_frame,
        x="Profit",
        nbins=12,
        color="Source",
        title=f"Profit Distribution for {current_country}",
        opacity=0.8,
        color_discrete_map={"Dataset": accent_palette["sage"], "Current Input": accent_palette["rose"]},
    )
    style_plotly_figure(fig_hist, f"Profit Distribution for {current_country}")
    fig_hist.update_traces(marker_line_color="#F6FBF7", marker_line_width=1.2)
    chart_col2.plotly_chart(fig_hist, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)
    fig_scatter = px.scatter(
        nearest_frame,
        x="Marketing Spend",
        y="Profit",
        color="Source",
        symbol="Source",
        hover_data=["R&D Spend", "Administration Cost", "Country", "Company Name", "Year"],
        title="Current Input vs Closest Dataset Rows",
        opacity=0.85,
        color_discrete_map={"Dataset": accent_palette["dataset"], "Current Input": accent_palette["current"]},
    )
    style_plotly_figure(fig_scatter, "Current Input vs Closest Dataset Rows")
    fig_scatter.update_traces(marker=dict(size=12, line=dict(width=1.4, color="#FFFFFF")))
    chart_col3.plotly_chart(fig_scatter, use_container_width=True)

    fig_values = px.bar(
        current_values,
        x="Metric",
        y="Amount",
        color="Metric",
        title="Current Input Values and Predicted Profit",
        color_discrete_sequence=[
            accent_palette["dataset"],
            accent_palette["gold"],
            accent_palette["sage"],
            accent_palette["rose"],
            accent_palette["current"],
        ],
    )
    style_plotly_figure(fig_values, "Current Input Values and Predicted Profit")
    fig_values.update_layout(showlegend=False, xaxis_tickangle=-20, bargap=0.24)
    fig_values.update_traces(marker_line_color="#F7FBF8", marker_line_width=1.2)
    chart_col4.plotly_chart(fig_values, use_container_width=True)

    chart_col5, chart_col6 = st.columns(2)
    fig_pie = px.pie(
        nearest_frame.groupby("Business Category", as_index=False)["Profit"].mean(),
        values="Profit",
        names="Business Category",
        title="Closest-Match Profit Share by Business Category",
        hole=0.35,
        color_discrete_sequence=[
            accent_palette["current"],
            accent_palette["dataset"],
            accent_palette["gold"],
            accent_palette["sage"],
            accent_palette["rose"],
            accent_palette["ink"],
        ],
    )
    style_plotly_figure(fig_pie, "Closest-Match Profit Share by Business Category")
    fig_pie.update_traces(textposition="inside", textinfo="percent+label", marker=dict(line=dict(color="#FFFFFF", width=2)))
    chart_col5.plotly_chart(fig_pie, use_container_width=True)

    fig_feature = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance from Best Model",
        color="Importance",
        color_continuous_scale=["#F2E8DC", "#D4A373", "#B56576", "#1F6E6B"],
    )
    style_plotly_figure(fig_feature, "Feature Importance from Best Model")
    fig_feature.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    fig_feature.update_traces(marker_line_color="#F7FBF8", marker_line_width=1.0)
    chart_col6.plotly_chart(fig_feature, use_container_width=True)

    chart_col7, chart_col8 = st.columns(2)
    correlation = artifact["correlation"]
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    fig.patch.set_facecolor("#F8FBF8")
    ax.set_facecolor("#F8FBF8")
    sns.heatmap(
        correlation,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        linewidths=0.6,
        linecolor="#F1F6F2",
        cbar_kws={"shrink": 0.8, "pad": 0.02},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap", loc="left", fontsize=18, color="#16351F", pad=16)
    ax.tick_params(axis="x", rotation=35, labelsize=10, colors="#4B6357")
    ax.tick_params(axis="y", rotation=0, labelsize=10, colors="#4B6357")
    chart_col7.pyplot(fig, use_container_width=True)
    plt.close(fig)

    fig_trend = px.line(
        dataset.groupby("Year", as_index=False)["Profit"].mean().sort_values("Year"),
        x="Year",
        y="Profit",
        markers=True,
        title="Average Profit Trend by Year",
    )
    style_plotly_figure(fig_trend, "Average Profit Trend by Year")
    fig_trend.update_traces(
        line=dict(color=accent_palette["current"], width=4),
        marker=dict(size=10, color=accent_palette["gold"], line=dict(width=1.5, color="#FFFFFF")),
    )
    chart_col8.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_data_preview(artifact, input_frame: pd.DataFrame) -> None:
    st.markdown("<div class='content-card'><div class='section-title'>Input Parameters</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    col1.dataframe(input_frame, use_container_width=True, hide_index=True)
    col2.dataframe(artifact["dataset"].head(12), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_dataset_notice(selected_preset: str) -> None:
    dataset_name = DATASET_PATH.name
    st.success(f"Prediction is using values aligned to {selected_preset} from `{dataset_name}`.")


def main() -> None:
    st.set_page_config(
        page_title="Profit Prediction Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
    )
    inject_styles()
    initialize_session_state()

    if not st.session_state["show_dashboard"]:
        render_home_page()
        return

    artifact = load_dashboard_artifact()
    render_header(artifact)
    prediction, input_frame, selected_preset = prediction_sidebar(artifact)
    render_dataset_notice(selected_preset)

    left_col, right_col = st.columns([0.9, 1.1])
    with left_col:
        render_prediction_result(prediction, artifact["thresholds"])
        render_model_performance(artifact)
    with right_col:
        render_data_preview(artifact, input_frame)

    render_visualizations(artifact, input_frame, prediction)


if __name__ == "__main__":
    if running_in_streamlit():
        main()
    else:
        print("Launching the dashboard with Streamlit...")
        launch_with_streamlit()
