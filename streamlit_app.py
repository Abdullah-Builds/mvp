"""
Cardiac Risk Prediction — Production MVP

Required files in the same folder:
    model.pkl
    mvp_hero.png
    app.py

Run:
    streamlit run app.py
"""

from pathlib import Path
import os
import io
import smtplib
from email.message import EmailMessage
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
    KeepTogether,
)
try:
    from groq import Groq
except ImportError:
    Groq = None
import streamlit.components.v1 as components


# Resolve files relative to this app, so it works on Windows, macOS and Linux.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
HERO_IMAGE = BASE_DIR / "mvp_hero.png"

st.set_page_config(
    page_title="Cardiac Risk MVP",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(""" <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css" > """, unsafe_allow_html=True, )
# ----------------------------- Styling --------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #06152d;
    --blue: #1687ff;
    --cyan: #39d7ff;
    --text: #10213a;
    --muted: #334155;
    --border: rgba(7,37,69,.20);
    --shadow: 0 18px 50px rgba(9,38,72,.10);
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 4%, rgba(22,135,255,.13), transparent 26%),
        radial-gradient(circle at 92% 17%, rgba(57,215,255,.11), transparent 23%),
        linear-gradient(180deg, #f5f9ff 0%, #ffffff 46%, #f4f8fd 100%);
}
.social-links { display: flex; gap: 12px; margin-top: 18px; flex-wrap: wrap; } .social-link { display: flex; align-items: center; gap: 10px; padding: 10px 15px; border: 1px solid #d7e2ec; border-radius: 10px; text-decoration: none; color: #0b1f3a; background: #f7fafd; transition: all 0.2s ease; } .social-link i { font-size: 24px; } .social-link span { display: flex; flex-direction: column; line-height: 1.2; } .social-link small { margin-top: 3px; color: #64748b; font-size: 11px; } .social-link:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(11, 31, 58, 0.10); } .github:hover { border-color: #24292f; } .linkedin:hover { border-color: #0a66c2; }
.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
    padding-bottom: 0;
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    min-height: 255px;
    padding: 42px 48px;
    border-radius: 28px;
    background:
        radial-gradient(circle at 82% 50%, rgba(57,215,255,.20), transparent 24%),
        linear-gradient(135deg, #041329 0%, #092b55 56%, #07172f 100%);
    box-shadow: 0 24px 70px rgba(4,25,55,.28);
    border: 1px solid rgba(255,255,255,.12);
    margin-bottom: 25px;
}

.hero-content {
    position: relative;
    z-index: 2;
    max-width: 690px;
}

.hero-badge {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    color: #ffffff !important;
    background: rgba(13, 111, 232, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.65);
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
    text-shadow: 0 1px 3px rgba(0, 0, 0, .45);
    box-shadow:
        0 4px 14px rgba(0, 0, 0, .18),
        inset 0 1px 0 rgba(255, 255, 255, .2);
}

.hero h1 {
    color: white;
    font-size: clamp(2.2rem, 4vw, 3.6rem);
    line-height: 1.02;
    margin: 17px 0 12px;
    font-weight: 800;
    letter-spacing: -.045em;
}

.hero p {
    color: #e1edf9;
    font-size: 1.03rem;
    line-height: 1.65;
    margin: 0;
}

.hero-orb {
    position: absolute;
    right: 4%;
    top: 22px;
    width: 235px;
    height: 235px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 28%, rgba(57,215,255,.23), rgba(15,89,165,.09) 45%, transparent 70%);
    box-shadow: inset 0 0 45px rgba(57,215,255,.11), 0 0 70px rgba(22,135,255,.11);
}

.hero-orb:after {
    content: "♥";
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    color: rgba(255,104,104,.9);
    font-size: 6rem;
    text-shadow: 0 0 28px rgba(255,72,72,.45);
}

.ekg {
    position: absolute;
    right: 29%;
    bottom: 55px;
    width: 380px;
    height: 85px;
    opacity: .85;
    border-bottom: 2px solid rgba(57,215,255,.16);
    filter: drop-shadow(0 0 9px rgba(57,215,255,.7));
}

.ekg:after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    top: 40px;
    height: 3px;
    background: linear-gradient(
        90deg,
        transparent 0 7%, #39d7ff 7% 22%, transparent 22% 31%,
        #39d7ff 31% 34%, transparent 34% 42%, #39d7ff 42% 43%,
        transparent 43% 50%, #39d7ff 50% 61%, transparent 61% 100%
    );
    transform: skewY(-16deg);
}

/* Form section headings */
.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #073b78;
    font-size: 1.05rem;
    font-weight: 800;
    margin: 22px 0 10px;
}

.section-title span {
    display: inline-grid;
    place-items: center;
    width: 34px;
    height: 34px;
    border-radius: 11px;
    color: #0b72e7;
    background: linear-gradient(145deg, #e8f4ff, #d5ebff);
    box-shadow:
        inset 1px 1px 4px rgba(255,255,255,.9),
        3px 5px 12px rgba(13,89,155,.10);
}

/* Inputs */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 11px !important;
    border-color: #aebed1 !important;
    background: #ffffff !important;
}

div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #1687ff !important;
    box-shadow: 0 0 0 3px rgba(22,135,255,.10) !important;
}

.stNumberInput label,
.stSelectbox label {
    color: #172b45 !important;
    font-weight: 600 !important;
    font-size: .84rem !important;
}

/* Button */
.stFormSubmitButton > button {
    width: 100%;
    min-height: 52px;
    border: 0 !important;
    border-radius: 15px !important;
    color: white !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #0d6fe8, #14a4f4) !important;
    box-shadow:
        0 12px 25px rgba(13,111,232,.27),
        inset 0 1px 0 rgba(255,255,255,.35) !important;
    transition: all .2s ease;
}

.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    box-shadow:
        0 18px 32px rgba(13,111,232,.34),
        inset 0 1px 0 rgba(255,255,255,.4) !important;
}

/* Result cards */
.metric-card {
    position: relative;
    overflow: hidden;
    min-height: 132px;
    padding: 21px;
    border-radius: 19px;
    background: linear-gradient(145deg, #ffffff, #f2f7fd);
    border: 1px solid #dce8f5;
    box-shadow:
        0 14px 35px rgba(8,45,87,.09),
        inset 1px 1px 0 rgba(255,255,255,.9);
}

.metric-card:before {
    content: "";
    position: absolute;
    width: 100px;
    height: 100px;
    right: -35px;
    top: -35px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(22,135,255,.12), transparent 65%);
}

.metric-label {
    color: #334155;
    font-size: .76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .06em;
}

.metric-value {
    color: #063b78;
    font-size: 1.72rem;
    font-weight: 800;
    margin-top: 10px;
}

.metric-sub {
    color: #475569;
    font-size: .77rem;
    margin-top: 3px;
}

.risk-panel {
    margin-top: 18px;
    padding: 22px;
    border-radius: 19px;
    background: linear-gradient(145deg, #ffffff, #f7fbff);
    border: 1px solid #dce8f5;
    box-shadow: 0 14px 35px rgba(8,45,87,.08);
}

.risk-track {
    height: 13px;
    border-radius: 999px;
    background: linear-gradient(
        90deg, #22a85a 0%, #9aca39 35%,
        #f0bd2f 62%, #ef5a4f 100%
    );
    box-shadow: inset 0 2px 5px rgba(0,0,0,.12);
    position: relative;
}

.risk-marker {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: white;
    border: 4px solid #0d78e8;
    box-shadow: 0 3px 10px rgba(0,0,0,.2);
}

.risk-scale {
    display: flex;
    justify-content: space-between;
    color: #475569;
    font-size: .74rem;
    margin-top: 8px;
}

/* Footer */
.footer {
    margin-top: 42px;
    padding: 34px 10px 24px;
    color: #d7e3f1;
    background: linear-gradient(135deg, #041329, #092a52);
    border-radius: 26px 26px 0 0;
    box-shadow: 0 -18px 55px rgba(4,25,55,.14);
}

.footer-grid {
    display: grid;
    grid-template-columns: 1.25fr 1fr 1fr;
    gap: 34px;
}

.footer h3 {
    color: white;
    margin: 0 0 9px;
    font-size: 1rem;
}

.footer p {
    margin: 0;
    line-height: 1.65;
    font-size: .78rem;
}

.footer-brand {
    color: white;
    font-size: 1.12rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.footer-bottom {
    margin-top: 26px;
    padding-top: 17px;
    border-top: 1px solid rgba(255,255,255,.10);
    text-align: center;
    font-size: .72rem;
    color: #7e98b6;
}


/* Enhanced accessibility / contrast */
.stAlert {
    color: #172033 !important;
    border: 1px solid #c58a00 !important;
    background: #fff8df !important;
}

.stAlert p,
.stAlert [data-testid="stMarkdownContainer"] {
    color: #172033 !important;
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] input {
    color: #071525 !important;
    -webkit-text-fill-color: #071525 !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] span {
    color: #071525 !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] svg {
    fill: #17324f !important;
}

.stNumberInput button {
    color: #17324f !important;
    background: #eef4fa !important;
    border-color: #aebed1 !important;
}

[data-testid="stCaptionContainer"] {
    color: #334155 !important;
}

[data-testid="stExpander"] {
    border: 1px solid #aebed1 !important;
}

[data-testid="stExpander"] summary {
    color: #0a315e !important;
    font-weight: 700 !important;
}

.stMarkdown, .stMarkdown p, .stMarkdown li {
    color: #172033;
}

.footer {
    color: #d7e3f1 !important;
}

.footer p {
    color: #d7e3f1 !important;
}

.footer-bottom {
    color: #a9bdd4 !important;
}


/* ---------------- AI MVP sidebar / navigation ---------------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #031126 0%, #062c56 55%, #04172e 100%);
    border-right: 1px solid rgba(120, 190, 255, .18);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.4rem;
}

.nav-brand {
    padding: 8px 8px 22px;
    color: #ffffff;
}

.nav-brand-title {
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -.02em;
}

.nav-brand-subtitle {
    color: #b9cee4;
    font-size: .72rem;
    margin-top: 4px;
}

.nav-divider {
    height: 1px;
    background: rgba(255,255,255,.12);
    margin: 4px 0 18px;
}

.nav-label {
    color: #9db8d3;
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .1em;
    text-transform: uppercase;
    padding: 0 8px 8px;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    min-height: 48px;
    margin: 4px 0;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    background: rgba(255,255,255,.055) !important;
    color: #f3f8ff !important;
    box-shadow: none !important;
    text-align: left !important;
    font-weight: 700 !important;
    transition: all .18s ease;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(22,135,255,.24) !important;
    border-color: rgba(57,215,255,.45) !important;
    transform: translateX(2px);
}

.ai-panel {
    margin-top: 10px;
    padding: 18px;
    border-radius: 18px;
    background: linear-gradient(145deg, #071b37, #0a3767);
    border: 1px solid rgba(94,190,255,.22);
    box-shadow: 0 16px 38px rgba(0,0,0,.18);
}

.ai-panel-title {
    color: #ffffff;
    font-weight: 800;
    font-size: 1rem;
}

.ai-panel-text {
    color: #c5d8eb;
    font-size: .78rem;
    line-height: 1.55;
    margin-top: 5px;
}

.chat-bubble-user {
    background: #0d6fe8;
    color: #ffffff;
    padding: 12px 15px;
    border-radius: 16px 16px 4px 16px;
    margin: 8px 0 8px 12%;
    line-height: 1.55;
}

.chat-bubble-ai {
    background: #edf5fc;
    color: #10233c;
    padding: 12px 15px;
    border: 1px solid #c9d9e8;
    border-radius: 16px 16px 16px 4px;
    margin: 8px 12% 8px 0;
    line-height: 1.55;
}

.chat-title {
    color: #073b78;
    font-size: 1.35rem;
    font-weight: 800;
    margin-bottom: 3px;
}

.chat-subtitle {
    color: #334155;
    font-size: .84rem;
    margin-bottom: 18px;
}

.report-card {
    padding: 22px;
    border-radius: 20px;
    background: linear-gradient(145deg, #ffffff, #f3f8fd);
    border: 1px solid #b8cadc;
    box-shadow: 0 16px 40px rgba(8,45,87,.09);
}

.report-title {
    color: #063b78;
    font-size: 1.35rem;
    font-weight: 800;
}

.report-text {
    color: #172033;
    line-height: 1.65;
}

.dev-card {
    padding: 25px;
    border-radius: 22px;
    background: linear-gradient(145deg, #ffffff, #f2f7fd);
    border: 1px solid #b8cadc;
    box-shadow: 0 16px 40px rgba(8,45,87,.09);
}

.dev-card h2 {
    color: #063b78;
}

.dev-card p {
    color: #24364d;
    line-height: 1.7;
}

@media (max-width: 850px) {
    .hero {
        padding: 32px 26px;
        min-height: 300px;
    }

    .hero-orb {
        opacity: .55;
        right: -55px;
    }

    .ekg {
        right: 8%;
        width: 270px;
    }

    .footer-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------- Model ---------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        f"model.pkl was not found.\n\n"
        f"Expected location:\n`{MODEL_PATH}`\n\n"
        "Put model.pkl in the same folder as app.py."
    )
    st.stop()

try:
    bundle = load_model()
except Exception as exc:
    st.error(f"Could not load model.pkl: {exc}")
    st.stop()

models = bundle["models"]
feature_cols = bundle["feature_cols"]
numeric_cols = set(bundle["numeric_cols"])
categorical_options = bundle["categorical_options"]
numeric_defaults = bundle["numeric_defaults"]


# ----------------------------- Navigation / MVP panels ----------------------
if "active_page" not in st.session_state:
    st.session_state.active_page = "Prediction"

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "report_pdf" not in st.session_state:
    st.session_state.report_pdf = None


def set_page(page):
    st.session_state.active_page = page


with st.sidebar:
    st.markdown(
        """
        <div class="nav-brand">
            <div class="nav-brand-title">♥ Cardiac Risk MVP</div>
            <div class="nav-brand-subtitle">AI-assisted cardiac risk platform</div>
        </div>
        <div class="nav-divider"></div>
        <div class="nav-label">Navigation</div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("⌂  Risk Prediction", use_container_width=True):
        set_page("Prediction")

    if st.button("◉  About the Developer", use_container_width=True):
        set_page("Developer")

    if st.button("▣  Get a Report", use_container_width=True):
        set_page("Report")

    if st.button("✦  AI Assistance", use_container_width=True):
        set_page("AI Assistance")

    st.markdown(
        """
        <div class="ai-panel">
            <div class="ai-panel-title">AI Assistance</div>
            <div class="ai-panel-text">
                Ask questions about the generated risk results and
                patient inputs using your Groq-powered assistant.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------- Header --------------------------------------
st.markdown(
    """
<div class="hero">
    <div class="hero-content">
        <div class="hero-badge">AI-Powered Clinical Risk MVP</div>
        <h1>Cardiac Risk<br>Prediction</h1>
        <p>
            Estimate cardiac-risk indicators from patient measurements,
            history, lifestyle, symptoms, ECG and laboratory data.
        </p>
    </div>
    <div class="ekg"></div>
    <div class="hero-orb"></div>
</div>
""",
    unsafe_allow_html=True,
)

st.warning(
    "For research/demo use only. This model is not a medical diagnosis or a "
    "substitute for assessment by a qualified clinician."
)


# ----------------------------- Prediction state ----------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if st.session_state.active_page == "Prediction":
    # ----------------------------- Input form ----------------------------------
    with st.form("patient_form"):
        values = {}

        sections = [
            (
                "Patient & Measurements",
                [
                    "Age (yrs)",
                    "Sex",
                    "Weight (kg)",
                    "Height (cm)",
                    "BMI (kg/m²)",
                    "Systolic BP (mmHg)",
                    "Diastolic BP (mmHg)",
                    "Resting HR (bpm)",
                ],
            ),
            (
                "Family History",
                [
                    "Fam Hx: MI",
                    "Fam Hx: SCD",
                    "Fam Hx: Hypertension",
                    "Fam Hx: Diabetes",
                ],
            ),
            (
                "Lifestyle",
                [
                    "Smoking Status",
                    "Alcohol",
                    "Physical Activity",
                    "Daily Sitting (hrs)",
                    "Sleep Duration (hrs)",
                    "Stress Level",
                ],
            ),
            (
                "Existing Diseases",
                [
                    "Hypertension",
                    "Diabetes",
                    "High Cholesterol",
                    "Kidney Disease",
                    "Previous Heart Disease",
                ],
            ),
            (
                "Symptoms",
                [
                    "Chest Pain",
                    "Shortness of Breath",
                    "Palpitations",
                    "Dizziness",
                    "Syncope",
                ],
            ),
            (
                "ECG & Lab Values",
                [
                    "ECG Findings",
                    "Total Cholesterol (mg/dL)",
                    "LDL (mg/dL)",
                    "HDL (mg/dL)",
                    "Triglycerides (mg/dL)",
                    "HbA1c (%)",
                    "Hemoglobin (g/dL)",
                ],
            ),
        ]

        for section_name, fields in sections:
            available = [field for field in fields if field in feature_cols]

            if not available:
                continue

            st.markdown(
                f'<div class="section-title"><span>✦</span>{section_name}</div>',
                unsafe_allow_html=True,
            )

            cols = st.columns(2)

            for index, field in enumerate(available):
                with cols[index % 2]:
                    if field in numeric_cols:
                        values[field] = st.number_input(
                            field,
                            value=float(numeric_defaults.get(field, 0.0)),
                            format="%.2f",
                            help=(
                                "Default is the training-set median. "
                                "Replace it with the patient's value."
                            ),
                        )
                    else:
                        options = categorical_options.get(field, [])

                        if not options:
                            options = ["Unknown"]

                        values[field] = st.selectbox(field, options)

        submitted = st.form_submit_button(
            "Predict cardiac risk",
            type="primary",
        )

    # ----------------------------- Prediction ----------------------------------
    if submitted:
        input_df = pd.DataFrame(
            [{column: values.get(column) for column in feature_cols}]
        )

        try:
            score = float(
                models["framingham_score"].predict(input_df)[0]
            )
            years = float(
                models["est_years_to_event"].predict(input_df)[0]
            )
            risk = float(
                models["cvd_risk_pct"].predict(input_df)[0]
            )
            category = str(
                models["risk_category"].predict(input_df)[0]
            )

            st.session_state.prediction = {
                "framingham_score": score,
                "years_to_event": years,
                "cvd_risk": risk,
                "risk_category": category,
                "inputs": input_df.iloc[0].to_dict(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }

            score = max(0.0, score)
            years = max(0.0, years)
            risk = max(0.0, min(100.0, risk))

            st.markdown("### Prediction Report")

            # General, non-diagnostic suggestions based on model output and
            # supplied lifestyle/risk-factor fields.
            suggestions = []
            if risk >= 20:
                suggestions.append(
                    "Consider discussing the elevated estimated CVD risk with a qualified clinician.")
            elif risk >= 10:
                suggestions.append(
                    "Consider a routine cardiovascular risk review with a qualified clinician.")
            else:
                suggestions.append(
                    "Continue regular preventive health reviews and monitoring of cardiovascular risk factors.")

            smoking = str(values.get("Smoking Status", "")).lower()
            if smoking and smoking not in {"never", "no", "non-smoker", "none"}:
                suggestions.append(
                    "If applicable, discuss evidence-based smoking cessation support with a healthcare professional.")

            activity = str(values.get("Physical Activity", "")).lower()
            if activity and any(x in activity for x in ["low", "sedentary", "none", "inactive"]):
                suggestions.append(
                    "If medically appropriate, gradually increase regular physical activity and discuss a suitable plan with a clinician.")

            if float(values.get("Daily Sitting (hrs)", 0) or 0) >= 8:
                suggestions.append(
                    "Consider reducing prolonged sitting with regular movement breaks.")

            if float(values.get("Sleep Duration (hrs)", 0) or 0) > 0 and (
                float(values.get("Sleep Duration (hrs)", 0) or 0) < 6
                or float(values.get("Sleep Duration (hrs)", 0) or 0) > 10
            ):
                suggestions.append(
                    "Review sleep duration and quality as part of general cardiovascular health.")

            st.session_state.prediction["suggestions"] = suggestions

            columns = st.columns(4)
            cards = [
                ("Framingham Score", f"{score:.1f}", "pts"),
                ("Est. Yrs to Cardiac Event", f"{years:.1f}", "years"),
                ("10-Yr CVD Risk", f"{risk:.1f}", "%"),
                ("Risk Category", category, "model classification"),
            ]

            for column, (label, value, subtitle) in zip(columns, cards):
                with column:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value">{value}</div>
                            <div class="metric-sub">{subtitle}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            marker = max(1.5, min(98.5, risk))
            st.markdown(
                f"""
                <div class="risk-panel">
                    <div class="metric-label">10-Year CVD Risk Profile</div>
                    <div style="height:18px"></div>
                    <div class="risk-track">
                        <div class="risk-marker" style="left:{marker}%"></div>
                    </div>
                    <div class="risk-scale">
                        <span>Lower risk</span>
                        <strong>{risk:.1f}%</strong>
                        <span>Higher risk</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="report-card" style="margin-top:18px;">
                    <div class="report-title">Possible Next Steps</div>
                    <ul class="report-text">
                        {''.join(f'<li>{s}</li>' for s in suggestions)}
                    </ul>
                    <p class="report-text">
                        These are general educational suggestions, not a diagnosis
                        or individualized treatment plan.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ----------------------------- MVP Panels ---------------------------------
def get_groq_client():
    if Groq is None:
        return None

    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

    if not api_key:
        return None

    return Groq(api_key=api_key)


def build_ai_context():
    prediction = st.session_state.get("prediction")

    if not prediction:
        return (
            "No prediction has been generated yet. Explain that the user should "
            "first complete Risk Prediction."
        )

    safe_inputs = prediction["inputs"]

    return f"""
The application generated these model outputs:
- Framingham Score: {prediction["framingham_score"]:.2f} points
- Estimated years to cardiac event: {prediction["years_to_event"]:.2f}
- 10-year CVD risk: {prediction["cvd_risk"]:.2f}%
- Risk category: {prediction["risk_category"]}

Patient input fields used by the model:
{safe_inputs}

Answer questions using only this supplied application context and general,
non-diagnostic health education. Do not claim to diagnose disease, prescribe
medication, or replace a clinician. If the user asks for a diagnosis or urgent
medical decision, clearly recommend professional medical assessment.
"""


def render_ai_assistance():
    st.markdown('<div class="chat-title">AI Assistance</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-subtitle">Ask the AI assistant to explain the model results in simple language.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.get("prediction"):
        st.info("Generate a cardiac-risk prediction first. The AI assistant will then have the model results as context.")

    client = get_groq_client()

    if client is None:
        st.warning(
            "Groq is not configured. Add your API key as GROQ_API_KEY in "
            "Streamlit secrets or as an environment variable."
        )

        with st.expander("Configure Groq API"):
            st.code(
                '[groq]\nGROQ_API_KEY = "your_groq_api_key"',
                language="toml",
            )
            st.caption(
                "Recommended for Streamlit Cloud: add GROQ_API_KEY in "
                "Settings → Secrets."
            )

    for message in st.session_state.chat_messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            st.markdown(
                f'<div class="chat-bubble-user">{content}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-bubble-ai">{content}</div>',
                unsafe_allow_html=True,
            )

    prompt = st.chat_input(
        "Ask about your cardiac-risk results...",
        disabled=(client is None),
    )

    if prompt and client:
        st.session_state.chat_messages.append(
            {"role": "user", "content": prompt}
        )

        system_prompt = """
You are the AI assistance layer of a cardiac-risk prediction MVP.
Your role is to explain the application's model outputs and patient-input
patterns clearly and cautiously.

Rules:
1. Do not diagnose the patient.
2. Do not prescribe medicines or treatment.
3. Do not invent patient data or model results.
4. Explain that model predictions are estimates, not clinical conclusions.
5. Use plain language unless the user asks for technical detail.
6. For potentially urgent symptoms, advise seeking appropriate professional
   medical attention rather than giving a definitive diagnosis.
"""

        try:
            with st.spinner("AI assistant is thinking..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt + "\n\nAPPLICATION CONTEXT:\n" + build_ai_context(),
                        },
                        *st.session_state.chat_messages[-10:],
                    ],
                    temperature=0.2,
                    max_tokens=700,
                )

            answer = response.choices[0].message.content
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()

        except Exception as exc:
            st.error(f"Groq request failed: {exc}")


def build_report_pdf(prediction):
    """Create a polished A4 PDF matching the visual MVP report style."""
    buffer = io.BytesIO()

    # ------------------------------------------------------------------
    # Local imports so the rest of your code structure does not change
    # ------------------------------------------------------------------
    from reportlab.platypus import Flowable
    from reportlab.pdfgen import canvas as pdfcanvas
    from reportlab.lib.enums import TA_CENTER

    # ------------------------------------------------------------------
    # Brand palette
    # ------------------------------------------------------------------
    NAVY = colors.HexColor("#0B1F3A")
    NAVY_2 = colors.HexColor("#123B6B")
    BLUE = colors.HexColor("#2563EB")
    SKY = colors.HexColor("#E8F1FC")
    SLATE = colors.HexColor("#475569")
    LINE = colors.HexColor("#D7E2EC")
    CARD_BG = colors.HexColor("#F7FAFD")

    GREEN = colors.HexColor("#16A34A")
    LIME = colors.HexColor("#84CC16")
    AMBER = colors.HexColor("#F59E0B")
    RED = colors.HexColor("#DC2626")

    RISK_BANDS = [
        (0, 20, "Low", GREEN),
        (20, 40, "Moderate", LIME),
        (40, 70, "High", AMBER),
        (70, 100.0001, "Very High", RED),
    ]

    # ------------------------------------------------------------------
    # Helper: determine risk band
    # ------------------------------------------------------------------
    def risk_band(risk_pct):
        for lo, hi, label, color in RISK_BANDS:
            if lo <= risk_pct < hi:
                return label, color
        return RISK_BANDS[-1][2], RISK_BANDS[-1][3]

    # ------------------------------------------------------------------
    # Custom risk gauge
    # ------------------------------------------------------------------
    class RiskGauge(Flowable):
        def __init__(self, risk_pct, width=172 * mm, height=26 * mm):
            Flowable.__init__(self)
            self.risk_pct = max(0.0, min(100.0, float(risk_pct)))
            self.width = width
            self.height = height

        def draw(self):
            c = self.canv

            bar_h = 7 * mm
            bar_y = self.height - bar_h - 9 * mm
            bar_w = self.width

            # Segmented risk bar
            x = 0

            for lo, hi, label, color in RISK_BANDS:
                hi_clamped = min(hi, 100)
                seg_w = bar_w * (hi_clamped - lo) / 100.0

                c.setFillColor(color)
                c.roundRect(
                    x,
                    bar_y,
                    seg_w,
                    bar_h,
                    2,
                    stroke=0,
                    fill=1,
                )

                x += seg_w

            # Outer border
            c.setStrokeColor(colors.white)
            c.setLineWidth(0.6)
            c.roundRect(
                0,
                bar_y,
                bar_w,
                bar_h,
                2,
                stroke=1,
                fill=0,
            )

            # Pointer
            px = bar_w * (self.risk_pct / 100.0)
            px = max(3 * mm, min(bar_w - 3 * mm, px))

            tri_h = 5 * mm

            c.setFillColor(NAVY)

            p = c.beginPath()
            p.moveTo(
                px,
                bar_y + bar_h + 1.5 * mm,
            )
            p.lineTo(
                px - 2.6 * mm,
                bar_y + bar_h + 1.5 * mm + tri_h,
            )
            p.lineTo(
                px + 2.6 * mm,
                bar_y + bar_h + 1.5 * mm + tri_h,
            )
            p.close()

            c.drawPath(
                p,
                stroke=0,
                fill=1,
            )

            # Risk value
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(NAVY)

            c.drawCentredString(
                px,
                bar_y + bar_h + 1.5 * mm + tri_h + 2 * mm,
                f"{self.risk_pct:.1f}%",
            )

            # Band labels
            c.setFont("Helvetica", 6.6)
            c.setFillColor(SLATE)

            x = 0

            for lo, hi, label, color in RISK_BANDS:
                hi_clamped = min(hi, 100)
                seg_w = bar_w * (hi_clamped - lo) / 100.0

                c.drawCentredString(
                    x + seg_w / 2,
                    bar_y - 4.2 * mm,
                    label,
                )

                x += seg_w

    # ------------------------------------------------------------------
    # KPI cards
    # ------------------------------------------------------------------
    def stat_cards(
        framingham,
        years,
        risk_pct,
        category,
        cat_color,
    ):
        label_style = ParagraphStyle(
            "CardLabel",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=9,
            textColor=SLATE,
            alignment=TA_CENTER,
            spaceAfter=2,
        )

        value_style = ParagraphStyle(
            "CardValue",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=20,
            textColor=NAVY,
            alignment=TA_CENTER,
        )

        cat_value_style = ParagraphStyle(
            "CardValueCat",
            parent=value_style,
            fontSize=13,
            textColor=colors.white,
        )

        cards = [
            (
                "FRAMINGHAM SCORE",
                f"{framingham:.1f}",
                "pts",
                None,
            ),
            (
                "EST. YEARS TO EVENT",
                f"{years:.1f}",
                "yrs",
                None,
            ),
            (
                "10-YR CVD RISK",
                f"{risk_pct:.1f}",
                "%",
                None,
            ),
            (
                "RISK CATEGORY",
                str(category),
                "",
                cat_color,
            ),
        ]

        cells = []

        for title_txt, val, unit, bg in cards:

            vstyle = (
                cat_value_style
                if bg
                else value_style
            )

            inner = Table(
                [
                    [
                        Paragraph(
                            title_txt,
                            label_style,
                        )
                    ],
                    [
                        Paragraph(
                            f"{val}<font size=9> {unit}</font>",
                            vstyle,
                        )
                    ],
                ],
                colWidths=[40 * mm],
            )

            inner.setStyle(
                TableStyle(
                    [
                        (
                            "TOPPADDING",
                            (0, 0),
                            (-1, 0),
                            9,
                        ),
                        (
                            "BOTTOMPADDING",
                            (0, 0),
                            (-1, 0),
                            2,
                        ),
                        (
                            "TOPPADDING",
                            (0, 1),
                            (-1, 1),
                            2,
                        ),
                        (
                            "BOTTOMPADDING",
                            (0, 1),
                            (-1, 1),
                            10,
                        ),
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, -1),
                            bg if bg else CARD_BG,
                        ),
                        (
                            "BOX",
                            (0, 0),
                            (-1, -1),
                            0.8,
                            bg if bg else LINE,
                        ),
                        (
                            "ROUNDEDCORNERS",
                            [6, 6, 6, 6],
                        ),
                        (
                            "ALIGN",
                            (0, 0),
                            (-1, -1),
                            "CENTER",
                        ),
                    ]
                )
            )

            cells.append(inner)

        row = Table(
            [cells],
            colWidths=[43 * mm] * 4,
            hAlign="CENTER",
        )

        row.setStyle(
            TableStyle(
                [
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        2,
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        2,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                ]
            )
        )

        return row

    # ------------------------------------------------------------------
    # Page header/footer
    # ------------------------------------------------------------------
    def decorate_page(c, doc):
        c.saveState()

        page_w, page_h = A4

        # Header band
        c.setFillColor(NAVY)
        c.rect(
            0,
            page_h - 20 * mm,
            page_w,
            20 * mm,
            stroke=0,
            fill=1,
        )

        # Blue accent line
        c.setFillColor(BLUE)
        c.rect(
            0,
            page_h - 20.8 * mm,
            page_w,
            0.8 * mm,
            stroke=0,
            fill=1,
        )

        # Logo circle
        c.setFillColor(colors.white)

        c.circle(
            20 * mm,
            page_h - 10 * mm,
            5.4 * mm,
            stroke=0,
            fill=1,
        )

        # Pulse line
        c.setStrokeColor(NAVY)
        c.setLineWidth(1.1)

        c.line(
            20 * mm - 3 * mm,
            page_h - 10 * mm,
            20 * mm - 1 * mm,
            page_h - 10 * mm,
        )

        c.line(
            20 * mm - 1 * mm,
            page_h - 10 * mm,
            20 * mm - 0.2 * mm,
            page_h - 12.4 * mm,
        )

        c.line(
            20 * mm - 0.2 * mm,
            page_h - 12.4 * mm,
            20 * mm + 1 * mm,
            page_h - 7.6 * mm,
        )

        c.line(
            20 * mm + 1 * mm,
            page_h - 7.6 * mm,
            20 * mm + 1.8 * mm,
            page_h - 10 * mm,
        )

        c.line(
            20 * mm + 1.8 * mm,
            page_h - 10 * mm,
            20 * mm + 3.4 * mm,
            page_h - 10 * mm,
        )

        # Brand name
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 12.5)

        c.drawString(
            29 * mm,
            page_h - 8.6 * mm,
            "CardiacRisk AI",
        )

        c.setFont("Helvetica", 7.6)
        c.setFillColor(colors.HexColor("#B9CCEA"))

        c.drawString(
            29 * mm,
            page_h - 12.6 * mm,
            "Predictive Cardiovascular Risk Report",
        )

        # Patient reference
        patient_ref = prediction.get(
            "patient_ref",
            "",
        )

        if patient_ref:
            c.setFont(
                "Helvetica-Bold",
                8,
            )

            c.setFillColor(colors.white)

            c.drawRightString(
                page_w - 16 * mm,
                page_h - 8.6 * mm,
                str(patient_ref),
            )

        # Footer line
        c.setStrokeColor(LINE)
        c.setLineWidth(0.6)

        c.line(
            16 * mm,
            12 * mm,
            page_w - 16 * mm,
            12 * mm,
        )

        # Footer text
        c.setFont(
            "Helvetica",
            7,
        )

        c.setFillColor(SLATE)

        c.drawString(
            16 * mm,
            8.5 * mm,
            "CardiacRisk AI  ·  Confidential patient report",
        )

        c.drawCentredString(
            page_w / 2,
            8.5 * mm,
            "Not a substitute for professional medical advice",
        )

        c.drawRightString(
            page_w - 16 * mm,
            8.5 * mm,
            f"Page {doc.page}",
        )

        c.restoreState()

    # ------------------------------------------------------------------
    # PDF document
    # ------------------------------------------------------------------
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=28 * mm,
        bottomMargin=18 * mm,
        title="Cardiac Risk Assessment",
        author="CardiacRisk AI",
    )

    styles = getSampleStyleSheet()

    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=SLATE,
        alignment=TA_CENTER,
        spaceAfter=10,
    )

    section = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=NAVY,
        spaceBefore=10,
        spaceAfter=6,
    )

    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#172033"),
        spaceAfter=4,
    )

    suggestion_style = ParagraphStyle(
        "Suggestion",
        parent=body,
        leftIndent=0,
        spaceAfter=5,
    )

    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=7.3,
        leading=10.5,
        textColor=SLATE,
        spaceAfter=0,
    )

    # ------------------------------------------------------------------
    # Risk values
    # ------------------------------------------------------------------
    risk = max(
        0.0,
        min(
            100.0,
            float(prediction["cvd_risk"]),
        ),
    )

    computed_label, cat_color = risk_band(risk)

    category = str(
        prediction.get("risk_category")
        or computed_label
    )

    # ------------------------------------------------------------------
    # Story
    # ------------------------------------------------------------------
    story = []

    story.append(
        Spacer(
            1,
            2 * mm,
        )
    )

    story.append(
        Paragraph(
            f"Generated {prediction['timestamp']}",
            subtitle,
        )
    )

    # ------------------------------------------------------------------
    # KPI cards
    # ------------------------------------------------------------------
    story.append(
        stat_cards(
            prediction["framingham_score"],
            prediction["years_to_event"],
            risk,
            category,
            cat_color,
        )
    )

    story.append(
        Spacer(
            1,
            8 * mm,
        )
    )

    # ------------------------------------------------------------------
    # Risk gauge
    # ------------------------------------------------------------------
    story.append(
        Paragraph(
            "10-Year CVD Risk Profile",
            section,
        )
    )

    gauge_wrap = Table(
        [
            [
                RiskGauge(
                    risk,
                    width=172 * mm,
                )
            ]
        ],
        colWidths=[172 * mm],
    )

    gauge_wrap.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    CARD_BG,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.8,
                    LINE,
                ),
                (
                    "ROUNDEDCORNERS",
                    [8, 8, 8, 8],
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    0,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    0,
                ),
            ]
        )
    )

    story.append(gauge_wrap)

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------
    story.append(
        Paragraph(
            "Recommended Next Steps",
            section,
        )
    )

    suggestions = (
        prediction.get("suggestions")
        or [
            "Discuss this result with a qualified healthcare professional.",
            "Continue monitoring established cardiovascular risk factors.",
            "Maintain healthy lifestyle habits appropriate to your circumstances.",
        ]
    )

    for suggestion in suggestions:

        check_style = ParagraphStyle(
            "Check",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=colors.white,
            alignment=TA_CENTER,
        )

        bullet = Table(
            [
                [
                    Paragraph(
                        "✓",
                        check_style,
                    ),
                    Paragraph(
                        str(suggestion),
                        suggestion_style,
                    ),
                ]
            ],
            colWidths=[
                8 * mm,
                164 * mm,
            ],
        )

        bullet.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, 0),
                        GREEN,
                    ),
                    (
                        "ROUNDEDCORNERS",
                        [4, 4, 4, 4],
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                    (
                        "ALIGN",
                        (0, 0),
                        (0, 0),
                        "CENTER",
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        3,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        3,
                    ),
                    (
                        "LEFTPADDING",
                        (1, 0),
                        (1, 0),
                        6,
                    ),
                ]
            )
        )

        story.append(bullet)

        story.append(
            Spacer(
                1,
                1.5 * mm,
            )
        )

    # ------------------------------------------------------------------
    # Model inputs
    # ------------------------------------------------------------------
    inputs = prediction.get(
        "inputs",
        {},
    )

    if inputs:

        story.append(
            Paragraph(
                "Model Inputs",
                section,
            )
        )

        input_rows = [
            [
                Paragraph(
                    "<b>Parameter</b>",
                    body,
                ),
                Paragraph(
                    "<b>Value</b>",
                    body,
                ),
            ]
        ]

        for key, value in inputs.items():

            try:
                if pd.isna(value):
                    value = "N/A"
            except (TypeError, ValueError):
                pass

            input_rows.append(
                [
                    Paragraph(
                        str(key),
                        body,
                    ),
                    Paragraph(
                        str(value),
                        body,
                    ),
                ]
            )

        input_table = Table(
            input_rows,
            colWidths=[
                90 * mm,
                82 * mm,
            ],
            repeatRows=1,
        )

        style_cmds = [
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                NAVY,
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.35,
                LINE,
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                6,
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                6,
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                4.5,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                4.5,
            ),
        ]

        for r in range(
            1,
            len(input_rows),
        ):
            if r % 2 == 0:
                style_cmds.append(
                    (
                        "BACKGROUND",
                        (0, r),
                        (-1, r),
                        CARD_BG,
                    )
                )

        input_table.setStyle(
            TableStyle(style_cmds)
        )

        story.append(input_table)

    # ------------------------------------------------------------------
    # Disclaimer
    # ------------------------------------------------------------------
    story.append(
        Spacer(
            1,
            6 * mm,
        )
    )

    disclaimer = Table(
        [
            [
                Paragraph(
                    "<b>Important Disclaimer</b><br/>"
                    "This report is generated by a machine-learning MVP for "
                    "research and educational purposes. The estimates are not "
                    "a medical diagnosis, do not establish that a cardiac event "
                    "will occur, and should not replace assessment by a qualified "
                    "healthcare professional. Seek appropriate medical care for "
                    "concerning or urgent symptoms.",
                    small,
                )
            ]
        ],
        colWidths=[
            172 * mm
        ],
    )

    disclaimer.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#FFF8E8"),
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor("#F0DFA8"),
                ),
                (
                    "ROUNDEDCORNERS",
                    [6, 6, 6, 6],
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(disclaimer)

    # ------------------------------------------------------------------
    # Build PDF
    # ------------------------------------------------------------------
    doc.build(
        story,
        onFirstPage=decorate_page,
        onLaterPages=decorate_page,
    )

    buffer.seek(0)

    return buffer.getvalue()


def send_report_email(recipient, pdf_bytes):
    sender = st.secrets.get(
        "GMAIL_SENDER",
        os.getenv("GMAIL_SENDER", "khan.abdullah135790@gmail.com"),
    )
    app_password = st.secrets.get(
        "GMAIL_APP_PASSWORD",
        os.getenv("GMAIL_APP_PASSWORD"),
    )

    if not app_password:
        raise RuntimeError(
            "GMAIL_APP_PASSWORD is not configured. Add it to "
            ".streamlit/secrets.toml or the hosting platform's secrets."
        )

    msg = EmailMessage()
    msg["Subject"] = "Cardiac Risk Assessment Report"
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(
        "Attached is your Cardiac Risk Assessment report generated by the "
        "Cardiac Risk MVP.\n\n"
        "This report is for research/educational use and is not a medical diagnosis."
    )
    msg.add_attachment(
        pdf_bytes,
        maintype="application",
        subtype="pdf",
        filename="cardiac_risk_assessment.pdf",
    )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
        smtp.login(sender, app_password)
        smtp.send_message(msg)


def render_report():
    st.markdown("## Get a Report")
    st.markdown(
        '<div class="chat-subtitle">Create the polished PDF assessment and optionally send it by email.</div>',
        unsafe_allow_html=True,
    )

    prediction = st.session_state.get("prediction")
    if not prediction:
        st.info("Generate a prediction first. The report will then contain the model results and possible suggestions.")
        return

    # Always rebuild the PDF from the latest prediction.
    if st.button("Generate PDF Report", type="primary", use_container_width=True):
        try:
            st.session_state.report_pdf = build_report_pdf(prediction)
            st.success("PDF report generated successfully.")
        except Exception as exc:
            st.error(f"Could not generate the PDF: {exc}")

    if st.session_state.report_pdf:
        st.download_button(
            "Download PDF Report",
            data=st.session_state.report_pdf,
            file_name="cardiac_risk_assessment.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        st.divider()
        st.markdown("### Email Report")
        recipient = st.text_input(
            "Recipient email address",
            placeholder="patient@example.com",
            help="The generated PDF will be attached to the email.",
        )

        if st.button("Send PDF to Email", use_container_width=True):
            if not recipient or "@" not in recipient:
                st.error("Enter a valid recipient email address.")
            else:
                try:
                    send_report_email(recipient.strip(),
                                      st.session_state.report_pdf)
                    st.success(
                        f"Report sent successfully to {recipient.strip()}.")
                except Exception as exc:
                    st.error(f"Could not send the report: {exc}")

        st.caption(
            "Sender: khan.abdullah135790@gmail.com · "
            "The sender password is never stored in app.py."
        )


def render_developer():
    st.markdown("## About the Developer")

    st.markdown(
        """
      <div class="dev-card"> <h2>Cardiac Risk MVP</h2> <p> This MVP combines a machine-learning cardiac-risk prediction pipeline. The platform demonstrates how predictive analytics, automated reporting, and conversational AI can be integrated into a healthcare-oriented application. </p>  <div class="social-links"> <a href="https://github.com/Abdullah-Builds" target="_blank" rel="noopener noreferrer" class="social-link github"> <i class="fab fa-github"></i> <span> <strong>GitHub</strong> <small>View the source code</small> </span> </a> <a href="https://linkedin.com/in/abdullah-khan-718a25299" target="_blank" rel="noopener noreferrer" class="social-link linkedin"> <i class="fab fa-linkedin"></i> <span> <strong>LinkedIn</strong> <small>Connect with the developer</small> </span> </a> </div> </div>
        """,
        unsafe_allow_html=True,
    )


if st.session_state.active_page == "AI Assistance":
    render_ai_assistance()
elif st.session_state.active_page == "Report":
    render_report()
elif st.session_state.active_page == "Developer":
    render_developer()


if st.session_state.active_page == "Prediction":
    # ----------------------------- Model information ---------------------------
    with st.expander("Model information"):
        st.write(
            f"Training rows: {bundle.get('training_rows', 'N/A')}"
        )
        st.write(
            f"Model version: {bundle.get('model_version', 'N/A')}"
        )
        st.json(bundle.get("metrics_holdout", {}))

    # ----------------------------- Footer --------------------------------------
    components.html(
        """
        <div style="
            box-sizing:border-box;
            width:100%;
            padding:34px 42px 22px;
            border-radius:26px 26px 0 0;
            background:linear-gradient(135deg,#020b1b 0%,#062a55 100%);
            color:#d7e3f1;
            font-family:Inter,Arial,sans-serif;
            box-shadow:0 -18px 55px rgba(4,25,55,.14);
        ">
            <div style="
                display:grid;
                grid-template-columns:1.25fr 1fr 1fr;
                gap:34px;
                align-items:start;
            ">
                <div>
                    <div style="
                        color:#ffffff;
                        font-size:19px;
                        font-weight:800;
                        margin-bottom:9px;
                    ">
                        ♥ Cardiac Risk MVP
                    </div>
                    <div style="
                        font-size:13px;
                        line-height:1.65;
                        color:#d7e3f1;
                    ">
                        AI-powered cardiac risk estimation with a
                        production-oriented interface.
                    </div>
                </div>

                <div>
                    <div style="
                        color:#ffffff;
                        font-size:15px;
                        font-weight:700;
                        margin-bottom:9px;
                    ">
                        About
                    </div>
                    <div style="
                        font-size:13px;
                        line-height:1.65;
                        color:#d7e3f1;
                    ">
                        This MVP uses machine learning to estimate the
                        four outputs generated by the supplied dataset.
                    </div>
                </div>

                <div>
                    <div style="
                        color:#ffffff;
                        font-size:15px;
                        font-weight:700;
                        margin-bottom:9px;
                    ">
                        Disclaimer
                    </div>
                    <div style="
                        font-size:13px;
                        line-height:1.65;
                        color:#d7e3f1;
                    ">
                        For research and educational use only. It is not
                        a substitute for professional medical assessment
                        or diagnosis.
                    </div>
                </div>
            </div>

            <div style="
                margin-top:26px;
                padding-top:17px;
                border-top:1px solid rgba(255,255,255,.10);
                text-align:center;
                font-size:12px;
                color:#a9bdd4;
            ">
                Cardiac Risk MVP · Machine Learning Demonstration
            </div>
        </div>

        <style>
            @media (max-width:850px) {
                div[style*="grid-template-columns:1.25fr"] {
                    grid-template-columns:1fr !important;
                }
            }
        </style>
        """,
        height=245,
        scrolling=False,
    )
