"""
Survival prediction web app for clinicians.
Run: streamlit run app.py
Requires: cox_model.pkl, app_config.json (from train_survival_models.py), survival_df.csv not needed at runtime.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patheffects import withStroke

from train_survival_models import load_cox_model

# --- Load config ---
CONFIG_PATH = Path(__file__).parent / "app_config.json"
if not CONFIG_PATH.exists():
    st.error(
        "App config not found. Run **train_survival_models.py** first to generate "
        "cox_model.pkl and app_config.json, then restart this app."
    )
    st.stop()

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)
FEATURE_COLS = CONFIG["feature_cols"]
AGE_MEAN = CONFIG["age_mean"]
AGE_STD = CONFIG["age_std"]
MSI_DEFAULT = CONFIG.get("msi_default", 0)
RISK_P25 = CONFIG.get("cox_risk_p25")
RISK_P50 = CONFIG.get("cox_risk_p50")
RISK_P75 = CONFIG.get("cox_risk_p75")

# --- Load Cox model (cached) ---
@st.cache_resource
def get_cox_model():
    path = Path(__file__).parent / "cox_model.pkl"
    if not path.exists():
        return None
    return load_cox_model(str(path))


st.set_page_config(page_title="Surviva — Survival Risk & Prognosis", layout="wide")
# --- NIGHT MODE ONLY CSS ---
night_mode_css = """
<style>
    /* Overall page background - DARK MODE */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        background-attachment: fixed;
    }
    
    /* DNA helix pattern overlay - removed for cleaner professional look */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: none;
        pointer-events: none;
        z-index: 0;
    }

        background-repeat: repeat;
        background-attachment: fixed;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Main content area */
    [data-testid="stAppViewContainer"] > [data-testid="stVerticalBlockBorderContainer"] {
        position: relative;
        z-index: 1;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%) !important;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* Header and title styling */
    h1, h2, h3 {
        color: #58a6ff !important;
        font-weight: 900 !important;
        letter-spacing: -0.5px !important;
        text-shadow: 0 0 10px rgba(88, 166, 255, 0.3) !important;
    }
    
    /* Button styling */
    button {
        background: linear-gradient(135deg, #0a7490 0%, #0d99ad 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(10, 116, 144, 0.4) !important;
    }
    
    button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(10, 116, 144, 0.6) !important;
    }
    
    /* Input fields */
    input, select {
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        background-color: #161b22 !important;
        color: #e0e0e0 !important;
        transition: all 0.3s ease !important;
    }
    
    input:focus, select:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 10px rgba(88, 166, 255, 0.3) !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-weight: 700 !important;
    }
    
    /* Markdown text - HIGH CONTRAST for dark mode */
    p, li, div {
        color: #e0e0e0 !important;
        line-height: 1.6 !important;
    }
    
    /* Info/warning boxes */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
        background-color: #161b22 !important;
    }
    
    /* Subheader text */
    [data-testid="stMarkdownContainer"] {
        color: #e0e0e0 !important;
    }
</style>
"""

# Apply night mode CSS
st.markdown(night_mode_css, unsafe_allow_html=True)

# --- Logo Creation Function ---
def create_logo():
    """Create a modern Surviva logo with survival curve and night mode styling"""
    fig, ax = plt.subplots(figsize=(2.2, 1), dpi=100)
    
    # Set dark background to match night mode
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    # Set axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.3, color='#30363d')
    
    # Style axes with blue color
    for spine in ax.spines.values():
        spine.set_color('#58a6ff')
        spine.set_linewidth(1.5)
    
    # Remove tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create smooth survival curve with gradient effect
    x = np.linspace(0, 10, 100)
    y = 10 * np.exp(-0.3 * x)  # Exponential decay
    
    # Plot survival curve with gradient effect
    for i in range(len(x)-1):
        alpha = 0.3 + (0.4 * (1 - i/len(x)))  # Gradient alpha
        ax.plot(x[i:i+2], y[i:i+2], color='#58a6ff', linewidth=2.5, alpha=alpha, zorder=1)
    
    # Add "Surviva" text with modern glow effect
    text = ax.text(5, 5, 'Surviva', 
                   fontsize=20, 
                   fontweight=900,
                   ha='center', 
                   va='center',
                   color='#ff1744',
                   style='italic',
                   zorder=2)
    
    # Add modern glow effect with multiple strokes
    text.set_path_effects([
        withStroke(linewidth=3, foreground='#ffffff', alpha=0.8),
        withStroke(linewidth=5, foreground='#ff6b9d', alpha=0.3),
        withStroke(linewidth=7, foreground='#ff1744', alpha=0.1)
    ])
    
    plt.tight_layout(pad=0)
    return fig

# --- Display logo at top center ---
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
logo_fig = create_logo()
st.pyplot(logo_fig, use_container_width=False)
plt.close()
st.markdown("</div>", unsafe_allow_html=True)

# --- Title and Description ---
st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🔬 Survival Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #e0e0e0;'>Advanced Cox Proportional Hazards Model for Patient Survival Assessment</p>", unsafe_allow_html=True)

cox = get_cox_model()
if cox is None:
    st.error("Cox model not found (cox_model.pkl). Run train_survival_models.py first.")
    st.stop()

st.markdown(
    "<p style='color: #e0e0e0; text-align: center;'>Enter patient characteristics to see predicted survival, risk level, and precision medicine considerations. "
    "Not a substitute for clinical judgment.</p>",
    unsafe_allow_html=True
)

# --- Inputs (sidebar or main) ---
with st.sidebar:
    st.markdown("### Surviva")
    st.caption("Survival Risk & Prognosis")
    st.header("Patient profile")
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    tnm_stage = st.selectbox("TNM stage", options=[1, 2, 3, 4], format_func=lambda x: f"Stage {x}")
    age_years = st.number_input("Age (years)", min_value=18, max_value=120, value=65, step=1)
    kras_mut = st.selectbox(
        "KRAS mutation",
        options=[0, 1],
        format_func=lambda x: "Wild-type" if x == 0 else "Mutated",
    )
    # MMR/MSI if in features (msi.h)
    msi_val = MSI_DEFAULT
    if "msi.h" in FEATURE_COLS:
        msi_val = st.selectbox(
            "MSI / MMR status",
            options=[-1, 0, 1],
            format_func=lambda x: "Unknown" if x == -1 else ("MSS / proficient" if x == 0 else "MSI-H / deficient"),
        )
        if msi_val == -1:
            msi_val = MSI_DEFAULT
    submit = st.button("Get prediction")

# --- Build feature row (same column names as training) ---
age_norm = (float(age_years) - AGE_MEAN) / AGE_STD if AGE_STD != 0 else 0.0
row = {}
for c in FEATURE_COLS:
    if c == "Sex":
        row[c] = sex
    elif c == "tnm.stage":
        row[c] = tnm_stage
    elif c == "age.norm":
        row[c] = age_norm
    elif c == "kras.mut":
        row[c] = kras_mut
    elif c == "msi.h":
        row[c] = msi_val
    else:
        row[c] = 0
patient_df = pd.DataFrame([row])

# --- On submit: predict and show results ---
if submit:
    # Survival curve
    surv_fn = cox.predict_survival_function(patient_df)
    risk = float(cox.predict_partial_hazard(patient_df).iloc[0])

    # Get times and survival values; prepend (0, 1) so curve starts at S(0)=1
    times = np.asarray(surv_fn.index, dtype=float)
    vals = np.asarray(surv_fn.iloc[:, 0], dtype=float)
    if len(times) == 0:
        times, vals = np.array([0.0, 12.0, 36.0]), np.array([1.0, 1.0, 1.0])
    elif times[0] > 0:
        times = np.concatenate([[0.0], times])
        vals = np.concatenate([[1.0], vals])
    # 1-year and 3-year survival (interpolate if needed)
    try:
        i_12 = np.where(times <= 12)[0]
        i_36 = np.where(times <= 36)[0]
        if len(i_12):
            s_1y = float(vals[i_12[-1]])
        else:
            s_1y = float(np.interp(12, times, vals)) if len(times) > 1 else None
        if len(i_36):
            s_3y = float(vals[i_36[-1]])
        else:
            s_3y = float(np.interp(36, times, vals)) if len(times) > 1 else None
    except Exception:
        s_1y = s_3y = None
    # Plot: explicit line so the curve always shows
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, vals, "b-", linewidth=2, label="This patient")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability S(t)")
    ax.set_title("Predicted survival for this patient profile")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    st.markdown("**What this graph means:** The curve shows the model’s estimated probability that a patient with this profile is still alive at each time (in months). It is based on patterns in the training cohort, not a guarantee for this individual.")

    if s_1y is not None:
        st.subheader("Survival estimates")
        st.markdown(f"- **1-year survival:** approximately **{s_1y:.0%}**")
        if s_3y is not None:
            st.markdown(f"- **3-year survival:** approximately **{s_3y:.0%}**")

    # Risk level
    st.subheader("Risk level")
    if RISK_P25 is not None and RISK_P75 is not None:
        if risk <= RISK_P25:
            risk_label = "Low"
            risk_display = "**Low** (relative to training cohort)"
        elif risk <= RISK_P75:
            risk_label = "Medium"
            risk_display = "**Medium**"
        else:
            risk_label = "High"
            risk_display = "**High** (relative to training cohort)"
        st.markdown(f"Estimated risk for this profile: {risk_display}")
    else:
        risk_label = "High" if (RISK_P50 is not None and risk > RISK_P50) else "Medium"
        st.markdown(f"Relative risk score: {risk:.3f} (higher = higher predicted risk).")

    # Precision medicine: risk-, age-, and gender-stratified recommendations
    st.subheader("Precision medicine considerations")
    sex_str = "Male" if sex == 1 else "Female"
    stage_str = f"Stage {tnm_stage}"
    age_cat = "older" if age_years >= 70 else "younger" if age_years < 50 else "middle-aged"

    parts = []
    parts.append(f"This **{sex_str}** patient, age **{age_years}** ({age_cat}), has **{stage_str}** disease with **{risk_label}** estimated risk.")

    # Risk-stratified therapy framing
    if risk_label == "High":
        parts.append("**Given high risk:** Prioritize clinical trials, combination or escalation strategies, and early discussion of second-line and supportive/palliative options where appropriate.")
    elif risk_label == "Medium":
        parts.append("**Given medium risk:** Individualize treatment intensity and sequencing; consider biomarker-directed therapy and trials when indicated.")
    else:
        parts.append("**Given low risk:** Standard and adjuvant strategies per guidelines are often appropriate; escalation or trials as per local practice.")

    # Survival-based (tied to risk)
    if s_1y is not None:
        if s_1y < 0.3:
            parts.append(f"Low estimated 1-year survival ({s_1y:.0%}); consider intensive or trial options and early palliative involvement.")
        elif s_1y < 0.6:
            parts.append(f"Moderate 1-year survival ({s_1y:.0%}); tailor intensity and sequencing to goals and tolerance.")
        else:
            parts.append(f"Higher estimated 1-year survival ({s_1y:.0%}); standard and adjuvant strategies per guidelines may apply.")

    # Age-stratified therapy guidance
    if age_years >= 75:
        parts.append("**Older adult (≥75):** Assess fitness and goals; consider geriatric assessment, dose modifications, and reduced-intensity regimens where appropriate. Discuss tolerance to combination therapy.")
    elif age_years >= 70:
        parts.append("**Older adult (70–74):** Fitness and comorbidity may guide intensity; consider dose/schedule per geriatric guidelines where relevant.")
    elif age_years < 50:
        parts.append("**Younger patient:** May tolerate intensive regimens; consider curative-intent and trial options where indicated.")

    # Biomarker: MMR/MSI (unchanged logic, add gender note where relevant)
    if msi_val == 1:
        parts.append("**MSI-H / MMR-deficient:** Immunotherapy (e.g. pembrolizumab) where approved and appropriate.")
        if sex == 1:
            parts.append("In **males**, efficacy/safety data for immunotherapy in this setting may inform dosing; discuss with oncology team.")
    elif msi_val == 0:
        parts.append("**MSS / MMR-proficient:** Immunotherapy benefit is limited; follow standard and biomarker-directed therapy.")

    # Biomarker: KRAS (qualify by risk and age)
    if kras_mut == 1:
        if risk_label == "High":
            parts.append("**KRAS mutated:** If **G12C**, consider **sotorasib** or **adagrasib** where approved; given high risk, also consider trials or combination strategies. Other variants: clinical trials or standard therapy.")
        else:
            parts.append("**KRAS mutated:** If **G12C**, consider **sotorasib** or **adagrasib** where approved. Other variants: clinical trials or standard therapy.")
        if age_years >= 70:
            parts.append("In **older adults**, assess tolerance to KRAS-targeted therapy; dose modifications per prescribing information where applicable.")
    else:
        parts.append("**KRAS wild-type:** In eligible indications (e.g. colorectal), **anti-EGFR** therapy may be considered per guidelines.")
        if sex == 0:
            parts.append("In **females**, anti-EGFR efficacy and toxicity may vary; refer to sex-specific data where available.")

    # Stage
    if tnm_stage >= 3:
        parts.append(f"**{stage_str}:** Multidisciplinary discussion and, where appropriate, neoadjuvant/adjuvant or metastatic protocols per guidelines.")

    for p in parts:
        st.markdown(f"- {p}")

    # Dosing considerations (qualitative; no specific doses—refer to PI and guidelines)
    st.subheader("Dosing considerations")
    dosing_parts = []
    dosing_parts.append("**General:** Use current **prescribing information (PI)** and **institutional protocols** for exact doses. Consider **age, sex, body size, renal and hepatic function**, and **stage** when choosing or adjusting dose.")
    if age_years >= 70:
        dosing_parts.append("**Age (older adult):** Consider dose modifications and reduced intensity per PI and geriatric guidelines; assess renal/hepatic function before and during therapy.")
    if sex == 1:
        dosing_parts.append("**Sex:** Where PI or guidelines specify sex-specific or weight-based dosing, follow institutional protocols; discuss with pharmacy/oncology as needed.")
    if tnm_stage >= 3:
        dosing_parts.append("**Stage:** Regimen intensity and duration may differ by (neo)adjuvant vs metastatic setting; refer to guideline-based dosing for the relevant setting.")
    if risk_label == "High" or (s_1y is not None and s_1y < 0.4):
        dosing_parts.append("**Risk/fitness:** In high-risk or frail patients, consider starting at lower intensity and escalating per tolerance; refer to PI for dose modifications.")
    if msi_val == 1:
        dosing_parts.append("**Immunotherapy (e.g. pembrolizumab):** Dosing per PI and guidelines; consider dose reduction in older adults and in impaired organ function.")
    if kras_mut == 1:
        dosing_parts.append("**KRAS inhibitors (sotorasib/adagrasib):** Dosing per PI and institutional protocols; consider age, renal/hepatic function, and body size when choosing or adjusting dose.")
    else:
        dosing_parts.append("**Anti-EGFR therapy:** Dosing per PI and guidelines; consider age, sex, and organ function when choosing or adjusting dose.")
    for d in dosing_parts:
        st.markdown(f"- {d}")

    st.caption("This is for educational support only. Not a substitute for clinical judgment or prescribing.")

else:
    st.info("Use the sidebar to enter patient profile and click **Get prediction**.")
