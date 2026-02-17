"""
Train survival models on survival_df.csv.
Performs 70/30 train-test split and fits Cox PH and Random Survival Forest.
Evaluates both models with Concordance Index (C-index).
Saves models to disk; use load_cox_model() and load_rsf_model() to load them later.

Note: scikit-survival is optional and only needed to train RSF. The app only uses Cox model.
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


def load_cox_model(path: str = "cox_model.pkl"):
    """Load a fitted Cox PH model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_rsf_model(path: str = "rsf_model.pkl"):
    """Load a fitted Random Survival Forest model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# Load data
df = pd.read_csv("survival_df.csv", index_col=0)

# Use standard column names: os_delay_months (duration), os_event (event)
df = df.rename(columns={"os.delay (months)": "os_delay_months", "os.event": "os_event"})
duration_col = "os_delay_months"
event_col = "os_event"

# Drop only identifier columns (if any); keep all other columns as features
id_like = [c for c in df.columns if c.lower() in ("id", "subject_id", "patient_id")]
df = df.drop(columns=id_like, errors="ignore")
feature_cols = [c for c in df.columns if c not in (duration_col, event_col)]
df = df.dropna(subset=[duration_col, event_col] + feature_cols)

# 70/30 train-test split (no leakage: split before any model sees the data)
train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)

# --- Cox Proportional Hazards model ---
# Fit only on training data (no test data used for fitting → no data leakage)
cph = CoxPHFitter()
cph.fit(train_df, duration_col=duration_col, event_col=event_col)

print("Cox Proportional Hazards model")
print("=" * 50)
print(cph.summary)

# Hazard ratios (exp(coef)), sorted by magnitude of effect
hr_df = pd.DataFrame({
    "variable": cph.params_.index,
    "coef": cph.params_.values,
    "HR": np.exp(cph.params_.values),
}).sort_values("coef", key=lambda s: np.abs(s), ascending=False)
hr_df = hr_df.reset_index(drop=True)
print("\nHazard ratios (HR = exp(coef)), sorted by |coef|")
print("-" * 50)
print(hr_df.to_string(index=False))
print("\nInterpretation: HR > 1 = increased risk (worse prognosis); HR < 1 = protective (better prognosis).")

# C-index: predictions use only the fitted model (test risk from coefficients + test features only)
cph_train_risk = cph.predict_partial_hazard(train_df).values
cph_test_risk = cph.predict_partial_hazard(test_df).values
cph_train_cindex = concordance_index(
    train_df[duration_col].values,
    cph_train_risk,
    train_df[event_col].values,
)
cph_test_cindex = concordance_index(
    test_df[duration_col].values,
    cph_test_risk,
    test_df[event_col].values,
)
print("\nConcordance Index (C-index)")
print("-" * 30)
print("Training C-index:", cph_train_cindex)
print("Test C-index:    ", cph_test_cindex)

with open("cox_model.pkl", "wb") as f:
    pickle.dump(cph, f)
print("Cox model saved as cox_model.pkl")

# --- App config (for Streamlit app: feature names, age norm, risk percentiles) ---
app_config = {
    "feature_cols": feature_cols,
    "age_mean": 65.0,
    "age_std": 15.0,
    "msi_default": 0,
    "cox_risk_p25": float(np.percentile(cph_train_risk, 25)),
    "cox_risk_p50": float(np.percentile(cph_train_risk, 50)),
    "cox_risk_p75": float(np.percentile(cph_train_risk, 75)),
}
with open("app_config.json", "w") as f:
    json.dump(app_config, f, indent=2)
print("App config saved as app_config.json")
print("\nDone! The app is ready to use with the Cox model.")
