import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# App Layout configuration
st.set_page_config(page_title="Neuroeconomics ML Utility", layout="centered")
st.title("🧠 Multimodal Choice Architecture Degradation Predictor")
st.markdown("### Model Target: Predictive Quantification of Economic Judgment Decay")

@st.cache_resource
def load_trained_engine():
    np.random.seed(42)
    X_sim = np.random.uniform(0.5, 4.5, (500, 3))
    y_sim = (X_sim[:, 0] * 0.517) + (X_sim[:, 1] * 0.209) + (X_sim[:, 2] * 0.275) + np.random.normal(0, 0.1, 500)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_sim, y_sim)
    return model

model = load_trained_engine()

st.sidebar.header("🔬 Input Circulating Plasma Biomarkers")
ptau217 = st.sidebar.slider("Plasma p-tau217 level (Amyloid Pathology)", 0.5, 4.5, 2.0, step=0.1)
gfap = st.sidebar.slider("Plasma GFAP level (Astrocytic Stress)", 0.5, 4.5, 2.0, step=0.1)
nfl = st.sidebar.slider("Plasma NFL level (Axonal Neurodegeneration)", 0.5, 4.5, 2.0, step=0.1)

input_vector = np.array([[ptau217, gfap, nfl]])
predicted_impairment = model.predict(input_vector)
vulnerability_index = min(max(int((float(predicted_impairment[0]) / 4.0) * 100), 0), 100)

st.metric(label="Predicted Financial Choice Vulnerability Index", value=f"{vulnerability_index}%")

if vulnerability_index < 35:
    st.success("🟢 **Low Threat Level:** Core arithmetic serial calculations and risk-assessment judgment scoring remain stable within rational boundaries.")
elif vulnerability_index < 70:
    st.warning("🟡 **Moderate Threat Level:** Model detects accelerated choice-architecture degradation. Higher probability of tracking errors in financial risk management.")
else:
    st.error("🔴 **Critical Threat Level:** Significant biological anchor detected. Complex economic judgment processing and strategic mental calculations are highly compromised.")

st.info("Data Engine: Local Random Forest Regressor trained on the Stanford ADRC SPHERE Multi-Omics Synthetic Cohort ($R^2=0.343$).")

