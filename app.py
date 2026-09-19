import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Dashboard Configuration ---
st.set_page_config(
    page_title="Medical Claim Intelligence Hub",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Executive Aesthetic Stylesheet ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top right, #0d1f2d, #050b14 80%);
        color: #e2e8f0;
    }

    /* Top Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(14, 116, 144, 0.15) 100%);
        border: 1px solid rgba(45, 212, 191, 0.25);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #34d399, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-desc {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Section Subheadings */
    .group-label {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #38bdf8;
        border-bottom: 1px solid rgba(56, 189, 248, 0.2);
        padding-bottom: 6px;
        margin-bottom: 16px;
    }

    /* Result Metric Display */
    .metric-panel {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(8, 14, 26, 0.95));
        border: 2px solid #10b981;
        border-radius: 18px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 12px 36px rgba(16, 185, 129, 0.18);
    }
    .metric-header {
        font-size: 13px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
    }
    .metric-number {
        font-size: 42px;
        font-weight: 800;
        color: #34d399;
        margin: 10px 0;
        letter-spacing: -0.03em;
    }

    /* Action Button */
    div.stButton > button {
        background: linear-gradient(90deg, #059669 0%, #0284c7 100%);
        color: #ffffff !important;
        font-weight: 700;
        font-size: 16px;
        padding: 12px 28px;
        border: none;
        border-radius: 10px;
        width: 100%;
        transition: all 0.25s ease;
        box-shadow: 0 4px 15px rgba(5, 150, 105, 0.3);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(2, 132, 199, 0.45);
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loader ---
@st.cache_resource
def get_model():
    model_path = "RandomForest_model.pkl"
    if not os.path.exists(model_path):
        return None, f"File `{model_path}` was not found in the root repository."
    try:
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        return loaded_model, None
    except Exception as err:
        return None, f"Deserialization Error: {str(err)}"

model, load_err = get_model()

# --- Header ---
st.markdown("""
<div class="hero-card">
    <h1 class="hero-title">Medical Risk & Insurance Cost Estimator</h1>
    <p class="hero-desc">Actuarial Risk Prediction Dashboard powered by Ensembled Decision Trees</p>
</div>
""", unsafe_allow_html=True)

if load_err:
    st.error(load_err)
    st.info("Ensure `RandomForest_model.pkl` is committed directly to your repository root.")

# --- Tab Layout ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Demographics & Socioeconomic",
    "Clinical & Vital Statistics",
    "Insurance Policy & History",
    "Conditions & Utilization"
])

inputs = {}

with tab1:
    st.markdown('<div class="group-label">Demographic Markers</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        inputs["age"] = st.number_input("Age", 0, 110, 35)
        inputs["sex"] = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        inputs["region"] = st.selectbox("Geographic Region", [0, 1, 2, 3], format_func=lambda x: f"Region {x+1}")
    with c2:
        inputs["urban_rural"] = st.selectbox("Residence Environment", [0, 1], format_func=lambda x: "Rural" if x == 0 else "Urban")
        inputs["income"] = st.number_input("Annual Income ($)", 0.0, 1000000.0, 52000.0, 2500.0)
        inputs["education"] = st.selectbox("Education Level", [0, 1, 2, 3], format_func=lambda x: ["High School", "Bachelor", "Master", "PhD"][x])
    with c3:
        inputs["marital_status"] = st.selectbox("Marital Status", [0, 1], format_func=lambda x: "Single" if x == 0 else "Married")
        inputs["employment_status"] = st.selectbox("Employment", [0, 1, 2], format_func=lambda x: ["Unemployed", "Employed", "Self-Employed"][x])
        inputs["household_size"] = st.number_input("Household Size", 1, 15, 3)
        inputs["dependents"] = st.number_input("Number of Dependents", 0, 10, 1)

with tab2:
    st.markdown('<div class="group-label">Vitals & Health Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        inputs["bmi"] = st.number_input("BMI (kg/m²)", 10.0, 65.0, 24.8, 0.1)
        inputs["smoker"] = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker")
        inputs["alcohol_freq"] = st.selectbox("Alcohol Frequency", [0, 1, 2, 3], format_func=lambda x: ["None", "Occasional", "Moderate", "Frequent"][x])
        inputs["risk_score"] = st.slider("Clinical Risk Score", 0.0, 100.0, 22.0, 0.5)
    with c2:
        inputs["systolic_bp"] = st.number_input("Systolic BP (mmHg)", 70, 240, 120)
        inputs["diastolic_bp"] = st.number_input("Diastolic BP (mmHg)", 40, 150, 80)
        inputs["ldl"] = st.number_input("LDL (mg/dL)", 30.0, 350.0, 105.0, 1.0)
        inputs["hba1c"] = st.number_input("HbA1c (%)", 3.0, 18.0, 5.4, 0.1)
    with c3:
        inputs["visits_last_year"] = st.number_input("Visits (Past 12 Mo)", 0, 60, 2)
        inputs["hospitalizations_last_3yrs"] = st.number_input("Hospitalizations (Past 3 Yrs)", 0, 25, 0)
        inputs["days_hospitalized_last_3yrs"] = st.number_input("Hospital Days (Past 3 Yrs)", 0, 150, 0)
        inputs["medication_count"] = st.number_input("Active Prescriptions", 0, 30, 1)

with tab3:
    st.markdown('<div class="group-label">Coverage & Claims</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        inputs["plan_type"] = st.selectbox("Policy Tier", [0, 1, 2], format_func=lambda x: ["Bronze", "Silver", "Gold"][x])
        inputs["network_tier"] = st.selectbox("Network Coverage", [0, 1, 2], format_func=lambda x: ["Tier 1 (In-Network)", "Tier 2", "Tier 3"][x])
        inputs["provider_quality"] = st.slider("Provider Rating", 1.0, 5.0, 4.2, 0.1)
    with c2:
        inputs["deductible"] = st.number_input("Deductible ($)", 0.0, 20000.0, 1500.0, 250.0)
        inputs["copay"] = st.number_input("Copay ($)", 0.0, 500.0, 35.0, 5.0)
        inputs["policy_term_years"] = st.number_input("Tenure (Years)", 1, 35, 3)
        inputs["policy_changes_last_2yrs"] = st.number_input("Plan Alterations (2 Yrs)", 0, 10, 0)
    with c3:
        inputs["annual_premium"] = st.number_input("Annual Premium ($)", 0.0, 50000.0, 4600.0, 200.0)
        inputs["monthly_premium"] = st.number_input("Monthly Premium ($)", 0.0, 5000.0, 385.0, 20.0)
        inputs["claims_count"] = st.number_input("Prior Claims Count", 0, 50, 1)
        inputs["avg_claim_amount"] = st.number_input("Average Claim Value ($)", 0.0, 100000.0, 1100.0, 100.0)
        inputs["total_claims_paid"] = st.number_input("Total Historical Claims ($)", 0.0, 500000.0, 1100.0, 250.0)

with tab4:
    st.markdown('<div class="group-label">Diagnoses & Clinical Procedures</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Active Diagnoses")
        inputs["hypertension"] = int(st.checkbox("Hypertension", False))
        inputs["diabetes"] = int(st.checkbox("Diabetes", False))
        inputs["asthma"] = int(st.checkbox("Asthma", False))
        inputs["copd"] = int(st.checkbox("COPD", False))
        inputs["cardiovascular_disease"] = int(st.checkbox("Cardiovascular Condition", False))
        inputs["cancer_history"] = int(st.checkbox("History of Cancer", False))
        inputs["kidney_disease"] = int(st.checkbox("Kidney Disease", False))
        inputs["liver_disease"] = int(st.checkbox("Liver Disease", False))
        inputs["arthritis"] = int(st.checkbox("Arthritis", False))
        inputs["mental_health"] = int(st.checkbox("Mental Health Condition", False))
        
        chronic_keys = ["hypertension", "diabetes", "asthma", "copd", "cardiovascular_disease",
                        "cancer_history", "kidney_disease", "liver_disease", "arthritis", "mental_health"]
        inputs["chronic_count"] = sum([inputs[k] for k in chronic_keys])
    with c2:
        st.caption("Medical Services & Utilization")
        inputs["proc_imaging_count"] = st.number_input("Imaging Tests", 0, 25, 0)
        inputs["proc_surgery_count"] = st.number_input("Surgical Procedures", 0, 15, 0)
        inputs["proc_physio_count"] = st.number_input("Physiotherapy Sessions", 0, 50, 0)
        inputs["proc_consult_count"] = st.number_input("Specialist Consultations", 0, 50, 1)
        inputs["proc_lab_count"] = st.number_input("Laboratory Panels", 0, 50, 2)
        inputs["is_high_risk"] = st.selectbox("Underwriting High Risk Flag", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        inputs["had_major_procedure"] = st.selectbox("Recent Major Procedure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

st.markdown("<br>", unsafe_allow_html=True)

# --- Predict Section ---
col_act, col_info = st.columns([1, 2])
with col_act:
    calculate = st.button("⚡ Generate Cost Prediction")

if calculate:
    if model is None:
        st.error("Model unavailable. Please verify model file status.")
    else:
        try:
            # Reorder strictly based on training signatures
            ordered_cols = list(model.feature_names_in_)
            row = [inputs[feat] for feat in ordered_cols]
            df_input = pd.DataFrame([row], columns=ordered_cols)
            
            output = model.predict(df_input)[0]
            
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-header">Estimated Actuarial Cost</div>
                <div class="metric-number">${output:,.2f}</div>
                <div style="color: #94a3b8; font-size: 13px;">Inference generated across 50 Decision Trees</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Inference Failure: {str(e)}")
