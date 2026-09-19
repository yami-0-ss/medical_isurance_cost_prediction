import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Health Risk & Claim Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Custom CSS Theme ---
st.markdown("""
<style>
    :root {
        --primary: #4F46E5;
        --primary-light: #818CF8;
        --bg-color: #0B0F19;
        --card-bg: #111827;
        --border-color: #1F2937;
        --text-color: #F3F4F6;
        --accent: #06B6D4;
    }
    
    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #0F172A 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        border: 1px solid #3730A3;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #FFFFFF;
        margin: 0;
    }
    
    .hero-sub {
        color: #94A3B8;
        font-size: 0.98rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Section Cards */
    .form-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
    }
    
    .section-header {
        color: #38BDF8;
        font-size: 1.05rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 0.5rem;
    }

    /* Metrics & Results */
    .metric-container {
        background: radial-gradient(circle at top, #1E1B4B 0%, #0F172A 100%);
        border: 1px solid #6366F1;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.2);
    }

    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #38BDF8;
        letter-spacing: -0.02em;
    }

    .metric-caption {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Streamlit Button Customization */
    div.stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        color: #FFFFFF;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.05rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
        transition: all 0.2s ease-in-out;
        width: 100%;
    }
    
    div.stButton > button:hover {
        opacity: 0.95;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- Load Pretrained Model ---
@st.cache_resource
def load_rf_model():
    model_filename = "RandomForest_model.pkl"
    if not os.path.exists(model_filename):
        st.error(f"Error: `{model_filename}` was not found in the root directory.")
        return None
    with open(model_filename, "rb") as f:
        return pickle.load(f)

model = load_rf_model()

# --- Application Header ---
st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">🛡️ Health Claim & Cost Intelligence</h1>
    <p class="hero-sub">Random Forest Predictive Engine • Multi-Variable Clinical & Actuarial Assessment</p>
</div>
""", unsafe_allow_html=True)

# --- Feature Input Tabs ---
tab_profile, tab_clinical, tab_policy, tab_proc = st.tabs([
    "👤 Profile & Demographics", 
    "🩺 Clinical & Vitals", 
    "📜 Policy & Financials", 
    "🔬 Chronic & Procedures"
])

# Input state collection
user_inputs = {}

with tab_profile:
    st.markdown('<div class="section-header">Demographic & Socioeconomic Markers</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        user_inputs["age"] = st.number_input("Age", min_value=0, max_value=110, value=38)
        user_inputs["sex"] = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        user_inputs["region"] = st.selectbox("Region Code", options=[0, 1, 2, 3], format_func=lambda x: f"Region {x+1}")
    with c2:
        user_inputs["urban_rural"] = st.selectbox("Living Area", options=[0, 1], format_func=lambda x: "Rural" if x == 0 else "Urban")
        user_inputs["income"] = st.number_input("Annual Income ($)", min_value=0.0, max_value=500000.0, value=55000.0, step=1000.0)
        user_inputs["education"] = st.selectbox("Education Level", options=[0, 1, 2, 3], format_func=lambda x: ["High School", "Bachelor", "Master", "PhD"][x])
    with c3:
        user_inputs["marital_status"] = st.selectbox("Marital Status", options=[0, 1], format_func=lambda x: "Single" if x == 0 else "Married")
        user_inputs["employment_status"] = st.selectbox("Employment Status", options=[0, 1, 2], format_func=lambda x: ["Unemployed", "Employed", "Self-Employed"][x])
        user_inputs["household_size"] = st.number_input("Household Size", min_value=1, max_value=15, value=3)
        user_inputs["dependents"] = st.number_input("Dependents", min_value=0, max_value=10, value=1)

with tab_clinical:
    st.markdown('<div class="section-header">Biometrics & Clinical History</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        user_inputs["bmi"] = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=25.4, step=0.1)
        user_inputs["smoker"] = st.selectbox("Smoker", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        user_inputs["alcohol_freq"] = st.selectbox("Alcohol Consumption Frequency", options=[0, 1, 2, 3], format_func=lambda x: ["None", "Occasional", "Moderate", "Frequent"][x])
        user_inputs["risk_score"] = st.slider("Clinical Risk Score", min_value=0.0, max_value=100.0, value=24.5, step=0.1)
    with c2:
        user_inputs["systolic_bp"] = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=220, value=120)
        user_inputs["diastolic_bp"] = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=140, value=80)
        user_inputs["ldl"] = st.number_input("LDL Cholesterol (mg/dL)", min_value=40.0, max_value=300.0, value=100.0, step=1.0)
        user_inputs["hba1c"] = st.number_input("HbA1c (%)", min_value=3.0, max_value=16.0, value=5.5, step=0.1)
    with c3:
        user_inputs["visits_last_year"] = st.number_input("Clinic Visits (Last Year)", min_value=0, max_value=50, value=2)
        user_inputs["hospitalizations_last_3yrs"] = st.number_input("Hospitalizations (Last 3 Yrs)", min_value=0, max_value=20, value=0)
        user_inputs["days_hospitalized_last_3yrs"] = st.number_input("Days Hospitalized (Last 3 Yrs)", min_value=0, max_value=120, value=0)
        user_inputs["medication_count"] = st.number_input("Active Prescription Medications", min_value=0, max_value=30, value=1)

with tab_policy:
    st.markdown('<div class="section-header">Insurance Policy & Claims History</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        user_inputs["plan_type"] = st.selectbox("Plan Type", options=[0, 1, 2], format_func=lambda x: ["Bronze", "Silver", "Gold"][x])
        user_inputs["network_tier"] = st.selectbox("Network Tier", options=[0, 1, 2], format_func=lambda x: ["Tier 1", "Tier 2", "Tier 3"][x])
        user_inputs["provider_quality"] = st.slider("Provider Quality Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    with c2:
        user_inputs["deductible"] = st.number_input("Deductible ($)", min_value=0.0, max_value=10000.0, value=1500.0, step=100.0)
        user_inputs["copay"] = st.number_input("Copay ($)", min_value=0.0, max_value=500.0, value=30.0, step=5.0)
        user_inputs["policy_term_years"] = st.number_input("Policy Term (Years)", min_value=1, max_value=30, value=3)
        user_inputs["policy_changes_last_2yrs"] = st.number_input("Policy Changes (Last 2 Yrs)", min_value=0, max_value=10, value=0)
    with c3:
        user_inputs["annual_premium"] = st.number_input("Annual Premium ($)", min_value=0.0, max_value=50000.0, value=4800.0, step=100.0)
        user_inputs["monthly_premium"] = st.number_input("Monthly Premium ($)", min_value=0.0, max_value=5000.0, value=400.0, step=10.0)
        user_inputs["claims_count"] = st.number_input("Historical Claims Count", min_value=0, max_value=50, value=1)
        user_inputs["avg_claim_amount"] = st.number_input("Avg Historical Claim ($)", min_value=0.0, max_value=100000.0, value=1200.0, step=100.0)
        user_inputs["total_claims_paid"] = st.number_input("Total Claims Paid ($)", min_value=0.0, max_value=500000.0, value=1200.0, step=250.0)

with tab_proc:
    st.markdown('<div class="section-header">Chronic Illnesses & Medical Procedures</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Chronic Diagnoses**")
        user_inputs["hypertension"] = st.checkbox("Hypertension", value=False)
        user_inputs["diabetes"] = st.checkbox("Diabetes", value=False)
        user_inputs["asthma"] = st.checkbox("Asthma", value=False)
        user_inputs["copd"] = st.checkbox("COPD", value=False)
        user_inputs["cardiovascular_disease"] = st.checkbox("Cardiovascular Disease", value=False)
        user_inputs["cancer_history"] = st.checkbox("Cancer History", value=False)
        user_inputs["kidney_disease"] = st.checkbox("Kidney Disease", value=False)
        user_inputs["liver_disease"] = st.checkbox("Liver Disease", value=False)
        user_inputs["arthritis"] = st.checkbox("Arthritis", value=False)
        user_inputs["mental_health"] = st.checkbox("Mental Health Condition", value=False)
        
        # Calculate chronic count automatically
        chronic_keys = ["hypertension", "diabetes", "asthma", "copd", "cardiovascular_disease", 
                        "cancer_history", "kidney_disease", "liver_disease", "arthritis", "mental_health"]
        user_inputs["chronic_count"] = sum([int(user_inputs[k]) for k in chronic_keys])
        
    with c2:
        st.write("**Procedure Utilization & Flags**")
        user_inputs["proc_imaging_count"] = st.number_input("Imaging Procedures Count", min_value=0, max_value=20, value=0)
        user_inputs["proc_surgery_count"] = st.number_input("Surgery Procedures Count", min_value=0, max_value=10, value=0)
        user_inputs["proc_physio_count"] = st.number_input("Physiotherapy Sessions", min_value=0, max_value=50, value=0)
        user_inputs["proc_consult_count"] = st.number_input("Consultations Count", min_value=0, max_value=50, value=1)
        user_inputs["proc_lab_count"] = st.number_input("Lab Test Count", min_value=0, max_value=50, value=2)
        user_inputs["is_high_risk"] = st.selectbox("High Risk Categorization", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        user_inputs["had_major_procedure"] = st.selectbox("Had Major Procedure", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Convert booleans to binary integers
for k, v in user_inputs.items():
    if isinstance(v, bool):
        user_inputs[k] = int(v)

st.markdown("<br>", unsafe_allow_html=True)

# --- Predict Execution ---
c_btn, _ = st.columns([1, 2])
with c_btn:
    run_prediction = st.button("🚀 Calculate Model Prediction")

if run_prediction:
    if model is not None:
        try:
            # Enforce exact feature order expected by the model
            feature_order = list(model.feature_names_in_)
            input_df = pd.DataFrame([[user_inputs[col] for col in feature_order]], columns=feature_order)
            
            prediction = model.predict(input_df)[0]
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-caption">Predicted Target Value / Estimated Cost</div>
                <div class="metric-value">${prediction:,.2f}</div>
                <p style="color: #94A3B8; margin-top: 0.5rem; font-size: 0.88rem;">Evaluated across 50 Decision Trees • Scikit-Learn Random Forest Engine</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🔍 View Processed Feature Array"):
                st.dataframe(input_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Inference Error: {str(e)}")
