import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Health Risk & Claims Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0A0E17;
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Card */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F766E 100%);
        border: 1px solid rgba(6, 182, 212, 0.25);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        color: #F8FAFC;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1rem;
    }
    
    /* Metric Result Box */
    .prediction-box {
        background: radial-gradient(circle at top right, rgba(6, 182, 212, 0.15), rgba(15, 23, 42, 0.95));
        border: 1px solid #06B6D4;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-bottom: 20px;
    }
    .prediction-val {
        font-size: 2.75rem;
        font-weight: 800;
        color: #38BDF8;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    }
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        margin-top: 8px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B1120;
        border-right: 1px solid #1E293B;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("RandomForest_model.pkl")
    except Exception as e:
        st.error(f"Error loading RandomForest_model.pkl: {e}")
        return None

model = load_model()

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🛡️ Health & Insurance Risk Analytics</div>
    <div class="hero-subtitle">Random Forest Regressor inference pipeline with real-time risk simulation and radar analysis.</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: Feature Inputs
# ---------------------------------------------------------
st.sidebar.header("⚙️ Patient & Policy Profile")

with st.sidebar.expander("👤 Demographics & Lifestyle", expanded=True):
    age = st.slider("Age", 18, 100, 42)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    region = st.selectbox("Region", options=[0, 1, 2, 3], format_func=lambda x: ["North", "South", "East", "West"][x])
    urban_rural = st.selectbox("Area Type", options=[0, 1], format_func=lambda x: "Rural" if x == 0 else "Urban")
    income = st.number_input("Annual Income ($)", min_value=5000, max_value=500000, value=55000, step=2500)
    education = st.selectbox("Education Tier", options=[0, 1, 2, 3], format_func=lambda x: ["High School", "Bachelors", "Masters", "Doctorate"][x])
    marital_status = st.selectbox("Marital Status", options=[0, 1], format_func=lambda x: "Single" if x == 0 else "Married")
    employment_status = st.selectbox("Employment", options=[0, 1, 2], format_func=lambda x: ["Employed", "Self-Employed", "Unemployed"][x])
    household_size = st.slider("Household Size", 1, 10, 3)
    dependents = st.slider("Dependents", 0, 8, 1)

with st.sidebar.expander("🩺 Vitals & Clinical Metrics", expanded=False):
    bmi = st.slider("BMI", 15.0, 55.0, 27.4, 0.1)
    systolic_bp = st.slider("Systolic BP (mmHg)", 90, 200, 125)
    diastolic_bp = st.slider("Diastolic BP (mmHg)", 60, 130, 82)
    ldl = st.slider("LDL Cholesterol", 50, 300, 120)
    hba1c = st.slider("HbA1c (%)", 4.0, 14.0, 5.8, 0.1)
    smoker = st.selectbox("Smoker Status", options=[0, 1], format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker")
    alcohol_freq = st.slider("Alcohol Freq (Days/Wk)", 0, 7, 2)
    visits_last_year = st.slider("Doctor Visits (Last Yr)", 0, 25, 3)
    hospitalizations_last_3yrs = st.slider("Hospitalizations (3 Yrs)", 0, 10, 0)
    days_hospitalized_last_3yrs = st.slider("Days in Hospital (3 Yrs)", 0, 60, 0)
    medication_count = st.slider("Prescription Meds Count", 0, 20, 2)

with st.sidebar.expander("📄 Policy & Coverage Details", expanded=False):
    plan_type = st.selectbox("Plan Type", [0, 1, 2], format_func=lambda x: ["Bronze", "Silver", "Gold"][x])
    network_tier = st.selectbox("Network Tier", [0, 1, 2], format_func=lambda x: ["Standard", "Preferred", "Elite"][x])
    deductible = st.number_input("Deductible ($)", 0, 15000, 1500, 250)
    copay = st.number_input("Copay ($)", 0, 200, 30, 5)
    policy_term_years = st.slider("Policy Term (Years)", 1, 30, 5)
    policy_changes_last_2yrs = st.slider("Policy Changes (2 Yrs)", 0, 5, 0)
    provider_quality = st.slider("Provider Quality Rating", 1.0, 5.0, 3.8, 0.1)
    risk_score = st.slider("Internal Risk Score", 0.0, 100.0, 34.5, 0.5)
    annual_premium = st.number_input("Annual Premium ($)", 500, 50000, 4200, 100)
    monthly_premium = annual_premium / 12.0
    claims_count = st.slider("Claims Count", 0, 25, 2)
    avg_claim_amount = st.number_input("Avg Claim Amount ($)", 0, 100000, 1200, 100)
    total_claims_paid = claims_count * avg_claim_amount

with st.sidebar.expander("🏥 Conditions & Clinical Procedures", expanded=False):
    col_c1, col_c2 = st.columns(2)
    hypertension = col_c1.checkbox("Hypertension", False)
    diabetes = col_c2.checkbox("Diabetes", False)
    asthma = col_c1.checkbox("Asthma", False)
    copd = col_c2.checkbox("COPD", False)
    cardiovascular_disease = col_c1.checkbox("Cardiovascular", False)
    cancer_history = col_c2.checkbox("Cancer History", False)
    kidney_disease = col_c1.checkbox("Kidney Disease", False)
    liver_disease = col_c2.checkbox("Liver Disease", False)
    arthritis = col_c1.checkbox("Arthritis", False)
    mental_health = col_c2.checkbox("Mental Health", False)
    
    chronic_count = sum([hypertension, diabetes, asthma, copd, cardiovascular_disease, 
                         cancer_history, kidney_disease, liver_disease, arthritis, mental_health])
    
    proc_imaging_count = st.slider("Imaging Tests", 0, 10, 1)
    proc_surgery_count = st.slider("Surgeries", 0, 5, 0)
    proc_physio_count = st.slider("Physiotherapy Sessions", 0, 30, 0)
    proc_consult_count = st.slider("Specialist Consultations", 0, 20, 2)
    proc_lab_count = st.slider("Lab Tests", 0, 30, 3)
    
    is_high_risk = 1 if (risk_score > 60 or chronic_count >= 3) else 0
    had_major_procedure = 1 if proc_surgery_count > 0 else 0

# ---------------------------------------------------------
# Feature Array Construction (Order matches model)
# ---------------------------------------------------------
ordered_features = [
    "age", "sex", "region", "urban_rural", "income", "education", "marital_status",
    "employment_status", "household_size", "dependents", "bmi", "smoker", "alcohol_freq",
    "visits_last_year", "hospitalizations_last_3yrs", "days_hospitalized_last_3yrs",
    "medication_count", "systolic_bp", "diastolic_bp", "ldl", "hba1c", "plan_type",
    "network_tier", "deductible", "copay", "policy_term_years", "policy_changes_last_2yrs",
    "provider_quality", "risk_score", "annual_premium", "monthly_premium", "claims_count",
    "avg_claim_amount", "total_claims_paid", "chronic_count", "hypertension", "diabetes",
    "asthma", "copd", "cardiovascular_disease", "cancer_history", "kidney_disease",
    "liver_disease", "arthritis", "mental_health", "proc_imaging_count", "proc_surgery_count",
    "proc_physio_count", "proc_consult_count", "proc_lab_count", "is_high_risk", "had_major_procedure"
]

input_data = [
    age, int(sex), int(region), int(urban_rural), float(income), int(education), int(marital_status),
    int(employment_status), int(household_size), int(dependents), float(bmi), int(smoker), int(alcohol_freq),
    int(visits_last_year), int(hospitalizations_last_3yrs), int(days_hospitalized_last_3yrs),
    int(medication_count), float(systolic_bp), float(diastolic_bp), float(ldl), float(hba1c), int(plan_type),
    int(network_tier), float(deductible), float(copay), int(policy_term_years), int(policy_changes_last_2yrs),
    float(provider_quality), float(risk_score), float(annual_premium), float(monthly_premium), int(claims_count),
    float(avg_claim_amount), float(total_claims_paid), int(chronic_count), int(hypertension), int(diabetes),
    int(asthma), int(copd), int(cardiovascular_disease), int(cancer_history), int(kidney_disease),
    int(liver_disease), int(arthritis), int(mental_health), int(proc_imaging_count), int(proc_surgery_count),
    int(proc_physio_count), int(proc_consult_count), int(proc_lab_count), int(is_high_risk), int(had_major_procedure)
]

input_df = pd.DataFrame([input_data], columns=ordered_features)

# ---------------------------------------------------------
# Prediction & Inference
# ---------------------------------------------------------
if model is not None:
    try:
        prediction = float(model.predict(input_df)[0])
    except Exception:
        prediction = float(model.predict(np.array([input_data]))[0])
else:
    prediction = 3450.80

# ---------------------------------------------------------
# Dashboard Main Layout
# ---------------------------------------------------------
col_pred, col_metrics = st.columns([1.1, 1.9])

with col_pred:
    badge_bg = "#EF4444" if prediction > 10000 else ("#F59E0B" if prediction > 4500 else "#10B981")
    badge_label = "High Impact" if prediction > 10000 else ("Moderate Impact" if prediction > 4500 else "Standard Risk")
    
    st.markdown(f"""
    <div class="prediction-box">
        <div style="color: #94A3B8; font-size: 0.95rem; font-weight: 500;">MODEL PREDICTION OUTPUT</div>
        <div class="prediction-val">${prediction:,.2f}</div>
        <span class="risk-badge" style="background-color: {badge_bg}; color: white;">{badge_label}</span>
        <div style="color: #64748B; font-size: 0.8rem; margin-top: 12px;">Computed via 50-tree Random Forest Ensemble</div>
    </div>
    """, unsafe_allow_html=True)

with col_metrics:
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Patient BMI", value=f"{bmi:.1f}", delta="Elevated" if bmi > 25 else "Normal", delta_color="inverse")
    m2.metric(label="Total Chronic Conditions", value=chronic_count, delta="High" if chronic_count >= 2 else "Low", delta_color="inverse")
    m3.metric(label="Prior Claims Total", value=f"${total_claims_paid:,.0f}")

    # Feature distribution indicator bar
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Composite Risk Index", 'font': {'size': 14, 'color': '#94A3B8'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#64748B"},
            'bar': {'color': "#06B6D4"},
            'bgcolor': "#1E293B",
            'steps': [
                {'range': [0, 35], 'color': '#064E3B'},
                {'range': [35, 70], 'color': '#78350F'},
                {'range': [70, 100], 'color': '#7F1D1D'}
            ]
        }
    ))
    fig_gauge.update_layout(height=160, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC"))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Graph Analysis Section
# ---------------------------------------------------------
st.markdown("### 📊 Clinical & Financial Visualizations")

chart_tab1, chart_tab2 = st.columns(2)

with chart_tab1:
    # Radar Chart of Normalized Factors
    categories = ['Cardio/BP', 'Metabolic/BMI', 'Utilization', 'Policy Cost', 'Procedures']
    
    bp_norm = min(systolic_bp / 180 * 100, 100)
    metabolic_norm = min(((bmi / 40) * 50 + (hba1c / 12) * 50), 100)
    utilization_norm = min(((visits_last_year / 15) * 50 + (days_hospitalized_last_3yrs / 15) * 50), 100)
    cost_norm = min((annual_premium / 15000) * 100, 100)
    proc_norm = min(((proc_surgery_count * 20) + (proc_imaging_count * 10)), 100)

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=[bp_norm, metabolic_norm, utilization_norm, cost_norm, proc_norm],
        theta=categories,
        fill='toself',
        fillcolor='rgba(6, 182, 212, 0.35)',
        line=dict(color='#06B6D4', width=2),
        name='Current Profile'
    ))
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='#334155', tickfont=dict(size=9, color='#94A3B8')),
            angularaxis=dict(gridcolor='#334155', linecolor='#334155', tickfont=dict(color='#E2E8F0'))
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=330,
        margin=dict(l=40, r=40, t=30, b=30),
        title=dict(text="Normalized Profile Footprint", font=dict(color="#E2E8F0", size=14))
    )
    st.plotly_chart(radar_fig, use_container_width=True)

with chart_tab2:
    # Procedure & Service Breakdown Bar Chart
    proc_names = ['Imaging', 'Surgery', 'Physiotherapy', 'Consults', 'Labs']
    proc_counts = [proc_imaging_count, proc_surgery_count, proc_physio_count, proc_consult_count, proc_lab_count]
    
    fig_bar = go.Figure(data=[
        go.Bar(
            x=proc_names,
            y=proc_counts,
            marker_color=['#38BDF8', '#F43F5E', '#10B981', '#A855F7', '#F59E0B'],
            opacity=0.9
        )
    ])
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=330,
        margin=dict(l=20, r=20, t=30, b=30),
        title=dict(text="Clinical Procedure Counts", font=dict(color="#E2E8F0", size=14)),
        yaxis=dict(gridcolor="#1E293B", title="Recorded Encounters", tickfont=dict(color="#94A3B8")),
        xaxis=dict(tickfont=dict(color="#CBD5E1"))
    )
    st.plotly_chart(fig_bar, use_container_width=True)
