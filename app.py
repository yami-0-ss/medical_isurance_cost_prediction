import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load model safely
model = None
try:
    model = joblib.load("RandomForest_model.pkl")
    print("RandomForest_model.pkl loaded successfully.")
except Exception as err:
    print(f"Warning: Model could not be loaded at startup: {err}")

ORDERED_FEATURES = [
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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Risk & Insurance Intelligence</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #0B0F19; color: #E2E8F0; }
        .glass-panel { background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); }
        .glow-accent { box-shadow: 0 0 35px -5px rgba(6, 182, 212, 0.25); }
        input, select { background-color: #1F2937 !important; border-color: #374151 !important; color: #F3F4F6 !important; }
    </style>
</head>
<body class="p-4 md:p-8 min-h-screen">
    <div class="max-w-7xl mx-auto space-y-6">
        <!-- Hero Header -->
        <header class="glass-panel glow-accent rounded-2xl p-6 md:p-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-gradient-to-r from-[#0F172A] via-[#111E36] to-[#042F2E]">
            <div>
                <span class="text-xs uppercase tracking-wider font-bold text-cyan-400">RandomForest Ensemble Model</span>
                <h1 class="text-2xl md:text-3xl font-extrabold text-white mt-1">Health Risk & Claims Analytics</h1>
                <p class="text-slate-400 text-sm mt-1">52-feature inference pipeline with multivariate radar and encounter breakdown</p>
            </div>
            <button onclick="runInference()" class="px-6 py-3 bg-gradient-to-r from-cyan-500 to-teal-400 hover:from-cyan-400 hover:to-teal-300 text-slate-950 font-bold rounded-xl shadow-lg transition duration-150">
                Run Model Simulation
            </button>
        </header>

        <!-- Main Dashboard View -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <!-- Left: Inputs Column -->
            <div class="lg:col-span-5 space-y-4">
                <div class="glass-panel rounded-2xl p-5 space-y-4 max-h-[750px] overflow-y-auto pr-3">
                    <h2 class="text-sm font-bold text-slate-300 uppercase tracking-wide">Patient Parameters</h2>
                    
                    <div class="grid grid-cols-2 gap-3 text-xs">
                        <div>
                            <label class="block text-slate-400 mb-1">Age</label>
                            <input id="age" type="number" value="45" class="w-full rounded-lg px-3 py-2">
                        </div>
                        <div>
                            <label class="block text-slate-400 mb-1">BMI</label>
                            <input id="bmi" type="number" step="0.1" value="28.4" class="w-full rounded-lg px-3 py-2">
                        </div>
                        <div>
                            <label class="block text-slate-400 mb-1">Systolic BP</label>
                            <input id="systolic_bp" type="number" value="130" class="w-full rounded-lg px-3 py-2">
                        </div>
                        <div>
                            <label class="block text-slate-400 mb-1">Diastolic BP</label>
                            <input id="diastolic_bp" type="number" value="85" class="w-full rounded-lg px-3 py-2">
                        </div>
                        <div>
                            <label class="block text-slate-400 mb-1">LDL Cholesterol</label>
                            <input id="ldl" type="number" value="130" class="w-full rounded-lg px-3 py-2">
                        </div>
                        <div>
                            <label class="block text-slate-400 mb-1">HbA1c (%)</label>
                            <input id="hba1c" type="number" step="0.1" value="6.1" class="w-full rounded-lg px-3 py-2">
                        </div>
                        <div>
                            <label class="block text-slate-400 mb-1">Annual Premium ($)</label>
                            <input id="annual_premium" type="number" value="4800" class="w-full rounded-lg px-3 py-2">
                        </div>
                        <div>
                            <label class="block text-slate-400 mb-1">Internal Risk Score</label>
                            <input id="risk_score" type="number" step="0.5" value="38.5" class="w-full rounded-lg px-3 py-2">
                        </div>
                    </div>

                    <h2 class="text-sm font-bold text-slate-300 uppercase tracking-wide pt-2">Clinical Encounters (Counts)</h2>
                    <div class="grid grid-cols-3 gap-2 text-xs">
                        <div>
                            <label class="block text-slate-400 mb-1">Imaging</label>
                            <input id="proc_imaging_count" type="number" value="1" class="w-full rounded-lg px-2 py-1.5">
                        </div>
                        <div>
                            <label class="block text-slate-400 mb-1">Surgeries</label>
                            <input id="proc_surgery_count" type="number" value="0" class="w-full rounded-lg px-2 py-1.5">
                        </div>
                        <div>
                            <label class="block text-slate-400 mb-1">Physio</label>
                            <input id="proc_physio_count" type="number" value="2" class="w-full rounded-lg px-2 py-1.5">
                        </div>
                        <div>
                            <label class="block text-slate-400 mb-1">Consults</label>
                            <input id="proc_consult_count" type="number" value="3" class="w-full rounded-lg px-2 py-1.5">
                        </div>
                        <div>
                            <label class="block text-slate-400 mb-1">Labs</label>
                            <input id="proc_lab_count" type="number" value="4" class="w-full rounded-lg px-2 py-1.5">
                        </div>
                        <div>
                            <label class="block text-slate-400 mb-1">Past Claims</label>
                            <input id="claims_count" type="number" value="2" class="w-full rounded-lg px-2 py-1.5">
                        </div>
                    </div>

                    <h2 class="text-sm font-bold text-slate-300 uppercase tracking-wide pt-2">Conditions Active</h2>
                    <div class="grid grid-cols-2 gap-2 text-xs text-slate-300">
                        <label class="flex items-center gap-2"><input id="hypertension" type="checkbox" class="rounded"> Hypertension</label>
                        <label class="flex items-center gap-2"><input id="diabetes" type="checkbox" class="rounded"> Diabetes</label>
                        <label class="flex items-center gap-2"><input id="asthma" type="checkbox" class="rounded"> Asthma</label>
                        <label class="flex items-center gap-2"><input id="cardiovascular_disease" type="checkbox" class="rounded"> Cardiovascular</label>
                        <label class="flex items-center gap-2"><input id="kidney_disease" type="checkbox" class="rounded"> Kidney Disease</label>
                        <label class="flex items-center gap-2"><input id="smoker" type="checkbox" class="rounded"> Smoker</label>
                    </div>
                </div>
            </div>

            <!-- Right: Prediction & Visualizations Column -->
            <div class="lg:col-span-7 space-y-6">
                <!-- Top Metric Cards -->
                <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div class="glass-panel rounded-2xl p-5 border-l-4 border-cyan-400">
                        <div class="text-xs uppercase text-slate-400 font-semibold">Predicted Impact</div>
                        <div id="predValue" class="text-2xl sm:text-3xl font-black text-cyan-400 mt-1">$0.00</div>
                        <span id="riskBadge" class="inline-block mt-2 text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">Standard Risk</span>
                    </div>
                    <div class="glass-panel rounded-2xl p-5 border-l-4 border-emerald-400">
                        <div class="text-xs uppercase text-slate-400 font-semibold">Active Conditions</div>
                        <div id="chronicCount" class="text-2xl sm:text-3xl font-black text-emerald-400 mt-1">0</div>
                        <span class="inline-block mt-2 text-[10px] text-slate-400">Chronic comorbidities</span>
                    </div>
                    <div class="glass-panel rounded-2xl p-5 border-l-4 border-violet-400">
                        <div class="text-xs uppercase text-slate-400 font-semibold">Claims History</div>
                        <div id="claimsTotal" class="text-2xl sm:text-3xl font-black text-violet-400 mt-1">$2,400</div>
                        <span class="inline-block mt-2 text-[10px] text-slate-400">Aggregated historical payouts</span>
                    </div>
                </div>

                <!-- Charts Container -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="glass-panel rounded-2xl p-4">
                        <h3 class="text-xs font-bold text-slate-300 uppercase tracking-wide mb-3">Multivariate Health Footprint</h3>
                        <div class="h-64 flex items-center justify-center">
                            <canvas id="radarChart"></canvas>
                        </div>
                    </div>
                    <div class="glass-panel rounded-2xl p-4">
                        <h3 class="text-xs font-bold text-slate-300 uppercase tracking-wide mb-3">Clinical Procedure Breakdown</h3>
                        <div class="h-64 flex items-center justify-center">
                            <canvas id="barChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let radarChart, barChart;

        function initCharts() {
            const ctxRadar = document.getElementById('radarChart').getContext('2d');
            radarChart = new Chart(ctxRadar, {
                type: 'radar',
                data: {
                    labels: ['Cardio/BP', 'Metabolic/BMI', 'Risk Score', 'Policy Cost', 'Procedures'],
                    datasets: [{
                        label: 'Patient Index',
                        data: [65, 50, 40, 30, 45],
                        backgroundColor: 'rgba(6, 182, 212, 0.25)',
                        borderColor: '#06B6D4',
                        borderWidth: 2,
                        pointBackgroundColor: '#22D3EE'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            angleLines: { color: '#1F2937' },
                            grid: { color: '#1F2937' },
                            pointLabels: { color: '#94A3B8', font: { size: 10 } },
                            ticks: { display: false, max: 100 }
                        }
                    },
                    plugins: { legend: { display: false } }
                }
            });

            const ctxBar = document.getElementById('barChart').getContext('2d');
            barChart = new Chart(ctxBar, {
                type: 'bar',
                data: {
                    labels: ['Imaging', 'Surgery', 'Physio', 'Consults', 'Labs'],
                    datasets: [{
                        data: [1, 0, 2, 3, 4],
                        backgroundColor: ['#38BDF8', '#F43F5E', '#10B981', '#A855F7', '#F59E0B'],
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { display: false }, ticks: { color: '#94A3B8' } },
                        y: { grid: { color: '#1F2937' }, ticks: { color: '#94A3B8' } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }

        async function runInference() {
            const payload = {
                age: parseFloat(document.getElementById('age').value) || 45,
                bmi: parseFloat(document.getElementById('bmi').value) || 28.4,
                systolic_bp: parseFloat(document.getElementById('systolic_bp').value) || 130,
                diastolic_bp: parseFloat(document.getElementById('diastolic_bp').value) || 85,
                ldl: parseFloat(document.getElementById('ldl').value) || 130,
                hba1c: parseFloat(document.getElementById('hba1c').value) || 6.1,
                annual_premium: parseFloat(document.getElementById('annual_premium').value) || 4800,
                risk_score: parseFloat(document.getElementById('risk_score').value) || 38.5,
                proc_imaging_count: parseInt(document.getElementById('proc_imaging_count').value) || 1,
                proc_surgery_count: parseInt(document.getElementById('proc_surgery_count').value) || 0,
                proc_physio_count: parseInt(document.getElementById('proc_physio_count').value) || 2,
                proc_consult_count: parseInt(document.getElementById('proc_consult_count').value) || 3,
                proc_lab_count: parseInt(document.getElementById('proc_lab_count').value) || 4,
                claims_count: parseInt(document.getElementById('claims_count').value) || 2,
                hypertension: document.getElementById('hypertension').checked ? 1 : 0,
                diabetes: document.getElementById('diabetes').checked ? 1 : 0,
                asthma: document.getElementById('asthma').checked ? 1 : 0,
                cardiovascular_disease: document.getElementById('cardiovascular_disease').checked ? 1 : 0,
                kidney_disease: document.getElementById('kidney_disease').checked ? 1 : 0,
                smoker: document.getElementById('smoker').checked ? 1 : 0
            };

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                
                document.getElementById('predValue').innerText = '$' + data.prediction.toLocaleString('en-US', {minimumFractionDigits: 2});
                document.getElementById('chronicCount').innerText = data.chronic_count;
                document.getElementById('claimsTotal').innerText = '$' + data.total_claims_paid.toLocaleString('en-US');

                const badge = document.getElementById('riskBadge');
                if (data.prediction > 10000) {
                    badge.className = "inline-block mt-2 text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-800";
                    badge.innerText = "High Impact";
                } else if (data.prediction > 4500) {
                    badge.className = "inline-block mt-2 text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-800";
                    badge.innerText = "Moderate Impact";
                } else {
                    badge.className = "inline-block mt-2 text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800";
                    badge.innerText = "Standard Risk";
                }

                const bpNorm = Math.min((payload.systolic_bp / 180) * 100, 100);
                const bmiNorm = Math.min((payload.bmi / 40) * 100, 100);
                const costNorm = Math.min((payload.annual_premium / 12000) * 100, 100);
                const procSum = Math.min((payload.proc_surgery_count * 25 + payload.proc_imaging_count * 15), 100);

                radarChart.data.datasets[0].data = [bpNorm, bmiNorm, payload.risk_score, costNorm, procSum];
                radarChart.update();

                barChart.data.datasets[0].data = [
                    payload.proc_imaging_count,
                    payload.proc_surgery_count,
                    payload.proc_physio_count,
                    payload.proc_consult_count,
                    payload.proc_lab_count
                ];
                barChart.update();
            } catch (e) {
                console.error("Prediction failed:", e);
            }
        }

        window.onload = () => {
            initCharts();
            runInference();
        };
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET", "HEAD"])
def index():
    if request.method == "HEAD":
        return "", 200
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    
    age = float(data.get("age", 45))
    sex = int(data.get("sex", 1))
    region = int(data.get("region", 1))
    urban_rural = int(data.get("urban_rural", 1))
    income = float(data.get("income", 55000.0))
    education = int(data.get("education", 1))
    marital_status = int(data.get("marital_status", 1))
    employment_status = int(data.get("employment_status", 0))
    household_size = int(data.get("household_size", 3))
    dependents = int(data.get("dependents", 1))
    bmi = float(data.get("bmi", 28.4))
    smoker = int(data.get("smoker", 0))
    alcohol_freq = int(data.get("alcohol_freq", 2))
    visits_last_year = int(data.get("visits_last_year", 3))
    hospitalizations_last_3yrs = int(data.get("hospitalizations_last_3yrs", 0))
    days_hospitalized_last_3yrs = int(data.get("days_hospitalized_last_3yrs", 0))
    medication_count = int(data.get("medication_count", 2))
    systolic_bp = float(data.get("systolic_bp", 130.0))
    diastolic_bp = float(data.get("diastolic_bp", 85.0))
    ldl = float(data.get("ldl", 130.0))
    hba1c = float(data.get("hba1c", 6.1))
    plan_type = int(data.get("plan_type", 1))
    network_tier = int(data.get("network_tier", 1))
    deductible = float(data.get("deductible", 1500.0))
    copay = float(data.get("copay", 30.0))
    policy_term_years = int(data.get("policy_term_years", 5))
    policy_changes_last_2yrs = int(data.get("policy_changes_last_2yrs", 0))
    provider_quality = float(data.get("provider_quality", 3.8))
    risk_score = float(data.get("risk_score", 38.5))
    annual_premium = float(data.get("annual_premium", 4800.0))
    monthly_premium = annual_premium / 12.0
    claims_count = int(data.get("claims_count", 2))
    avg_claim_amount = float(data.get("avg_claim_amount", 1200.0))
    total_claims_paid = claims_count * avg_claim_amount

    hypertension = int(data.get("hypertension", 0))
    diabetes = int(data.get("diabetes", 0))
    asthma = int(data.get("asthma", 0))
    copd = int(data.get("copd", 0))
    cardiovascular_disease = int(data.get("cardiovascular_disease", 0))
    cancer_history = int(data.get("cancer_history", 0))
    kidney_disease = int(data.get("kidney_disease", 0))
    liver_disease = int(data.get("liver_disease", 0))
    arthritis = int(data.get("arthritis", 0))
    mental_health = int(data.get("mental_health", 0))

    chronic_conditions = [
        hypertension, diabetes, asthma, copd, cardiovascular_disease,
        cancer_history, kidney_disease, liver_disease, arthritis, mental_health
    ]
    chronic_count = sum(chronic_conditions)

    proc_imaging_count = int(data.get("proc_imaging_count", 1))
    proc_surgery_count = int(data.get("proc_surgery_count", 0))
    proc_physio_count = int(data.get("proc_physio_count", 2))
    proc_consult_count = int(data.get("proc_consult_count", 3))
    proc_lab_count = int(data.get("proc_lab_count", 4))

    is_high_risk = 1 if (risk_score > 60 or chronic_count >= 3) else 0
    had_major_procedure = 1 if proc_surgery_count > 0 else 0

    row = [
        age, sex, region, urban_rural, income, education, marital_status,
        employment_status, household_size, dependents, bmi, smoker, alcohol_freq,
        visits_last_year, hospitalizations_last_3yrs, days_hospitalized_last_3yrs,
        medication_count, systolic_bp, diastolic_bp, ldl, hba1c, plan_type,
        network_tier, deductible, copay, policy_term_years, policy_changes_last_2yrs,
        provider_quality, risk_score, annual_premium, monthly_premium, claims_count,
        avg_claim_amount, total_claims_paid, chronic_count, hypertension, diabetes,
        asthma, copd, cardiovascular_disease, cancer_history, kidney_disease,
        liver_disease, arthritis, mental_health, proc_imaging_count, proc_surgery_count,
        proc_physio_count, proc_consult_count, proc_lab_count, is_high_risk, had_major_procedure
    ]

    if model is not None:
        try:
            df = pd.DataFrame([row], columns=ORDERED_FEATURES)
            val = float(model.predict(df)[0])
        except Exception:
            val = float(model.predict(np.array([row]))[0])
    else:
        val = 3540.50

    return jsonify({
        "prediction": round(val, 2),
        "chronic_count": chronic_count,
        "total_claims_paid": total_claims_paid
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
