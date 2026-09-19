import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# --- Load Pretrained Model ---
MODEL_PATH = "RandomForest_model.pkl"
model = None
model_error = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        model_error = f"Model load error: {str(e)}"
else:
    model_error = f"Model file '{MODEL_PATH}' not found in repository root."

# --- Dashboard Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Cost Intelligence & Risk Analytics</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0a0f1d;
            --bg-card: #111827;
            --border: #1f2937;
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.25);
            --accent: #06b6d4;
            --text-main: #f3f4f6;
            --text-muted: #94a3b8;
            --emerald: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2.5rem 1rem;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        /* Hero Banner */
        .hero {
            background: linear-gradient(135deg, #1e1b4b 0%, #1e293b 60%, #0f172a 100%);
            border: 1px solid #3730a3;
            border-radius: 16px;
            padding: 2.2rem 2.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1.5rem;
        }

        .hero h1 {
            font-size: 2rem;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.02em;
        }

        .hero p {
            color: var(--text-muted);
            margin-top: 0.4rem;
            font-size: 0.95rem;
        }

        .status-badge {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid var(--emerald);
            color: #34d399;
            padding: 0.4rem 0.9rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        /* Form Layout */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.8rem;
            margin-bottom: 1.5rem;
        }

        .section-title {
            color: var(--accent);
            font-size: 1rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 1.25rem;
            border-bottom: 1px solid #1f2937;
            padding-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .grid-3 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.2rem;
        }

        .grid-4 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
        }

        input, select {
            background: #0b1120;
            border: 1px solid #26334d;
            border-radius: 8px;
            color: #ffffff;
            padding: 0.65rem 0.85rem;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s;
        }

        input:focus, select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.2);
        }

        .checkbox-container {
            background: #0b1120;
            border: 1px solid #26334d;
            border-radius: 8px;
            padding: 0.65rem 0.9rem;
            display: flex;
            align-items: center;
            gap: 0.65rem;
            cursor: pointer;
        }

        .checkbox-container input {
            accent-color: var(--emerald);
            cursor: pointer;
            width: 16px;
            height: 16px;
        }

        /* Result Area */
        .result-box {
            display: none;
            background: radial-gradient(circle at top, #1e1b4b 0%, #0f172a 100%);
            border: 1.5px solid var(--primary);
            border-radius: 14px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 30px var(--primary-glow);
        }

        .result-title {
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .result-price {
            font-size: 3rem;
            font-weight: 800;
            color: #38bdf8;
            margin: 0.6rem 0;
        }

        /* Submit Button */
        .btn-submit {
            background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
            color: #ffffff;
            border: none;
            border-radius: 10px;
            padding: 0.95rem 2rem;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            width: 100%;
            box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
            transition: all 0.2s;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(6, 182, 212, 0.45);
        }
    </style>
</head>
<body>

<div class="container">
    <div class="hero">
        <div>
            <h1>🛡️ Medical Insurance Cost Estimator</h1>
            <p>52-Feature Actuarial & Clinical Prediction Hub</p>
        </div>
        <div class="status-badge">Random Forest Engine Active</div>
    </div>

    <div id="resultBox" class="result-box">
        <div class="result-title">Predicted Healthcare Cost</div>
        <div id="predictedCost" class="result-price">$0.00</div>
        <p style="color: var(--text-muted); font-size: 0.85rem;">Ensemble output generated across 50 Decision Trees</p>
    </div>

    <form id="predictionForm">
        <!-- Profile & Demographics -->
        <div class="card">
            <div class="section-title">👤 Profile & Socioeconomic Details</div>
            <div class="grid-3">
                <div class="form-group">
                    <label>Age</label>
                    <input type="number" name="age" value="38" min="0" max="110" required>
                </div>
                <div class="form-group">
                    <label>Sex</label>
                    <select name="sex"><option value="0">Female</option><option value="1">Male</option></select>
                </div>
                <div class="form-group">
                    <label>Region</label>
                    <select name="region">
                        <option value="0">Region 1</option><option value="1">Region 2</option>
                        <option value="2">Region 3</option><option value="3">Region 4</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Living Area</label>
                    <select name="urban_rural"><option value="1">Urban</option><option value="0">Rural</option></select>
                </div>
                <div class="form-group">
                    <label>Annual Income ($)</label>
                    <input type="number" step="500" name="income" value="55000" required>
                </div>
                <div class="form-group">
                    <label>Education</label>
                    <select name="education">
                        <option value="0">High School</option><option value="1" selected>Bachelor</option>
                        <option value="2">Master</option><option value="3">PhD</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Marital Status</label>
                    <select name="marital_status"><option value="0">Single</option><option value="1" selected>Married</option></select>
                </div>
                <div class="form-group">
                    <label>Employment</label>
                    <select name="employment_status"><option value="0">Unemployed</option><option value="1" selected>Employed</option><option value="2">Self-Employed</option></select>
                </div>
                <div class="form-group">
                    <label>Household Size</label>
                    <input type="number" name="household_size" value="3" min="1" max="15">
                </div>
                <div class="form-group">
                    <label>Dependents</label>
                    <input type="number" name="dependents" value="1" min="0" max="10">
                </div>
            </div>
        </div>

        <!-- Clinical & Vitals -->
        <div class="card">
            <div class="section-title">🩺 Clinical Markers & Biometrics</div>
            <div class="grid-3">
                <div class="form-group">
                    <label>BMI (kg/m²)</label>
                    <input type="number" step="0.1" name="bmi" value="25.4" required>
                </div>
                <div class="form-group">
                    <label>Smoker</label>
                    <select name="smoker"><option value="0">No</option><option value="1">Yes</option></select>
                </div>
                <div class="form-group">
                    <label>Alcohol Frequency</label>
                    <select name="alcohol_freq">
                        <option value="0">None</option><option value="1" selected>Occasional</option>
                        <option value="2">Moderate</option><option value="3">Frequent</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Systolic Blood Pressure (mmHg)</label>
                    <input type="number" name="systolic_bp" value="120">
                </div>
                <div class="form-group">
                    <label>Diastolic Blood Pressure (mmHg)</label>
                    <input type="number" name="diastolic_bp" value="80">
                </div>
                <div class="form-group">
                    <label>LDL Cholesterol (mg/dL)</label>
                    <input type="number" step="0.1" name="ldl" value="100.0">
                </div>
                <div class="form-group">
                    <label>HbA1c (%)</label>
                    <input type="number" step="0.1" name="hba1c" value="5.5">
                </div>
                <div class="form-group">
                    <label>Clinical Risk Score</label>
                    <input type="number" step="0.1" name="risk_score" value="24.5">
                </div>
                <div class="form-group">
                    <label>Prescription Count</label>
                    <input type="number" name="medication_count" value="1">
                </div>
                <div class="form-group">
                    <label>Visits (Last Year)</label>
                    <input type="number" name="visits_last_year" value="2">
                </div>
                <div class="form-group">
                    <label>Hospitalizations (Last 3 Yrs)</label>
                    <input type="number" name="hospitalizations_last_3yrs" value="0">
                </div>
                <div class="form-group">
                    <label>Days In Hospital (Last 3 Yrs)</label>
                    <input type="number" name="days_hospitalized_last_3yrs" value="0">
                </div>
            </div>
        </div>

        <!-- Policy & Financials -->
        <div class="card">
            <div class="section-title">📜 Insurance Policy & Claim History</div>
            <div class="grid-3">
                <div class="form-group">
                    <label>Plan Type</label>
                    <select name="plan_type"><option value="0">Bronze</option><option value="1" selected>Silver</option><option value="2">Gold</option></select>
                </div>
                <div class="form-group">
                    <label>Network Tier</label>
                    <select name="network_tier"><option value="0">Tier 1</option><option value="1" selected>Tier 2</option><option value="2">Tier 3</option></select>
                </div>
                <div class="form-group">
                    <label>Provider Quality Score (1.0 - 5.0)</label>
                    <input type="number" step="0.1" name="provider_quality" value="4.0">
                </div>
                <div class="form-group">
                    <label>Deductible ($)</label>
                    <input type="number" step="100" name="deductible" value="1500">
                </div>
                <div class="form-group">
                    <label>Copay ($)</label>
                    <input type="number" step="5" name="copay" value="30">
                </div>
                <div class="form-group">
                    <label>Policy Term (Years)</label>
                    <input type="number" name="policy_term_years" value="3">
                </div>
                <div class="form-group">
                    <label>Plan Adjustments (2 Yrs)</label>
                    <input type="number" name="policy_changes_last_2yrs" value="0">
                </div>
                <div class="form-group">
                    <label>Annual Premium ($)</label>
                    <input type="number" step="100" name="annual_premium" value="4800">
                </div>
                <div class="form-group">
                    <label>Monthly Premium ($)</label>
                    <input type="number" step="10" name="monthly_premium" value="400">
                </div>
                <div class="form-group">
                    <label>Historical Claims Count</label>
                    <input type="number" name="claims_count" value="1">
                </div>
                <div class="form-group">
                    <label>Average Claim Amount ($)</label>
                    <input type="number" step="100" name="avg_claim_amount" value="1200">
                </div>
                <div class="form-group">
                    <label>Total Claims Paid ($)</label>
                    <input type="number" step="100" name="total_claims_paid" value="1200">
                </div>
            </div>
        </div>

        <!-- Chronic & Procedures -->
        <div class="card">
            <div class="section-title">🔬 Chronic Conditions & Medical Services</div>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1rem;">Select all confirmed medical conditions:</p>
            <div class="grid-4" style="margin-bottom: 1.5rem;">
                <label class="checkbox-container"><input type="checkbox" name="hypertension"> Hypertension</label>
                <label class="checkbox-container"><input type="checkbox" name="diabetes"> Diabetes</label>
                <label class="checkbox-container"><input type="checkbox" name="asthma"> Asthma</label>
                <label class="checkbox-container"><input type="checkbox" name="copd"> COPD</label>
                <label class="checkbox-container"><input type="checkbox" name="cardiovascular_disease"> Cardiovascular</label>
                <label class="checkbox-container"><input type="checkbox" name="cancer_history"> Cancer History</label>
                <label class="checkbox-container"><input type="checkbox" name="kidney_disease"> Kidney Disease</label>
                <label class="checkbox-container"><input type="checkbox" name="liver_disease"> Liver Disease</label>
                <label class="checkbox-container"><input type="checkbox" name="arthritis"> Arthritis</label>
                <label class="checkbox-container"><input type="checkbox" name="mental_health"> Mental Health</label>
            </div>

            <div class="grid-3">
                <div class="form-group">
                    <label>Imaging Procedures</label>
                    <input type="number" name="proc_imaging_count" value="0">
                </div>
                <div class="form-group">
                    <label>Surgeries</label>
                    <input type="number" name="proc_surgery_count" value="0">
                </div>
                <div class="form-group">
                    <label>Physiotherapy Count</label>
                    <input type="number" name="proc_physio_count" value="0">
                </div>
                <div class="form-group">
                    <label>Doctor Consultations</label>
                    <input type="number" name="proc_consult_count" value="1">
                </div>
                <div class="form-group">
                    <label>Laboratory Panels</label>
                    <input type="number" name="proc_lab_count" value="2">
                </div>
                <div class="form-group">
                    <label>High Risk Categorization</label>
                    <select name="is_high_risk"><option value="0">No</option><option value="1">Yes</option></select>
                </div>
                <div class="form-group">
                    <label>Had Major Procedure</label>
                    <select name="had_major_procedure"><option value="0">No</option><option value="1">Yes</option></select>
                </div>
            </div>
        </div>

        <button type="submit" class="btn-submit">⚡ Compute Estimated Insurance Cost</button>
    </form>
</div>

<script>
    document.getElementById("predictionForm").addEventListener("submit", async function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        const data = {};
        
        formData.forEach((value, key) => {
            data[key] = value;
        });

        // Convert checkboxes to binary
        const chronicDiseases = [
            "hypertension", "diabetes", "asthma", "copd", "cardiovascular_disease",
            "cancer_history", "kidney_disease", "liver_disease", "arthritis", "mental_health"
        ];
        
        let chronicCount = 0;
        chronicDiseases.forEach(d => {
            const hasDisease = formData.has(d) ? 1 : 0;
            data[d] = hasDisease;
            chronicCount += hasDisease;
        });
        data["chronic_count"] = chronicCount;

        try {
            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });
            const result = await res.json();
            
            if (result.success) {
                document.getElementById("resultBox").style.display = "block";
                document.getElementById("predictedCost").innerText = "$" + result.prediction.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
                window.scrollTo({ top: 0, behavior: 'smooth' });
            } else {
                alert("Prediction Error: " + result.error);
            }
        } catch (err) {
            alert("Network Error: " + err.message);
        }
    });
</script>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": model_error or "Model is not loaded."}), 500

    try:
        payload = request.get_json(force=True)
        # Extract features adhering strictly to the model's feature ordering
        feature_names = list(model.feature_names_in_)
        input_data = []
        for feat in feature_names:
            val = float(payload.get(feat, 0))
            input_data.append(val)

        df_features = pd.DataFrame([input_data], columns=feature_names)
        prediction = float(model.predict(df_features)[0])

        return jsonify({"success": True, "prediction": round(prediction, 2)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
