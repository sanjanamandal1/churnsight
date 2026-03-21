import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import run_preprocessing
from explain import get_shap_explainer, get_top_drivers
from recommender import get_recommendations

st.set_page_config(page_title="ChurnSight Simulator", page_icon="lightning", layout="wide")

@st.cache_resource
def load_artifacts():
    with open("models/ensemble_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

def make_gauge(probability):
    color = "#ef4444" if probability >= 0.7 else "#f59e0b" if probability >= 0.4 else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        title={"text": "Churn Probability %", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#dcfce7"},
                {"range": [40, 70], "color": "#fef9c3"},
                {"range": [70, 100], "color": "#fee2e2"},
            ]
        }
    ))
    fig.update_layout(height=300)
    return fig

def build_input_vector(inputs, feature_names, scaler):
    row = {f: 0 for f in feature_names}
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "charges_per_tenure"]
    raw_num = np.array([[
        inputs["tenure"],
        inputs["MonthlyCharges"],
        inputs["tenure"] * inputs["MonthlyCharges"],
        inputs["MonthlyCharges"] / (inputs["tenure"] + 1)
    ]])
    scaled = scaler.transform(raw_num)[0]
    for i, col in enumerate(num_cols):
        if col in row:
            row[col] = scaled[i]
    row["gender"] = 1 if inputs["gender"] == "Male" else 0
    row["SeniorCitizen"] = inputs["SeniorCitizen"]
    row["Partner"] = 1 if inputs["Partner"] == "Yes" else 0
    row["Dependents"] = 1 if inputs["Dependents"] == "Yes" else 0
    row["PhoneService"] = 1 if inputs["PhoneService"] == "Yes" else 0
    row["PaperlessBilling"] = 1 if inputs["PaperlessBilling"] == "Yes" else 0
    row["high_value"] = 1 if inputs["MonthlyCharges"] > 64.76 else 0
    tg_map = {"0-1yr": 0, "1-2yr": 1, "2-4yr": 2, "4-5yr": 3, "5-6yr": 4}
    row["tenure_group"] = tg_map.get(inputs["tenure_group"], 0)
    contract_col = "Contract_" + inputs["Contract"]
    if contract_col in row:
        row[contract_col] = 1
    internet_col = "InternetService_" + inputs["InternetService"]
    if internet_col in row:
        row[internet_col] = 1
    tech_col = "TechSupport_" + inputs["TechSupport"]
    if tech_col in row:
        row[tech_col] = 1
    security_col = "OnlineSecurity_" + inputs["OnlineSecurity"]
    if security_col in row:
        row[security_col] = 1
    payment_col = "PaymentMethod_" + inputs["PaymentMethod"]
    if payment_col in row:
        row[payment_col] = 1
    return pd.DataFrame([row])[feature_names]

model, scaler, feature_names = load_artifacts()

st.title("What-if Churn Simulator")
st.markdown("Adjust customer attributes and see churn probability change in real time.")
st.markdown("---")

col_inputs, col_result = st.columns([1, 1])

with col_inputs:
    st.subheader("Customer Attributes")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    if tenure <= 12: tg = "0-1yr"
    elif tenure <= 24: tg = "1-2yr"
    elif tenure <= 48: tg = "2-4yr"
    elif tenure <= 60: tg = "4-5yr"
    else: tg = "5-6yr"
    inputs = {
        "gender": gender, "SeniorCitizen": senior,
        "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone,
        "InternetService": internet, "Contract": contract,
        "TechSupport": tech_support, "OnlineSecurity": security,
        "PaymentMethod": payment, "PaperlessBilling": paperless,
        "MonthlyCharges": monthly_charges, "tenure_group": tg
    }

with col_result:
    st.subheader("Live Churn Prediction")
    try:
        X_input = build_input_vector(inputs, feature_names, scaler)
        prob = model.predict_proba(X_input)[0][1]
        st.plotly_chart(make_gauge(prob), use_container_width=True)
        if prob >= 0.70:
            st.error("HIGH RISK - Immediate action recommended!")
        elif prob >= 0.40:
            st.warning("MEDIUM RISK - Monitor and engage proactively")
        else:
            st.success("LOW RISK - Customer is likely to stay")
        st.markdown("---")
        st.subheader("Recommended Actions")
        shap_explainer = get_shap_explainer(model, X_input)
        shap_vals = shap_explainer.shap_values(X_input)[0]
        top_pos, _ = get_top_drivers(shap_vals, feature_names)
        recs = get_recommendations(top_pos)
        for r in recs[:3]:
            st.info(r["icon"] + " " + r["action"] + " — " + r["reason"] + " [" + r["priority"] + " Priority]")
    except Exception as e:
        st.error("Error: " + str(e))