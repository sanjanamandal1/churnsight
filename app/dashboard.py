import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

from preprocess import run_preprocessing
from explain import batch_explain, get_shap_explainer, get_shap_values, get_top_drivers
from risk_segmentor import assign_risk_tiers, estimate_revenue_saved
from recommender import get_recommendations, get_bulk_recommendations
from analytics import compute_health_score, get_health_label, simulate_churn_trend

st.set_page_config(page_title="ChurnSight — Executive Intelligence", page_icon="🔭", layout="wide")

@st.cache_resource
def load_artifacts():
    with open("models/ensemble_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("models/metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    return model, scaler, feature_names, metrics

@st.cache_data
def load_and_score_data():
    X_train, X_test, y_train, y_test, scaler, feature_names = run_preprocessing("data/telco_churn.csv")
    model, _, _, _ = load_artifacts()
    df_raw = pd.read_csv("data/telco_churn.csv")
    df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce")
    df_raw["TotalCharges"] = df_raw["TotalCharges"].fillna(df_raw["TotalCharges"].median())
    df_raw["Churn"] = df_raw["Churn"].map({"Yes": 1, "No": 0})
    probs = model.predict_proba(X_test)[:, 1]
    scored = df_raw.iloc[X_test.index].copy()
    scored["churn_probability"] = probs
    scored["risk_tier"] = assign_risk_tiers(pd.Series(probs))
    scored = scored.reset_index(drop=True)
    shap_df = batch_explain(model, feature_names, X_test)
    shap_df = shap_df.reset_index(drop=True)
    scored = get_bulk_recommendations(scored, shap_df)
    return scored, X_test, shap_df, y_test

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def score_custom_dataframe(df_raw, model, scaler, feature_names):
    from preprocess import clean_data, feature_engineering, encode_features
    df_clean = clean_data(df_raw.copy())
    df_fe = feature_engineering(df_clean)
    df_enc = encode_features(df_fe)

    for col in feature_names:
        if col not in df_enc.columns:
            df_enc[col] = 0

    X_custom = df_enc[feature_names].copy()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_tenure']
    X_custom[num_cols] = scaler.transform(X_custom[num_cols])

    probs = model.predict_proba(X_custom)[:, 1]
    df_scored = df_raw.copy()
    df_scored["churn_probability"] = probs
    df_scored["risk_tier"] = assign_risk_tiers(probs)

    shap_df = batch_explain(model, feature_names, X_custom)
    df_scored = get_bulk_recommendations(df_scored, shap_df)
    return df_scored

def apply_executive_theme(fig, title=""):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#9ca3af", size=12),
        title=dict(text=title, font=dict(color="#f3f4f6", size=15, weight=600)),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
        legend=dict(font=dict(color="#d1d5db"))
    )
    return fig

# Sidebar Branding
st.sidebar.markdown("<h2 style='margin-bottom:0; font-size:22px; font-weight:700; letter-spacing:-0.03em;'>CHURNSIGHT</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#6b7280; font-size:11px; margin-top:0; margin-bottom:15px; letter-spacing:0.05em; text-transform:uppercase; font-weight:600;'>Intelligence Platform</p>", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", [
    "Overview",
    "Customer Risk Table",
    "Bulk Scorer",
    "Model Performance",
    "Advanced Analytics",
    "What-if Simulator",
    "ROI Simulator"
])

model, scaler, feature_names, metrics = load_artifacts()
load_css()
scored_df, X_test, shap_df, y_test = load_and_score_data()

# -----------------------------------------------------------------------
# PAGE 1 - OVERVIEW
# -----------------------------------------------------------------------
if page == "Overview":
    st.title("Overview")
    st.markdown("<p class='section-subtitle'>High-level portfolio churn health metrics, risk distribution, and revenue impact.</p>", unsafe_allow_html=True)

    total = len(scored_df)
    churned = scored_df["Churn"].sum()
    churn_rate = round(churned / total * 100, 1)
    revenue_at_risk = round(scored_df[scored_df["risk_tier"] == "High Risk"]["MonthlyCharges"].sum(), 2)
    revenue_saved = estimate_revenue_saved(scored_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churn Rate", f"{churn_rate}%")
    c3.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}/mo")
    c4.metric("Est. Revenue Saveable", f"${revenue_saved:,.0f}/mo")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        contract_cols = [c for c in scored_df.columns if "Contract" in c]
        if contract_cols:
            fig = px.histogram(scored_df, x=contract_cols[0],
                               color=scored_df["Churn"].map({1: "Churned", 0: "Retained"}),
                               barmode="group",
                               color_discrete_map={"Churned": "#f43f5e", "Retained": "#10b981"})
            apply_executive_theme(fig, "Churn Rate by Contract Type")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        tier_counts = scored_df["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        colors = {"High Risk": "#f43f5e", "Medium Risk": "#f59e0b", "Low Risk": "#10b981"}
        fig2 = px.pie(tier_counts, names="tier", values="count",
                      color="tier", color_discrete_map=colors, hole=0.45)
        apply_executive_theme(fig2, "Risk Tier Breakdown")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.box(scored_df,
                      x=scored_df["Churn"].map({1: "Churned", 0: "Retained"}),
                      y="MonthlyCharges",
                      color=scored_df["Churn"].map({1: "Churned", 0: "Retained"}),
                      color_discrete_map={"Churned": "#f43f5e", "Retained": "#10b981"})
        apply_executive_theme(fig3, "Monthly Charges Distribution")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        scored_df["tenure_band"] = pd.cut(scored_df["tenure"],
                                           bins=[-1,12,24,48,60,72],
                                           labels=["0-1yr","1-2yr","2-4yr","4-5yr","5-6yr"])
        tenure_churn = scored_df.groupby("tenure_band", observed=False)["Churn"].mean().reset_index()
        tenure_churn.columns = ["Tenure Group", "Churn Rate"]
        fig4 = px.bar(tenure_churn, x="Tenure Group", y="Churn Rate",
                      color="Churn Rate", color_continuous_scale=["#fecdd3", "#f43f5e"])
        apply_executive_theme(fig4, "Churn Proportion by Customer Tenure")
        st.plotly_chart(fig4, use_container_width=True)

# -----------------------------------------------------------------------
# PAGE 2 - CUSTOMER RISK TABLE
# -----------------------------------------------------------------------
elif page == "Customer Risk Table":
    st.title("Customer Risk Table")
    st.markdown("<p class='section-subtitle'>Filter accounts by risk tier and perform individual SHAP driver analysis.</p>", unsafe_allow_html=True)

    tier_filter = st.multiselect("Filter by Risk Tier",
                                  scored_df["risk_tier"].unique().tolist(),
                                  default=scored_df["risk_tier"].unique().tolist())

    filtered = scored_df[scored_df["risk_tier"].isin(tier_filter)].copy()
    filtered["churn_probability"] = filtered["churn_probability"].apply(lambda x: f"{x:.1%}")

    display_cols = ["gender", "tenure", "MonthlyCharges", "TotalCharges",
                    "churn_probability", "risk_tier", "top_recommendation"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True, height=350)

    st.markdown("---")
    st.subheader("Individual Account Analysis")
    customer_idx = st.number_input("Customer Index (Row Number)",
                                    min_value=0, max_value=len(scored_df)-1, value=0)

    if st.button("Analyze Selected Customer"):
        customer = scored_df.iloc[customer_idx]
        prob = customer["churn_probability"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Churn Probability", f"{prob:.1%}")
        col2.metric("Risk Classification", customer["risk_tier"])
        col3.metric("Monthly Billing", f"${customer['MonthlyCharges']:.2f}")

        if customer_idx < len(shap_df):
            shap_row = shap_df.iloc[customer_idx]
            top_pos = shap_row.nlargest(5)
            top_neg = shap_row.nsmallest(5)

            st.markdown("#### Key Feature Drivers (SHAP)")
            fig_shap = go.Figure(go.Bar(
                x=top_pos.values.tolist() + top_neg.values.tolist(),
                y=top_pos.index.tolist() + top_neg.index.tolist(),
                orientation="h",
                marker_color=["#f43f5e"]*5 + ["#10b981"]*5
            ))
            apply_executive_theme(fig_shap, "Feature Impact on Churn Score")
            st.plotly_chart(fig_shap, use_container_width=True)

            st.markdown("#### Recommended Intervention Strategies")
            recs = get_recommendations(top_pos)
            for r in recs:
                st.info(f"{r['icon']} **{r['action']}** — _{r['reason']}_ [{r['priority']} Priority]")

# -----------------------------------------------------------------------
# PAGE 3 - BULK SCORER
# -----------------------------------------------------------------------
elif page == "Bulk Scorer":
    st.title("Bulk Batch Scorer")
    st.markdown("<p class='section-subtitle'>Upload customer datasets to generate churn predictions, risk tiers, and retention plans.</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])
    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded)
            st.subheader("Uploaded Data Preview")
            st.dataframe(df_upload.head(5), use_container_width=True)

            with st.spinner("Executing ChurnSight Inference & SHAP Pipeline..."):
                scored_custom = score_custom_dataframe(df_upload, model, scaler, feature_names)

            st.success(f"Successfully scored {len(scored_custom):,} customer records.")

            c1, c2, c3 = st.columns(3)
            c1.metric("High Risk Accounts", len(scored_custom[scored_custom["risk_tier"] == "High Risk"]))
            c2.metric("Medium Risk Accounts", len(scored_custom[scored_custom["risk_tier"] == "Medium Risk"]))
            c3.metric("Low Risk Accounts", len(scored_custom[scored_custom["risk_tier"] == "Low Risk"]))

            st.markdown("---")
            st.subheader("Scored Portfolio Output")
            display_cols = [c for c in ["customerID", "gender", "tenure", "MonthlyCharges", "TotalCharges", "churn_probability", "risk_tier", "top_recommendation"] if c in scored_custom.columns]
            st.dataframe(scored_custom[display_cols].reset_index(drop=True), use_container_width=True, height=350)

            csv_out = scored_custom.to_csv(index=False)
            st.download_button("Export Results (CSV)",
                                data=csv_out, file_name="churnsight_scored_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Processing error: {e}")
    else:
        st.markdown("### Export Baseline Scored Dataset")
        sample = scored_df[["gender", "tenure", "MonthlyCharges",
                              "churn_probability", "risk_tier", "top_recommendation"]].head(50)
        csv = sample.to_csv(index=False)
        st.download_button("Download Sample Output (CSV)",
                            data=csv, file_name="churnsight_results.csv", mime="text/csv")

# -----------------------------------------------------------------------
# PAGE 4 - MODEL PERFORMANCE
# -----------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Diagnostics")
    st.markdown("<p class='section-subtitle'>Out-of-sample evaluation metrics for the Stacking Ensemble classifier.</p>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", metrics["roc_auc"])
    c2.metric("F1 Score", metrics["f1"])
    c3.metric("Precision", metrics["precision"])
    c4.metric("Recall", metrics["recall"])
    c5.metric("Accuracy", metrics["accuracy"])

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=metrics["fpr"], y=metrics["tpr"],
                                      mode="lines", name=f'AUC = {metrics["roc_auc"]}',
                                      line=dict(color="#6366f1", width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                      line=dict(dash="dash", color="#4b5563")))
        apply_executive_theme(fig_roc, "ROC Receiver Operating Curve")
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        cm = metrics["confusion_matrix"]
        fig_cm = px.imshow(cm, text_auto=True,
                           labels=dict(x="Predicted Class", y="Actual Class"),
                           x=["Retained", "Churned"],
                           y=["Retained", "Churned"],
                           color_continuous_scale=[[0, "#141c2b"], [1, "#3b82f6"]])
        apply_executive_theme(fig_cm, "Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("### Model Architecture")
    st.markdown("""
    - **Base Models**: XGBoost Classifier + LightGBM Classifier
    - **Meta-Learner**: Logistic Regression
    - **Optimization**: 3-fold cross-validated Optuna hyperparameter tuning
    - **Resampling**: SMOTE training balancing
    - **Explainability Engine**: TreeSHAP
    """)

# -----------------------------------------------------------------------
# PAGE 5 - ADVANCED ANALYTICS
# -----------------------------------------------------------------------
elif page == "Advanced Analytics":
    st.title("Advanced Portfolio Analytics")
    st.markdown("<p class='section-subtitle'>Customer health scoring, decision threshold optimization, and trend forecasting.</p>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Health Score", "Threshold Tuner", "Churn Trajectory"])

    with tab1:
        st.subheader("Customer Health Index")
        health_df = scored_df.copy()
        health_df["health_score"] = compute_health_score(health_df)
        health_df["health_label"] = health_df["health_score"].apply(get_health_label)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio Health Avg", f"{health_df['health_score'].mean():.1f}/100")
        c2.metric("Healthy Accounts", len(health_df[health_df["health_label"] == "Healthy"]))
        c3.metric("At-Risk Accounts", len(health_df[health_df["health_label"] == "At Risk"]))
        c4.metric("Critical Accounts", len(health_df[health_df["health_label"] == "Critical"]))

        fig_health = px.histogram(
            health_df, x="health_score", color="health_label",
            color_discrete_map={"Healthy": "#10b981", "At Risk": "#f59e0b",
                                "Struggling": "#f97316", "Critical": "#f43f5e"},
            nbins=30
        )
        apply_executive_theme(fig_health, "Health Score Distribution Across Portfolio")
        st.plotly_chart(fig_health, use_container_width=True)

        st.markdown("#### High-Priority Attention List (Bottom 10)")
        bottom10 = health_df.nsmallest(10, "health_score")[
            ["gender", "tenure", "MonthlyCharges", "churn_probability",
             "risk_tier", "health_score", "health_label"]
        ].reset_index(drop=True)
        st.dataframe(bottom10, use_container_width=True)

    with tab2:
        st.subheader("Decision Cutoff Tuner")
        threshold = st.slider("Classification Threshold", 0.10, 0.90, 0.50, 0.01)
        probs = scored_df["churn_probability"]
        y_true = scored_df["Churn"].values
        y_pred = (probs >= threshold).astype(int)

        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{prec:.3f}")
        c2.metric("Recall", f"{rec:.3f}")
        c3.metric("F1 Score", f"{f1:.3f}")

        cm2 = confusion_matrix(y_true, y_pred)
        fig_cm2 = px.imshow(cm2, text_auto=True,
                            labels=dict(x="Predicted", y="Actual"),
                            x=["Retained", "Churned"],
                            y=["Retained", "Churned"],
                            color_continuous_scale=[[0, "#141c2b"], [1, "#3b82f6"]])
        apply_executive_theme(fig_cm2, f"Confusion Matrix at Threshold {threshold:.2f}")
        st.plotly_chart(fig_cm2, use_container_width=True)

    with tab3:
        st.subheader("12-Month Churn Forecast")
        avg_prob = scored_df["churn_probability"].mean()
        retention = st.slider("Intervention Capture Rate (%)", 0.0, 0.9, 0.3, 0.05)

        no_action = simulate_churn_trend(avg_prob, months=12, retention_rate=0.0)
        with_action = simulate_churn_trend(avg_prob, months=12, retention_rate=retention)

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=no_action["month"], y=no_action["remaining_customers"],
            mode="lines+markers", name="Baseline (No Action)",
            line=dict(color="#f43f5e", width=2)
        ))
        fig_trend.add_trace(go.Scatter(
            x=with_action["month"], y=with_action["remaining_customers"],
            mode="lines+markers", name=f"With {int(retention*100)}% Retention Action",
            line=dict(color="#10b981", width=2)
        ))
        apply_executive_theme(fig_trend, "Projected Customer Base (Per 1,000 Accounts)")
        st.plotly_chart(fig_trend, use_container_width=True)

# -----------------------------------------------------------------------
# PAGE 6 - WHAT-IF SIMULATOR
# -----------------------------------------------------------------------
elif page == "What-if Simulator":
    st.title("What-if Churn Simulator")
    st.markdown("<p class='section-subtitle'>Simulate live churn probability changes by adjusting customer profile variables.</p>", unsafe_allow_html=True)

    def make_gauge(probability):
        color = "#f43f5e" if probability >= 0.7 else "#f59e0b" if probability >= 0.4 else "#10b981"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(probability * 100, 1),
            title={"text": "Churn Risk %", "font": {"size": 16, "color": "#9ca3af"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#4b5563"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#064e3b"},
                    {"range": [40, 70], "color": "#78350f"},
                    {"range": [70, 100], "color": "#881337"},
                ]
            }
        ))
        apply_executive_theme(fig, "")
        fig.update_layout(height=260)
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
        median_charge = getattr(scaler, 'monthly_charges_median_', metrics.get('monthly_charges_median', 64.76))
        row["high_value"] = 1 if inputs["MonthlyCharges"] > median_charge else 0
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

    col_inputs, col_result = st.columns([1, 1])

    with col_inputs:
        st.subheader("Account Parameters")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
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
        st.subheader("Real-Time Prediction")
        try:
            X_input = build_input_vector(inputs, feature_names, scaler)
            prob = model.predict_proba(X_input)[0][1]
            st.plotly_chart(make_gauge(prob), use_container_width=True)
            if prob >= 0.70:
                st.error("HIGH RISK — Action Required")
            elif prob >= 0.40:
                st.warning("MEDIUM RISK — Monitor Profile")
            else:
                st.success("LOW RISK — Stable Customer")
            st.markdown("---")
            st.subheader("Recommended Retention Interventions")
            explainer = get_shap_explainer(model, X_input)
            shap_vals = explainer.shap_values(X_input)[0]
            top_pos, _ = get_top_drivers(shap_vals, feature_names)
            recs = get_recommendations(top_pos)
            for r in recs[:3]:
                st.info(f"{r['icon']} **{r['action']}** — _{r['reason']}_")
        except Exception as e:
            st.error("Simulation error: " + str(e))

# -----------------------------------------------------------------------
# PAGE 7 - ROI SIMULATOR
# -----------------------------------------------------------------------
elif page == "ROI Simulator":
    st.title("ROI & Financial Impact Simulator")
    st.markdown("<p class='section-subtitle'>Estimate financial return on retention program investments.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Financial Model Parameters")
        total_customers = st.number_input("Total Customer Base", value=7000, step=100)
        avg_monthly_revenue = st.number_input("Avg Monthly Revenue ($)", value=65, step=5)
        churn_rate = st.slider("Monthly Churn Rate (%)", 1.0, 30.0, 26.5, 0.5)
        retention_budget = st.number_input("Monthly Retention Budget ($)", value=10000, step=500)
        cost_per_intervention = st.number_input("Cost per Intervention ($)", value=50, step=5)
        success_rate = st.slider("Intervention Success Rate (%)", 10, 90, 30, 5)

    with col2:
        st.subheader("Calculated Financial Impact")
        monthly_churners = int(total_customers * churn_rate / 100)
        revenue_lost_no_action = monthly_churners * avg_monthly_revenue
        max_interventions = int(retention_budget / cost_per_intervention)
        customers_targeted = min(monthly_churners, max_interventions)
        customers_saved = int(customers_targeted * success_rate / 100)
        revenue_saved = customers_saved * avg_monthly_revenue
        net_roi = revenue_saved - retention_budget
        roi_percent = round((net_roi / retention_budget) * 100, 1) if retention_budget > 0 else 0

        c1, c2 = st.columns(2)
        c1.metric("Monthly Churners", f"{monthly_churners:,}")
        c2.metric("Revenue at Risk", f"${revenue_lost_no_action:,.0f}")
        c1.metric("Targeted Accounts", f"{customers_targeted:,}")
        c2.metric("Saved Accounts", f"{customers_saved:,}")
        c1.metric("Monthly Saved Revenue", f"${revenue_saved:,.0f}")
        c2.metric("Net Retention ROI", f"${net_roi:,.0f}", delta=f"{roi_percent}%")

    st.markdown("---")
    st.subheader("12-Month Cumulative ROI Curve")

    months = list(range(1, 13))
    cumulative_saved = [revenue_saved * m for m in months]
    cumulative_spent = [retention_budget * m for m in months]
    cumulative_net = [s - c for s, c in zip(cumulative_saved, cumulative_spent)]

    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(x=months, y=cumulative_saved,
                                  mode="lines+markers", name="Saved Revenue",
                                  line=dict(color="#10b981", width=2)))
    fig_roi.add_trace(go.Scatter(x=months, y=cumulative_spent,
                                  mode="lines+markers", name="Retention Budget Spend",
                                  line=dict(color="#f43f5e", width=2)))
    fig_roi.add_trace(go.Scatter(x=months, y=cumulative_net,
                                  mode="lines+markers", name="Net Portfolio Return",
                                  line=dict(color="#6366f1", width=2, dash="dash")))
    apply_executive_theme(fig_roi, "12-Month Cumulative Financial Return ($)")
    st.plotly_chart(fig_roi, use_container_width=True)

    st.markdown("---")
    st.subheader("Sensitivity Analysis")

    success_rates = list(range(10, 100, 10))
    rois = []
    for sr in success_rates:
        cs = int(customers_targeted * sr / 100)
        rs = cs * avg_monthly_revenue
        rois.append(round((rs - retention_budget) / retention_budget * 100, 1))

    fig_sens = px.bar(x=success_rates, y=rois,
                      labels={"x": "Intervention Success Rate (%)", "y": "Return on Investment (%)"},
                      color=rois,
                      color_continuous_scale=["#f43f5e", "#f59e0b", "#10b981"])
    apply_executive_theme(fig_sens, "ROI Sensitivity to Success Rate")
    st.plotly_chart(fig_sens, use_container_width=True)