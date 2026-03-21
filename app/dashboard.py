import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import run_preprocessing
from explain import batch_explain, get_shap_explainer, get_shap_values, get_top_drivers
from risk_segmentor import assign_risk_tiers, estimate_revenue_saved
from recommender import get_recommendations, get_bulk_recommendations

st.set_page_config(page_title="ChurnSight", page_icon="telescope", layout="wide")

# -- Load model artifacts ----------------------------------------------
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
    df_raw["TotalCharges"].fillna(df_raw["TotalCharges"].median(), inplace=True)
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

# -- Sidebar navigation ------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/telescope.png", width=60)
st.sidebar.title("ChurnSight")
st.sidebar.markdown("*Explainable Churn Intelligence*")
page = st.sidebar.radio("Navigate", [
    "Overview",
    "Customer Risk Table",
    "Bulk Scorer",
    "Model Performance"
])

model, scaler, feature_names, metrics = load_artifacts()

# ── Custom CSS ────────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()
scored_df, X_test, shap_df, y_test = load_and_score_data()

# -----------------------------------------------------------------------
# PAGE 1 — OVERVIEW
# -----------------------------------------------------------------------
if page == "Overview":
    st.title("ChurnSight — Overview")
    st.markdown("---")

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
        st.subheader("Churn by Contract Type")
        contract_cols = [c for c in scored_df.columns if "Contract" in c]
        if contract_cols:
            fig = px.histogram(scored_df, x=contract_cols[0],
                               color=scored_df["Churn"].map({1: "Churned", 0: "Retained"}),
                               barmode="group", color_discrete_map={"Churned": "#ef4444", "Retained": "#22c55e"})
            st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Risk Tier Distribution")
        tier_counts = scored_df["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        colors = {"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#22c55e"}
        fig2 = px.pie(tier_counts, names="tier", values="count",
                      color="tier", color_discrete_map=colors, hole=0.4)
        st.plotly_chart(fig2, width='stretch')

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Monthly Charges Distribution")
        fig3 = px.box(scored_df, x=scored_df["Churn"].map({1: "Churned", 0: "Retained"}),
                      y="MonthlyCharges",
                      color=scored_df["Churn"].map({1: "Churned", 0: "Retained"}),
                      color_discrete_map={"Churned": "#ef4444", "Retained": "#22c55e"})
        st.plotly_chart(fig3, width='stretch')

    with col4:
        st.subheader("Churn by Tenure Group")
        scored_df["tenure_band"] = pd.cut(scored_df["tenure"],
                                           bins=[0,12,24,48,60,72],
                                           labels=["0-1yr","1-2yr","2-4yr","4-5yr","5-6yr"])
        tenure_churn = scored_df.groupby("tenure_band", observed=False)["Churn"].mean().reset_index()
        tenure_churn.columns = ["Tenure Group", "Churn Rate"]
        fig4 = px.bar(tenure_churn, x="Tenure Group", y="Churn Rate",
                      color="Churn Rate", color_continuous_scale="Reds")
        st.plotly_chart(fig4, width='stretch')

# -----------------------------------------------------------------------
# PAGE 2 — CUSTOMER RISK TABLE
# -----------------------------------------------------------------------
elif page == "Customer Risk Table":
    st.title("Customer Risk Table")
    st.markdown("---")

    tier_filter = st.multiselect("Filter by Risk Tier",
                                  ["High Risk", "Medium Risk", "Low Risk"],
                                  default=["High Risk", "Medium Risk", "Low Risk"])

    filtered = scored_df[scored_df["risk_tier"].isin(tier_filter)].copy()
    filtered["churn_probability"] = filtered["churn_probability"].apply(lambda x: f"{x:.1%}")

    display_cols = ["gender", "tenure", "MonthlyCharges", "TotalCharges",
                    "churn_probability", "risk_tier", "top_recommendation"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        width='stretch',
        height=400
    )

    st.markdown("---")
    st.subheader("?? Deep Dive — Individual Customer")
    customer_idx = st.number_input("Enter customer index (row number above)",
                                    min_value=0, max_value=len(scored_df)-1, value=0)

    if st.button("Analyze Customer"):
        customer = scored_df.iloc[customer_idx]
        prob = float(customer["churn_probability"].strip("%")) / 100 if isinstance(customer["churn_probability"], str) else customer["churn_probability"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Churn Probability", f"{prob:.1%}")
        col2.metric("Risk Tier", customer["risk_tier"])
        col3.metric("Monthly Charges", f"${customer['MonthlyCharges']:.2f}")

        st.markdown("#### ?? Top Churn Drivers (SHAP)")
        if customer_idx < len(shap_df):
            shap_row = shap_df.iloc[customer_idx]
            top_pos = shap_row.nlargest(5)
            top_neg = shap_row.nsmallest(5)

            fig_shap = go.Figure(go.Bar(
                x=top_pos.values.tolist() + top_neg.values.tolist(),
                y=top_pos.index.tolist() + top_neg.index.tolist(),
                orientation="h",
                marker_color=["#ef4444"]*5 + ["#22c55e"]*5
            ))
            fig_shap.update_layout(title="SHAP Feature Contributions",
                                   xaxis_title="SHAP Value", height=400)
            st.plotly_chart(fig_shap, width='stretch')

            st.markdown("#### ?? Retention Recommendations")
            recs = get_recommendations(top_pos)
            for r in recs:
                priority_color = {"High": "??", "Medium": "??", "Low": "??"}
                pc = priority_color.get(r["priority"], "?")
                st.info(f"{r['icon']} **{r['action']}**  \n_{r['reason']}_ {pc} {r['priority']} Priority")

# -----------------------------------------------------------------------
# PAGE 3 — BULK SCORER
# -----------------------------------------------------------------------
elif page == "Bulk Scorer":
    st.title("?? Bulk CSV Scorer")
    st.markdown("Upload a customer CSV to get churn scores and recommendations.")
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.write("Preview:", df_upload.head())
        st.info("? File uploaded! Full bulk scoring pipeline coming in next version.")
    else:
        st.markdown("### ?? Download sample scored data")
        sample = scored_df[["gender", "tenure", "MonthlyCharges",
                              "churn_probability", "risk_tier", "top_recommendation"]].head(50)
        csv = sample.to_csv(index=False)
        st.download_button("?? Download Sample Results (CSV)",
                            data=csv, file_name="churnsight_results.csv", mime="text/csv")

# -----------------------------------------------------------------------
# PAGE 4 — MODEL PERFORMANCE
# -----------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", metrics["roc_auc"])
    c2.metric("F1 Score", metrics["f1"])
    c3.metric("Precision", metrics["precision"])
    c4.metric("Recall", metrics["recall"])
    c5.metric("Accuracy", metrics["accuracy"])

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=metrics["fpr"], y=metrics["tpr"],
                                      mode="lines", name=f'AUC = {metrics["roc_auc"]}',
                                      line=dict(color="#6366f1", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                      line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(xaxis_title="False Positive Rate",
                               yaxis_title="True Positive Rate", height=400)
        st.plotly_chart(fig_roc, width='stretch')

    with col2:
        st.subheader("Confusion Matrix")
        cm = metrics["confusion_matrix"]
        fig_cm = px.imshow(cm, text_auto=True,
                           labels=dict(x="Predicted", y="Actual"),
                           x=["Not Churned", "Churned"],
                           y=["Not Churned", "Churned"],
                           color_continuous_scale="Blues")
        st.plotly_chart(fig_cm, width='stretch')

    st.subheader("?? Model Notes")
    st.markdown("""
    - **Model**: XGBoost + LightGBM Soft Voting Ensemble
    - **Tuning**: Optuna (20 trials per model)
    - **Imbalance**: Handled via SMOTE on training set only
    - **Explainability**: SHAP TreeExplainer on XGBoost sub-model
    - **Dataset**: Telco Customer Churn (Kaggle)
    """)
