import shap
import pickle
import numpy as np
import pandas as pd

def load_model():
    with open('models/ensemble_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

def get_shap_explainer(model, X_train_sample):
    """Create SHAP explainer using the XGB sub-model from ensemble"""
    xgb_model = model.estimators_[0]  # XGBoost is first in ensemble
    explainer = shap.TreeExplainer(xgb_model)
    return explainer

def get_shap_values(explainer, X):
    """Get SHAP values for a dataframe"""
    shap_values = explainer.shap_values(X)
    return shap_values

def get_top_drivers(shap_values_row, feature_names, top_n=5):
    """Get top positive and negative SHAP drivers for a single customer"""
    shap_series = pd.Series(shap_values_row, index=feature_names)

    top_positive = shap_series.nlargest(top_n)   # pushing toward churn
    top_negative = shap_series.nsmallest(top_n)  # pushing away from churn

    return top_positive, top_negative

def explain_customer(model, feature_names, customer_df):
    """
    Full explanation pipeline for a single customer row.
    Returns shap values + top drivers.
    """
    explainer = get_shap_explainer(model, customer_df)
    shap_vals = get_shap_values(explainer, customer_df)

    if len(customer_df) == 1:
        row_shap = shap_vals[0]
    else:
        row_shap = shap_vals

    top_pos, top_neg = get_top_drivers(row_shap, feature_names)

    return {
        'shap_values': row_shap,
        'top_churn_drivers': top_pos,
        'top_retention_drivers': top_neg,
        'feature_names': feature_names
    }

def batch_explain(model, feature_names, X_df):
    """Get SHAP values for entire dataframe — used in dashboard table"""
    explainer = get_shap_explainer(model, X_df)
    shap_vals = explainer.shap_values(X_df)
    shap_df = pd.DataFrame(shap_vals, columns=feature_names, index=X_df.index)
    return shap_df