import pandas as pd
import numpy as np

def assign_risk_tiers(churn_probabilities):
    probs = list(churn_probabilities)
    tiers = []
    for prob in probs:
        if prob >= 0.70:
            tiers.append('High Risk')
        elif prob >= 0.40:
            tiers.append('Medium Risk')
        else:
            tiers.append('Low Risk')
    return tiers

def get_risk_summary(df):
    summary = df.groupby('risk_tier').agg(
        customer_count=('churn_probability', 'count'),
        avg_monthly_charges=('MonthlyCharges', 'mean'),
        total_revenue_at_risk=('MonthlyCharges', 'sum')
    ).reset_index()
    return summary

def get_high_risk_customers(df, top_n=50):
    return df[df['risk_tier'] == 'High Risk'].sort_values('churn_probability', ascending=False).head(top_n)

def estimate_revenue_saved(df, retention_rate=0.3):
    high_risk = df[df['risk_tier'] == 'High Risk']
    return round(high_risk['MonthlyCharges'].sum() * retention_rate, 2)
