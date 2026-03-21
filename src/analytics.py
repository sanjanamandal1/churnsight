import pandas as pd
import numpy as np

def compute_health_score(df):
    score = pd.Series(100.0, index=df.index)
    score -= df["churn_probability"] * 40
    score += (df["tenure"] / 72) * 20
    score -= (df["MonthlyCharges"] / 120.0) * 20
    if "Contract_One year" in df.columns:
        score += df["Contract_One year"] * 10
    if "Contract_Two year" in df.columns:
        score += df["Contract_Two year"] * 15
    if "TechSupport_Yes" in df.columns:
        score += df["TechSupport_Yes"] * 5
    return score.clip(0, 100).round(1)

def get_health_label(score):
    if score >= 75:
        return "Healthy"
    elif score >= 50:
        return "At Risk"
    elif score >= 25:
        return "Struggling"
    else:
        return "Critical"

def simulate_churn_trend(churn_probability_mean, months=12, retention_rate=0.0):
    remaining = 1000.0
    trend = []
    for month in range(1, months + 1):
        effective_churn = churn_probability_mean * (1 - retention_rate)
        churned = remaining * effective_churn
        remaining -= churned
        trend.append({
            "month": month,
            "remaining_customers": round(remaining),
            "churned_this_month": round(churned)
        })
    return pd.DataFrame(trend)