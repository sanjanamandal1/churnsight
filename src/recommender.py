import pandas as pd

def get_recommendations(top_churn_drivers):
    recommendations = []
    drivers = top_churn_drivers.index.tolist()

    for driver in drivers:
        if "MonthlyCharges" in driver:
            recommendations.append({"icon": "💰", "action": "Offer 15% discount on current plan", "reason": "High monthly charges are pushing this customer away", "priority": "High"})
        if "Contract" in driver:
            recommendations.append({"icon": "📋", "action": "Propose annual contract with waived setup fee", "reason": "Month-to-month customers churn 3x more than annual", "priority": "High"})
        if "tenure" in driver:
            recommendations.append({"icon": "🎁", "action": "Send loyalty reward — free month or service upgrade", "reason": "Low tenure customers have not yet built brand loyalty", "priority": "Medium"})
        if "TechSupport" in driver:
            recommendations.append({"icon": "🛠️", "action": "Offer free 3-month TechSupport add-on", "reason": "Lack of tech support correlates strongly with churn", "priority": "High"})
        if "OnlineSecurity" in driver:
            recommendations.append({"icon": "🔒", "action": "Offer complimentary Online Security package for 2 months", "reason": "Customers without security features churn more often", "priority": "Medium"})
        if "PaymentMethod" in driver:
            recommendations.append({"icon": "💳", "action": "Incentivize switch to auto-pay with a $5 monthly credit", "reason": "Manual payment methods correlate with higher churn", "priority": "Medium"})
        if "InternetService" in driver:
            recommendations.append({"icon": "🌐", "action": "Offer free speed upgrade for 3 months", "reason": "Internet service dissatisfaction is a key churn trigger", "priority": "High"})
        if "Dependents" in driver or "Partner" in driver:
            recommendations.append({"icon": "👨‍👩‍👧", "action": "Offer family plan bundle discount", "reason": "Customers without family ties are more likely to switch", "priority": "Medium"})
        if "Streaming" in driver:
            recommendations.append({"icon": "🎬", "action": "Offer free streaming bundle upgrade for 1 month", "reason": "Streaming dissatisfaction is linked to churn", "priority": "Low"})

    seen = set()
    unique_recs = []
    for r in recommendations:
        if r["action"] not in seen:
            seen.add(r["action"])
            unique_recs.append(r)

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    unique_recs.sort(key=lambda x: priority_order.get(x["priority"], 3))

    if not unique_recs:
        unique_recs.append({"icon": "📞", "action": "Schedule proactive retention call", "reason": "General churn risk detected", "priority": "Medium"})

    return unique_recs

def get_bulk_recommendations(df, shap_df):
    recs = []
    for idx in df.index:
        if idx in shap_df.index:
            top_drivers = shap_df.loc[idx].nlargest(5)
            customer_recs = get_recommendations(top_drivers)
            top_rec = customer_recs[0]["action"] if customer_recs else "Schedule retention call"
        else:
            top_rec = "Schedule retention call"
        recs.append(top_rec)
    df = df.copy()
    df["top_recommendation"] = recs
    return df
