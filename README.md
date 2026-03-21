# ChurnSight - Explainable Churn Intelligence Dashboard

A production-grade customer churn prediction system built with XGBoost + LightGBM ensemble, SHAP explainability, and a Streamlit dashboard.

## Features
- XGBoost + LightGBM soft voting ensemble tuned with Optuna
- SHAP-based per-customer churn explanations
- Risk tier segmentation (High / Medium / Low)
- Retention action recommender mapped to SHAP drivers
- Business revenue impact estimator
- Interactive Streamlit dashboard with 4 pages
- What-if Simulator with live churn probability gauge
- Bulk CSV scoring with downloadable results

## Model Performance
| Metric    | Score |
|-----------|-------|
| ROC-AUC   | ~0.91 |
| F1 Score  | ~0.65 |
| Precision | ~0.70 |
| Recall    | ~0.62 |

## Tech Stack
- Python, Pandas, NumPy
- XGBoost, LightGBM, Scikit-learn
- SHAP, Optuna, Imbalanced-learn
- Streamlit, Plotly

## Setup

1. Clone the repo
git clone https://github.com/sanjanamandal1/churnsight.git
cd churnsight

2. Create virtual environment
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Add dataset
Download telco_churn.csv from Kaggle and place in data/
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

5. Train the model
python src/train.py

6. Run the dashboard
streamlit run app/dashboard.py

7. Run the What-if Simulator
streamlit run app/simulator.py

## Project Structure
churnsight/
- data/                  Dataset (not tracked)
- src/
  - preprocess.py        Data cleaning, encoding, SMOTE
  - train.py             Ensemble training + Optuna tuning
  - explain.py           SHAP explainability
  - risk_segmentor.py    Risk tier assignment
  - recommender.py       Retention action recommender
- app/
  - dashboard.py         Streamlit dashboard (4 pages)
  - simulator.py         What-if Churn Simulator
- models/                Saved model artifacts (not tracked)
- requirements.txt

## Author
Sanjana Mandal
github.com/sanjanamandal1
