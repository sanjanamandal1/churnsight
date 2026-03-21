import numpy as np
import pickle
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix,
                             roc_curve)
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from preprocess import run_preprocessing

def objective_xgb(trial, X_train, y_train, X_test, y_test):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)

def objective_lgbm(trial, X_train, y_train, X_test, y_test):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbose': -1
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)

def tune_models(X_train, y_train, X_test, y_test, n_trials=20):
    print("🔍 Tuning XGBoost...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(
        lambda trial: objective_xgb(trial, X_train, y_train, X_test, y_test),
        n_trials=n_trials
    )
    best_xgb_params = study_xgb.best_params
    best_xgb_params.update({'use_label_encoder': False,
                             'eval_metric': 'logloss', 'random_state': 42})

    print("🔍 Tuning LightGBM...")
    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(
        lambda trial: objective_lgbm(trial, X_train, y_train, X_test, y_test),
        n_trials=n_trials
    )
    best_lgbm_params = study_lgbm.best_params
    best_lgbm_params.update({'random_state': 42, 'verbose': -1})

    print(f"✅ Best XGB AUC: {study_xgb.best_value:.4f}")
    print(f"✅ Best LGBM AUC: {study_lgbm.best_value:.4f}")

    return best_xgb_params, best_lgbm_params

def build_ensemble(best_xgb_params, best_lgbm_params):
    xgb = XGBClassifier(**best_xgb_params)
    lgbm = LGBMClassifier(**best_lgbm_params)

    ensemble = VotingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm)],
        voting='soft'
    )
    return ensemble

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'roc_auc': round(roc_auc_score(y_test, y_prob), 4),
        'f1': round(f1_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall': round(recall_score(y_test, y_pred), 4),
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    metrics['fpr'] = fpr.tolist()
    metrics['tpr'] = tpr.tolist()

    print("\n📊 Model Performance:")
    print(f"  ROC-AUC  : {metrics['roc_auc']}")
    print(f"  F1 Score : {metrics['f1']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall   : {metrics['recall']}")
    print(f"  Accuracy : {metrics['accuracy']}")

    return metrics

def save_model(model, scaler, feature_names, metrics):
    os.makedirs('models', exist_ok=True)
    with open('models/ensemble_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    with open('models/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print("\n✅ Model and artifacts saved to models/")

def train_pipeline(data_path='data/telco_churn.csv'):
    print("🚀 Starting ChurnSight Training Pipeline\n")

    # Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names = run_preprocessing(data_path)

    # Tune
    best_xgb, best_lgbm = tune_models(X_train, y_train, X_test, y_test, n_trials=20)

    # Build ensemble
    ensemble = build_ensemble(best_xgb, best_lgbm)

    # Train final model
    print("\n🏋️ Training final ensemble...")
    ensemble.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(ensemble, X_test, y_test)

    # Save
    save_model(ensemble, scaler, feature_names, metrics)

    return ensemble, metrics

if __name__ == "__main__":
    import sys
    sys.path.append('src')
    train_pipeline()