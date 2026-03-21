import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Fix TotalCharges — it has spaces instead of nulls
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Drop customerID — not a feature
    df.drop(columns=['customerID'], inplace=True)

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Tenure buckets
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5-6yr']
    )

    # High value customer flag
    df['high_value'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)

    # Interaction feature
    df['charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

    return df

def encode_features(df: pd.DataFrame):
    df = df.copy()

    # Label encode binary columns
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'tenure_group']
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # One-hot encode multi-class columns
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    return df

def scale_and_split(df: pd.DataFrame):
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Scale numerical features
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_tenure']
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE — fix class imbalance on training set only
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    print(f"✅ Training samples after SMOTE: {X_train_res.shape[0]}")
    print(f"✅ Test samples: {X_test.shape[0]}")
    print(f"✅ Features: {X_train_res.shape[1]}")

    return X_train_res, X_test, y_train_res, y_test, scaler, X.columns.tolist()

def run_preprocessing(path: str):
    df = load_data(path)
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_features(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = scale_and_split(df)
    return X_train, X_test, y_train, y_test, scaler, feature_names

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, features = run_preprocessing("data/telco_churn.csv")
    print("Preprocessing complete!")
    print(f"Features: {features}")