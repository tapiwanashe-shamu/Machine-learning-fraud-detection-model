"""
Fraud Detection on PaySim Data — Steps 1-4
1. Load + inspect data
2. Clean + prep
3. Baseline model (Logistic Regression) + XGBoost comparison
4. Exports traceable predictions to csv for AI audit review

Run: pip install pandas numpy scikit-learn xgboost
     python fraud_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# ---------- STEP 1: LOAD + INSPECT ----------
print("Loading data...")

# Filename
DATA_FILE = "C:/Users/User/Documents/DATA ANALYTICS/AI-ML Projects/paysim_fraud_detection_dataset/PS_20174392719_1491204439457_log.csv"
df = pd.read_csv(DATA_FILE, engine = "pyarrow")

print(f"\nShape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFraud rate: {df['isFraud'].mean()*100:.4f}%")
print(f"Fraud count: {df['isFraud'].sum()} out of {len(df)}")
print(f"\nTransaction types:\n{df['type'].value_counts()}")
print(f"\nFraud by type:\n{df[df['isFraud']==1]['type'].value_counts()}")

# ---------- STEP 2: CLEAN + PREP ----------
print("\n" + "="*50)
print("Cleaning and prepping data...")

# Engineered balance-delta features. PaySim's balance fields don't always
# update consistently, and the size of that inconsistency can be one of the
# strongest indicators of fraud in the dataset.
df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

# Kept identifiers and the original index off to the side, 
# but will be used later to trace flagged predictions back to
# their specific transactions for the audit-style report. 
id_columns = df[["nameOrig", "nameDest"]].copy()
df_clean = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"])

# One-hot encode transaction type
df_clean = pd.get_dummies(df_clean, columns=["type"], drop_first=True)

# Separate features and target
X = df_clean.drop(columns=["isFraud"])
y = df_clean["isFraud"]

print(f"Final feature columns: {list(X.columns)}")

# Stratified split keeps the same fraud ratio in train and test.
train_idx, test_idx = train_test_split(
    X.index, test_size=0.2, stratify=y, random_state=42
)
X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]
ids_train, ids_test = id_columns.loc[train_idx], id_columns.loc[test_idx]

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Train fraud rate: {y_train.mean()*100:.4f}%")
print(f"Test fraud rate: {y_test.mean()*100:.4f}%")

# ---------- STEP 3: BASELINE MODEL ----------
print("\n" + "="*50)
print("Training Logistic Regression baseline...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)

print("\nLogistic Regression Results:")
print(classification_report(y_test, y_pred_lr, target_names=["Not Fraud", "Fraud"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

print("\n" + "="*50)
print("Training XGBoost...")

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # handles imbalance
    eval_metric="aucpr",
    random_state=42,
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("\nXGBoost Results:")
print(classification_report(y_test, y_pred_xgb, target_names=["Not Fraud", "Fraud"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

print("\nAnalysis complete. Compare precision/recall/F1 between the two models above.")

# ---------- STEP 4: SAVE TRACEABLE PREDICTIONS ----------
# Used to audit every test transaction, its true label, both models' predictions, 
# and the account identifiers, so a flagged row can be traced back to an actual transaction.
predictions_out = X_test.copy()
predictions_out["nameOrig"] = ids_test["nameOrig"]
predictions_out["nameDest"] = ids_test["nameDest"]
predictions_out["isFraud_actual"] = y_test
predictions_out["isFraud_pred_logreg"] = y_pred_lr
predictions_out["isFraud_pred_xgb"] = y_pred_xgb

predictions_out.to_csv("test_predictions_traceable.csv", index=True)
print(f"\nSaved traceable predictions to test_predictions_traceable.csv ({len(predictions_out)} rows).")