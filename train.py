"""Credit Scoring Model: train and evaluate classifiers on a labeled CSV dataset.
For a reproducible demo, run: python generate_demo_data.py
Then: python train.py --data data/credit_demo.csv
Label convention: creditworthy=1, not creditworthy=0.
This is an educational demonstration, not a real lending decision system.
"""
import argparse, json, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib

TARGET = "creditworthy"
NUMERIC = ["age", "annual_income", "debt_to_income", "credit_utilization", "late_payments_12m", "credit_history_years"]
CATEGORICAL = ["employment_type", "home_ownership"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/credit_demo.csv")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.data)
    required = set(NUMERIC + CATEGORICAL + [TARGET])
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df[NUMERIC + CATEGORICAL + [TARGET]].copy()
    df[TARGET] = df[TARGET].astype(int)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", num_pipe, NUMERIC), ("cat", cat_pipe, CATEGORICAL)])
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", min_samples_leaf=2)
    }
    results = {}
    best_name, best_model, best_f1 = None, None, -1
    for name, clf in models.items():
        pipe = Pipeline([("preprocess", prep), ("model", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        result = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "confusion_matrix_labels_0_1": confusion_matrix(y_test, pred, labels=[0,1]).tolist(),
            "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0)
        }
        results[name] = result
        if result["f1_score"] > best_f1:
            best_name, best_model, best_f1 = name, pipe, result["f1_score"]
    joblib.dump(best_model, os.path.join(args.out, "credit_scoring_model.joblib"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "test_rows": len(y_test), "results": results}, f, indent=2)
    print(f"Best model by F1: {best_name}")
    for name, r in results.items():
        print(f"{name}: accuracy={r['accuracy']:.3f}, precision={r['precision']:.3f}, recall={r['recall']:.3f}, F1={r['f1_score']:.3f}, ROC-AUC={r['roc_auc']:.3f}")
    print(f"Saved model and metrics in: {args.out}")

if __name__ == "__main__": main()
