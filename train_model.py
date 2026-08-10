"""
Train the cardiac-risk MVP model and create model.pkl.

Usage:
    python train_model.py "Karachi_Cardiac_Risk_Dataset_1500.xlsx - Patient Dataset.csv"

The script intentionally excludes the four requested prediction targets and ID-only
columns from model inputs to reduce target leakage.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

DEFAULT_CSV = "Karachi_Cardiac_Risk_Dataset_1500.xlsx - Patient Dataset.csv"

TARGETS = [
    "Framingham Score (pts)",
    "Est. Yrs to Cardiac Event",
    "10-Yr CVD Risk (%)",
    "Risk Category",
]
ID_COLS = ["Patient ID", "Karachi Age Alert"]


def load_dataset(path):
    raw = pd.read_csv(path, header=None)
    header_row = next(
        i for i in range(min(20, len(raw)))
        if "Patient ID" in raw.iloc[i].astype(str).tolist()
    )
    df = raw.iloc[header_row + 1:].copy()
    df.columns = raw.iloc[header_row].tolist()
    df = df.loc[:, ~df.columns.isna()].copy().reset_index(drop=True)
    return df


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def main():
    csv_file = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV)
    out_file = Path("model.pkl")

    df = load_dataset(csv_file)
    feature_cols = [c for c in df.columns if c not in TARGETS + ID_COLS]
    X = df[feature_cols].copy()

    # Convert columns that are overwhelmingly numeric.
    for c in feature_cols:
        if X[c].dtype == "object":
            converted = pd.to_numeric(X[c], errors="coerce")
            non_null = X[c].notna().sum()
            if non_null and converted.notna().sum() / non_null >= 0.90:
                X[c] = converted

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    y = {
        "framingham_score": pd.to_numeric(df[TARGETS[0]], errors="coerce"),
        "est_years_to_event": pd.to_numeric(df[TARGETS[1]], errors="coerce"),
        "cvd_risk_pct": pd.to_numeric(df[TARGETS[2]], errors="coerce"),
        "risk_category": df[TARGETS[3]].astype(str),
    }

    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.20,
        random_state=42,
        stratify=y["risk_category"],
    )

    def regressor():
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=350, min_samples_leaf=2,
                random_state=42, n_jobs=-1
            )),
        ])

    def classifier():
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=350, min_samples_leaf=2,
                random_state=42, class_weight="balanced", n_jobs=-1
            )),
        ])

    models = {
        "framingham_score": regressor(),
        "est_years_to_event": regressor(),
        "cvd_risk_pct": regressor(),
        "risk_category": classifier(),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X.iloc[train_idx], y[name].iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        if name == "risk_category":
            metrics[name] = {"accuracy": float(accuracy_score(
                y[name].iloc[test_idx], pred
            ))}
        else:
            metrics[name] = {
                "mae": float(mean_absolute_error(y[name].iloc[test_idx], pred)),
                "r2": float(r2_score(y[name].iloc[test_idx], pred)),
            }

    # Production fit on all rows.
    for name, model in models.items():
        model.fit(X, y[name])

    numeric_defaults = {}
    for c in numeric_cols:
        vals = pd.to_numeric(X[c], errors="coerce")
        numeric_defaults[c] = float(vals.median()) if vals.notna().any() else 0.0

    categorical_options = {
        c: sorted(X[c].dropna().astype(str).unique().tolist())
        for c in categorical_cols
    }

    bundle = {
        "models": models,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_defaults": numeric_defaults,
        "categorical_options": categorical_options,
        "targets": TARGETS,
        "training_rows": int(len(X)),
        "metrics_holdout": metrics,
        "model_version": "MVP-cardiac-risk-rf-v1",
    }
    joblib.dump(bundle, out_file)
    print(f"Saved: {out_file.resolve()}")
    print("Holdout metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
