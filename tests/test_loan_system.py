"""Automated tests for the loan-prediction system.

Run with:  pytest
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import MODEL_PATH  # noqa: E402
from explain import background_reference, reason_codes  # noqa: E402
from evaluate import metrics_at_threshold, optimize_threshold_cost  # noqa: E402
from generate_dataset import generate_dataset  # noqa: E402
from models import all_candidates  # noqa: E402
from preprocessing import build_pipeline, engineer_features  # noqa: E402


def _toy_frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Gender": rng.choice(["Male", "Female"], n),
            "Married": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["0", "1", "2", "3+"], n),
            "Education": rng.choice(["Graduate", "Not Graduate"], n),
            "Self_Employed": rng.choice(["No", "Yes"], n),
            "ApplicantIncome": rng.integers(2000, 12000, n).astype(float),
            "CoapplicantIncome": rng.integers(0, 6000, n).astype(float),
            "LoanAmount": rng.integers(30000, 500000, n).astype(float),
            "Loan_Amount_Term": rng.choice([180, 240, 360], n).astype(float),
            "Credit_History": rng.choice([1.0, 0.0], n, p=[0.8, 0.2]),
            "Property_Area": rng.choice(["Urban", "Semiurban", "Rural"], n),
        }
    )
    df["Loan_Status"] = np.where(df["Credit_History"] == 1.0, 1, 0)
    flip = rng.random(n) < 0.1
    df.loc[flip, "Loan_Status"] = 1 - df.loc[flip, "Loan_Status"]
    return df


# ----------------------------------------------------------------- data
def test_generate_dataset_columns_and_missing():
    df = generate_dataset(n=300, seed=1)
    assert len(df) == 300
    for col in ["Loan_ID", "Gender", "Credit_History", "LoanAmount", "Loan_Status"]:
        assert col in df.columns
    assert df["Loan_Status"].isin(["Y", "N"]).all()
    assert df.isnull().sum().sum() > 0  # raw data must contain missing values


# --------------------------------------------------- feature engineering
def test_engineer_features_adds_derived_columns():
    df = _toy_frame().drop(columns=["Loan_Status"])
    out = engineer_features(df)
    for col in ["TotalIncome", "LoanAmount_log", "EMI", "Income_to_Loan_Ratio"]:
        assert col in out.columns
    assert np.isclose(out["TotalIncome"].iloc[0], df["ApplicantIncome"].iloc[0] + df["CoapplicantIncome"].iloc[0])


# ------------------------------------------------------ pipeline training
def test_pipeline_fit_and_predict():
    df = _toy_frame()
    X, y = df.drop(columns=["Loan_Status"]), df["Loan_Status"]
    pipe = build_pipeline(all_candidates()["Logistic Regression"])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    assert set(np.unique(preds)).issubset({0, 1})
    assert proba.shape[0] == len(X)
    assert 0.5 < (preds == y).mean()  # better than chance


def test_pipeline_handles_missing_values():
    df = _toy_frame()
    df.loc[df.index[:15], "Credit_History"] = np.nan
    df.loc[df.index[:10], "LoanAmount"] = np.nan
    X, y = df.drop(columns=["Loan_Status"]), df["Loan_Status"]
    pipe = build_pipeline(all_candidates()["Logistic Regression"])
    pipe.fit(X, y)  # must not raise
    assert pipe.predict(X).shape[0] == len(X)


# ------------------------------------------------------------ explanation
def test_reason_codes_linear_and_occlusion():
    df = _toy_frame()
    X, y = df.drop(columns=["Loan_Status"]), df["Loan_Status"]
    record = X.iloc[[0]]

    linear = build_pipeline(all_candidates()["Logistic Regression"])
    linear.fit(X, y)
    r1 = reason_codes(linear, record, background=X, top_k=5)
    assert len(r1["top_reasons"]) == 5
    assert r1["method"].startswith("linear")

    ensemble = build_pipeline(all_candidates()["Soft Voting Ensemble"])
    ensemble.fit(X, y)
    r2 = reason_codes(ensemble, record, background=X, top_k=5)
    assert len(r2["top_reasons"]) == 5
    assert r2["method"].startswith("occlusion")
    assert all("feature" in r for r in r2["top_reasons"])


def test_background_reference_types():
    ref = background_reference(_toy_frame().drop(columns=["Loan_Status"]))
    assert isinstance(ref["ApplicantIncome"], float)
    assert ref["Gender"] in {"Male", "Female"}


# ------------------------------------------------------------- threshold
def test_threshold_optimisation_and_metrics():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 400)
    y_prob = np.clip(0.20 + 0.60 * y_true + rng.normal(0, 0.22, 400), 0, 1)
    cost = optimize_threshold_cost(y_true, y_prob, cost_fn=5, cost_fp=1)
    assert 0.05 <= cost["best_threshold"] <= 0.95
    assert cost["min_expected_cost"] <= (5 * y_true.sum())  # better than rejecting everyone
    m = metrics_at_threshold(y_true, y_prob, 0.5)
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(m)
    assert 0.0 <= m["accuracy"] <= 1.0


# ----------------------------------------------------- saved artefacts
@pytest.mark.skipif(not MODEL_PATH.exists(), reason="production model not built yet")
def test_saved_model_predicts():
    import joblib

    model = joblib.load(MODEL_PATH)
    record = _toy_frame().drop(columns=["Loan_Status"]).iloc[[0]]
    proba = model.predict_proba(record)[:, 1]
    assert 0.0 <= proba[0] <= 1.0
