"""Batch prediction helper.

Loads the production pipeline and scores a CSV of out-of-sample applications
(the file Motz Financial Services supplies on presentation day).

Usage:
    python src/predict.py path\\to\\out_of_sample.csv [output.csv]

The input must contain the raw applicant columns (Loan_ID optional). Missing
values are tolerated because imputation lives inside the saved pipeline.
"""
import sys

import joblib
import pandas as pd

from config import MODEL_PATH


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    X = df.drop(columns=[c for c in ("Loan_Status", "Loan_ID") if c in df.columns])
    proba = model.predict_proba(X)[:, 1]
    label = model.predict(X)
    out = df.copy()
    out["Prediction"] = ["Approved" if p == 1 else "Rejected" for p in label]
    out["Approval_Probability"] = proba.round(4)
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <input.csv> [output.csv]")
        raise SystemExit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.rsplit(".", 1)[0] + "_predictions.csv"

    df = pd.read_csv(input_path)
    result = predict_dataframe(df)
    result.to_csv(output_path, index=False)
    print(f"Scored {len(result)} applications -> {output_path}")
    print(result["Prediction"].value_counts())


if __name__ == "__main__":
    main()
