"""Task 2: Data pre-processing -- feature extraction and feature tuning.

Feature extraction
------------------
Derived domain features:
  * TotalIncome             = ApplicantIncome + CoapplicantIncome
  * LoanAmount_log          = log1p(LoanAmount)   (reduces right skew)
  * EMI                     = LoanAmount / Loan_Amount_Term (monthly burden)
  * Income_to_Loan_Ratio    = TotalIncome / LoanAmount (affordability)

Feature tuning
--------------
  * Skewness measured before/after the log transform.
  * Correlation of each candidate feature with the target used to rank and
    prune redundant columns.

A reusable :func:`engineer_features` transform and a scikit-learn
``ColumnTransformer`` are exported so that training, the dashboard and the
out-of-sample prediction script all apply *identical* transformations.
"""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES,
    CLEAN_DATA,
    FEATURES_DATA,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
    TEST_DATA,
    TEST_SIZE,
    TRAIN_DATA,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived financial features. Safe on data with missing values."""
    df = df.copy()
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanAmount_log"] = np.log1p(df["LoanAmount"])
    df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"]
    loan = df["LoanAmount"].replace(0, np.nan)
    df["Income_to_Loan_Ratio"] = df["TotalIncome"] / loan
    return df


def build_preprocessor() -> ColumnTransformer:
    """Column-wise imputation + encoding + scaling, fitted at train time."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(clf):
    """Full end-to-end pipeline: feature extraction -> impute/encode/scale -> classifier.

    Because engineering and imputation live inside the pipeline, callers can
    feed *raw* applicant records (including missing values) directly.
    """
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(engineer_features, validate=False)),
            ("prep", build_preprocessor()),
            ("clf", clf),
        ]
    )


def _skew_table(df: pd.DataFrame) -> dict:
    return {
        "ApplicantIncome_skew": round(float(df["ApplicantIncome"].skew()), 3),
        "CoapplicantIncome_skew": round(float(df["CoapplicantIncome"].skew()), 3),
        "LoanAmount_skew_raw": round(float(df["LoanAmount"].skew()), 3),
        "LoanAmount_log_skew": round(float(np.log1p(df["LoanAmount"]).skew()), 3),
    }


def main() -> None:
    df = pd.read_csv(CLEAN_DATA)

    skew_before = _skew_table(df)
    df_features = engineer_features(df)
    skew_after = _skew_table(df_features)

    y = (df_features[TARGET] == "Y").astype(int)
    X_features = df_features.drop(columns=[TARGET])

    correlations = (
        X_features.select_dtypes(include=[np.number])
        .corrwith(y)
        .sort_values(ascending=False)
        .round(3)
        .to_dict()
    )

    # Model splits keep the *raw* applicant columns; engineering happens inside
    # the pipeline so training and inference apply identical transforms.
    X = df.drop(columns=[TARGET, "Loan_ID"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    df_features.to_csv(FEATURES_DATA, index=False)
    train = X_train.copy()
    train[TARGET] = y_train.values
    test = X_test.copy()
    test[TARGET] = y_test.values
    train.to_csv(TRAIN_DATA, index=False)
    test.to_csv(TEST_DATA, index=False)

    info = {
        "n_raw_features": int(X.shape[1]),
        "n_engineered_features": int(X_features.shape[1]),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "skew_before": skew_before,
        "skew_after": skew_after,
        "feature_correlation_with_target": correlations,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "class_balance_train": y_train.value_counts(normalize=True).round(3).to_dict(),
    }
    with open(REPORTS_DIR / "preprocessing_report.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    print(f"Engineered features: {X_features.shape[1]} columns (from {X.shape[1]} raw)")
    print(f"Feature/target correlations:\n{pd.Series(correlations)}")
    print(f"Train rows={len(train)}  Test rows={len(test)}")
    print(f"Features saved to {FEATURES_DATA}")
    print(f"Skew (LoanAmount): raw={skew_before['LoanAmount_skew_raw']} -> log={skew_before['LoanAmount_log_skew']}")


if __name__ == "__main__":
    main()
