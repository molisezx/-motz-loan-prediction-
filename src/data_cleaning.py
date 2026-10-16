"""Task 1: Data gathering and cleaning.

Performs:
  a. Missing value treatment
  b. Outlier detection and treatment (IQR winsorisation)
  c. Descriptive / statistical analysis

Outputs a cleaned dataset, a JSON cleaning report and exploratory figures.
"""
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import (
    CLEAN_DATA,
    CLEANING_REPORT,
    FIGURES_DIR,
    RAW_DATA,
    TARGET,
)

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120

NUMERIC_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area", "Credit_History"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA)
    return df


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({"missing_count": missing, "missing_pct": pct}).query("missing_count > 0")


def plot_missing(df: pd.DataFrame, path) -> None:
    missing = report_missing(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(missing.index, missing["missing_pct"], color="#e07a5f")
    ax.set_ylabel("Missing (%)")
    ax.set_title("Missing values per column (raw data)")
    for i, v in enumerate(missing["missing_pct"]):
        ax.text(i, v + 0.1, f"{v}%", ha="center", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def detect_outliers_iqr(series: pd.Series) -> tuple[float, float, int]:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((series < lower) | (series > upper)).sum())
    return float(lower), float(upper), count


def impute(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Mode-impute categoricals, median-impute numerics."""
    strategy = {}
    for col in [c for c in CATEGORICAL_COLS if c in df.columns]:
        if df[col].isnull().any():
            fill = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(fill)
            strategy[col] = f"mode -> {fill}"
    for col in [c for c in NUMERIC_COLS if c in df.columns]:
        if df[col].isnull().any():
            fill = float(df[col].median())
            df[col] = df[col].fillna(fill)
            strategy[col] = f"median -> {fill:.0f}"
    return df, strategy


def treat_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Winsorise numeric columns at the 1.5*IQR fences.

    Columns with a zero inter-quartile range (near-constant, e.g. a loan term
    that is 360 months for most applications) are left untouched: their fences
    collapse onto the median and clipping would flatten the column to a single
    value, destroying real signal rather than removing outliers.
    """
    details = {}
    for col in NUMERIC_COLS:
        lower, upper, n_out = detect_outliers_iqr(df[col])
        if upper <= lower:
            details[col] = {
                "lower_fence": round(lower, 1),
                "upper_fence": round(upper, 1),
                "outliers_capped": 0,
                "note": "skipped (zero inter-quartile range)",
            }
            continue
        df[col] = df[col].clip(lower=lower, upper=upper)
        details[col] = {"lower_fence": round(lower, 1), "upper_fence": round(upper, 1), "outliers_capped": n_out}
    return df, details


def plot_boxplots(before: pd.DataFrame, after: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_COLS, strict=False):
        sns.boxplot(x=before[col], ax=ax, color="#f4a261")
        ax.set_title(f"{col} (raw)")
    fig.suptitle("Outlier inspection before cleaning", fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_histograms(df: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_COLS, strict=False):
        sns.histplot(df[col], kde=True, ax=ax, color="#2a9d8f")
        ax.set_title(col)
    fig.suptitle("Distribution of numeric features (clean data)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, path) -> None:
    num = df[NUMERIC_COLS + ["Credit_History"]].copy()
    num[TARGET] = (df[TARGET] == "Y").astype(int)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_target(df: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(x=TARGET, data=df, ax=axes[0], hue=TARGET, legend=False)
    axes[0].set_title("Loan status distribution")
    order = ["Semiurban", "Urban", "Rural"]
    sns.countplot(x="Property_Area", hue=TARGET, data=df, order=order, ax=axes[1])
    axes[1].set_title("Loan status by property area")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    df = load_raw()
    raw = df.copy()

    missing_report = report_missing(df)
    plot_missing(df, FIGURES_DIR / "01_missing_values.png")

    outlier_details = {
        col: {
            "lower_fence": round(detect_outliers_iqr(df[col])[0], 1),
            "upper_fence": round(detect_outliers_iqr(df[col])[1], 1),
            "outliers_detected": detect_outliers_iqr(df[col])[2],
        }
        for col in NUMERIC_COLS
    }
    plot_boxplots(raw, raw, FIGURES_DIR / "02_outliers_raw.png")

    df, impute_strategy = impute(df)
    df, outlier_details = treat_outliers(df)

    plot_histograms(df, FIGURES_DIR / "03_feature_distributions.png")
    plot_correlation(df, FIGURES_DIR / "04_correlation_heatmap.png")
    plot_target(df, FIGURES_DIR / "05_target_distribution.png")

    descriptive = df[NUMERIC_COLS].describe().round(2).to_dict()
    value_counts = {c: df[c].value_counts().to_dict() for c in CATEGORICAL_COLS}
    value_counts = {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in value_counts.items()}

    df.to_csv(CLEAN_DATA, index=False)

    report = {
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(df)),
        "missing_before": missing_report["missing_count"].to_dict(),
        "missing_after": int(df.isnull().sum().sum()),
        "imputation_strategy": impute_strategy,
        "outliers": outlier_details,
        "descriptive_stats": descriptive,
        "categorical_value_counts": value_counts,
        "target_distribution": {k: int(v) for k, v in df[TARGET].value_counts().items()},
    }
    with open(CLEANING_REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"Cleaned data written to {CLEAN_DATA}  shape={df.shape}")
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    print(f"Cleaning report written to {CLEANING_REPORT}")


if __name__ == "__main__":
    main()
