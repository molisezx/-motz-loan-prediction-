"""Explainable AI for the loan model.

Provides:
  * per-applicant *reason codes* (additive feature contributions) for the linear
    production model, so every decision can be justified to a customer/auditor;
  * global *permutation importance*, which is model-agnostic;
  * reusable plotting helpers.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120

# Human-friendly renaming of the one-hot encoded feature names.
_PRETTY = {
    "num__ApplicantIncome": "Applicant income",
    "num__CoapplicantIncome": "Co-applicant income",
    "num__LoanAmount": "Loan amount",
    "num__Loan_Amount_Term": "Loan term",
    "num__Credit_History": "Credit history",
    "num__TotalIncome": "Total income",
    "num__LoanAmount_log": "Loan amount (log)",
    "num__EMI": "Monthly instalment (EMI)",
    "num__Income_to_Loan_Ratio": "Income-to-loan ratio",
    "cat__Gender_Female": "Gender: female",
    "cat__Gender_Male": "Gender: male",
    "cat__Married_No": "Not married",
    "cat__Married_Yes": "Married",
    "cat__Dependents_0": "Dependents: 0",
    "cat__Dependents_1": "Dependents: 1",
    "cat__Dependents_2": "Dependents: 2",
    "cat__Dependents_3+": "Dependents: 3+",
    "cat__Education_Graduate": "Graduate",
    "cat__Education_Not Graduate": "Not a graduate",
    "cat__Self_Employed_No": "Not self-employed",
    "cat__Self_Employed_Yes": "Self-employed",
    "cat__Property_Area_Rural": "Property: rural",
    "cat__Property_Area_Semiurban": "Property: semiurban",
    "cat__Property_Area_Urban": "Property: urban",
}


def pretty(name: str) -> str:
    return _PRETTY.get(name, name.split("__", 1)[-1].replace("_", " "))


def feature_names(pipe) -> list[str]:
    return list(pipe.named_steps["prep"].get_feature_names_out())


def background_reference(df: pd.DataFrame) -> dict:
    """Reference values used by the occlusion explainer (median / mode)."""
    ref = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            ref[col] = float(df[col].median())
        else:
            ref[col] = df[col].mode(dropna=True).iloc[0]
    return ref


def _linear_reason_codes(pipe, record: pd.DataFrame, top_k: int) -> dict:
    clf = pipe.named_steps["clf"]
    z = pipe.named_steps["prep"].transform(pipe.named_steps["features"].transform(record))[0]
    coef = clf.coef_.ravel()
    contributions = coef * z
    names = feature_names(pipe)
    rows = [
        {"feature": pretty(n), "contribution": float(c), "value": float(v),
         "direction": "approve" if c >= 0 else "reject"}
        for n, c, v in zip(names, contributions, z, strict=False)
    ]
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return {
        "method": "linear additive (log-odds)",
        "intercept": float(clf.intercept_[0]),
        "decision_score": float(clf.intercept_[0] + contributions.sum()),
        "top_reasons": rows[:top_k],
    }


def _occlusion_reason_codes(pipe, record: pd.DataFrame, reference: dict, top_k: int) -> dict:
    """Model-agnostic explanation: change in approval probability when each raw
    feature is replaced by its reference (median/mode) value."""
    base_p = float(pipe.predict_proba(record)[:, 1][0])
    rows = []
    for col in record.columns:
        modified = record.copy()
        modified[col] = reference[col]
        p_ref = float(pipe.predict_proba(modified)[:, 1][0])
        contribution = base_p - p_ref
        rows.append(
            {
                "feature": col.replace("_", " "),
                "contribution": float(contribution),
                "value": record[col].iloc[0],
                "direction": "approve" if contribution >= 0 else "reject",
            }
        )
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return {
        "method": "occlusion (probability delta)",
        "base_probability": base_p,
        "top_reasons": rows[:top_k],
    }


def reason_codes(pipe, record: pd.DataFrame, background: pd.DataFrame | None = None, top_k: int = 8) -> dict:
    """Explain a single prediction.

    Uses exact additive log-odds contributions for linear models, and a
    model-agnostic occlusion method (probability change vs a reference profile)
    for ensembles and other non-linear models.
    """
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        return _linear_reason_codes(pipe, record, top_k)
    ref = background_reference(background if background is not None else record)
    return _occlusion_reason_codes(pipe, record, ref, top_k)


def global_permutation_importance(pipe, X: pd.DataFrame, y, n_repeats: int = 10, seed: int = 42) -> pd.DataFrame:
    """Model-agnostic permutation importance on the raw applicant columns."""
    result = permutation_importance(pipe, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=-1, scoring="accuracy")
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_reason_codes(reason: dict, path, title: str = "Why this decision?") -> None:
    rows = sorted(reason["top_reasons"], key=lambda r: r["contribution"])
    names = [r["feature"] for r in rows]
    vals = [r["contribution"] for r in rows]
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution to approval score  (green = pushes toward approval)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_permutation_importance(imp: pd.DataFrame, path, top: int = 12) -> None:
    data = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(data["feature"], data["importance_mean"], xerr=data["importance_std"], color="#264653")
    ax.set_xlabel("Drop in accuracy when the feature is shuffled")
    ax.set_title(f"Permutation importance (top {top})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
