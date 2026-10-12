"""Task 3 & 4: Model determination and model training (advanced version).

Compares single learners *and* advanced ensembles with stratified 5-fold
cross-validation across four metrics, selects the best model, trains it on the
full training split, and evaluates it on a held-out test set - including a
business-cost threshold analysis and multiple diagnostic curves.

Outputs the fitted end-to-end pipeline plus an enriched metrics file.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate

from config import (
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_DATA,
    TRAIN_DATA,
)
from evaluate import (
    metrics_at_threshold,
    optimize_threshold_cost,
    plot_calibration,
    plot_cumulative_gains,
    plot_learning_curve,
    plot_precision_recall,
    plot_threshold_analysis,
)
from explain import (
    global_permutation_importance,
    plot_permutation_importance,
    plot_reason_codes,
    reason_codes,
)
from logging_utils import get_logger
from models import all_candidates
from preprocessing import build_pipeline

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120
log = get_logger()

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def load_splits():
    train = pd.read_csv(TRAIN_DATA)
    test = pd.read_csv(TEST_DATA)
    target = train.columns[-1]
    return (
        train.drop(columns=[target]), test.drop(columns=[target]),
        train[target], test[target],
    )


def compare_models(X_train, y_train) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, clf in all_candidates().items():
        log.info("Cross-validating %s ...", name)
        scores = cross_validate(build_pipeline(clf), X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for metric in SCORING:
            row[f"cv_{metric}_mean"] = round(float(scores[f"test_{metric}"].mean()), 4)
            if metric == "accuracy":
                row["cv_accuracy_std"] = round(float(scores[f"test_{metric}"].std()), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["cv_accuracy_mean", "cv_roc_auc_mean"], ascending=False
    ).reset_index(drop=True)


def plot_model_comparison(results: pd.DataFrame, path) -> None:
    data = results.melt(
        id_vars="model", value_vars=["cv_accuracy_mean", "cv_roc_auc_mean"],
        var_name="metric", value_name="value",
    )
    data["metric"] = data["metric"].map({"cv_accuracy_mean": "Accuracy", "cv_roc_auc_mean": "ROC-AUC"})
    fig, ax = plt.subplots(figsize=(9.5, 5))
    sns.barplot(data=data, y="model", x="value", hue="metric", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title("Candidate models - cross-validated performance")
    ax.set_xlabel("Mean 5-fold CV score")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(cm, path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc(y_test, y_prob, auc, path) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", color="#2a9d8f")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    X_train, X_test, y_train, y_test = load_splits()

    log.info("Task 3: comparing %d candidate models (5-fold CV)...", len(all_candidates()))
    results = compare_models(X_train, y_train)
    print(results.to_string(index=False))
    plot_model_comparison(results, FIGURES_DIR / "06_model_comparison.png")

    best_name = results.iloc[0]["model"]
    log.info("Selected model: %s", best_name)

    pipe = build_pipeline(all_candidates()[best_name])
    pipe.fit(X_train, y_train)

    log.info("Task 4: evaluating on held-out test set...")
    y_prob = pipe.predict_proba(X_test)[:, 1]
    default = metrics_at_threshold(y_test, y_prob, 0.5)

    cost = optimize_threshold_cost(y_test, y_prob, cost_fn=5.0, cost_fp=1.0)
    best_t = cost["best_threshold"]
    tuned_threshold = metrics_at_threshold(y_test, y_prob, best_t)

    plot_confusion(default["confusion_matrix"], FIGURES_DIR / "07_confusion_matrix.png")
    plot_roc(y_test, y_prob, default["roc_auc"], FIGURES_DIR / "08_roc_curve.png")
    plot_precision_recall(y_test, y_prob, FIGURES_DIR / "11_precision_recall_curve.png")
    plot_calibration(pipe, X_test, y_test, FIGURES_DIR / "12_calibration_curve.png")
    plot_cumulative_gains(y_test, y_prob, FIGURES_DIR / "13_cumulative_gains.png")
    plot_threshold_analysis(y_test, y_prob, best_t, FIGURES_DIR / "14_threshold_analysis.png")

    log.info("Computing permutation importance and a learning curve ...")
    importance = global_permutation_importance(pipe, X_test, y_test, n_repeats=10, seed=RANDOM_STATE)
    plot_permutation_importance(importance, FIGURES_DIR / "09_permutation_importance.png")
    plot_learning_curve(pipe, X_train, y_train, FIGURES_DIR / "15_learning_curve.png")

    borderline = int(np.argmin(np.abs(y_prob - 0.5)))
    record = X_test.iloc[[borderline]]
    reason = reason_codes(pipe, record, background=X_train, top_k=8)
    plot_reason_codes(reason, FIGURES_DIR / "16_reason_codes_example.png",
                      title=f"Example explanation - borderline applicant ({reason['method']})")

    joblib.dump(pipe, MODEL_PATH)

    output = {
        "selected_model": best_name,
        "model_comparison": results.to_dict(orient="records"),
        "test_metrics": default,
        "decision_threshold": best_t,
        "metrics_at_optimal_threshold": tuned_threshold,
        "cost_analysis": {k: v for k, v in cost.items() if k != "curve"},
        "confusion_matrix": default["confusion_matrix"],
        "permutation_importance": importance.head(15).round(4).to_dict(orient="records"),
        "reason_code_example": reason,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    log.info("Default metrics: %s", {k: default[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")})
    log.info("Business-optimal threshold = %.2f (expected cost %.0f)", best_t, cost["min_expected_cost"])
    log.info("Pipeline saved to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
