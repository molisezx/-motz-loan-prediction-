"""Threshold analysis and rich model evaluation.

Beyond the default 0.5 cut-off this module finds the *business-optimal* decision
threshold (the point that minimises the expected cost of wrong decisions) and
produces the diagnostic curves used in the report and dashboard.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import learning_curve

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 9


def metrics_at_threshold(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def optimize_threshold_cost(y_true, y_prob, cost_fn: float = 5.0, cost_fp: float = 1.0) -> dict:
    """Find the threshold minimising expected cost = cost_FN*FN + cost_FP*FP.

    In lending, approving a bad loan (false negative for the *rejection* class /
    false positive for approval) is far more expensive than wrongly rejecting a
    good applicant, so ``cost_fn`` > ``cost_fp``.
    """
    thresholds = np.unique(np.round(np.linspace(0.05, 0.95, 91), 2))
    best = None
    curve = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = cost_fn * fn + cost_fp * fp
        curve.append((float(t), float(cost)))
        if best is None or cost < best[1]:
            best = (float(t), float(cost), int(fn), int(fp))
    return {
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "best_threshold": round(best[0], 3),
        "min_expected_cost": best[1],
        "false_negatives": best[2],
        "false_positives": best[3],
        "curve": curve,
    }


def plot_threshold_analysis(y_true, y_prob, best_t: float, path) -> None:
    thresholds = np.linspace(0.05, 0.95, 91)
    acc, prec, rec, f1 = [], [], [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        acc.append(accuracy_score(y_true, y_pred))
        prec.append(precision_score(y_true, y_pred, zero_division=0))
        rec.append(recall_score(y_true, y_pred, zero_division=0))
        f1.append(f1_score(y_true, y_pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for series, label in [(acc, "Accuracy"), (prec, "Precision"), (rec, "Recall"), (f1, "F1")]:
        ax.plot(thresholds, series, label=label)
    ax.axvline(best_t, color="black", linestyle="--", label=f"Optimal = {best_t:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs decision threshold")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_precision_recall(y_true, y_prob, path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, color="#2a9d8f")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(pipe, X, y_true, path) -> None:
    prob = pipe.predict_proba(X)[:, 1]
    prob_true, prob_pred = calibration_curve(y_true, prob, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(prob_pred, prob_true, "o-", color="#264653", label="Model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_learning_curve(pipe, X, y, path) -> None:
    sizes, train_scores, test_scores = learning_curve(
        pipe, X, y, cv=5, scoring="accuracy", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 6), random_state=42,
    )
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(sizes, train_scores.mean(axis=1), "o-", label="Training score")
    ax.plot(sizes, test_scores.mean(axis=1), "s-", label="Cross-validation score")
    ax.fill_between(sizes, test_scores.mean(1) - test_scores.std(1), test_scores.mean(1) + test_scores.std(1), alpha=0.2)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning curve")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cumulative_gains(y_true, y_prob, path) -> None:
    order = np.argsort(-y_prob)
    y_sorted = np.asarray(y_true)[order]
    cum_pos = np.cumsum(y_sorted) / y_sorted.sum()
    frac = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(frac, cum_pos, color="#2a9d8f", label="Model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random")
    ax.set_xlabel("Fraction of applicants contacted")
    ax.set_ylabel("Fraction of approvals captured")
    ax.set_title("Cumulative gains chart")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
