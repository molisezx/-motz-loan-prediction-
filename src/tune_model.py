"""Task 5: Model tuning (advanced version).

Hyper-parameters of the model selected in Task 4 are optimised with an
exhaustive grid search (stratified 5-fold CV). The tuned pipeline is compared
against the baseline; the better of the two becomes the production model. The
business-optimal decision threshold is recomputed for the adopted model.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from config import FIGURES_DIR, METRICS_PATH, MODEL_PATH, RANDOM_STATE, TEST_DATA, TRAIN_DATA, TUNING_PATH
from evaluate import metrics_at_threshold, optimize_threshold_cost
from logging_utils import get_logger
from models import all_candidates, param_grids
from preprocessing import build_pipeline

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120
log = get_logger()


def load_splits():
    train = pd.read_csv(TRAIN_DATA)
    test = pd.read_csv(TEST_DATA)
    target = train.columns[-1]
    return (
        train.drop(columns=[target]), test.drop(columns=[target]),
        train[target], test[target],
    )


def plot_comparison(baseline, tuned, path) -> None:
    rows = []
    for name, m in [("Baseline", baseline), ("Tuned", tuned)]:
        for metric in ["accuracy", "f1", "roc_auc"]:
            rows.append({"stage": name, "metric": metric.upper(), "value": m[metric]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="metric", y="value", hue="stage", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Baseline vs tuned model performance")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    with open(METRICS_PATH, encoding="utf-8") as fh:
        metrics = json.load(fh)

    selected = metrics["selected_model"]
    baseline = {k: metrics["test_metrics"][k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    log.info("Tuning: %s", selected)

    X_train, X_test, y_train, y_test = load_splits()
    grid_def = param_grids().get(selected)

    if not grid_def:
        log.warning("No parameter grid registered for %s; keeping baseline.", selected)
        final = baseline
        best_params, best_cv, n_candidates, adopted = {}, metrics.get("test_metrics", {}), 0, "baseline"
    else:
        pipe = build_pipeline(all_candidates()[selected])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        grid = GridSearchCV(pipe, grid_def, cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
        grid.fit(X_train, y_train)

        best = grid.best_estimator_
        y_prob = best.predict_proba(X_test)[:, 1]
        tuned = metrics_at_threshold(y_test, y_prob, 0.5)
        best_params = {k: (v if not isinstance(v, float) else round(v, 6)) for k, v in grid.best_params_.items()}
        best_cv = round(float(grid.best_score_), 4)
        n_candidates = int(len(grid.cv_results_["params"]))
        log.info("Best params: %s (CV accuracy %.4f)", best_params, best_cv)

        adopted = "tuned" if tuned["accuracy"] >= baseline["accuracy"] else "baseline"
        final_model = best if adopted == "tuned" else joblib.load(MODEL_PATH)
        joblib.dump(final_model, MODEL_PATH)
        final = {k: tuned[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")} if adopted == "tuned" else baseline

        y_prob_final = joblib.load(MODEL_PATH).predict_proba(X_test)[:, 1]
        cost = optimize_threshold_cost(y_test, y_prob_final, cost_fn=5.0, cost_fp=1.0)
        metrics["decision_threshold"] = cost["best_threshold"]
        metrics["metrics_at_optimal_threshold"] = metrics_at_threshold(y_test, y_prob_final, cost["best_threshold"])
        metrics["cost_analysis"] = {k: v for k, v in cost.items() if k != "curve"}
        plot_comparison(baseline, final, FIGURES_DIR / "10_baseline_vs_tuned.png")

    metrics["final_metrics"] = {
        **final,
        "threshold": metrics.get("decision_threshold", 0.5),
        "best_params": best_params,
        "best_cv_score": best_cv,
    }
    metrics["adopted_model"] = adopted
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    result = {
        "selected_model": selected,
        "baseline_metrics": baseline,
        "tuned_metrics": {"accuracy": final["accuracy"], "f1": final["f1"], "roc_auc": final["roc_auc"],
                          "best_params": best_params, "best_cv_score": best_cv},
        "adopted": adopted,
        "n_candidates": n_candidates,
        "decision_threshold": metrics.get("decision_threshold", 0.5),
    }
    with open(TUNING_PATH, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    log.info("Adopted model: %s | final metrics: %s", adopted, final)
    log.info("Production model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
