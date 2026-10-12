"""Central registry of candidate models and their hyper-parameter grids.

Keeping the model zoo in one place guarantees that training and tuning operate
on exactly the same definitions.
"""
import warnings

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning, message=".*probability.*deprecated.*")

RANDOM_STATE = 42


def base_models() -> dict:
    """Single, explainable baseline algorithms."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "K-Nearest Neighbours": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def ensemble_models() -> dict:
    """Advanced ensembles that typically outperform single learners."""
    estimators = [
        ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ("svm", SVC(probability=True, random_state=RANDOM_STATE)),
    ]
    return {
        "Soft Voting Ensemble": VotingClassifier(estimators=estimators, voting="soft"),
        "Stacking Ensemble": StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5,
            n_jobs=-1,
        ),
    }


def all_candidates() -> dict:
    return {**base_models(), **ensemble_models()}


def param_grids() -> dict:
    return {
        "Logistic Regression": {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
        "Decision Tree": {
            "clf__max_depth": [3, 5, 10, None],
            "clf__min_samples_leaf": [1, 2, 5],
            "clf__criterion": ["gini", "entropy"],
        },
        "Random Forest": {
            "clf__n_estimators": [200, 300, 400],
            "clf__max_depth": [None, 5, 10, 15],
            "clf__min_samples_split": [2, 5, 10],
        },
        "K-Nearest Neighbours": {
            "clf__n_neighbors": [3, 5, 7, 9, 11],
            "clf__weights": ["uniform", "distance"],
        },
        "Gaussian Naive Bayes": {"clf__var_smoothing": [1e-9, 1e-6, 1e-3]},
        "Support Vector Machine": {"clf__C": [0.1, 1.0, 10.0], "clf__kernel": ["rbf", "linear"]},
        "Gradient Boosting": {
            "clf__n_estimators": [100, 200, 300],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__max_depth": [2, 3, 4],
        },
        "Soft Voting Ensemble": {
            "clf__weights": [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)],
        },
        "Stacking Ensemble": {
            "clf__final_estimator__C": [0.01, 0.1, 1.0, 10.0],
            "clf__final_estimator__class_weight": [None, "balanced"],
        },
    }
