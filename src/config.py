"""Central configuration for the Motz Financial Services loan prediction system.

All paths are resolved relative to the project root so the pipeline can be run
from any working directory.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA = RAW_DIR / "loan_data.csv"
CLEAN_DATA = PROCESSED_DIR / "loan_clean.csv"
FEATURES_DATA = PROCESSED_DIR / "loan_features.csv"
TRAIN_DATA = PROCESSED_DIR / "train.csv"
TEST_DATA = PROCESSED_DIR / "test.csv"

MODEL_PATH = MODELS_DIR / "loan_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
TUNING_PATH = REPORTS_DIR / "tuning_results.json"
CLEANING_REPORT = REPORTS_DIR / "cleaning_report.json"

TARGET = "Loan_Status"
POSITIVE_LABEL = "Y"
NEGATIVE_LABEL = "N"
RANDOM_STATE = 42
TEST_SIZE = 0.20

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "TotalIncome",
    "LoanAmount_log",
    "EMI",
    "Income_to_Loan_Ratio",
]

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
