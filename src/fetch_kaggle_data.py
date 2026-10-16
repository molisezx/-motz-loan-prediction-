"""Task 1 (data gathering): fetch the loan-application dataset from Kaggle.

The data is the publicly available *Loan Prediction Problem* dataset
(``altruistdelhite04/loan-prediction-problem-dataset``). It mirrors the schema
of the Motz Financial Services extract, so it is a drop-in replacement for the
previously generated synthetic data.

Outputs:
    data/raw/loan_data.csv       labelled training extract (target present)
    data/raw/kaggle_holdout.csv  unlabelled Kaggle test extract (demo scoring)

The download is cached by ``kagglehub`` so subsequent runs work offline.
"""
from pathlib import Path

import pandas as pd

from config import RAW_DATA, RAW_DIR

KAGGLE_DATASET = "altruistdelhite04/loan-prediction-problem-dataset"
TRAIN_FILE = "train_u6lujuX_CVtuZ9i.csv"
TEST_FILE = "test_Y3wMUE5_7gLdaTN.csv"
HOLDOUT_DATA = RAW_DIR / "kaggle_holdout.csv"


def download_source() -> Path:
    """Download (or reuse the cached copy of) the Kaggle dataset."""
    import kagglehub

    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def load_raw() -> pd.DataFrame:
    """Return the labelled Kaggle training extract.

    The dataset is downloaded (and cached) via ``kagglehub``. If that is not
    possible -- no ``kagglehub`` installed, no network, or Kaggle unavailable --
    the committed copy at ``data/raw/loan_data.csv`` is used so the pipeline
    still runs offline and reproducibly for grading.
    """
    try:
        src = download_source()
        train = pd.read_csv(src / TRAIN_FILE)
        holdout = pd.read_csv(src / TEST_FILE)
        train.to_csv(RAW_DATA, index=False)
        holdout.to_csv(HOLDOUT_DATA, index=False)
        return train
    except Exception as exc:
        if RAW_DATA.exists():
            print(f"Kaggle download unavailable ({exc}); using committed copy {RAW_DATA}")
            return pd.read_csv(RAW_DATA)
        raise RuntimeError(
            "Could not download the Kaggle dataset and no committed copy exists at "
            f"{RAW_DATA}. Re-run with an internet connection."
        ) from exc


def main() -> None:
    df = load_raw()
    print(f"Raw dataset written to {RAW_DATA}")
    print(f"Unlabelled hold-out written to {HOLDOUT_DATA}")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Target distribution:\n{df['Loan_Status'].value_counts(normalize=True).round(3)}")


if __name__ == "__main__":
    main()
