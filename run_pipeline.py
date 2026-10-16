"""Run the complete loan-prediction pipeline end-to-end.

    python run_pipeline.py

Steps: dataset download -> cleaning -> pre-processing -> training -> tuning.
A failure in any step aborts the run.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

STEPS = [
    "fetch_kaggle_data.py",
    "data_cleaning.py",
    "preprocessing.py",
    "train_model.py",
    "tune_model.py",
    "build_report.py",
    "build_presentation.py",
]


def main() -> None:
    for step in STEPS:
        print(f"\n{'=' * 70}\n>>> {step}\n{'=' * 70}")
        subprocess.run([sys.executable, str(SRC / step)], check=True, cwd=ROOT)
    print("\nPipeline complete. Launch the dashboard with:")
    print("    streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
