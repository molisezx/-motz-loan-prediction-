# Motz Loan Prediction — Explainable Machine Learning System

[![CI](https://github.com/molisezx/-motz-loan-prediction-/actions/workflows/ci.yml/badge.svg)](https://github.com/molisezx/-motz-loan-prediction-/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-8%20passing-brightgreen)

**DIAL3110 · Fundamentals of Artificial Intelligence · Group Assignment 2026**

An end-to-end, production-grade machine-learning system that predicts whether a
loan application to **Motz Financial Services** will be **approved** or
**rejected** — and *explains every decision*. Built with a reproducible pipeline,
nine candidate algorithms, business-cost optimisation, explainable AI, an
automated test suite, a REST API, and a themed interactive dashboard.

---

## Why this project stands out

| Feature | What it does |
|---------|--------------|
| **9-model benchmark** | 7 base learners + soft-voting and stacking ensembles, compared by multi-metric cross-validation |
| **Cost-sensitive decisions** | Finds the profit-optimal threshold (FN:FP cost ratio) instead of assuming 0.5 |
| **Explainable AI** | Per-applicant *reason codes* + model-agnostic permutation importance |
| **Rigorous evaluation** | ROC, precision-recall, calibration, cumulative-gains, learning curves |
| **Professional engineering** | Typed modules, logging, tests, Docker, CI, Makefile, REST API |
| **Reproducibility** | One command regenerates data, model, figures, report and slides (seed = 42) |

## Results

| Metric | Value |
|--------|-------|
| Selected model | **Support Vector Machine** (linear kernel) |
| Cross-validated accuracy | **79.2 %** |
| Test accuracy | **85.4 %** |
| Recall / Precision | **98.8 % / 83.2 %** |
| ROC-AUC | **0.787** |
| Majority-class baseline | 68.7 % |
| Cost-optimal threshold | 0.10 (FN:FP = 5:1) |

Data: the public Kaggle **Loan Prediction Problem** dataset (614 labelled
applications, fetched at build time via `src/fetch_kaggle_data.py`).

## Repository structure

```
DIAL3110_Loan_Prediction/
├── run_pipeline.py            # one-command end-to-end pipeline
├── Makefile · Dockerfile · pyproject.toml · requirements.txt
├── .github/workflows/ci.yml   # continuous integration
├── data/{raw,processed}/      # downloaded & processed datasets
├── src/
│   ├── config.py              # paths & constants
│   ├── logging_utils.py       # project-wide logging
│   ├── fetch_kaggle_data.py   # Task 1  data gathering (Kaggle download)
│   ├── data_cleaning.py       # Task 1  missing values, outliers, statistics
│   ├── preprocessing.py       # Task 2  feature engineering + pipeline factory
│   ├── models.py              # candidate model registry + hyper-parameter grids
│   ├── train_model.py         # Task 3&4 model determination, training, diagnostics
│   ├── tune_model.py          # Task 5  hyper-parameter tuning
│   ├── evaluate.py            # threshold optimisation + evaluation curves
│   ├── explain.py             # Task 6  explainable AI (reason codes)
│   ├── predict.py             # batch scoring of a CSV
│   ├── api.py                 # FastAPI REST service
│   ├── build_report.py        # Task 7  generates the Word report
│   └── build_presentation.py  # Task 8  generates the slide deck
├── dashboard/app.py           # Task 6  themed multi-view Streamlit app
├── tests/                     # pytest suite
├── docs/                      # Word report · MODEL_CREATION.md · MODEL_CARD.md
├── reports/figures/           # 16 auto-generated diagnostic figures
└── presentation/              # generated slide deck
```

## Quickstart

```powershell
python -m pip install -r requirements.txt

python run_pipeline.py            # data → cleaning → features → train → tune → report → slides
streamlit run dashboard/app.py    # launch the dashboard
```

## How to run (for graders)

The trained model, processed datasets, figures, report and slides are all
committed, so **no pipeline step is required** to inspect the results.

```powershell
# 1. install dependencies
python -m pip install -r requirements.txt

# 2. launch the dashboard (opens http://localhost:8501)
streamlit run dashboard/app.py
```

Optionally reproduce everything from scratch (deterministic, `RANDOM_STATE = 42`):

```powershell
python run_pipeline.py            # downloads the Kaggle data, or uses the committed copy if offline
```

Docker (no Python setup needed):

```powershell
docker build -t motz-loan .
docker run -p 8501:8501 motz-loan
```

No internet is required: if the Kaggle download is unavailable, the pipeline
falls back to the committed `data/raw/loan_data.csv`.

## Dashboard

A themed, multi-view decision-support app (`Home · Predict · Batch · Insights ·
Performance · Explainability · About`). The **Predict** view returns an approval
probability, the cost-optimal decision, and a live reason-code chart.

## REST API

```powershell
uvicorn src.api:app --reload --port 8000
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Readiness probe |
| `/predict` | POST | Score one applicant |
| `/predict/batch` | POST | Score a list of applicants |

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"Gender":"Male","Married":"Yes","Dependents":"0","Education":"Graduate",
       "Self_Employed":"No","ApplicantIncome":5000,"CoapplicantIncome":1500,
       "LoanAmount":150000,"Loan_Amount_Term":360,"Credit_History":1.0,
       "Property_Area":"Semiurban"}'
```

## Testing & quality

```powershell
pytest -q                       # 8 tests
ruff check src dashboard        # lint
```

CI (`.github/workflows/ci.yml`) runs lint, the full pipeline and the tests on
Python 3.11 and 3.12 for every push.

## Docker

```powershell
docker build -t motz-loan .
docker run -p 8501:8501 motz-loan                       # dashboard
docker run -p 8000:8000 motz-loan uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Documentation

- [`docs/MODEL_CREATION.md`](docs/MODEL_CREATION.md) — how the model was built.
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — intended use, limitations, ethics.
- [`docs/DIAL3110_Loan_Prediction_Report.docx`](docs) — full written report (24 IEEE references).
- Generated slide deck in [`presentation/`](presentation).

## Assignment task coverage

| Task | Requirement | Where |
|------|-------------|-------|
| 1 | Data gathering, cleaning, missing values, outliers, statistics | `src/fetch_kaggle_data.py` · `src/data_cleaning.py` |
| 2 | Feature extraction and tuning | `src/preprocessing.py` |
| 3 | Model determination with reasoning | `src/train_model.py` |
| 4 | Model training | `src/train_model.py` |
| 5 | Model tuning | `src/tune_model.py` |
| 6 | Prediction dashboard | `dashboard/app.py` |
| 7 | Report format (IEEE, 24 refs) | `docs/…Report.docx` |
| 8 | Presentation + performance matrix | `presentation/…Slides.pptx` |

## Reproducibility

Every random operation uses `RANDOM_STATE = 42`. `python run_pipeline.py`
downloads the Kaggle dataset and deterministically recreates the splits, model,
metrics, figures, report and slides.

## Authors

Fill in before submitting:

| Name | Student ID | Responsibilities |
|------|------------|------------------|
| `<your name>` | `<student id>` | Data gathering & cleaning (Tasks 1–2) |
| `<your name>` | `<student id>` | Model determination, training & tuning (Tasks 3–5) |
| `<your name>` | `<student id>` | Dashboard & explainability (Task 6) |
| `<your name>` | `<student id>` | Report & presentation (Tasks 7–8) |
