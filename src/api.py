"""Optional REST API for the loan model (FastAPI).

Run with:
    uvicorn src.api:app --reload --port 8000

Then:
    POST /predict        - score a single applicant
    POST /predict/batch  - score a list of applicants
    GET  /health         - readiness probe
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import METRICS_PATH, MODEL_PATH  # noqa: E402


class Applicant(BaseModel):
    Gender: str = Field(..., examples=["Male"])
    Married: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["0"])
    Education: str = Field(..., examples=["Graduate"])
    Self_Employed: str = Field(..., examples=["No"])
    ApplicantIncome: float = Field(..., examples=[5000])
    CoapplicantIncome: float = Field(..., examples=[1500])
    LoanAmount: float = Field(..., examples=[150000])
    Loan_Amount_Term: float = Field(..., examples=[360])
    Credit_History: float = Field(..., examples=[1.0])
    Property_Area: str = Field(..., examples=["Semiurban"])


app = FastAPI(title="Motz Loan Prediction API", version="1.0.0")
_ARTIFACTS: dict = {}


def _load():
    if not _ARTIFACTS:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not trained. Run `python run_pipeline.py`.")
        _ARTIFACTS["model"] = joblib.load(MODEL_PATH)
        import json

        _ARTIFACTS["threshold"] = (
            json.loads(METRICS_PATH.read_text(encoding="utf-8")).get("decision_threshold", 0.5)
            if METRICS_PATH.exists()
            else 0.5
        )
    return _ARTIFACTS


def _score(records: list[dict]) -> list[dict]:
    art = _load()
    df = pd.DataFrame(records)
    proba = art["model"].predict_proba(df)[:, 1]
    threshold = art["threshold"]
    return [
        {
            "approval_probability": round(float(p), 4),
            "decision": "Approved" if p >= threshold else "Rejected",
        }
        for p in proba
    ]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(applicant: Applicant):
    return _score([applicant.model_dump()])[0]


@app.post("/predict/batch")
def predict_batch(applicants: list[Applicant]):
    if not applicants:
        raise HTTPException(status_code=400, detail="Empty batch")
    return _score([a.model_dump() for a in applicants])
