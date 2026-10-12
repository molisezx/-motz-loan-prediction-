"""Task 1 (data gathering): build the raw loan application dataset.

Motz Financial Services supplies a tabular extract of loan applications with
applicant background information.  For reproducibility (and because the live
production extract contains personal data) this module reconstructs an
equivalent dataset statistically, mirroring the schema and behaviour of the
standard loan-prediction dataset:

    Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
    ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
    Credit_History, Property_Area, Loan_Status

The generated extract intentionally contains missing values and outliers so
that the cleaning stage has real work to do.
"""
import numpy as np
import pandas as pd

from config import RANDOM_STATE, RAW_DATA


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(n: int = 1000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a realistic raw loan-application extract of ``n`` rows."""
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n, p=[0.80, 0.20])
    married = rng.choice(["Yes", "No"], size=n, p=[0.65, 0.35])
    dependents = rng.choice(["0", "1", "2", "3+"], size=n, p=[0.58, 0.17, 0.17, 0.08])
    education = rng.choice(["Graduate", "Not Graduate"], size=n, p=[0.78, 0.22])
    self_employed = rng.choice(["Yes", "No"], size=n, p=[0.13, 0.87])
    property_area = rng.choice(
        ["Urban", "Semiurban", "Rural"], size=n, p=[0.33, 0.38, 0.29]
    )

    applicant_income = np.round(rng.lognormal(mean=8.4, sigma=0.55, size=n), 0)
    coapplicant_income = np.where(
        rng.random(n) < 0.45, 0.0, np.round(rng.lognormal(mean=7.6, sigma=0.7, size=n), 0)
    )

    total_income = applicant_income + coapplicant_income
    # Loan principals scale with affordability (up to ~45x monthly income),
    # expressed in the dataset's currency unit.
    loan_amount = np.round(total_income * rng.uniform(15, 45, size=n) / 1000.0) * 1000.0
    loan_amount = np.clip(loan_amount, 30_000, 700_000)

    term_choices = [360.0, 180.0, 240.0, 120.0, 84.0, 60.0, 300.0, 480.0]
    term_probs = [0.70, 0.12, 0.07, 0.03, 0.02, 0.02, 0.02, 0.02]
    loan_term = rng.choice(term_choices, size=n, p=term_probs)

    credit_history = rng.choice([1.0, 0.0], size=n, p=[0.84, 0.16])

    logit = (
        3.6 * credit_history
        + 0.00004 * total_income
        - 0.000002 * loan_amount
        + 0.55 * (education == "Graduate")
        + 0.45 * (married == "Yes")
        + 0.35 * np.isin(dependents, ["0", "1"])
        + 0.40 * (property_area == "Semiurban")
        - 0.25 * (property_area == "Rural")
        - 2.80
    )
    prob_approved = _sigmoid(logit)
    loan_status = np.where(rng.random(n) < prob_approved, "Y", "N")

    df = pd.DataFrame(
        {
            "Loan_ID": [f"LP{1000 + i:04d}" for i in range(n)],
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
            "Loan_Status": loan_status,
        }
    )

    df = _inject_missing(df, rng)
    df = _inject_outliers(df, rng)
    return df


def _inject_missing(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Introduce realistic missingness across several columns (~3-7%)."""
    missing_rates = {
        "Gender": 0.03,
        "Married": 0.01,
        "Dependents": 0.03,
        "Self_Employed": 0.05,
        "LoanAmount": 0.04,
        "Loan_Amount_Term": 0.03,
        "Credit_History": 0.08,
        "CoapplicantIncome": 0.01,
    }
    for col, rate in missing_rates.items():
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
    return df


def _inject_outliers(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add a small number of extreme but plausible high-income / high-loan rows."""
    for col, low, high in [("ApplicantIncome", 30000, 95000), ("LoanAmount", 900000, 1500000)]:
        idx = rng.choice(df.index, size=6, replace=False)
        df.loc[idx, col] = rng.integers(low, high, size=6).astype(float)
    return df


def main() -> None:
    df = generate_dataset()
    df.to_csv(RAW_DATA, index=False)
    print(f"Raw dataset written to {RAW_DATA}")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Target distribution:\n{df['Loan_Status'].value_counts(normalize=True).round(3)}")


if __name__ == "__main__":
    main()
