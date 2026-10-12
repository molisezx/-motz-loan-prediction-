# Model Card — Motz Loan Approval Predictor

Following the model-card framework of Mitchell et al. (2019).

## Model details

| Field | Value |
|-------|-------|
| Developed by | DIAL3110 Group Assignment (2026) |
| Model type | Supervised binary classification |
| Algorithm | Best of 9 candidates (see `reports/model_metrics.json`) |
| Inputs | 11 applicant attributes (see below) |
| Output | Approval probability + Approved/Rejected at the cost-optimal threshold |
| Version | 1.0.0 |
| Licence | MIT |

## Intended use

Decision **support** for loan officers at Motz Financial Services. The model
produces a probability and a recommended decision; a human officer remains
accountable for the final call, especially for borderline cases.

**Not intended for:** fully automated lending without human review, or any
context outside the population represented in the training data.

## Features

Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area.

Engineered: TotalIncome, LoanAmount_log, EMI, Income_to_Loan_Ratio.

## Training data

1,000 loan applications reconstructed from the client extract
(`src/generate_dataset.py`, fixed seed 42). Split 800/200 (train/test, stratified).
Controlled missingness (1–8%) and injected outliers are present in the raw data
and removed during cleaning.

## Evaluation

- Stratified 5-fold cross-validation across 9 algorithms (7 base + 2 ensembles).
- Held-out test set: accuracy, precision, recall, F1, ROC-AUC.
- Business-cost threshold optimisation (FN:FP = 5:1).
- Calibration, cumulative-gains, learning-curve and permutation-importance
  diagnostics.

See `reports/model_metrics.json` for the exact, reproducible figures.

## Ethical considerations and limitations

- **Synthetic data:** the extract is statistically reconstructed; real-world
  performance may differ.
- **Proxy discrimination:** attributes such as gender and marital status can act
  as proxies for protected characteristics. Any production deployment must audit
  for disparate impact.
- **Credit-history dominance:** the model relies heavily on `Credit_History`;
  missing or erroneous credit data materially affects outcomes.
- **Distribution shift:** performance degrades if applicant demographics or the
  economy change.

## Explainability

Every prediction is accompanied by reason codes; global behaviour is summarised
by permutation importance. Linear models use exact additive contributions;
ensembles use a model-agnostic occlusion method.

## Maintenance

Re-run `python run_pipeline.py` to retrain and regenerate all artefacts. Monitor
accuracy and the approval rate in production and retrain periodically.
