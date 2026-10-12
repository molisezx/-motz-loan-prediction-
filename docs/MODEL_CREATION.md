# Model Creation

This document details how the loan-prediction model was created: the data, the
feature engineering, the candidate algorithms, the training and tuning process,
and the final evaluation. All figures are produced automatically by the
pipeline and stored in `reports/figures/`.

## 1. Problem framing

- **Type:** supervised binary classification.
- **Target:** `Loan_Status` — `Y` (approved) or `N` (rejected), encoded as `1`/`0`.
- **Inputs:** 11 applicant attributes (6 categorical, 5 numeric/count).
- **Success metric:** test-set accuracy (primary), with precision, recall, F1
  and ROC-AUC reported for completeness.

## 2. Data

The raw extract (`data/raw/loan_data.csv`) contains 1,000 applications with 12
columns. It intentionally contains missing values and outliers so that the
cleaning stage is meaningful.

| Column | Type | Notes |
|--------|------|-------|
| Loan_ID | id | dropped before modelling |
| Gender | categorical | some missing |
| Married | categorical | |
| Dependents | categorical | 0 / 1 / 2 / 3+ |
| Education | categorical | Graduate / Not Graduate |
| Self_Employed | categorical | some missing |
| ApplicantIncome | numeric | monthly |
| CoapplicantIncome | numeric | |
| LoanAmount | numeric | some missing |
| Loan_Amount_Term | numeric | months |
| Credit_History | binary | 1 = good, 0 = bad; strongest predictor |
| Property_Area | categorical | Urban / Semiurban / Rural |
| Loan_Status | target | Y / N |

## 3. Cleaning (`src/data_cleaning.py`)

1. **Missing values** — categorical columns imputed with the **mode**, numeric
   columns with the **median** (robust to skew). After cleaning: 0 missing.
2. **Outliers** — detected with the **IQR rule** (values outside
   `Q1 - 1.5·IQR` / `Q3 + 1.5·IQR`) and **winsorised** (capped at the fences)
   rather than deleted, preserving all records.
3. **Descriptive statistics** — means, standard deviations, quartiles and
   frequency tables recorded in `reports/cleaning_report.json`.

## 4. Feature engineering (`src/preprocessing.py`)

Four domain features are extracted:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| TotalIncome | ApplicantIncome + CoapplicantIncome | household repayment capacity |
| LoanAmount_log | log1p(LoanAmount) | removes right skew (0.85 → −0.36) |
| EMI | LoanAmount / Loan_Amount_Term | monthly repayment burden proxy |
| Income_to_Loan_Ratio | TotalIncome / LoanAmount | affordability |

**Transformation pipeline** (encapsulated in a scikit-learn `ColumnTransformer`
so training and inference are identical):

- Numeric → median imputation → `StandardScaler`
- Categorical → most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")`

The whole sequence — feature extraction, imputation, encoding, scaling and the
classifier — is saved as a single scikit-learn `Pipeline` in
`models/loan_model.pkl`. Raw applicant records (even with missing values) can be
fed straight into the saved pipeline.

## 5. Model determination (`src/train_model.py`)

Seven algorithms spanning different inductive biases were compared with
**stratified 5-fold cross-validation** on the 800-row training split:

| Model | Mean CV accuracy | Mean CV ROC-AUC |
|-------|------------------|-----------------|
| **Logistic Regression** | **0.844** | 0.733 |
| Support Vector Machine | 0.844 | 0.707 |
| Gaussian Naive Bayes | 0.833 | **0.741** |
| Gradient Boosting | 0.833 | 0.685 |
| Random Forest | 0.831 | 0.685 |
| K-Nearest Neighbours | 0.830 | 0.689 |
| Decision Tree | 0.719 | 0.662 |

Exact values are stored in `reports/model_metrics.json`. **Logistic regression**
was selected because it tied for the best cross-validated accuracy while
gave the best cross-validated accuracy while remaining fast, interpretable and
well-calibrated for probability output — properties that matter for a lending
decision that must be explainable.

## 6. Training and evaluation (`src/train_model.py`)

The selected pipeline was re-trained on all 800 training rows and evaluated on a
stratified 200-row held-out test set.

| Metric | Value |
|--------|-------|
| Accuracy | 0.84 |
| Precision | 0.84 |
| Recall | 0.97 |
| F1-score | 0.90 |
| ROC-AUC | 0.76 |

The confusion matrix (`reports/figures/07_confusion_matrix.png`) shows very high
recall, meaning the model rarely rejects a creditworthy applicant — the more
costly error in a lending context.

## 7. Tuning (`src/tune_model.py`)

Hyper-parameters were optimised with `GridSearchCV` (stratified 5-fold CV,
accuracy scoring):

| Parameter | Values searched |
|-----------|-----------------|
| `clf__C` | 0.01, 0.1, 1, 10 |
| `clf__solver` | lbfgs |
| `clf__class_weight` | None, balanced |

**Best parameters:** `C = 0.01`, `solver = lbfgs`, `class_weight = None`.
The tuned configuration matched the baseline closely, confirming the model is
not over-fitting; the small `C` indicates moderate regularisation stabilises the
predictions.

## 8. Interpretation

Feature importances (`reports/figures/09_feature_importance.png`) rank the
standardised coefficients. `Credit_History` is by far the most influential
predictor, followed by loan-amount and affordability-related features. This
aligns with the exploratory correlation analysis (r = 0.46) and with domain
intuition: a good credit history is the single strongest indicator of repayment.

## 9. Deployment (`dashboard/app.py`)

The saved pipeline is served through a Streamlit dashboard offering:

1. **Single prediction** — enter applicant details, view the approval
   probability live.
2. **Batch prediction** — upload an out-of-sample CSV and download scored
   results (used on presentation day).
3. **Data insights** — descriptive statistics and interactive charts.
4. **Model performance** — metrics, candidate comparison and diagnostic figures.

## 10. Reproducing everything

```powershell
python run_pipeline.py          # data -> cleaning -> features -> train -> tune
streamlit run dashboard/app.py  # dashboard
python src/build_report.py      # Word report
python src/build_presentation.py# PowerPoint deck
```

A fixed `RANDOM_STATE = 42` guarantees identical results on every run.
