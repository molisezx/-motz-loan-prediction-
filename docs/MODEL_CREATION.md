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

The raw extract (`data/raw/loan_data.csv`) is the public Kaggle **Loan Prediction
Problem** dataset: 614 applications with 13 columns, fetched by
`src/fetch_kaggle_data.py`. It contains genuine missing values and outliers so
that the cleaning stage is meaningful.

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
   rather than deleted, preserving all records. Columns with a **zero IQR**
   (near-constant, e.g. `Loan_Amount_Term`) are skipped, since clipping them
   would flatten a genuine feature to a single value.
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

Nine candidate algorithms spanning different inductive biases were compared with
**stratified 5-fold cross-validation** on the 491-row training split:

| Model | Mean CV accuracy | Mean CV ROC-AUC |
|-------|------------------|-----------------|
| **Support Vector Machine** | **0.792** | 0.764 |
| Soft Voting Ensemble | 0.790 | 0.774 |
| Logistic Regression | 0.790 | 0.727 |
| Stacking Ensemble | 0.786 | 0.776 |
| Random Forest | 0.782 | 0.781 |
| K-Nearest Neighbours | 0.770 | 0.728 |
| Gradient Boosting | 0.770 | 0.734 |
| Gaussian Naive Bayes | 0.770 | 0.720 |
| Decision Tree | 0.682 | 0.631 |

Exact values are stored in `reports/model_metrics.json`. The **support vector
machine** (linear kernel) was selected: it achieved the best cross-validated
accuracy and, once tuned, the strongest held-out ROC-AUC while remaining
robust in the high-dimensional one-hot feature space.

## 6. Training and evaluation (`src/train_model.py`)

The selected pipeline was re-trained on all 491 training rows and evaluated on a
stratified 123-row held-out test set.

| Metric | Value |
|--------|-------|
| Accuracy | 0.854 |
| Precision | 0.832 |
| Recall | 0.988 |
| F1-score | 0.903 |
| ROC-AUC | 0.787 |

The confusion matrix (`reports/figures/07_confusion_matrix.png`) shows very high
recall, meaning the model rarely rejects a creditworthy applicant — the more
costly error in a lending context.

## 7. Tuning (`src/tune_model.py`)

Hyper-parameters were optimised with `GridSearchCV` (stratified 5-fold CV,
accuracy scoring):

| Parameter | Values searched |
|-----------|----------------|
| `clf__C` | 0.1, 1, 10 |
| `clf__kernel` | linear, rbf |

**Best parameters:** `C = 0.1`, `kernel = linear`. The tuned configuration kept
test accuracy unchanged while improving ROC-AUC from 0.787 to 0.814, confirming
the model generalises rather than over-fitting; the small `C` indicates moderate
regularisation stabilises the decision boundary.

## 8. Interpretation

Feature importances (`reports/figures/09_permutation_importance.png`) rank the
features by the drop in accuracy when shuffled. `Credit_History` is by far the
most influential predictor (permutation importance 0.236), followed by
loan-amount and affordability-related features. This
aligns with the exploratory correlation analysis (r = 0.541) and with domain
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
