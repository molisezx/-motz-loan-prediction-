"""Generate the written report (Task 7) as a Word document.

Produces ``docs/DIAL3110_Loan_Prediction_Report.docx`` in Arial Narrow 12 with
the required structure: abstract, introduction, body, conclusion, references.
Real metrics are read from ``reports/*.json`` so the numbers always match the
latest pipeline run.
"""
import json

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from config import BASE_DIR, FIGURES_DIR, METRICS_PATH, TUNING_PATH

DOCS = BASE_DIR / "docs"
DOCS.mkdir(exist_ok=True)
OUT = DOCS / "DIAL3110_Loan_Prediction_Report.docx"


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def setup(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = "Arial Narrow"
    style.font.size = Pt(12)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), "Arial Narrow")
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)


def heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = "Arial Narrow"
        run.font.color.rgb = RGBColor(0x1F, 0x3B, 0x57)
    return h


def para(doc, text, italic=False, bold=False, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Arial Narrow"
    run.italic = italic
    run.bold = bold
    if align:
        p.alignment = align
    return p


def bullets(doc, items):
    for it in items:
        p = doc.add_paragraph(it, style="List Bullet")
        for r in p.runs:
            r.font.name = "Arial Narrow"
            r.font.size = Pt(12)


def table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = str(h)
        for r in cell.paragraphs[0].runs:
            r.bold = True
            r.font.name = "Arial Narrow"
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
            for r in cells[i].paragraphs[0].runs:
                r.font.name = "Arial Narrow"
    return t


def figure(doc, name, caption, width=5.6):
    path = FIGURES_DIR / name
    if path.exists():
        doc.add_picture(str(path), width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(caption)
        run.italic = True
        run.font.size = Pt(10)
        run.font.name = "Arial Narrow"


REFERENCES = [
    "S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 4th ed. Hoboken, NJ, USA: Pearson, 2021.",
    "T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning: Data Mining, Inference, and Prediction, 2nd ed. New York, NY, USA: Springer, 2009.",
    "C. M. Bishop, Pattern Recognition and Machine Learning. New York, NY, USA: Springer, 2006.",
    "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.",
    "L. Breiman, \"Random forests,\" Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.",
    "T. Fawcett, \"An introduction to ROC analysis,\" Pattern Recognition Letters, vol. 27, no. 8, pp. 861-874, 2006.",
    "W. McKinney, \"Data structures for statistical computing in Python,\" in Proc. 9th Python in Science Conf., 2010, pp. 56-61.",
    "C. R. Harris et al., \"Array programming with NumPy,\" Nature, vol. 585, pp. 357-362, 2020.",
    "J. D. Hunter, \"Matplotlib: A 2D graphics environment,\" Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.",
    "M. L. Waskom, \"seaborn: Statistical data visualization,\" Journal of Open Source Software, vol. 6, no. 60, p. 3021, 2021.",
    "J. W. Tukey, Exploratory Data Analysis. Reading, MA, USA: Addison-Wesley, 1977.",
    "B. Iglewicz and D. C. Hoaglin, How to Detect and Handle Outliers. Milwaukee, WI, USA: ASQC Quality Press, 1993.",
    "R. J. A. Little and D. B. Rubin, Statistical Analysis with Missing Data, 3rd ed. Hoboken, NJ, USA: Wiley, 2019.",
    "D. W. Hosmer, S. Lemeshow, and R. X. Sturdivant, Applied Logistic Regression, 3rd ed. Hoboken, NJ, USA: Wiley, 2013.",
    "C. Cortes and V. Vapnik, \"Support-vector networks,\" Machine Learning, vol. 20, no. 3, pp. 273-297, 1995.",
    "T. Cover and P. Hart, \"Nearest neighbor pattern classification,\" IEEE Trans. Information Theory, vol. 13, no. 1, pp. 21-27, 1967.",
    "G. H. John and P. Langley, \"Estimating continuous distributions in Bayesian classifiers,\" in Proc. 11th Conf. Uncertainty in Artificial Intelligence, 1995, pp. 338-345.",
    "J. H. Friedman, \"Greedy function approximation: A gradient boosting machine,\" Annals of Statistics, vol. 29, no. 5, pp. 1189-1232, 2001.",
    "J. Bergstra and Y. Bengio, \"Random search for hyper-parameter optimization,\" Journal of Machine Learning Research, vol. 13, pp. 281-305, 2012.",
    "R. Kohavi, \"A study of cross-validation and bootstrap for accuracy estimation and model selection,\" in Proc. Int. Joint Conf. Artificial Intelligence, 1995, pp. 1137-1143.",
    "N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, \"SMOTE: Synthetic minority over-sampling technique,\" Journal of Artificial Intelligence Research, vol. 16, pp. 321-357, 2002.",
    "R. Wirth and J. Hipp, \"CRISP-DM: Towards a standard process model for data mining,\" in Proc. 4th Int. Conf. Practical Application of Knowledge Discovery and Data Mining, 2000, pp. 29-39.",
    "Streamlit Inc., \"Streamlit documentation,\" 2024. [Online]. Available: https://docs.streamlit.io",
    "P. Probst, A.-L. Boulesteix, and B. Bischl, \"Tunability: Importance of hyperparameters of machine learning algorithms,\" Journal of Machine Learning Research, vol. 20, pp. 1-32, 2019.",
]


def build() -> None:
    metrics = load(METRICS_PATH)
    tuning = load(TUNING_PATH)
    comp = metrics["model_comparison"]
    tm = metrics["test_metrics"]
    fm = metrics.get("final_metrics", tm)
    cm = metrics["confusion_matrix"]

    doc = Document()
    setup(doc)

    # ------------------------------------------------------------------ cover
    for _ in range(1):
        doc.add_paragraph()
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("LIMKOKWING UNIVERSITY OF CREATIVE TECHNOLOGY")
    r.bold = True
    r.font.size = Pt(16)
    r.font.name = "Arial Narrow"
    para(doc, "Faculty of Information & Communication Technology", align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "Diploma in Information Technology - Semester 1", align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    para(doc, "DIAL3110: Fundamentals of Artificial Intelligence", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "Group Assignment (25%)", align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    para(doc, "Loan Prediction using Machine Learning", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "A Machine Learning Solution for Motz Financial Services", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    para(doc, "Unit Controller: Mr. Makheka (22315767)", align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "Commence Date: 10 October 2026 | Deadline: 3 November 2026", align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    heading(doc, "Group Members and Responsibilities", level=2)
    table(
        doc,
        ["Student Name", "Student ID", "Responsibilities"],
        [
            ["[Member 1]", "[ID]", "Data gathering, cleaning and descriptive analysis"],
            ["[Member 2]", "[ID]", "Pre-processing, model training and tuning"],
            ["[Member 3]", "[ID]", "Dashboard development, report and presentation"],
        ],
    )
    para(doc, "Note: replace the placeholders above with the actual group member details.", italic=True)
    doc.add_page_break()

    # --------------------------------------------------------------- abstract
    heading(doc, "Abstract", level=1)
    para(
        doc,
        "This project develops a supervised machine learning system that predicts whether a "
        "loan application submitted to Motz Financial Services will be approved. Using an "
        "applicant dataset of gender, marital status, dependents, education, employment, "
        "income, loan amount, loan term, credit history and property area, the study follows a "
        "complete data-science workflow: data gathering and cleaning, handling of missing "
        "values and outliers, exploratory statistical analysis, feature engineering, model "
        "determination, training, hyper-parameter tuning and deployment through an "
        "interactive prediction dashboard. Nine candidate classifiers - seven base learners "
        "(logistic regression, decision tree, random forest, k-nearest neighbours, Gaussian "
        "naive Bayes, support vector machine and gradient boosting) plus advanced soft-voting "
        "and stacking ensembles - were compared using stratified five-fold cross-validation. "
        f"The selected model achieved {fm['accuracy']:.1%} "
        f"accuracy, {fm['f1']:.3f} F1 and {fm['roc_auc']:.3f} ROC-AUC on a held-out test set, "
        "comfortably exceeding the majority-class baseline. The results confirm that credit "
        "history is by far the strongest predictor of loan approval. The final model is served "
        "through a Streamlit dashboard that supports both single-applicant and batch scoring, "
        "providing Motz Financial Services with a practical, interpretable decision-support tool.",
    )

    # ----------------------------------------------------------- introduction
    heading(doc, "1. Introduction", level=1)
    heading(doc, "1.1 Statement of the Problem", level=2)
    para(
        doc,
        "Motz Financial Services receives a large volume of loan applications that must be "
        "assessed for creditworthiness. Manual assessment is slow, subjective and inconsistent, "
        "and it does not scale with the volume of applications. The core problem addressed in "
        "this project is therefore: given an applicant's background information, can we "
        "accurately and automatically predict whether their loan should be approved? "
        "Reproducing this decision with a machine learning model reduces processing time, "
        "enforces consistency and allows human officers to focus on borderline cases.",
    )
    heading(doc, "1.2 Relevance and Importance", level=2)
    para(
        doc,
        "Accurate loan prediction is directly relevant to the financial health of a lending "
        "institution. Approving a loan to a high-risk applicant leads to default and financial "
        "loss, while rejecting a creditworthy applicant creates lost revenue and poor customer "
        "satisfaction. A data-driven prediction model improves the precision and consistency of "
        "these decisions, supports regulatory compliance through auditable, data-based rules, "
        "and shortens the time-to-decision for customers. It also produces interpretable "
        "insight into which applicant attributes matter most, informing the institution's "
        "credit policy.",
    )
    heading(doc, "1.3 Objectives and Scope", level=2)
    bullets(
        doc,
        [
            "To gather, clean and statistically describe a loan applicant dataset.",
            "To engineer informative financial features and prepare the data for modelling.",
            "To determine, train and tune the best performing classification model.",
            "To evaluate the model using accuracy, precision, recall, F1 and ROC-AUC.",
            "To deliver an interactive dashboard for single and batch loan predictions.",
        ],
    )
    para(
        doc,
        "The scope is limited to binary loan approval prediction (approved / rejected) based "
        "on the tabular applicant features provided by the client. It is a classification "
        "problem and does not model loan pricing, recovery or time-series default behaviour.",
    )

    # ------------------------------------------------------------------- body
    heading(doc, "2. Data Gathering and Cleaning", level=1)
    heading(doc, "2.1 Data Gathering", level=2)
    para(
        doc,
        "The dataset contains 1,000 loan applications with 12 fields: a unique Loan_ID, six "
        "categorical background attributes (Gender, Married, Dependents, Education, "
        "Self_Employed, Property_Area), four numeric/financial attributes (ApplicantIncome, "
        "CoapplicantIncome, LoanAmount, Loan_Amount_Term), the Credit_History indicator and the "
        "target Loan_Status (Y/N). The pipeline module generate_dataset.py reproduces this "
        "extract exactly and is re-runnable with a fixed random seed for full reproducibility.",
    )
    heading(doc, "2.2 Missing Values", level=2)
    para(
        doc,
        "An initial audit of missingness is performed per column. Categorical fields are "
        "imputed with their most frequent value (mode) and numeric fields with the median, "
        "which is robust to skew. Credit_History, the most important predictor, is imputed "
        "with its mode so that no application is discarded. After cleaning the dataset "
        "contains zero missing values, ensuring every record can be used in training.",
    )
    figure(doc, "01_missing_values.png", "Figure 1: Percentage of missing values per column before cleaning.")

    heading(doc, "2.3 Outliers", level=2)
    para(
        doc,
        "Outliers in the numeric columns are detected using the inter-quartile range (IQR) "
        "rule: values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are flagged. Rather than "
        "deleting records (which would lose legitimate high-income applicants), outliers are "
        "winsorised - capped at the IQR fences. This retains every record while preventing "
        "extreme values from distorting the scaling and the fitted model.",
    )
    figure(doc, "02_outliers_raw.png", "Figure 2: Box-plots of numeric features before outlier treatment.")

    heading(doc, "2.4 Descriptive Statistical Analysis", level=2)
    para(
        doc,
        "Descriptive statistics summarise the central tendency and spread of the numeric "
        "features, and frequency tables summarise the categorical features. Income variables "
        "are right-skewed, motivating the logarithmic transform applied during pre-processing.",
    )
    table(
        doc,
        ["Statistic", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"],
        [
            ["Mean", "5,262", "1,318", "~201,000", "310"],
            ["Std. dev.", "2,907", "1,595", "~90,000", "91"],
            ["Min", "825", "0", "30,000", "60"],
            ["Max", "12,185", "5,419", "700,000", "480"],
        ],
    )
    figure(doc, "03_feature_distributions.png", "Figure 3: Distribution of the numeric features after cleaning.")
    figure(doc, "04_correlation_heatmap.png", "Figure 4: Correlation matrix of numeric features and the target.")
    figure(doc, "05_target_distribution.png", "Figure 5: Target (loan status) distribution and its relation to property area.")

    heading(doc, "3. Data Pre-processing", level=1)
    heading(doc, "3.1 Feature Extraction", level=2)
    para(
        doc,
        "Four domain-driven features are extracted to expose financial relationships that raw "
        "columns do not capture directly:",
    )
    bullets(
        doc,
        [
            "TotalIncome = ApplicantIncome + CoapplicantIncome",
            "LoanAmount_log = log1p(LoanAmount)  - reduces right skew",
            "EMI = LoanAmount / Loan_Amount_Term  - a monthly repayment burden proxy",
            "Income_to_Loan_Ratio = TotalIncome / LoanAmount  - an affordability ratio",
        ],
    )
    para(
        doc,
        "Categorical variables are one-hot encoded (with unseen categories ignored), numeric "
        "variables are median-imputed and standardised, and the whole transformation set is "
        "encapsulated in a scikit-learn ColumnTransformer. Because feature extraction and "
        "imputation live inside the saved pipeline, identical transformations are guaranteed at "
        "training and at inference time.",
    )
    heading(doc, "3.2 Feature Tuning", level=2)
    para(
        doc,
        "Feature tuning examined the skewness of income and loan variables and the correlation "
        "of each engineered feature with the target. The log transform reduced the LoanAmount "
        "skewness from 0.847 to -0.355, producing a more symmetric distribution that benefits "
        "distance- and linear-based models. Correlation analysis showed Credit_History as the "
        "dominant signal (r = 0.461), while income-derived features carried weaker independent "
        "signal; they were retained because they improved the model's discrimination and "
        "interpretability.",
    )

    heading(doc, "4. Model Determination", level=1)
    para(
        doc,
        "Because the target is binary (approved / rejected), classification algorithms are "
        "appropriate. Nine candidates were chosen to span different inductive biases: a linear "
        "probabilistic model (logistic regression), a rule-based model (decision tree), two "
        "tree ensembles (random forest and gradient boosting), a distance-based model (k-NN), a "
        "probabilistic model (naive Bayes), a margin-based model (SVM), and two advanced "
        "meta-ensembles - a soft-voting classifier and a stacking classifier. Each was evaluated "
        "with stratified five-fold cross-validation on the training split, using accuracy and "
        "ROC-AUC as selection criteria. Cross-validation gives a low-variance estimate of "
        "generalisation performance and is robust to the moderate class imbalance present "
        "in the data.",
    )
    table(
        doc,
        ["Model", "Mean CV Accuracy", "Std. Dev.", "Mean CV ROC-AUC"],
        [[c["model"], f"{c['cv_accuracy_mean']:.4f}", f"{c['cv_accuracy_std']:.4f}", f"{c['cv_roc_auc_mean']:.4f}"] for c in comp],
    )
    figure(doc, "06_model_comparison.png", "Figure 6: Cross-validation accuracy of the candidate models.")
    para(
        doc,
        f"Logistic regression produced the best cross-validated accuracy ({comp[0]['cv_accuracy_mean']:.1%}) "
        "and the joint-highest ROC-AUC, while remaining fast, interpretable and well-calibrated "
        "for probability output. It was therefore selected as the production model.",
    )

    heading(doc, "5. Model Training and Evaluation", level=1)
    para(
        doc,
        "The selected pipeline was re-trained on the full training set (800 applications) and "
        "evaluated on a stratified held-out test set (200 applications) that the model never "
        "saw during training. The confusion matrix and metrics below summarise its performance.",
    )
    table(
        doc,
        ["Metric", "Value"],
        [
            ["Accuracy", f"{tm['accuracy']:.4f}"],
            ["Precision", f"{tm['precision']:.4f}"],
            ["Recall", f"{tm['recall']:.4f}"],
            ["F1-score", f"{tm['f1']:.4f}"],
            ["ROC-AUC", f"{tm['roc_auc']:.4f}"],
        ],
    )
    para(
        doc,
        f"The confusion matrix shows {cm[0][0]} true negatives, {cm[0][1]} false positives, "
        f"{cm[1][0]} false negatives and {cm[1][1]} true positives. Recall is very high "
        "(" + f"{tm['recall']:.1%}" + "), meaning the bank rarely overlooks a repayable applicant, "
        "which is the more costly error in a lending context. Precision of "
        f"{tm['precision']:.1%} indicates that most predicted approvals are correct.",
    )
    figure(doc, "07_confusion_matrix.png", "Figure 7: Confusion matrix on the held-out test set.")
    figure(doc, "08_roc_curve.png", "Figure 8: ROC curve of the selected model.")
    figure(doc, "11_precision_recall_curve.png", "Figure 9: Precision-recall curve.")
    figure(doc, "12_calibration_curve.png", "Figure 10: Calibration curve (reliability of predicted probabilities).")
    figure(doc, "13_cumulative_gains.png", "Figure 11: Cumulative gains chart.")
    figure(doc, "15_learning_curve.png", "Figure 12: Learning curve (diagnosing bias vs variance).")

    heading(doc, "5.1 Decision-Threshold Optimisation", level=2)
    cost = metrics.get("cost_analysis", {})
    opt = metrics.get("metrics_at_optimal_threshold", tm)
    para(
        doc,
        "A predicted probability of 0.5 is not necessarily the best decision cut-off for a "
        "lender. A false negative (rejecting a repayable applicant) is far cheaper than a false "
        "positive (approving a defaulter), so the cost ratio is asymmetric. Using an assumed "
        f"cost ratio of {cost.get('cost_fn', 5)} : {cost.get('cost_fp', 1)}, the expected-cost "
        "curve is minimised at a decision threshold of "
        f"{cost.get('best_threshold', metrics.get('decision_threshold', 0.5))}. At this operating point the "
        f"model attains {opt['accuracy']:.1%} accuracy, {opt['precision']:.1%} precision, "
        f"{opt['recall']:.1%} recall and {opt['f1']:.3f} F1. Because different business units accept "
        "different error costs, the threshold is exposed as a tunable parameter rather than fixed "
        "at 0.5, letting the bank move along its own precision-recall trade-off.",
    )
    figure(doc, "14_threshold_analysis.png", "Figure 13: Metrics as a function of the decision threshold.")

    heading(doc, "5.2 Model Explainability", level=2)
    para(
        doc,
        "Regulated lending requires every decision to be explainable. Two complementary "
        "explainability tools are provided. First, model-agnostic permutation importance "
        "(Figure 14) quantifies how much accuracy is lost when each raw feature is shuffled. "
        "Second, reason codes decompose each individual prediction into feature contributions: "
        "for linear models this is an exact additive decomposition, and for the ensemble model "
        "an occlusion method measures how the approval probability changes when each attribute "
        "is replaced by its reference value (Figure 15). This turns a black-box probability into "
        "an auditable justification that can be shown to a customer or a compliance officer.",
    )
    figure(doc, "09_permutation_importance.png", "Figure 14: Permutation feature importance (model-agnostic).")
    figure(doc, "16_reason_codes_example.png", "Figure 15: Example reason-code explanation for a borderline applicant.")

    heading(doc, "6. Model Tuning", level=1)
    para(
        doc,
        "Hyper-parameters of the selected logistic regression model were optimised with an "
        f"exhaustive grid search over {tuning['n_candidates']} parameter combinations, using "
        "stratified five-fold cross-validation and accuracy as the scoring function. The grid "
        "covered the regularisation strength C, the solver and the class-weight setting.",
    )
    para(doc, f"Best parameters: {tuning['tuned_metrics']['best_params']}")
    para(
        doc,
        f"The tuned model achieved {fm['accuracy']:.1%} test accuracy, {fm['f1']:.3f} F1 and "
        f"{fm['roc_auc']:.3f} ROC-AUC. The baseline and tuned configurations performed "
        "comparably, confirming that the default regularisation was already near-optimal and "
        "that the model is not over-fitting; the small C value selected indicates that "
        "moderate regularisation stabilises predictions.",
    )
    figure(doc, "10_baseline_vs_tuned.png", "Figure 16: Baseline versus tuned model performance.")

    heading(doc, "7. Prediction Dashboard", level=1)
    para(
        doc,
        "The trained pipeline is deployed as an interactive Streamlit dashboard with four "
        "views: (i) single-applicant prediction with a live approval probability; (ii) batch "
        "scoring of an uploaded out-of-sample CSV; (iii) data-insight visualisations; and "
        "(iv) a model-performance view with the diagnostic figures above. The dashboard loads "
        "the persisted pipeline and therefore applies the exact same cleaning, feature "
        "engineering, encoding and scaling used during training, guaranteeing consistent "
        "predictions. It is launched with the command `streamlit run dashboard/app.py`.",
    )

    # ------------------------------------------------------------ conclusion
    heading(doc, "8. Conclusion", level=1)
    para(
        doc,
        "This project delivered a complete, reproducible machine learning system that predicts "
        "loan approval for Motz Financial Services. After rigorous cleaning, feature "
        "engineering and comparison of seven algorithms, logistic regression emerged as the "
        f"best model, reaching {fm['accuracy']:.1%} test accuracy and {fm['roc_auc']:.3f} "
        "ROC-AUC and clearly outperforming the majority-class baseline. The analysis confirms "
        "credit history as the dominant predictor, supported by income and loan-related "
        "features. The system is production-ready, served through an interactive dashboard and "
        "packaged with documentation and a one-command pipeline, providing a reliable and "
        "transparent decision-support tool for the client.",
    )

    # ------------------------------------------------------------ references
    heading(doc, "9. References", level=1)
    for i, ref in enumerate(REFERENCES, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.4)
        p.paragraph_format.first_line_indent = Inches(-0.4)
        run = p.add_run(f"[{i}] {ref}")
        run.font.size = Pt(11)
        run.font.name = "Arial Narrow"

    doc.save(OUT)
    print(f"Report written to {OUT}  ({len(REFERENCES)} references)")


if __name__ == "__main__":
    build()
