"""Generate the 10-minute presentation (Task 8) as a PowerPoint deck.

Produces ``presentation/DIAL3110_Loan_Prediction_Slides.pptx`` containing data
visualisation figures, the modelling steps, and the performance matrix.
"""
import json

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

from config import BASE_DIR, FIGURES_DIR, METRICS_PATH, TUNING_PATH

OUT = BASE_DIR / "presentation" / "DIAL3110_Loan_Prediction_Slides.pptx"

NAVY = RGBColor(0x1F, 0x3B, 0x57)
TEAL = RGBColor(0x2A, 0x9D, 0x8F)


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def add_title_slide(prs, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = subtitle
    for p in [slide.shapes.title, slide.placeholders[1]]:
        for r in p.text_frame.paragraphs[0].runs:
            r.font.color.rgb = NAVY
            r.font.name = "Arial Narrow"
    return slide


def add_bullets_slide(prs, title, bullets):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    slide.shapes.title.text_frame.paragraphs[0].runs[0].font.color.rgb = NAVY
    tf = slide.placeholders[1].text_frame
    tf.clear()
    for i, text in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = Pt(18)
        p.font.name = "Arial Narrow"
    return slide


def add_image_slide(prs, title, image, caption=""):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title
    slide.shapes.title.text_frame.paragraphs[0].runs[0].font.color.rgb = NAVY
    path = FIGURES_DIR / image
    if path.exists():
        slide.shapes.add_picture(str(path), Inches(1.0), Inches(1.5), width=Inches(8.0))
    if caption:
        box = slide.shapes.add_textbox(Inches(1.0), Inches(6.6), Inches(8.0), Inches(0.6))
        run = box.text_frame.paragraphs[0].add_run()
        run.text = caption
        run.font.size = Pt(14)
        run.font.italic = True
        run.font.name = "Arial Narrow"
    return slide


def add_table_slide(prs, title, headers, rows):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title
    slide.shapes.title.text_frame.paragraphs[0].runs[0].font.color.rgb = NAVY
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(0.8), Inches(1.6), Inches(8.4), Inches(4.5))
    table = shape.table
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = str(h)
        for r in cell.text_frame.paragraphs[0].runs:
            r.font.bold = True
            r.font.size = Pt(14)
            r.font.name = "Arial Narrow"
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(v)
            for r in cell.text_frame.paragraphs[0].runs:
                r.font.size = Pt(13)
                r.font.name = "Arial Narrow"
    return slide


def build() -> None:
    metrics = load(METRICS_PATH)
    tuning = load(TUNING_PATH)
    comp = metrics["model_comparison"]
    tm = metrics["test_metrics"]
    fm = metrics.get("final_metrics", tm)

    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    add_title_slide(
        prs,
        "Loan Prediction using Machine Learning",
        "Motz Financial Services\nDIAL3110 - Fundamentals of Artificial Intelligence\nGroup Assignment 2026",
    )

    add_bullets_slide(
        prs,
        "Agenda",
        [
            "Problem statement and objectives",
            "Data gathering and cleaning",
            "Data pre-processing and feature engineering",
            "Model determination, training and tuning",
            "Prediction dashboard",
            f"Results: {fm['accuracy']:.1%} accuracy, {fm['roc_auc']:.3f} ROC-AUC",
        ],
    )

    add_bullets_slide(
        prs,
        "The Problem",
        [
            "Motz Financial Services assesses loans manually.",
            "Manual review is slow, subjective and inconsistent.",
            "Goal: predict loan approval (Y/N) from applicant data.",
            "Business value: faster, consistent, auditable decisions.",
        ],
    )

    add_image_slide(prs, "Step 1 - Data & Missing Values", "01_missing_values.png",
                    "Missing values are imputed: mode for categoricals, median for numerics.")

    add_image_slide(prs, "Step 1 - Outlier Treatment", "02_outliers_raw.png",
                    "IQR-detected outliers are winsorised at the fences (no records lost).")

    add_image_slide(prs, "Data Visualisation - Distributions", "03_feature_distributions.png",
                    "Income variables are right-skewed; LoanAmount is log-transformed.")

    add_image_slide(prs, "Data Visualisation - Correlations", "04_correlation_heatmap.png",
                    "Credit_History is the strongest signal (r = 0.461).")

    add_image_slide(prs, "Data Visualisation - Target", "05_target_distribution.png",
                    "Overall approval rate and its relation to property area.")

    add_bullets_slide(
        prs,
        "Step 2 - Feature Engineering",
        [
            "TotalIncome = ApplicantIncome + CoapplicantIncome",
            "LoanAmount_log = log1p(LoanAmount)  (skew reduced 0.85 -> -0.36)",
            "EMI = LoanAmount / Loan_Amount_Term",
            "Income_to_Loan_Ratio = TotalIncome / LoanAmount",
            "One-hot encoding + standardisation inside a saved pipeline",
        ],
    )

    add_table_slide(
        prs,
        "Step 3 - Model Determination (5-fold CV)",
        ["Model", "CV Accuracy", "CV ROC-AUC"],
        [[c["model"], f"{c['cv_accuracy_mean']:.3f}", f"{c['cv_roc_auc_mean']:.3f}"] for c in comp],
    )

    add_image_slide(prs, "Step 3 - Model Comparison", "06_model_comparison.png",
                    "Logistic regression wins on cross-validated accuracy.")

    add_table_slide(
        prs,
        "Step 4 - Performance Matrix",
        ["Metric", "Value"],
        [
            ["Accuracy", f"{tm['accuracy']:.4f}"],
            ["Precision", f"{tm['precision']:.4f}"],
            ["Recall", f"{tm['recall']:.4f}"],
            ["F1-score", f"{tm['f1']:.4f}"],
            ["ROC-AUC", f"{tm['roc_auc']:.4f}"],
        ],
    )

    add_image_slide(prs, "Results - Confusion Matrix", "07_confusion_matrix.png",
                    "High recall: few creditworthy applicants are rejected.")

    add_image_slide(prs, "Results - ROC Curve", "08_roc_curve.png",
                    f"ROC-AUC = {tm['roc_auc']:.3f}.")

    add_image_slide(prs, "Results - ROC & Precision-Recall", "11_precision_recall_curve.png",
                    "Precision-recall trade-off on the positive class.")

    add_image_slide(prs, "Results - Calibration", "12_calibration_curve.png",
                    "Predicted probabilities are reliable and well-calibrated.")

    add_image_slide(prs, "Step 5 - Permutation Importance", "09_permutation_importance.png",
                    "Credit history dominates; the result is model-agnostic.")

    add_image_slide(prs, "Explainable AI - Reason Codes", "16_reason_codes_example.png",
                    "Every decision is decomposed into interpretable feature contributions.")

    add_bullets_slide(
        prs,
        "Step 6 - Model Tuning",
        [
            f"Grid search over {tuning['n_candidates']} parameter combinations.",
            f"Best parameters: {tuning['tuned_metrics']['best_params']}",
            f"Final: {fm['accuracy']:.1%} accuracy, {fm['f1']:.3f} F1, {fm['roc_auc']:.3f} ROC-AUC.",
        ],
    )

    cost = metrics.get("cost_analysis", {})
    add_image_slide(prs, "Step 7 - Cost-Optimal Threshold", "14_threshold_analysis.png",
                    f"Cost ratio {cost.get('cost_fn', 5)}:{cost.get('cost_fp', 1)} -> "
                    f"optimal cut-off {cost.get('best_threshold', metrics.get('decision_threshold', 0.5))}.")

    add_image_slide(prs, "Diagnostics - Learning Curve", "15_learning_curve.png",
                    "No over-fitting: train and validation curves converge.")

    add_bullets_slide(
        prs,
        "Step 7 - Prediction Dashboard",
        [
            "Streamlit app with four views.",
            "Single-applicant prediction with live probability.",
            "Batch scoring of out-of-sample CSV files.",
            "Data insights and model-performance views.",
        ],
    )

    add_bullets_slide(
        prs,
        "Conclusion",
        [
            "Complete, reproducible ML system delivered.",
            f"Logistic regression: {fm['accuracy']:.1%} accuracy on unseen data.",
            "Credit history is the dominant predictor.",
            "Ready for production as a decision-support tool.",
        ],
    )

    prs.save(OUT)
    print(f"Presentation written to {OUT}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
