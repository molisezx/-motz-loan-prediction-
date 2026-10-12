"""Task 6: Professional loan-prediction dashboard (Streamlit).

A themed, multi-view decision-support application:
  Home · Predict · Batch scoring · Data insights · Model performance ·
  Explainability · About

Run with:  streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import CLEAN_DATA, FIGURES_DIR, METRICS_PATH, MODEL_PATH, TRAIN_DATA  # noqa: E402
from explain import reason_codes  # noqa: E402

PRIMARY = "#1F7A8C"
GOOD = "#2A9D8F"
BAD = "#E76F51"


# ---------------------------------------------------------------- utilities
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def inject_css():
    st.markdown(
        f"""
        <style>
        .main {{ background: linear-gradient(180deg,#F7F9FC 0%,#EEF3F8 100%); }}
        .block-container {{ padding-top: 1.6rem; max-width: 1180px; }}
        h1, h2, h3 {{ color: #14303B; letter-spacing: -0.3px; }}
        .brand {{
            background: linear-gradient(90deg,{PRIMARY} 0%,#2A9D8F 100%);
            padding: 22px 26px; border-radius: 16px; color: white; margin-bottom: 8px;
            box-shadow: 0 6px 20px rgba(31,122,140,0.25);
        }}
        .brand h1 {{ color: white; margin: 0; font-size: 1.7rem; }}
        .brand p {{ margin: 4px 0 0; opacity: .92; }}
        .card {{
            background: white; border-radius: 14px; padding: 18px 20px;
            box-shadow: 0 4px 16px rgba(20,48,59,0.08); border: 1px solid #E6EDF3;
        }}
        .metric-card {{
            background: white; border-radius: 14px; padding: 14px 16px; text-align: center;
            box-shadow: 0 4px 14px rgba(20,48,59,0.08); border-top: 4px solid {PRIMARY};
        }}
        .metric-card .v {{ font-size: 1.7rem; font-weight: 700; color:#14303B; }}
        .metric-card .l {{ font-size: .78rem; text-transform: uppercase; color:#5C7183; letter-spacing:.04em; }}
        .pill {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size:.75rem; font-weight:600; }}
        .pill-good {{ background:#E4F5F1; color:#1E7A6B; }}
        .pill-bad {{ background:#FDEAE4; color:#B5432A; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, border=PRIMARY):
    return (
        f'<div class="metric-card" style="border-top-color:{border}">'
        f'<div class="v">{value}</div><div class="l">{label}</div></div>'
    )


def reason_chart(reason):
    rows = sorted(reason["top_reasons"], key=lambda r: r["contribution"])
    names = [r["feature"] for r in rows]
    vals = [r["contribution"] for r in rows]
    colors = [GOOD if v >= 0 else BAD for v in vals]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("Contribution to approval score")
    ax.set_title("Decision drivers for this applicant")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- pages
def page_home(metrics):
    st.markdown(
        '<div class="brand"><h1>Motz Financial Services</h1>'
        "<p>AI-powered loan decision support &nbsp;·&nbsp; DIAL3110 Fundamentals of Artificial Intelligence</p></div>",
        unsafe_allow_html=True,
    )
    fm = metrics.get("final_metrics", metrics.get("test_metrics", {}))
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Accuracy", f"{fm.get('accuracy', 0):.1%}"), unsafe_allow_html=True)
    c2.markdown(metric_card("Precision", f"{fm.get('precision', fm.get('precision', 0)):.1%}" if fm.get("precision") else "—"), unsafe_allow_html=True)
    c3.markdown(metric_card("Recall", f"{fm.get('recall', 0):.1%}" if fm.get("recall") else "—"), unsafe_allow_html=True)
    c4.markdown(metric_card("ROC-AUC", f"{fm.get('roc_auc', 0):.3f}", GOOD), unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("What this system does")
        st.write(
            "It predicts whether a loan application will be **approved** or **rejected** from the "
            "applicant's background, then explains every decision in plain terms. The model was "
            "selected from nine algorithms and tuned for the bank's asymmetric cost of errors."
        )
        st.markdown(
            '<span class="pill pill-good">Explainable AI</span> &nbsp; '
            '<span class="pill pill-good">Cost-optimal threshold</span> &nbsp; '
            '<span class="pill pill-good">Ensemble modelling</span>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Pipeline at a glance")
        st.markdown(
            "1. Data cleaning &nbsp;→&nbsp; 2. Feature engineering &nbsp;→&nbsp; "
            "3. Model selection &nbsp;→&nbsp; 4. Training &nbsp;→&nbsp; 5. Tuning &nbsp;→&nbsp; "
            "6. Threshold optimisation &nbsp;→&nbsp; 7. Dashboard"
        )
        st.caption("Use the sidebar to predict an application, score a batch, or inspect the model.")
        st.markdown("</div>", unsafe_allow_html=True)


def page_predict(model, metrics, background):
    st.subheader("Single applicant prediction")
    threshold = float(metrics.get("decision_threshold", 0.5))
    st.caption(f"Decisions use the cost-optimal threshold of {threshold:.2f}.")

    with st.form("applicant"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with col2:
            self_employed = st.selectbox("Self employed", ["No", "Yes"])
            property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
            credit_history = st.selectbox("Credit history", ["Good (1)", "Bad (0)"])
            loan_term = st.selectbox("Loan term (months)", [360, 180, 240, 120, 84, 60, 300, 480])
        with col3:
            applicant_income = st.number_input("Applicant income / month", 0, 200000, 5000, step=500)
            coapplicant_income = st.number_input("Co-applicant income / month", 0, 200000, 1500, step=500)
            loan_amount = st.number_input("Loan amount", 20000, 700000, 150000, step=10000)
        submitted = st.form_submit_button("Assess application", type="primary")

    if submitted:
        record = pd.DataFrame([{
            "Gender": gender, "Married": married, "Dependents": dependents,
            "Education": education, "Self_Employed": self_employed,
            "ApplicantIncome": float(applicant_income), "CoapplicantIncome": float(coapplicant_income),
            "LoanAmount": float(loan_amount), "Loan_Amount_Term": float(loan_term),
            "Credit_History": 1.0 if credit_history.startswith("Good") else 0.0,
            "Property_Area": property_area,
        }])
        proba = float(model.predict_proba(record)[:, 1][0])
        decision = "Approved" if proba >= threshold else "Rejected"
        color = GOOD if proba >= threshold else BAD

        r1, r2 = st.columns([1, 1.3])
        with r1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### Decision: <span style='color:{color}'>{decision}</span>", unsafe_allow_html=True)
            st.metric("Approval probability", f"{proba:.1%}")
            st.progress(min(max(proba, 0.0), 1.0))
            st.caption(f"Threshold {threshold:.2f} · model: {metrics.get('selected_model', 'n/a')}")
            st.markdown("</div>", unsafe_allow_html=True)
        with r2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            reason = reason_codes(model, record, background=background, top_k=8)
            st.caption(f"Explanation method: {reason['method']}")
            st.pyplot(reason_chart(reason), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


def page_batch(model, metrics):
    st.subheader("Batch scoring")
    threshold = float(metrics.get("decision_threshold", 0.5))
    st.write("Upload an out-of-sample CSV file (the same columns as the training data). "
             "Missing values are handled automatically.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(8), use_container_width=True)
        X = df.drop(columns=[c for c in ("Loan_Status", "Loan_ID") if c in df.columns])
        proba = model.predict_proba(X)[:, 1]
        out = df.copy()
        out["Approval_Probability"] = proba.round(4)
        out["Decision"] = np.where(proba >= threshold, "Approved", "Rejected")
        approved = int((out["Decision"] == "Approved").sum())
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("Applications", len(out)), unsafe_allow_html=True)
        c2.markdown(metric_card("Approved", approved, GOOD), unsafe_allow_html=True)
        c3.markdown(metric_card("Rejected", len(out) - approved, BAD), unsafe_allow_html=True)
        st.write("")
        st.dataframe(out, use_container_width=True)
        st.bar_chart(out["Decision"].value_counts())
        st.download_button("Download scored CSV", out.to_csv(index=False), "loan_predictions.csv", "text/csv")


def page_insights():
    st.subheader("Data insights")
    try:
        df = load_csv(CLEAN_DATA)
    except FileNotFoundError:
        st.warning("Run `python run_pipeline.py` first.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Applications", f"{len(df):,}"), unsafe_allow_html=True)
    c2.markdown(metric_card("Approval rate", f"{(df['Loan_Status']=='Y').mean():.1%}"), unsafe_allow_html=True)
    c3.markdown(metric_card("Avg. income", f"{df['ApplicantIncome'].mean():,.0f}"), unsafe_allow_html=True)
    c4.markdown(metric_card("Avg. loan", f"{df['LoanAmount'].mean():,.0f}"), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown("**Approval rate by credit history**")
        st.bar_chart(df.groupby("Credit_History")["Loan_Status"].apply(lambda s: (s == "Y").mean()))
        st.markdown("**Applicant income distribution**")
        st.bar_chart(df["ApplicantIncome"].value_counts(bins=20).sort_index())
    with right:
        st.markdown("**Approval rate by property area**")
        st.bar_chart(df.groupby("Property_Area")["Loan_Status"].apply(lambda s: (s == "Y").mean()))
        st.markdown("**Loan status split**")
        st.bar_chart(df["Loan_Status"].value_counts())
    st.markdown("**Descriptive statistics**")
    st.dataframe(df.describe().round(1), use_container_width=True)


def page_performance(metrics):
    st.subheader("Model performance")
    fm = metrics.get("final_metrics", metrics.get("test_metrics", {}))
    st.markdown(
        f"**Selected model:** `{metrics.get('selected_model', 'n/a')}` &nbsp;·&nbsp; "
        f"**Adopted:** `{metrics.get('adopted_model', 'n/a')}` &nbsp;·&nbsp; "
        f"**Decision threshold:** `{metrics.get('decision_threshold', 0.5)}`"
    )
    c = st.columns(5)
    for col, key in zip(c, ["accuracy", "precision", "recall", "f1", "roc_auc"], strict=False):
        if key in fm:
            col.markdown(metric_card(key.upper(), f"{fm[key]:.3f}"), unsafe_allow_html=True)

    st.write("")
    st.markdown("**Candidate model comparison (5-fold cross-validation)**")
    comp = pd.DataFrame(metrics.get("model_comparison", []))
    if not comp.empty:
        show = comp[["model", "cv_accuracy_mean", "cv_roc_auc_mean", "cv_precision_mean", "cv_recall_mean", "cv_f1_mean"]]
        st.dataframe(show.rename(columns=lambda c: c.replace("cv_", "").replace("_mean", "").upper()),
                     use_container_width=True)

    cost = metrics.get("cost_analysis", {})
    if cost:
        st.info(
            f"Cost-optimal threshold **{cost.get('best_threshold')}** "
            f"(assumed cost ratio FN:FP = {cost.get('cost_fn')}:{cost.get('cost_fp')}); "
            f"expected cost {cost.get('min_expected_cost'):.0f}."
        )

    st.markdown("**Diagnostic figures**")
    figs = [
        "06_model_comparison.png", "07_confusion_matrix.png", "08_roc_curve.png",
        "11_precision_recall_curve.png", "12_calibration_curve.png",
        "13_cumulative_gains.png", "14_threshold_analysis.png", "15_learning_curve.png",
        "10_baseline_vs_tuned.png",
    ]
    cols = st.columns(2)
    for i, fig in enumerate(figs):
        p = FIGURES_DIR / fig
        if p.exists():
            cols[i % 2].image(str(p), caption=fig.replace("_", " ").replace(".png", ""), use_container_width=True)


def page_explainability(metrics):
    st.subheader("Explainability")
    st.write(
        "Explainable AI makes each prediction auditable. **Permutation importance** shows which "
        "features matter globally; **reason codes** decompose one applicant's decision into "
        "individual contributions."
    )
    p = FIGURES_DIR / "09_permutation_importance.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    pr = FIGURES_DIR / "16_reason_codes_example.png"
    if pr.exists():
        st.image(str(pr), use_container_width=True)
    imp = metrics.get("permutation_importance", [])
    if imp:
        st.markdown("**Top features (drop in accuracy when shuffled)**")
        st.dataframe(pd.DataFrame(imp).rename(columns={"feature": "Feature", "importance_mean": "Importance"}),
                     use_container_width=True)


def page_about(metrics):
    st.subheader("About the model")
    fm = metrics.get("final_metrics", {})
    st.markdown(
        f"""
- **Algorithm:** {metrics.get('selected_model', 'n/a')}
- **Features:** 11 raw applicant attributes → 4 engineered financial features
- **Training data:** 1,000 applications (800 train / 200 test, stratified)
- **Validation:** stratified 5-fold cross-validation across 9 candidate models
- **Best parameters:** `{fm.get('best_params', {})}`
- **Intended use:** decision support for loan officers; human review required for edge cases
- **Limitations:** synthetic/reconstructed extract; credit history is the dominant signal
        """
    )
    st.caption("DIAL3110 Fundamentals of Artificial Intelligence · Group Assignment 2026")


# -------------------------------------------------------------------- main
def main():
    st.set_page_config(page_title="Motz Loan Predictor", page_icon="🏦", layout="wide")
    inject_css()

    model = load_model()
    metrics = load_json(METRICS_PATH)
    background = load_csv(TRAIN_DATA) if Path(TRAIN_DATA).exists() else None

    with st.sidebar:
        st.markdown("### 🏦 Motz Loan AI")
        page = st.radio(
            "Navigation",
            ["Home", "Predict", "Batch scoring", "Data insights", "Model performance", "Explainability", "About"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"Model: {metrics.get('selected_model', 'n/a')}")
        st.caption(f"Accuracy: {metrics.get('final_metrics', {}).get('accuracy', 0):.1%}")

    if page == "Home":
        page_home(metrics)
    elif page == "Predict":
        page_predict(model, metrics, background)
    elif page == "Batch scoring":
        page_batch(model, metrics)
    elif page == "Data insights":
        page_insights()
    elif page == "Model performance":
        page_performance(metrics)
    elif page == "Explainability":
        page_explainability(metrics)
    else:
        page_about(metrics)


if __name__ == "__main__":
    main()
