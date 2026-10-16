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

PRIMARY = "#22D3EE"
ACCENT = "#A855F7"
GOOD = "#34D399"
BAD = "#F87171"


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
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        :root{
            --primary:#22D3EE; --accent:#A855F7; --good:#34D399; --bad:#F87171;
            --ink:#E6F1FF; --muted:#8AA0B8; --line:rgba(255,255,255,.08);
        }

        html, body, .stApp, .stMarkdown, p, span, label { font-family:'Inter', system-ui, sans-serif; }

        .stApp{
            background:
                radial-gradient(1000px 620px at 10% -12%, rgba(34,211,238,.16), transparent 60%),
                radial-gradient(900px 620px at 102% -5%, rgba(168,85,247,.20), transparent 55%),
                radial-gradient(900px 900px at 50% 125%, rgba(34,211,238,.10), transparent 60%),
                #070B14;
            color:var(--ink);
        }
        .block-container{ padding-top:1.4rem; max-width:1200px; }

        h1,h2,h3,h4,h5{ color:var(--ink); letter-spacing:-.02em; }
        a{ color:var(--primary); }

        /* ---------------------------------------------------------- brand */
        .brand{
            position:relative; overflow:hidden;
            background:linear-gradient(120deg, rgba(34,211,238,.16), rgba(168,85,247,.18));
            border:1px solid var(--line); padding:24px 28px; border-radius:18px; margin-bottom:10px;
            box-shadow:0 0 0 1px rgba(34,211,238,.14), 0 18px 50px rgba(0,0,0,.55),
                       inset 0 1px 0 rgba(255,255,255,.08);
            backdrop-filter:blur(14px);
        }
        .brand::after{
            content:''; position:absolute; inset:0; pointer-events:none;
            background:radial-gradient(420px 220px at 88% -30%, rgba(168,85,247,.40), transparent 72%);
        }
        .brand h1{
            margin:0; font-size:1.75rem; font-weight:800; letter-spacing:-.02em;
            background:linear-gradient(90deg,#ffffff,#22D3EE 58%,#A855F7);
            -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
        }
        .brand p{ margin:7px 0 0; color:#AFC4DA; }
        .brand .tagline{ font-family:'JetBrains Mono', monospace; font-size:.76rem; letter-spacing:.14em;
            text-transform:uppercase; color:#7FE9F5; margin-top:10px; }
        .ai-dot{ display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--good);
            margin-right:8px; vertical-align:middle;
            box-shadow:0 0 0 0 rgba(52,211,153,.7); animation:pulse 2s infinite; }
        @keyframes pulse{
            0%{ box-shadow:0 0 0 0 rgba(52,211,153,.7); }
            70%{ box-shadow:0 0 0 10px rgba(52,211,153,0); }
            100%{ box-shadow:0 0 0 0 rgba(52,211,153,0); }
        }

        /* ---------------------------------------------------------- cards */
        .card{
            background:rgba(255,255,255,.035); border:1px solid var(--line);
            border-radius:16px; padding:18px 20px;
            box-shadow:0 12px 40px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.05);
            backdrop-filter:blur(12px);
        }
        .metric-card{
            position:relative; overflow:hidden; text-align:center; padding:16px;
            background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.015));
            border:1px solid var(--line); border-radius:16px;
            box-shadow:0 12px 34px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.06);
        }
        .metric-card::before{
            content:''; position:absolute; top:0; left:0; right:0; height:3px;
            background:linear-gradient(90deg,#22D3EE,#A855F7);
        }
        .metric-card .v{ font-size:1.75rem; font-weight:700; color:#fff;
            font-family:'JetBrains Mono', monospace; line-height:1.15; }
        .metric-card .l{ font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; color:var(--muted); }

        /* ---------------------------------------------------------- pills */
        .pill{ display:inline-block; padding:4px 12px; border-radius:999px; font-size:.74rem;
            font-weight:600; border:1px solid var(--line); }
        .pill-good{ background:rgba(52,211,153,.12); color:#6EE7B7; border-color:rgba(52,211,153,.35);
            box-shadow:0 0 18px rgba(52,211,153,.18); }
        .pill-bad{ background:rgba(248,113,113,.12); color:#FCA5A5; border-color:rgba(248,113,113,.35); }

        /* ------------------------------------------------------- sidebar */
        section[data-testid="stSidebar"]{
            background:linear-gradient(180deg, rgba(14,22,38,.92), rgba(7,11,20,.96));
            border-right:1px solid var(--line);
        }
        section[data-testid="stSidebar"] *{ color:#DCE8F5; }
        section[data-testid="stSidebar"] [role="radiogroup"] label{
            padding:8px 10px; border-radius:10px; transition:all .15s ease;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:hover{
            background:rgba(34,211,238,.10);
        }

        /* ------------------------------------------------------- widgets */
        .stButton>button, .stFormSubmitButton>button, .stDownloadButton>button{
            background:linear-gradient(90deg,#22D3EE,#A855F7); color:#06121F; border:0;
            border-radius:12px; font-weight:700; padding:.55rem 1.1rem;
            box-shadow:0 8px 30px rgba(34,211,238,.35); transition:all .15s ease;
        }
        .stButton>button:hover, .stFormSubmitButton>button:hover, .stDownloadButton>button:hover{
            filter:brightness(1.08); transform:translateY(-1px);
        }
        .stTextInput input, .stNumberInput input,
        .stSelectbox div[data-baseweb="select"]>div, .stMultiSelect div[data-baseweb="select"]>div{
            background:rgba(255,255,255,.05)!important; color:var(--ink)!important;
            border:1px solid rgba(255,255,255,.12)!important; border-radius:10px!important;
        }
        .stTextInput input:focus, .stNumberInput input:focus{
            border-color:var(--primary)!important; box-shadow:0 0 0 3px rgba(34,211,238,.18)!important;
        }
        [data-testid="stDataFrame"], [data-testid="stTable"]{
            border:1px solid var(--line); border-radius:12px; overflow:hidden;
        }
        .stProgress > div > div > div > div{
            background:linear-gradient(90deg,#22D3EE,#A855F7);
        }
        [data-testid="stMetricValue"]{ color:#fff; font-family:'JetBrains Mono', monospace; }
        [data-testid="stMetricLabel"]{ color:var(--muted); }
        .stAlert{ background:rgba(34,211,238,.08); border:1px solid rgba(34,211,238,.25); border-radius:12px; }
        hr{ border-color:var(--line); }

        ::-webkit-scrollbar{ width:10px; height:10px; }
        ::-webkit-scrollbar-thumb{ background:rgba(34,211,238,.35); border-radius:999px; }
        ::-webkit-scrollbar-track{ background:transparent; }
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
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor="#0B1220")
    ax.set_facecolor("#0B1220")
    ax.barh(names, vals, color=colors, edgecolor="none")
    ax.axvline(0, color="#3B4A63", linewidth=0.9)
    ax.tick_params(colors="#AFC4DA")
    for spine in ax.spines.values():
        spine.set_color("#243247")
    ax.set_xlabel("Contribution to approval score", color="#AFC4DA")
    ax.set_title("Decision drivers for this applicant", color="#E6F1FF")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- pages
def page_home(metrics):
    st.markdown(
        '<div class="brand"><h1>Motz Financial Services</h1>'
        "<p><span class='ai-dot'></span>Neural decision engine online &nbsp;·&nbsp; "
        "AI-powered loan decision support</p>"
        "<p class='tagline'>DIAL3110 · Fundamentals of Artificial Intelligence</p></div>",
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
- **Training data:** Kaggle loan-prediction dataset (614 applications; 491 train / 123 test, stratified)
- **Validation:** stratified 5-fold cross-validation across 9 candidate models
- **Best parameters:** `{fm.get('best_params', {})}`
- **Intended use:** decision support for loan officers; human review required for edge cases
- **Limitations:** public Kaggle extract; credit history is the dominant signal
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
