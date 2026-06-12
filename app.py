"""Explainable Credit Risk Scorer — every decision comes with its reasons.

A loan-application form scored by XGBoost on the classic German Credit
dataset, with SHAP waterfalls, plain-English reasons for and against,
and actionable guidance. Protected attributes are deliberately excluded.
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap
import streamlit as st

from data_prep import DECODE
from model import score_application, train_model

st.set_page_config(page_title="Explainable Credit Risk Scorer",
                   page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Training credit model on the German Credit dataset…")
def load_artifacts():
    return train_model()


ART = load_artifacts()

PERSONAS = {
    "— choose a preset —": None,
    "🎓 Recent graduate, first loan": dict(
        checking="low balance (0–200 DM)", duration=24, amount=4500,
        credit_history="no credits taken", purpose="education",
        savings="under 100 DM", employment="under 1 year", installment_rate=4,
        property="none", age=23, housing="renting", existing_credits=1,
        job="skilled employee"),
    "🏠 Established homeowner, car loan": dict(
        checking="healthy balance (≥ 200 DM)", duration=18, amount=3200,
        credit_history="existing credits paid duly", purpose="used car",
        savings="500–1,000 DM", employment="≥ 7 years", installment_rate=2,
        property="real estate", age=44, housing="homeowner",
        existing_credits=1, job="management / self-employed"),
    "⚠️ Overextended borrower": dict(
        checking="overdrawn (below 0 DM)", duration=48, amount=9800,
        credit_history="past payment delays", purpose="business",
        savings="none / unknown", employment="1–4 years", installment_rate=4,
        property="none", age=31, housing="renting", existing_credits=3,
        job="unskilled, resident"),
}

# ---------------- sidebar: the application form ----------------

with st.sidebar:
    st.title("⚖️ Loan application")

    persona = st.selectbox("Try a preset applicant", list(PERSONAS))
    p = PERSONAS[persona] or {}

    def sel(label, key, default_idx=0):
        opts = list(DECODE[key].values())
        idx = opts.index(p[key]) if p.get(key) in opts else default_idx
        return st.selectbox(label, opts, index=idx)

    checking = sel("Checking account status", "checking", 1)
    savings = sel("Savings", "savings", 1)
    employment = sel("Years in current employment", "employment", 2)
    credit_history = sel("Credit history", "credit_history", 2)
    purpose = sel("Loan purpose", "purpose")
    housing = sel("Housing", "housing", 1)
    prop = sel("Property owned", "property")
    job = sel("Job type", "job", 2)

    amount = st.slider("Loan amount (DM)", 250, 18000, p.get("amount", 3000), 250)
    duration = st.slider("Loan term (months)", 6, 72, p.get("duration", 24), 6)
    installment_rate = st.slider("Repayment, % of disposable income", 1, 4,
                                 p.get("installment_rate", 3))
    age = st.slider("Age", 19, 75, p.get("age", 35))
    existing_credits = st.slider("Existing credits at this bank", 1, 4,
                                 p.get("existing_credits", 1))

    threshold = st.slider("Approval threshold (max default risk)", 0.10, 0.60,
                          0.35, 0.05,
                          help="Applications with predicted default probability "
                               "below this are approved.")

    m = ART["metrics"]
    st.divider()
    st.subheader("Model card")
    st.markdown(
        f"""
- **Model:** XGBoost · SHAP explanations
- **Data:** [UCI German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) (1,000 applications)
- **ROC-AUC:** {m['roc_auc']:.3f} · **PR-AUC:** {m['avg_precision']:.3f}
- **Fairness:** sex and nationality are **deliberately excluded** from the model
- Decisions here are illustrative — not financial advice
"""
    )
    st.markdown(
        "Built by **[Drishtant Leuva](https://www.linkedin.com/in/drishtant-leuva/)** · "
        "[source code](https://github.com/drishtantleuva/credit-risk-scorer)"
    )

application = dict(
    checking=checking, duration=duration, credit_history=credit_history,
    purpose=purpose, amount=amount, savings=savings, employment=employment,
    installment_rate=installment_rate, property=prop, age=age,
    housing=housing, existing_credits=existing_credits, job=job,
)

result = score_application(ART, application)
pd_prob = result["pd"]
approved = pd_prob < threshold

# ---------------- main panel ----------------

st.title("Explainable Credit Risk Scorer")
st.caption(
    "Every decision comes with its reasons — SHAP-powered explanations on the "
    "classic German Credit dataset. Adjust the application on the left and watch "
    "the decision, and its explanation, update live."
)

left, right = st.columns([2, 3], gap="large")

with left:
    verdict = "✅ APPROVED" if approved else "❌ DECLINED"
    color = "#21c98d" if approved else "#ff5c5c"
    st.markdown(
        f"<h2 style='color:{color};margin-bottom:0'>{verdict}</h2>",
        unsafe_allow_html=True,
    )
    st.caption(f"Predicted default probability vs. threshold of {threshold:.0%}")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pd_prob * 100,
        number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Default risk"},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": color},
            "steps": [
                {"range": [0, threshold * 100], "color": "rgba(33,201,141,0.15)"},
                {"range": [threshold * 100, 100], "color": "rgba(255,92,92,0.12)"},
            ],
            "threshold": {
                "line": {"color": "#888", "width": 3},
                "value": threshold * 100,
            },
        },
    ))
    gauge.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=10))
    st.plotly_chart(gauge, use_container_width=True)

    if result["hurts"]:
        st.markdown("**Working against this application:**")
        for c in result["hurts"]:
            st.markdown(f"- 🔻 {c['text']}")
    if result["helps"]:
        st.markdown("**Working in its favour:**")
        for c in result["helps"]:
            st.markdown(f"- 🟢 {c['text']}")

    if not approved and result["advice"]:
        st.markdown("**What could change the outcome:**")
        for a in result["advice"]:
            st.markdown(f"- 💡 {a.capitalize()}")

with right:
    st.subheader("How the model weighed it — SHAP waterfall")
    st.caption(
        "Each bar shows how one attribute pushed the risk estimate up (red) "
        "or down (blue) from the average applicant."
    )
    fig, ax = plt.subplots()
    shap.plots.waterfall(result["explanation"], max_display=12, show=False)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close("all")

st.divider()
st.caption(
    "Trained on the [Statlog German Credit dataset]"
    "(https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) "
    "(UCI Machine Learning Repository, 1,000 anonymised loan applications, 1990s "
    "German bank; amounts in Deutsche Mark). Educational demonstration of "
    "explainable credit decisioning — not a real lending system and not "
    "financial advice."
)
