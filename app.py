"""Explainable Credit Risk Scorer — every decision comes with its reasons.

A loan-application form scored by XGBoost on the classic German Credit
dataset, with SHAP waterfalls, plain-English reasons for and against,
and actionable guidance. Protected attributes are deliberately excluded.
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap
import streamlit as st

import branding
from data_prep import DECODE, load_data
from model import score_application, train_model

st.set_page_config(page_title="Explainable Credit Risk Scorer",
                   page_icon=":material/balance:", layout="wide")
branding.inject()


@st.cache_resource(show_spinner="Training the credit model on the German Credit dataset…")
def load_artifacts():
    return train_model()


@st.cache_data
def load_decoded():
    return load_data()


ART = load_artifacts()

PERSONAS = {
    "Choose a preset": None,
    "Recent graduate — first loan": dict(
        checking="low balance (0–200 DM)", duration=24, amount=4500,
        credit_history="no credits taken", purpose="education",
        savings="under 100 DM", employment="under 1 year", installment_rate=4,
        property="none", age=23, housing="renting", existing_credits=1,
        job="skilled employee"),
    "Established homeowner — car loan": dict(
        checking="healthy balance (≥ 200 DM)", duration=18, amount=3200,
        credit_history="existing credits paid duly", purpose="used car",
        savings="500–1,000 DM", employment="≥ 7 years", installment_rate=2,
        property="real estate", age=44, housing="homeowner",
        existing_credits=1, job="management / self-employed"),
    "Overextended borrower": dict(
        checking="overdrawn (below 0 DM)", duration=48, amount=9800,
        credit_history="past payment delays", purpose="business",
        savings="none / unknown", employment="1–4 years", installment_rate=4,
        property="none", age=31, housing="renting", existing_credits=3,
        job="unskilled, resident"),
}

# ---------------- sidebar: the application form ----------------

with st.sidebar:
    st.title("Loan application")

    persona = st.selectbox("Preset applicants", list(PERSONAS))
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

application = dict(
    checking=checking, duration=duration, credit_history=credit_history,
    purpose=purpose, amount=amount, savings=savings, employment=employment,
    installment_rate=installment_rate, property=prop, age=age,
    housing=housing, existing_credits=existing_credits, job=job,
)

result = score_application(ART, application)
pd_prob = result["pd"]
approved = pd_prob < threshold
m = ART["metrics"]

# ---------------- header ----------------

branding.eyebrow("Explainable ML · Credit Risk · Responsible AI")
st.title("Explainable Credit Risk Scorer")
st.caption(
    "Every decision comes with its reasons — an interactive study in explainable "
    "credit decisioning. Adjust the application in the sidebar; the decision and "
    "its explanation update live."
)

tab_score, tab_how, tab_data = st.tabs(
    ["Score an application", "How it works", "The data"]
)

# ================= TAB 1: scoring =================

with tab_score:
    left, right = st.columns([2, 3], gap="large")

    with left:
        branding.verdict_pill(approved)
        st.caption(f"Predicted default probability vs. threshold of {threshold:.0%}")

        color = "#21c98d" if approved else "#ff5c5c"
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pd_prob * 100,
            number={"suffix": "%", "font": {"size": 44, "color": "#f2f2f5"}},
            title={"text": "Default risk", "font": {"color": "#b9b9c2", "size": 15}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%",
                         "tickcolor": "#8a8a92", "tickfont": {"color": "#b9b9c2"}},
                "bar": {"color": color},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, threshold * 100], "color": "rgba(33,201,141,0.12)"},
                    {"range": [threshold * 100, 100], "color": "rgba(255,92,92,0.10)"},
                ],
                "threshold": {
                    "line": {"color": "#b9b9c2", "width": 3},
                    "value": threshold * 100,
                },
            },
        ))
        gauge.update_layout(height=290, margin=dict(l=30, r=30, t=50, b=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            font={"color": "#e8e8ec"})
        st.plotly_chart(gauge, use_container_width=True)

        if result["hurts"]:
            st.markdown("**Working against this application**")
            for c in result["hurts"]:
                branding.reason(c["text"], "neg")
        if result["helps"]:
            st.markdown("**Working in its favour**")
            for c in result["helps"]:
                branding.reason(c["text"], "pos")

        if not approved and result["advice"]:
            st.markdown("**What could change the outcome**")
            for a in result["advice"]:
                branding.reason(a.capitalize(), "tip")

    with right:
        st.subheader("How the model weighed it")
        st.caption(
            "SHAP waterfall — each bar shows how one attribute pushed the risk "
            "estimate up (red) or down (blue) from the average applicant."
        )
        fig, ax = plt.subplots()
        shap.plots.waterfall(result["explanation"], max_display=12, show=False)
        st.pyplot(branding.darken(plt.gcf()), use_container_width=True)
        plt.close("all")

# ================= TAB 2: how it works =================

with tab_how:
    st.subheader("From application to explained decision")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        branding.step(1, "Encode the application",
                      "13 attributes — 8 categorical, 5 numeric — are one-hot "
                      "encoded with a fixed column order so training and live "
                      "scoring always see identical features.")
    with c2:
        branding.step(2, "Score with XGBoost",
                      "400 shallow trees (depth 3) estimate the probability of "
                      "default. No class re-weighting, so the output stays "
                      "calibrated to the dataset's 30% base default rate.")
    with c3:
        branding.step(3, "Explain with SHAP",
                      "TreeExplainer computes exact Shapley values — how much "
                      "each feature moved this decision from the average. "
                      "Local accuracy is guaranteed: contributions sum to the score.")
    with c4:
        branding.step(4, "Translate to English",
                      "One-hot contributions are aggregated back to their parent "
                      "attribute and ranked, generating reasons for and against — "
                      "and adverse-action guidance when declined.")

    st.write("")
    st.subheader("What drives decisions overall")
    st.caption(
        "SHAP beeswarm across 250 held-out applicants. Each dot is one person; "
        "position shows how strongly that feature pushed their risk up or down. "
        "The well-known German Credit result is plainly visible: checking-account "
        "status dominates everything else."
    )
    fig, ax = plt.subplots()
    shap.plots.beeswarm(ART["global_explanation"], max_display=12, show=False)
    st.pyplot(branding.darken(plt.gcf()), use_container_width=True)
    plt.close("all")

    st.write("")
    st.subheader("Design decisions, and their why")
    with st.expander("Why XGBoost, and not logistic regression or a neural net?"):
        st.markdown(
            "Gradient-boosted trees are the workhorse of tabular credit modelling: "
            "they capture non-linear effects and interactions (e.g. *high amount* "
            "is riskier *for short employment tenure*) that logistic regression "
            "misses, while remaining cheap to train and — crucially — exactly "
            "explainable through tree SHAP. A neural net would add opacity and "
            "infrastructure for no measurable lift on 1,000 rows."
        )
    with st.expander("Why are the probabilities not re-balanced?"):
        st.markdown(
            "It's common to up-weight the minority class, but that inflates "
            "predicted probabilities and breaks their meaning. Here a displayed "
            "**24% default risk means 24%** — calibrated against the dataset's "
            "30% base rate. The trade-off between catching defaults and approving "
            "good customers is exposed explicitly through the threshold slider "
            "instead of being hidden inside the loss function."
        )
    with st.expander("Why SHAP rather than LIME or feature importance?"):
        st.markdown(
            "Global feature importance can't explain *one* decision, and LIME's "
            "local surrogates are unstable across runs. Shapley values have the "
            "properties a regulator would ask for: **local accuracy** (the "
            "explanation sums exactly to the prediction) and **consistency** "
            "(a feature that matters more never gets a smaller attribution). "
            "For trees they can be computed exactly, not approximated."
        )
    with st.expander("Fairness: what was removed from the data, and what wasn't"):
        st.markdown(
            "The raw dataset includes `personal_status_sex` and `foreign_worker`. "
            "Both are **excluded from the model** — protected characteristics have "
            "no place in a credit decision. Honest caveat: exclusion alone doesn't "
            "guarantee fairness, because remaining features can act as proxies. "
            "A production system would add bias audits across protected groups "
            "(demographic parity, equalised odds) on top of feature exclusion."
        )
    with st.expander("Known limitations"):
        st.markdown(
            "- 1,000 applications from a 1990s German bank — amounts in Deutsche "
            "Mark, social patterns of its era. A teaching dataset, not a deployable "
            "scorecard.\n"
            "- Held-out ROC-AUC ≈ 0.80 — honest performance for this data; beware "
            "any demo claiming 0.99 on German Credit.\n"
            "- Counter-intuitive SHAP signals (e.g. *'no savings account'* mildly "
            "lowering risk) are genuine quirks of the data worth interrogating, "
            "not bugs to hide."
        )

    st.write("")
    st.subheader("Read the code")
    st.markdown(
        "The whole system is about 520 lines — small enough to review in one "
        "sitting, and structured the way production code is: data preparation, "
        "modelling and presentation kept strictly apart.\n\n"
        "| Module | Responsibility | |\n"
        "|---|---|---|\n"
        "| [`data_prep.py`](https://github.com/drishtantleuva/credit-risk-scorer/blob/main/data_prep.py) | Decodes the UCI attribute codes, drops protected attributes, one-hot encodes with a column order shared by training and inference | 90 lines |\n"
        "| [`model.py`](https://github.com/drishtantleuva/credit-risk-scorer/blob/main/model.py) | Trains the classifier, computes SHAP attributions, aggregates one-hot contributions back to human terms and renders them as reasons and advice | 125 lines |\n"
        "| [`app.py`](https://github.com/drishtantleuva/credit-risk-scorer/blob/main/app.py) | Everything you are looking at | 308 lines |\n"
    )

# ================= TAB 3: the data =================

with tab_data:
    df = load_decoded()
    st.subheader("Statlog German Credit Data")
    st.markdown(
        "1,000 anonymised loan applications from a German bank (1990s), published "
        "by Prof. Hans Hofmann and hosted by the "
        "[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data). "
        "It is *the* classic credit-risk research benchmark. 700 loans were repaid; "
        "300 defaulted. The raw attribute codes (`A11`, `A34`, …) ship in "
        "[`data/german.data`](https://github.com/drishtantleuva/credit-risk-scorer/blob/main/data/german.data) "
        "and are decoded to the readable labels below."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Applications", "1,000")
    c2.metric("Repaid", "700")
    c3.metric("Defaulted", "300")

    st.dataframe(df.head(40), use_container_width=True, height=420)
    st.caption(
        "First 40 rows after decoding. The `default` column is the model target "
        "(1 = the loan was not repaid). Sex and nationality columns are dropped "
        "during preparation and never reach the model."
    )

branding.footer("credit-risk-scorer")
