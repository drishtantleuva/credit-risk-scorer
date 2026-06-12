"""Credit risk model: XGBoost + SHAP + plain-English decision explanations."""

from __future__ import annotations

import pandas as pd
import shap
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_prep import CATEGORICAL, FEATURES, encode, load_data

# encoded-feature prefix -> how to phrase its influence
PHRASES = {
    "checking": "Checking account: {val}",
    "credit_history": "Credit history: {val}",
    "purpose": "Loan purpose: {val}",
    "savings": "Savings: {val}",
    "employment": "Employment length: {val}",
    "property": "Property owned: {val}",
    "housing": "Housing: {val}",
    "job": "Job type: {val}",
    "duration": "Loan term of {val:.0f} months",
    "amount": "Loan amount of {val:,.0f} DM",
    "installment_rate": "Repayments at {val:.0f}% of disposable income",
    "age": "Applicant age: {val:.0f}",
    "existing_credits": "{val:.0f} existing credit(s) at this bank",
}

# advice templates for the "what could improve this" panel
ADVICE = {
    "checking": "an established checking balance (≥ 200 DM) is the single strongest positive signal",
    "savings": "building savings above 500 DM materially lowers predicted risk",
    "duration": "a shorter loan term reduces predicted risk",
    "amount": "a smaller loan amount relative to your profile reduces risk",
    "installment_rate": "keeping repayments below ~3% of disposable income helps",
    "employment": "longer tenure in current employment lowers predicted risk",
    "credit_history": "this reflects past repayment behaviour and changes slowly",
    "existing_credits": "fewer concurrent credits at the bank lowers risk",
    "age": "risk estimates shift with age; this cannot be acted upon",
    "purpose": "some purposes (e.g. used car) historically default less than others",
    "property": "holding property or savings contracts acts as a stabilising signal",
    "housing": "homeowners historically default less in this dataset",
    "job": "skill level correlates with income stability in this dataset",
}


def _group(feature_name: str) -> str:
    """Map an encoded column ('savings_< 100 DM') back to its base feature."""
    for cat in CATEGORICAL:
        if feature_name.startswith(cat + "_"):
            return cat
    return feature_name


def train_model(seed: int = 42) -> dict:
    df = load_data()
    X = encode(df)
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    model = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=seed,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "default_rate": float(y.mean()),
    }
    explainer = shap.TreeExplainer(model)
    return {"model": model, "explainer": explainer, "metrics": metrics,
            "columns": list(X.columns)}


def score_application(art: dict, application: dict) -> dict:
    """Score one application; return default probability, SHAP, and reasons."""
    row = pd.DataFrame([application])
    X = encode(row)[art["columns"]]
    pd_prob = float(art["model"].predict_proba(X)[:, 1][0])
    explanation = art["explainer"](X.astype(float))[0]

    contribs = []
    for name, val in zip(art["columns"], explanation.values):
        if abs(val) < 1e-6:
            continue
        base = _group(name)
        raw = application[base]
        contribs.append({
            "base": base,
            "shap": float(val),
            "text": PHRASES[base].format(val=raw),
        })
    # aggregate one-hot siblings into their base feature
    agg: dict[str, dict] = {}
    for c in contribs:
        if c["base"] in agg:
            agg[c["base"]]["shap"] += c["shap"]
        else:
            agg[c["base"]] = dict(c)
    ranked = sorted(agg.values(), key=lambda c: c["shap"])

    helps = [c for c in ranked if c["shap"] < 0][:4]              # toward approval
    hurts = [c for c in reversed(ranked) if c["shap"] > 0][:4]    # toward decline
    advice = [ADVICE[c["base"]] for c in hurts[:3]]

    return {
        "pd": pd_prob,
        "explanation": explanation,
        "helps": helps,
        "hurts": hurts,
        "advice": advice,
    }
