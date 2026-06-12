"""Load and decode the UCI Statlog German Credit dataset.

Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
1,000 loan applications from a German bank (1990s), 700 good / 300 bad.

We deliberately EXCLUDE the `personal_status_sex` and `foreign_worker`
attributes from the model: protected characteristics have no place in a
credit decision. This is stated in the app's model card.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "german.data"

RAW_COLUMNS = [
    "checking", "duration", "credit_history", "purpose", "amount",
    "savings", "employment", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age",
    "other_installments", "housing", "existing_credits", "job",
    "dependents", "telephone", "foreign_worker", "target",
]

# attribute code -> human-readable label
DECODE = {
    "checking": {
        "A11": "overdrawn (below 0 DM)",
        "A12": "low balance (0–200 DM)",
        "A13": "healthy balance (≥ 200 DM)",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits taken",
        "A31": "all credits paid back duly",
        "A32": "existing credits paid duly",
        "A33": "past payment delays",
        "A34": "critical account / other credits",
    },
    "purpose": {
        "A40": "new car", "A41": "used car", "A42": "furniture",
        "A43": "radio/TV", "A44": "appliances", "A45": "repairs",
        "A46": "education", "A48": "retraining", "A49": "business",
        "A410": "other",
    },
    "savings": {
        "A61": "under 100 DM", "A62": "100–500 DM", "A63": "500–1,000 DM",
        "A64": "≥ 1,000 DM", "A65": "none / unknown",
    },
    "employment": {
        "A71": "unemployed", "A72": "under 1 year", "A73": "1–4 years",
        "A74": "4–7 years", "A75": "≥ 7 years",
    },
    "property": {
        "A121": "real estate", "A122": "insurance / savings contract",
        "A123": "car or other", "A124": "none",
    },
    "housing": {
        "A151": "renting", "A152": "homeowner", "A153": "rent-free",
    },
    "job": {
        "A171": "unskilled, non-resident", "A172": "unskilled, resident",
        "A173": "skilled employee", "A174": "management / self-employed",
    },
}

# features the model uses (protected attributes deliberately excluded)
CATEGORICAL = ["checking", "credit_history", "purpose", "savings",
               "employment", "property", "housing", "job"]
NUMERIC = ["duration", "amount", "installment_rate", "age", "existing_credits"]
FEATURES = CATEGORICAL + NUMERIC


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=" ", header=None, names=RAW_COLUMNS)
    for col, mapping in DECODE.items():
        df[col] = df[col].map(mapping)
    # target: 1 = good payer, 2 = defaulted -> default flag
    df["default"] = (df["target"] == 2).astype(int)
    return df[FEATURES + ["default"]]


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode with stable column order (train and inference match)."""
    base = load_data()[FEATURES]
    combined = pd.concat([base, df[FEATURES]], ignore_index=True)
    dummies = pd.get_dummies(combined, columns=CATEGORICAL)
    return dummies.iloc[len(base):].reset_index(drop=True)
