from pathlib import Path
import numpy as np
import pandas as pd

try:
    from lifelines import CoxPHFitter
except ImportError:
    CoxPHFitter = None

DATA_PATH = Path("data/cases.csv")

FEATURES = [
    "case_type",
    "court",
    "hearings",
    "adjournments",
    "filing_year",
]

def _prepare(df):
    x = df[FEATURES + ["duration_months", "event"]].copy()
    x["filing_year"] = x["filing_year"].astype(int)
    x = pd.get_dummies(x, columns=["case_type", "court"], drop_first=False, dtype=float)
    return x

def load_or_train_model():
    if CoxPHFitter is None:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt")

    df = pd.read_csv(DATA_PATH)
    model_df = _prepare(df)

    model = CoxPHFitter(penalizer=0.1)
    model.fit(model_df, duration_col="duration_months", event_col="event")

    feature_columns = [c for c in model_df.columns if c not in ["duration_months", "event"]]
    return model, feature_columns, df

def predict_case(model, court, case_type, filing_date, hearings, adjournments):
    filing_year = filing_date.year

    row = pd.DataFrame([{
        "case_type": case_type,
        "court": court,
        "hearings": hearings,
        "adjournments": adjournments,
        "filing_year": filing_year,
    }])

    row = pd.get_dummies(row, columns=["case_type", "court"], drop_first=False, dtype=float)

    # Align with model training columns.
    for col in model.params_.index:
        if col not in row.columns:
            row[col] = 0.0
    row = row[model.params_.index]

    sf = model.predict_survival_function(row, times=np.arange(1, 121))
    survival = sf.iloc[:, 0].clip(0, 1)
    curve = pd.DataFrame({
        "months": survival.index.astype(float),
        "probability_disposed": 1 - survival.values
    })

    def horizon(target):
        hit = curve[curve["probability_disposed"] >= target]
        return float(hit.iloc[0]["months"]) if not hit.empty else 120.0

    return {
        "p50": horizon(0.50),
        "p75": horizon(0.75),
        "p90": horizon(0.90),
        "curve": curve,
        "factors": [
            f"Case type: {case_type}",
            f"Court: {court}",
            f"Hearings so far: {hearings}",
            f"Adjournments so far: {adjournments}",
            f"Filing year: {filing_year}",
        ],
    }
