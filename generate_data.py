from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1800

courts = ["Delhi District Court", "Noida District Court", "Jaipur District Court", "Lucknow District Court"]
case_types = ["Cheque Bounce", "Civil Dispute", "Property", "Family", "Motor Accident"]

court_effect = {
    "Delhi District Court": 1.10,
    "Noida District Court": 0.95,
    "Jaipur District Court": 1.00,
    "Lucknow District Court": 1.08,
}
type_effect = {
    "Cheque Bounce": 1.15,
    "Civil Dispute": 1.25,
    "Property": 1.35,
    "Family": 0.90,
    "Motor Accident": 1.05,
}

rows = []
for i in range(N):
    court = rng.choice(courts)
    case_type = rng.choice(case_types, p=[.28, .20, .18, .16, .18])
    filing_year = int(rng.integers(2018, 2026))
    filing_month = int(rng.integers(1, 13))
    filing_date = pd.Timestamp(year=filing_year, month=filing_month, day=1)

    hearings = int(max(0, rng.poisson(7)))
    adjournments = int(min(25, rng.poisson(2.5)))

    base = rng.gamma(shape=3.0, scale=7.0)
    duration = base * court_effect[court] * type_effect[case_type]
    duration += hearings * 0.7 + adjournments * 1.8
    duration = float(max(3, duration))

    # Simulate observation cutoff. Some cases are still pending.
    cutoff = pd.Timestamp("2026-07-01")
    observed_age = max(1, (cutoff - filing_date).days / 30.44)

    event = int(duration <= observed_age)
    observed_duration = min(duration, observed_age)

    disposal_date = filing_date + pd.to_timedelta(duration * 30.44, unit="D") if event else pd.NaT
    current_age = observed_age if not event else duration

    rows.append({
        "case_id": f"NG-{i+1:05d}",
        "court": court,
        "case_type": case_type,
        "filing_date": filing_date,
        "filing_year": filing_year,
        "disposal_date": disposal_date,
        "duration_months": round(observed_duration, 2),
        "event": event,
        "hearings": hearings,
        "adjournments": adjournments,
        "current_age_months": round(current_age, 2),
    })

df = pd.DataFrame(rows)
Path("data").mkdir(exist_ok=True)
df.to_csv("data/cases.csv", index=False)
print(df.head())
print("Rows:", len(df), "Disposed:", df.event.sum(), "Pending:", (df.event == 0).sum())
