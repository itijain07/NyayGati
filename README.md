# NyayGati — SIH Prototype

NyayGati is a prototype decision-support platform for:
1. Citizen case-duration estimation
2. Court backlog and delay-pattern analytics
3. What-if backlog simulation

## Important prototype note

The included dataset is **synthetic demonstration data**. It is not presented as real court performance.

This prototype intentionally avoids:
- making judicial decisions
- giving legal advice
- claiming an exact disposal date
- automatically scheduling or reordering real cases

## Run locally

### 1. Create a virtual environment
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 2. Install packages
```bash
pip install -r requirements.txt
```

### 3. Generate the synthetic dataset
```bash
python generate_data.py
```

### 4. Start the prototype
```bash
streamlit run app.py
```

The browser will open the NyayGati dashboard.

## How the prediction works

The prototype uses a Cox Proportional Hazards survival model.

Why survival analysis?
Some historical cases are disposed and have a known duration. Other cases are still pending, so their final duration is unknown. Those pending observations are right-censored. Survival analysis can use both kinds of observations.

The UI converts the estimated survival curve into approximate 50%, 75%, and 90% disposal horizons.

## How the simulation works

The sandbox compares:
- baseline backlog / baseline monthly disposal capacity
- simulated backlog / increased monthly capacity

It is deliberately transparent so the team can explain the logic during judging.

## Suggested presentation flow

1. Citizen Portal: enter a sample case.
2. Show 50/75/90% disposal horizons and the probability curve.
3. Switch to Court Analytics.
4. Show a potential bottleneck cluster.
5. Open Admin Sandbox.
6. Change monthly capacity and run the scenario.
7. Compare baseline vs simulated clearance time.
8. State that all results are statistical/simulated and require human oversight.

## Prototype-to-production roadmap

For a production version:
- replace synthetic data with legally accessible official data
- add authentication and role-based access
- validate the model using time-aware evaluation
- add richer survival models if justified by the dataset
- replace the simplified sandbox with a validated discrete-event court-calendar model
- audit for bias and monitor model performance
