def run_simulation(backlog, monthly_capacity, extra_capacity, priority_share, priority_category):
    baseline_capacity = max(1, monthly_capacity)
    simulated_capacity = baseline_capacity + extra_capacity

    baseline_months = backlog / baseline_capacity
    simulated_months = backlog / simulated_capacity

    improvement_pct = max(0, (baseline_months - simulated_months) / baseline_months * 100)

    return {
        "baseline_months": baseline_months,
        "simulated_months": simulated_months,
        "improvement_pct": improvement_pct,
        "priority_category": priority_category,
        "priority_share": priority_share,
    }
