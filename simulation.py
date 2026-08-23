"""
Case-level scheduling simulation.

The old version of this file only did backlog / capacity arithmetic on
aggregate totals — it never looked at individual cases, so nothing was
actually being "scheduled." This version builds a real per-case queue and
simulates month-by-month hearing allocation under two orderings:

  - "fifo"      : cases get hearings in filing-date order (oldest first).
                  This mirrors how many courts default to working today.
  - "optimized" : cases are sorted by estimated remaining hearings needed
                  (shortest-remaining-first), with an optional quota of
                  capacity reserved for a chosen priority case type each
                  month. Clearing low-effort cases first frees hearing
                  slots faster, which is the actual mechanism behind the
                  backlog-reduction claim — not just "more capacity."
"""

import pandas as pd
import numpy as np


def build_case_queue(pending_df, case_type_avg_hearings):
    """
    Attach an estimated 'remaining_hearings' figure to each pending case.

    We don't know how many hearings a still-open case needs to finish, so
    we estimate it as (typical total hearings for this case_type, based on
    already-disposed cases) minus (hearings already held for this case),
    floored at 1 so every case needs at least one more hearing.
    """
    queue = pending_df.copy()
    queue["expected_total_hearings"] = queue["case_type"].map(case_type_avg_hearings)
    queue["expected_total_hearings"] = queue["expected_total_hearings"].fillna(
        case_type_avg_hearings.mean()
    )
    queue["remaining_hearings"] = (
        queue["expected_total_hearings"] - queue["hearings"]
    ).clip(lower=1)
    return queue


def simulate_schedule(
    queue,
    monthly_hearing_capacity,
    order="fifo",
    priority_category=None,
    priority_share=0.0,
    months=120,
):
    """
    Simulate month-by-month hearing allocation and case closures for ONE
    ordering strategy. Returns a DataFrame with one row per month:
    cumulative cases closed and remaining backlog.
    """
    q = queue.copy()
    q["remaining_hearings"] = q["remaining_hearings"].astype(float)
    q["closed"] = False

    if order == "fifo":
        q = q.sort_values("filing_date")
    else:  # "optimized" — shortest remaining work first clears cases fastest
        q = q.sort_values("remaining_hearings")

    total = len(q)
    history = []

    for month in range(1, months + 1):
        capacity_left = monthly_hearing_capacity

        # Priority quota: reserve a slice of this month's capacity for the
        # chosen case type before touching the general queue.
        if priority_category and priority_share > 0:
            reserved = int(round(monthly_hearing_capacity * priority_share))
            active_priority = q[(~q["closed"]) & (q["case_type"] == priority_category)]
            for idx in active_priority.index:
                if reserved <= 0 or capacity_left <= 0:
                    break
                q.loc[idx, "remaining_hearings"] -= 1
                reserved -= 1
                capacity_left -= 1
                if q.loc[idx, "remaining_hearings"] <= 0:
                    q.loc[idx, "closed"] = True

        # Remaining capacity flows through the queue in its sorted order.
        active = q[~q["closed"]]
        for idx in active.index:
            if capacity_left <= 0:
                break
            q.loc[idx, "remaining_hearings"] -= 1
            capacity_left -= 1
            if q.loc[idx, "remaining_hearings"] <= 0:
                q.loc[idx, "closed"] = True

        closed_count = int(q["closed"].sum())
        history.append(
            {"month": month, "closed_cumulative": closed_count, "backlog": total - closed_count}
        )

        if closed_count >= total:
            break

    return pd.DataFrame(history)


def _clearance_month(curve):
    cleared = curve[curve["backlog"] == 0]
    return int(cleared["month"].min()) if len(cleared) else None


def run_simulation(
    pending_df,
    case_type_avg_hearings,
    monthly_hearing_capacity,
    priority_category=None,
    priority_share=0.0,
    months=120,
):
    """
    Compare FIFO scheduling (baseline) against priority/shortest-first
    scheduling (optimized) on the SAME pending caseload and SAME monthly
    capacity — isolating the effect of scheduling order, not capacity.
    """
    queue = build_case_queue(pending_df, case_type_avg_hearings)

    baseline_curve = simulate_schedule(
        queue, monthly_hearing_capacity, order="fifo", months=months
    )
    optimized_curve = simulate_schedule(
        queue,
        monthly_hearing_capacity,
        order="optimized",
        priority_category=priority_category,
        priority_share=priority_share,
        months=months,
    )

    baseline_months = _clearance_month(baseline_curve)
    optimized_months = _clearance_month(optimized_curve)

    improvement_pct = None
    if baseline_months and optimized_months:
        improvement_pct = max(0.0, (baseline_months - optimized_months) / baseline_months * 100)

    return {
        "baseline_curve": baseline_curve,
        "optimized_curve": optimized_curve,
        "baseline_months": baseline_months,
        "optimized_months": optimized_months,
        "improvement_pct": improvement_pct,
        "priority_category": priority_category,
        "priority_share": priority_share,
    }
