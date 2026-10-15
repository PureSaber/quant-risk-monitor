"""Decision cadence must not inherit research label-maturity filters."""

import numpy as np
import pandas as pd
import pytest

from study_v3_cross_audit import recompute_cross
from study_v3_portfolio_models import decision_grid
from study_v3_report import verify_decision_calendar
from study_v3_risk import forecast_grid


def test_rebalance_grid_keeps_development_boundary_and_unmatured_tail():
    dates = pd.bdate_range("2022-12-01", "2026-09-30")
    decisions = decision_grid(dates)
    positions = [i for i, _ in decisions]
    assert np.diff(positions).tolist() == [5] * (len(positions) - 1)
    assert len(dates) - 1 - positions[-1] < 5
    labelled = {day for _, day, _ in forecast_grid(dates)}
    boundary = [day for i, day in decisions if day.year == 2023 and dates[i + 5].year == 2024]
    assert boundary and not set(boundary) & labelled
    assert decisions[-1][1] not in labelled


def test_last_decision_has_risk_predictions_without_inventing_mature_returns():
    dates = pd.bdate_range("2026-09-28", periods=3)
    times = [d.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15) for d in dates]
    panel = {"dates": dates, "prices": {"returns": pd.DataFrame({"A": 0.01}, index=dates)}}
    model = {"exposures": pd.DataFrame({"market": [1.0]}, index=["A"]),
             "F_daily_rate": np.array([[0.01]]), "D_daily_rate": np.array([0.001])}
    models = {dates[0]: {"benchmark": pd.Series({"A": 1.0}), "models": {"one": model}}}
    snapshots = pd.DataFrame({"event_time": times, "nav_units": 1000, "nav_scale": 0})
    decisions = [{"session": "2026-09-28", "status": "infeasible"}]
    frame = recompute_cross(decisions, pd.DataFrame(), snapshots, models, panel)
    assert frame.status.tolist() == ["outcome_pending"]
    assert frame.end.isna().all()
    assert frame.actual_frozen_weight_return.isna().all()
    assert frame.active_qlike.isna().all()
    assert frame.active_variance.iloc[0] > 0


def test_native_account_calendar_rejects_maturity_filtered_decisions():
    dates = pd.bdate_range("2022-12-01", "2026-09-30")
    full = {day: {} for _, day in decision_grid(dates)}
    records = [{"session": str(day.date())} for day in full]
    verify_decision_calendar(records, pd.Series(list(full)), full)
    old = [{"session": str(day.date())} for _, day, _ in forecast_grid(dates)]
    with pytest.raises(ValueError, match="omitted"):
        verify_decision_calendar(old, pd.Series(list(full)), full)
