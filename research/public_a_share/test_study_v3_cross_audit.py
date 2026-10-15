"""Failed decisions must not disappear from actual-account risk evaluation."""

import numpy as np
import pandas as pd
import pytest

from study_v3_cross_audit import recompute_cross


@pytest.mark.parametrize("holding", [None, "A", "MISSING"])
def test_failed_target_retains_actual_portfolio_and_risk_coverage(holding):
    dates = pd.bdate_range("2023-01-03", periods=6)
    times = [day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15) for day in dates]
    returns = pd.DataFrame({"A": 0.01, "MISSING": 0.02}, index=dates)
    panel = {"dates": dates, "prices": {"returns": returns}}
    snapshots = pd.DataFrame({"event_time": times, "nav_units": 1000, "nav_scale": 0})
    positions = pd.DataFrame(
        [] if holding is None else [{
            "event_time": times[0], "instrument_id": holding,
            "base_market_value_units": 500, "base_market_value_scale": 0,
        }]
    )
    model = {"exposures": pd.DataFrame({"market": [1.0]}, index=["A"]),
             "F_daily_rate": np.array([[0.01]]), "D_daily_rate": np.array([0.001])}
    models = {dates[0]: {"benchmark": pd.Series({"A": 1.0}), "models": {"one": model, "two": model}}}
    decisions = [{"session": str(dates[0].date()), "status": "input_unavailable"}]
    result = recompute_cross(decisions, positions, snapshots, models, panel)
    assert len(result) == 2
    assert set(result.target_kind) == {"actual_close"}
    assert result.portfolio_id.nunique() == 1
    assert set(result.decision_status) == {"input_unavailable"}
    if holding == "MISSING":
        assert set(result.status) == {"risk_union_incomplete"}
        assert result.variance.isna().all()
        assert set(result.missing_risk_assets) == {"MISSING"}
        assert result.actual_frozen_weight_return.iloc[0] == pytest.approx(0.05)
    elif holding is None:
        assert result.zero_risk_cash.all()
        assert result.qlike.isna().all()
        assert (result.annualized_predicted_te > 0).all()
    else:
        assert result.actual_frozen_weight_return.iloc[0] == pytest.approx(0.025)
        assert result.qlike.notna().all()
    # No outcome after the end of the actually replayed safe prefix is admitted.
    assert recompute_cross(decisions, positions, snapshots.iloc[:-1], models, panel).empty
