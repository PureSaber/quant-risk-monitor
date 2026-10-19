from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from held_risk_policy import held_exposure_candidate
from recovery_lifecycle import state_overlay


def fixture():
    dates = pd.bdate_range("2022-01-01", periods=270)
    frame = pd.DataFrame(1.0, index=dates, columns=["a"])
    fields = [
        "size_log_mcap",
        "beta_252d",
        "residual_vol_252d",
        "liquidity_log_turnover_20d",
        "book_to_price",
        "earnings_yield_ttm",
        "earnings_growth_yoy",
        "book_leverage",
    ]
    panel = {
        "dates": dates,
        "descriptors": {k: frame.copy() for k in fields},
        "groups": pd.DataFrame("industry", index=dates, columns=["a"]),
        "prices": {"adjusted_close": frame.copy()},
    }
    day = dates[260]
    panel["prices"]["adjusted_close"].at[dates[238], "a"] = np.nan
    status = frame.copy()
    status.at[dates[238], "a"] = 0
    prior = SimpleNamespace(
        values=pd.DataFrame({"market": [1.0]}, index=["a"]), industries=pd.Series({"a": "industry"})
    )
    return day, panel, {dates[259]: prior}, status


def test_carry_is_explicit_causal_and_never_fills_returns_or_prices():
    day, panel, snapshots, statuses = fixture()
    before = panel["prices"]["adjusted_close"].copy()
    result, info = held_exposure_candidate(day, "a", panel, snapshots, statuses, max_sessions=5)
    assert result.loc["a", "market"] == 1
    assert info["estimated_exposure"] and not info["price_or_return_imputed"]
    assert info["exposure_age_sessions"] == 1
    pd.testing.assert_frame_equal(before, panel["prices"]["adjusted_close"])
    panel["prices"]["adjusted_close"].loc[panel["dates"][261] :] = 1e9
    repeated, _ = held_exposure_candidate(day, "a", panel, snapshots, statuses, max_sessions=5)
    pd.testing.assert_frame_equal(result, repeated)


@pytest.mark.parametrize("failure", ["unknown", "other_gap", "industry", "financial", "expired"])
def test_uncertain_or_unreliable_assets_are_not_silently_covered(failure):
    day, panel, snapshots, statuses = fixture()
    if failure == "unknown":
        statuses.at[day, "a"] = np.nan
    if failure == "other_gap":
        panel["prices"]["adjusted_close"].iloc[240, 0] = np.nan
    if failure == "industry":
        panel["groups"].at[day, "a"] = "changed"
    if failure == "financial":
        panel["descriptors"]["book_to_price"].at[day, "a"] = np.nan
    if failure == "expired":
        snapshots = {panel["dates"][250]: next(iter(snapshots.values()))}
    result, info = held_exposure_candidate(day, "a", panel, snapshots, statuses, max_sessions=5)
    assert result is None and info["reason"]


def test_late_exit_evidence_never_backdates_availability():
    day, panel, snapshots, status = fixture()
    date = panel["dates"][259]
    event = {
        "asset": "a",
        "effective_date": str(date.date()),
        "available_at": str(day.date()) + "T00:00:00+08:00",
        "source_sha256": "evidence",
        "review_basis": "verified notice",
    }
    overlay = state_overlay(status, [event])
    assert overlay.at[date, "a"] == "observed_trading"
    assert overlay.at[day, "a"] == "exited_a_share"
