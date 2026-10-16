import pandas as pd
import pytest
from repair_account_quality import entitlement_stop_context, holdings_evidence


def fixture():
    dates = pd.date_range("2024-01-02", periods=4)
    positions = pd.DataFrame([
        {"event_time": day.tz_localize("UTC") + pd.Timedelta(hours=7),
         "instrument_id": asset, "quantity_units": 200 if asset == "B" and i == 2 else 100,
         "quantity_scale": 0, "base_market_value_units": value, "base_market_value_scale": 0}
        for i, day in enumerate(dates[:3]) for asset, value in (("A", 1000), ("B", 6000))
    ])
    snapshots = pd.DataFrame({"event_time": dates[:3].tz_localize("UTC"),
                              "nav_units": [10000] * 3, "nav_scale": [0] * 3})
    bars = pd.DataFrame({"date": [dates[0], dates[3], *dates[:3]],
                        "symbol": ["A", "A", "B", "B", "B"], "close": [10] * 5})
    statuses = pd.DataFrame({"A": [1, 0, None, 1], "B": [1] * 4}, index=dates)
    return positions, snapshots, bars, statuses


def test_stale_valuation_uses_full_nav_and_does_not_see_future_quote():
    positions, snapshots, bars, statuses = fixture()
    summary, stale, intervals = holdings_evidence(positions, snapshots, bars, statuses)
    assert summary["stale_valuation_days"] == 2
    assert summary["endpoint_stale_assets"] == 1
    assert summary["endpoint_stale_nav_weight"] == pytest.approx(.1)
    assert stale.state.tolist() == ["confirmed_suspension", "trading_status_unknown"]
    assert stale.last_observed_trading_quote.tolist() == ["2024-01-02"] * 2
    assert summary["longest_unchanged_nonempty_holding_sessions"] == 2
    assert intervals.sessions.tolist() == [2, 1]
    without_future = holdings_evidence(positions, snapshots, bars.iloc[[0, 2, 3, 4]], statuses)
    pd.testing.assert_frame_equal(stale, without_future[1])


def test_all_cash_is_not_reported_as_frozen_nonempty_holdings():
    positions, snapshots, bars, statuses = fixture()
    summary, stale, intervals = holdings_evidence(positions.iloc[:0], snapshots, bars, statuses)
    assert summary["stale_valuation_days"] == summary["endpoint_stale_assets"] == 0
    assert summary["longest_unchanged_nonempty_holding_sessions"] == 0
    assert stale.empty
    assert intervals.iloc[0].sessions == 3 and intervals.iloc[0].held_assets == 0


def test_known_trading_day_without_quote_is_separate_from_suspension():
    positions, snapshots, bars, statuses = fixture()
    statuses.iloc[1, 0] = 1
    _, stale, _ = holdings_evidence(positions, snapshots, bars, statuses)
    assert stale.iloc[0].state == "quote_missing"


def test_stop_context_must_match_saved_record_date_holding():
    positions, _, _, _ = fixture()
    stop = {"asset": "A", "diagnostics": {"terms": {"record_date": "2024-01-03",
            "eligible_quantity": {"units": 10000, "scale": 2}}}}
    assert entitlement_stop_context(stop, positions)["status"] == "passed"
    stop["diagnostics"]["terms"]["eligible_quantity"]["units"] = 10001
    with pytest.raises(ValueError, match="safe prefix differs"):
        entitlement_stop_context(stop, positions)


def test_legacy_stop_does_not_claim_quantity_was_verified():
    assert entitlement_stop_context({}, pd.DataFrame())["status"] == "unavailable_in_legacy_diagnostic"
