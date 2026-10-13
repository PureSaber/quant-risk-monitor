import json

import numpy as np
import pandas as pd
import pytest

from confirm_exposure_v2 import merge_prices
from confirmation_context import parse_sector_rows
from exposure_v2_study import history_binding, load_histories, seal_histories, summarize


def protocol():
    return {
        "bootstrap": {"seed": 42, "block_dates": 10, "replications": 100},
        "minimum_selection_dates": 40,
        "maximum_log_bias_deterioration": 0.1,
        "maximum_coverage_drop": 0.05,
        "inputs": {"fixture": "a" * 64},
    }


def forecasts():
    rows = []
    for date in pd.bdate_range("2023-01-01", periods=50):
        for variant in ("repaired", "growth"):
            for portfolio in ("equal", "capitalization", "industry:a", "industry:b"):
                rows.append(
                    {
                        "date": str(date.date()),
                        "variant": variant,
                        "estimator": "common_ols",
                        "family": "industry" if portfolio.startswith("industry:") else portfolio,
                        "portfolio": portfolio,
                        "qlike": -5 if variant == "repaired" else -5.1,
                        "return": 0.1,
                        "variance": 0.01,
                        "assets": 100,
                        "baseline_assets": 100,
                    }
                )
    return pd.DataFrame(rows)


def test_joint_complete_cases_do_not_hide_missing_industry_returns():
    rows = forecasts()
    rows.loc[
        (rows.date == rows.date.iloc[0]) & rows.portfolio.eq("industry:b"), ["qlike", "return"]
    ] = np.nan
    result = summarize(rows, protocol())["common_ols"]
    assert result["complete_dates"] == 49
    assert result["comparisons"]["growth"]["passes_selection"]
    assert result["comparisons"]["growth"]["mean_delta"] == pytest.approx(-0.1)
    assert summarize(rows.iloc[:8], protocol())["common_ols"]["complete_dates"] == 0
    json.dumps(summarize(rows.iloc[:8], protocol()), allow_nan=False)
    assert summarize(pd.DataFrame(), protocol())["common_ols"]["complete_dates"] == 0


def test_frozen_selection_guardrails_are_applied_without_retuning():
    rows = forecasts()
    rows.loc[rows.variant.eq("growth"), "variance"] = 0.0001
    comparison = summarize(rows, protocol())["common_ols"]["comparisons"]["growth"]
    assert not comparison["bias_guardrail"]
    assert not comparison["passes_selection"]
    rows = forecasts()
    rows.loc[rows.variant.eq("growth"), "assets"] = 94
    assert not summarize(rows, protocol())["common_ols"]["comparisons"]["growth"][
        "coverage_guardrail"
    ]


def test_cached_histories_reject_changed_inputs_or_content(tmp_path):
    import pickle

    (tmp_path / "histories.pkl").write_bytes(pickle.dumps({"test": 1}))
    config = protocol()
    seal_histories(tmp_path, config)
    assert load_histories(tmp_path, config) == {"test": 1}
    config["inputs"]["fixture"] = "b" * 64
    with pytest.raises(ValueError, match="binding changed"):
        load_histories(tmp_path, config)
    (tmp_path / "histories.pkl").write_bytes(pickle.dumps({"test": 2}))
    with pytest.raises(ValueError, match="binding changed"):
        load_histories(tmp_path, protocol())
    assert history_binding(protocol())["history_functions_sha256"]


def test_dated_industry_rows_handle_missing_spaces_without_guessing_subindustry():
    members = {f"{n:06}" for n in range(100)}
    text = "\n".join(f"{asset}公司名 C 制造业\n35 专用设备" for asset in sorted(members))
    parsed = parse_sector_rows(text, members)
    assert len(parsed) == 100
    assert set(parsed.values()) == {"C"}
    with pytest.raises(ValueError, match="conflicting"):
        parse_sector_rows(text + "\n000001公司名 J 金融业", members)
    with pytest.raises(ValueError, match="coverage"):
        parse_sector_rows("000001 公司名 C 制造业", members)


def test_confirmation_adjustment_bridge_does_not_fill_suspended_returns(tmp_path):
    from exposure_v2_study import digest

    old_dates = pd.bdate_range("2025-12-22", periods=6)
    future_dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
    dates = old_dates.append(future_dates)
    closes = pd.DataFrame({"000001": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}, index=old_dates)
    closes["000002"] = 20.0
    panel = {
        "assets": ["000001", "000002"],
        "prices": {n: closes.copy() for n in ("close", "adjusted_close", "volume", "turnover")},
    }
    market = pd.Series(range(len(dates)), index=dates)
    (tmp_path / "raw/prices").mkdir(parents=True)
    (tmp_path / "raw/bao-prices").mkdir(parents=True)
    quotes = {"data": {"klines": [f"{d.date()},100,100,100,100,20,2000,0,0,0,1" for d in dates]}}
    (tmp_path / "raw/prices/000001-0.json").write_text(json.dumps(quotes))
    path = tmp_path / "raw/bao-prices/000001.csv"
    bao = pd.DataFrame(
        {
            "date": dates,
            "close": [50.0, 50.5, 51.0, 51.5, 52.0, 52.5, 53.0, 53.0],
            "tradestatus": [1] * 7 + [0],
        }
    )
    bao.to_csv(path, index=False)
    path.with_suffix(".receipt.json").write_text(json.dumps({"sha256": digest(path)}))
    (tmp_path / "raw/prices/000002-0.json").write_text(json.dumps({"data": None}))
    empty = tmp_path / "raw/bao-prices/000002.csv"
    pd.DataFrame(columns=["date", "close", "tradestatus"]).to_csv(empty, index=False)
    empty.with_suffix(".receipt.json").write_text(json.dumps({"sha256": digest(empty)}))
    prices, audit = merge_prices(panel, tmp_path, market)
    pd.testing.assert_frame_equal(
        prices["adjusted_close"].loc[old_dates], closes, check_names=False, check_freq=False
    )
    assert prices["adjusted_close"].loc[future_dates[0], "000001"] == 106.0
    assert np.isnan(prices["adjusted_close"].loc[future_dates[1], "000001"])
    assert np.isnan(prices["returns"].loc[future_dates[1], "000001"])
    assert prices["close"].loc[future_dates, "000002"].isna().all()
    assert all(dtype == np.dtype(float) for dtype in prices["close"].dtypes)
    assert audit[0]["adjustment_scale"] == 2.0
    bao.loc[0, "close"] = 45.0
    bao.to_csv(path, index=False)
    path.with_suffix(".receipt.json").write_text(json.dumps({"sha256": digest(path)}))
    with pytest.raises(ValueError, match="nonconstant"):
        merge_prices(panel, tmp_path, market)
