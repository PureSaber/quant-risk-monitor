from copy import deepcopy

import numpy as np
import pandas as pd

from quant_risk_monitor.factor_model import ExposureSnapshot, FactorModelConfig
from repair_risk_coverage import extend_day


def test_risk_union_extension_keeps_candidates_benchmark_and_complete_assets_unchanged():
    dates = pd.bdate_range("2022-01-01", periods=260)
    day = dates[252]
    all_assets = ["a", "b", "c", "d", "e"]
    x = pd.DataFrame({"market": 1.0}, index=all_assets)

    def snap(values, at):
        return ExposureSnapshot(
            effective_at=at.tz_localize("UTC"),
            available_at=at.tz_localize("UTC"),
            values=values,
            market_caps=pd.Series(1.0, index=values.index),
            industries=pd.Series("industry", index=values.index),
        )

    current = snap(x.loc[["a", "b", "c", "d"]], day)
    older = snap(x, dates[251])
    residuals = pd.DataFrame(
        np.random.default_rng(7).normal(0, 0.02, (260, 5)), index=dates, columns=all_assets
    )
    residuals.loc[dates[20], "d"] = np.nan
    histories = {
        "snapshots": {"repaired": {dates[251]: older, day: current}},
        "histories": {"repaired": {"equal": {"residuals": residuals}}},
    }
    config = FactorModelConfig(model_kind="fundamental_style")
    definitions = {"repaired_ols": ("repaired", config, "baseline")}
    entry = {
        "assets": ["a", "b", "c"],
        "benchmark": pd.Series({"a": 0.6, "b": 0.3, "c": 0.1}),
        "models": {
            "repaired_ols": {
                "exposures": x.loc[["a", "b", "c"]],
                "F_daily_rate": np.array([[0.001]]),
                "D_daily_rate": np.array([0.002, 0.003, 0.004]),
            }
        },
    }
    panel = {
        "dates": dates,
        "assets": all_assets,
        "prices": {"close": pd.DataFrame(10.0, index=dates, columns=all_assets)},
    }
    statuses = pd.DataFrame(1, index=dates, columns=all_assets)
    statuses.at[day, "e"] = 0
    result, _ = extend_day(day, entry, panel, histories, definitions, statuses)
    assert result["assets"] == entry["assets"]
    pd.testing.assert_series_equal(result["benchmark"], entry["benchmark"])
    model = result["models"]["repaired_ols"]
    assert model["coverage"]["d"]["valid_observations"] == 251
    assert model["coverage"]["e"]["exposure_status"] == "stale_confirmed_suspension"
    np.testing.assert_array_equal(
        model["D_daily_rate"][:3], entry["models"]["repaired_ols"]["D_daily_rate"]
    )
    future = deepcopy(histories)
    future["histories"]["repaired"]["equal"]["residuals"].loc[dates[252] :] = 1e6
    repeated, _ = extend_day(day, entry, panel, future, definitions, statuses)
    np.testing.assert_array_equal(
        repeated["models"]["repaired_ols"]["D_daily_rate"], model["D_daily_rate"]
    )
    statuses.at[day, "e"] = np.nan
    unknown, _ = extend_day(day, entry, panel, histories, definitions, statuses)
    assert "e" not in unknown["models"]["repaired_ols"]["exposures"].index
    assert unknown["models"]["repaired_ols"]["coverage"]["e"]["status"] == "unavailable"
