import json

import numpy as np
import pandas as pd
import pytest

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
