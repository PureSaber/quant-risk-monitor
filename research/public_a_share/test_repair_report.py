import json

import pandas as pd
import pytest
from repair_report import compare_branch_prefixes, cross_metrics


def forecast_rows():
    return [
        {
            "date": day, "scenario": 1, "generator": "repaired_ols", "predictor": model,
            "portfolio_id": str(day), "target_kind": "integer", "qlike": loss,
            "active_qlike": loss + 1, "variance": 0.01, "active_variance": 0.02,
            "actual_frozen_weight_return": 0.1, "active_frozen_weight_return": 0.12,
        }
        for day in pd.date_range("2024-01-01", periods=21)
        for model, loss in (("legacy_ols", 2), ("repaired_ols", 1))
    ]


def test_cross_risk_rejects_different_actual_returns_for_same_portfolio():
    rows = forecast_rows()
    rows[0]["actual_frozen_weight_return"] = 0.2
    with pytest.raises(ValueError, match="different realized"):
        cross_metrics(pd.DataFrame(rows))


def test_cross_risk_retains_missing_date_and_minimum_block_gate():
    rows = forecast_rows()
    for row in rows[:4]:
        row["qlike"] = None
        row["active_qlike"] = None
    summary, _ = cross_metrics(pd.DataFrame(rows))
    for objective in ("absolute", "active"):
        result = summary[objective]["integer"]
        assert result["complete_dates"] == 19
        assert result["comparisons"]["repaired_ols"]["simultaneous_95_interval"] is None
        assert not result["comparisons"]["repaired_ols"]["historical_improvement"]


def test_branch_check_exposes_differences_before_rights_without_seeing_later_outcomes(tmp_path):
    folders = [tmp_path / label for label in ("strict", "sensitivity_floor")]
    for folder in folders:
        folder.mkdir()
        pd.DataFrame({"event_time": [pd.Timestamp("2024-07-04", tz="UTC")],
                      "nav_units": [1000000], "nav_scale": [0]}).to_parquet(
            folder / "portfolio_snapshots.parquet", index=False
        )
        (folder / "allocation-decisions.json").write_text(json.dumps([
            {"session": "2024-07-04", "targets": {"600763": 100}},
            {"session": "2024-07-05", "targets": {"600763": 200 if folder == folders[0] else 500}},
        ]))
    accounts = pd.DataFrame({"branch": [p.name for p in folders], "scenario": [16, 16],
                             "folder": [str(p) for p in folders]})
    assert compare_branch_prefixes(accounts).iloc[0].pre_rights_economic_prefix_equal
    path = folders[1] / "allocation-decisions.json"
    rows = json.loads(path.read_text())
    rows[0]["targets"]["600763"] = 0
    path.write_text(json.dumps(rows))
    checked = compare_branch_prefixes(accounts).iloc[0]
    assert not checked.pre_rights_economic_prefix_equal
    assert checked.first_difference == "2024-07-04"
