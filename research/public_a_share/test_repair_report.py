import pandas as pd
import pytest
from repair_report import cross_metrics


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
