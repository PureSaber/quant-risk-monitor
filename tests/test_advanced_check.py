from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from quant_risk_monitor.cli import run_check


def test_advanced_check_outputs_risk_and_factor_contributions(tmp_path) -> None:
    positions = tmp_path / "positions.csv"
    returns = tmp_path / "asset_returns.csv"
    factors = tmp_path / "factors.csv"
    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01"],
            "symbol": ["A", "B"],
            "weight": [0.6, 0.4],
            "market_value": [600_000, 400_000],
        }
    ).to_csv(positions, index=False)
    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "A": [0.01, -0.01, 0.02],
            "B": [0.0, 0.01, -0.01],
        }
    ).to_csv(returns, index=False)
    pd.DataFrame({"symbol": ["A", "B"], "value": [1.0, -0.5], "momentum": [0.2, 0.8]}).to_csv(
        factors, index=False
    )
    config = tmp_path / "risk.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "advanced": {
                    "positions": {"path": str(positions)},
                    "asset_returns": {"path": str(returns)},
                    "factor_exposures": {"path": str(factors)},
                }
            }
        ),
        encoding="utf-8",
    )
    result = run_check(config)
    assert len(result.metrics["risk_contributions"]) == 2
    assert (
        abs(sum(row["risk_contribution"] for row in result.metrics["risk_contributions"]) - 1.0)
        < 1e-8
    )
    assert result.metrics["factor_exposures"]["value"] == pytest.approx(0.4)


def test_advanced_check_reports_parametric_tail_risk_and_limit(tmp_path) -> None:
    returns = tmp_path / "portfolio_returns.csv"
    pd.DataFrame({"net_return": [-0.10, -0.04, 0.01, 0.02, 0.03]}).to_csv(returns, index=False)
    config = tmp_path / "risk.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "advanced": {"returns": {"path": str(returns), "confidence": 0.95}},
                "rules": {"parametric_var_limit": 0.0, "parametric_cvar_limit": 0.0},
            }
        ),
        encoding="utf-8",
    )

    result = run_check(config)

    assert result.metrics["tail_risk"]["parametric"]["var"] > 0
    assert {alert.rule_id for alert in result.alerts} == {
        "tail_parametric_var",
        "tail_parametric_cvar",
    }


def test_advanced_check_reports_missing_risk_inputs_as_critical(tmp_path) -> None:
    positions = tmp_path / "positions.csv"
    returns = tmp_path / "asset_returns.csv"
    factors = tmp_path / "factors.csv"
    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01", "2025-01-01"],
            "symbol": ["A", "B", "UNHELD"],
            "weight": [0.5, 0.5, 0.0],
            "market_value": [500_000, 500_000, 0],
        }
    ).to_csv(positions, index=False)
    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "A": [0.01, -0.01, 0.02],
            "B": [None, None, None],
        }
    ).to_csv(returns, index=False)
    pd.DataFrame({"symbol": ["A", "B"], "beta": [1.0, None]}).to_csv(factors, index=False)
    config = tmp_path / "risk.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "advanced": {
                    "positions": {"path": str(positions)},
                    "asset_returns": {"path": str(returns)},
                    "factor_exposures": {"path": str(factors)},
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_check(config)

    assert result.has_critical
    assert {alert.rule_id for alert in result.alerts} == {
        "risk_contributions_not_evaluable",
        "factor_exposures_not_evaluable",
    }
    assert result.metrics["risk_contributions"] == {"status": "not_evaluable"}
    assert result.metrics["factor_exposures"] == {"status": "not_evaluable"}
    assert result.metrics["factor_exposure_coverage"]["missing_assets"] == ["B"]
    assert result.metrics["risk_contribution_coverage"]["insufficient_assets"] == ["B"]


def test_advanced_check_rejects_nonfinite_position_weight(tmp_path) -> None:
    positions = tmp_path / "positions.csv"
    pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "weight": [0.5, np.inf],
            "market_value": [500_000, 500_000],
        }
    ).to_csv(positions, index=False)
    config = tmp_path / "risk.yaml"
    config.write_text(
        yaml.safe_dump({"advanced": {"positions": {"path": str(positions)}}}),
        encoding="utf-8",
    )

    result = run_check(config)

    assert result.has_critical
    assert result.alerts[0].rule_id == "positions_input_not_evaluable"
    assert result.metrics["positions_input"]["status"] == "not_evaluable"
