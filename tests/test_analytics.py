import numpy as np
import pandas as pd
import pytest

from quant_risk_monitor.analytics import (
    factor_exposures,
    historical_var_cvar,
    liquidity_days_to_exit,
    risk_contributions,
    shrink_covariance,
    stress_test,
)


def test_tail_risk_and_covariance_risk_contributions() -> None:
    returns = pd.Series([-0.10, -0.04, 0.01, 0.02, 0.03])
    tail = historical_var_cvar(returns, confidence=0.8)
    assert tail.var >= 0.04
    assert tail.cvar >= tail.var

    matrix = pd.DataFrame({"A": [0.01, -0.01, 0.02, 0.0], "B": [0.0, 0.01, -0.01, 0.02]})
    covariance = shrink_covariance(matrix)
    contributions = risk_contributions(pd.Series({"A": 0.6, "B": 0.4}), covariance)
    assert contributions["risk_contribution"].sum() == pytest.approx(1.0)
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_stress_liquidity_and_factor_exposure() -> None:
    weights = pd.Series({"A": 0.6, "B": 0.4})
    scenarios = pd.DataFrame(
        {"A": [-0.2, -0.05], "B": [-0.1, -0.3]}, index=["market_crash", "sector_crash"]
    )
    stressed = stress_test(weights, scenarios)
    assert stressed.loc["market_crash", "portfolio_return"] == pytest.approx(-0.16)

    liquidity = liquidity_days_to_exit(
        pd.Series({"A": 1_000_000.0, "B": 500_000.0}),
        pd.Series({"A": 10_000_000.0, "B": 1_000_000.0}),
        max_participation=0.1,
    )
    assert liquidity.loc["B", "days_to_exit"] == pytest.approx(5.0)

    exposures = pd.DataFrame({"value": [1.0, -0.5], "momentum": [0.2, 0.8]}, index=["A", "B"])
    portfolio = factor_exposures(weights, exposures)
    assert portfolio["value"] == pytest.approx(0.4)
