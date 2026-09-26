import numpy as np
import pandas as pd
import pytest

from quant_risk_monitor.analytics import (
    factor_exposure_drift,
    factor_exposures,
    historical_var_cvar,
    liquidity_days_to_exit,
    parametric_var_cvar,
    risk_contributions,
    shrink_covariance,
    stress_test,
)


def test_tail_risk_and_covariance_risk_contributions() -> None:
    returns = pd.Series([-0.10, -0.04, 0.01, 0.02, 0.03])
    tail = historical_var_cvar(returns, confidence=0.8)
    assert tail.var >= 0.04
    assert tail.cvar >= tail.var
    parametric = parametric_var_cvar(returns, confidence=0.8)
    assert parametric.var > 0
    assert parametric.cvar >= parametric.var

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

    drift = factor_exposure_drift(
        pd.Series({"value": 0.8, "momentum": 0.2}),
        pd.Series({"value": 0.4, "momentum": 0.2}),
        pd.Series({"value": 0.2, "momentum": 0.1}),
    )
    assert drift.loc["value", "z_score"] == pytest.approx(2.0)


@pytest.mark.parametrize("function", [historical_var_cvar, parametric_var_cvar])
def test_tail_risk_rejects_insufficient_data_and_invalid_confidence(function) -> None:
    with pytest.raises(ValueError, match="at least two"):
        function(pd.Series([0.1]))
    with pytest.raises(ValueError, match="confidence"):
        function(pd.Series([0.1, -0.1]), confidence=1.0)


@pytest.mark.parametrize("function", [historical_var_cvar, parametric_var_cvar])
@pytest.mark.parametrize("invalid", [np.inf, -np.inf, "not-a-return"])
def test_tail_risk_rejects_nonfinite_or_nonnumeric_observations(function, invalid) -> None:
    with pytest.raises(ValueError, match="finite numeric"):
        function(pd.Series([0.1, invalid, -0.1]))


def test_analytics_fail_closed_on_incomplete_inputs() -> None:
    with pytest.raises(ValueError, match="shrinkage"):
        shrink_covariance(pd.DataFrame({"A": [0.1, -0.1]}), shrinkage=1.1)
    with pytest.raises(ValueError, match="missing portfolio assets"):
        risk_contributions(
            pd.Series({"A": 1.0, "B": 1.0}),
            pd.DataFrame([[1.0]], index=["A"], columns=["A"]),
        )
    with pytest.raises(ValueError, match="variance"):
        risk_contributions(
            pd.Series({"A": 0.0}),
            pd.DataFrame([[1.0]], index=["A"], columns=["A"]),
        )
    with pytest.raises(ValueError, match="weights must be finite"):
        risk_contributions(
            pd.Series({"A": np.inf}),
            pd.DataFrame([[1.0]], index=["A"], columns=["A"]),
        )
    with pytest.raises(ValueError, match="symmetric"):
        risk_contributions(
            pd.Series({"A": 0.5, "B": 0.5}),
            pd.DataFrame([[1.0, 0.2], [0.1, 1.0]], index=["A", "B"], columns=["A", "B"]),
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        risk_contributions(
            pd.Series({"A": 1.0, "B": 0.0}),
            pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=["A", "B"], columns=["A", "B"]),
        )
    with pytest.raises(ValueError, match="missing assets"):
        stress_test(pd.Series({"A": 1.0}), pd.DataFrame({"B": [-0.1]}))
    with pytest.raises(ValueError, match="finite for every non-zero holding"):
        stress_test(pd.Series({"A": 1.0}), pd.DataFrame({"A": [np.nan]}))
    with pytest.raises(ValueError, match="max_participation"):
        liquidity_days_to_exit(pd.Series({"A": 1.0}), pd.Series({"A": 10.0}), max_participation=0)
    with pytest.raises(ValueError, match="ADV"):
        liquidity_days_to_exit(pd.Series({"A": 1.0}), pd.Series({"A": 0.0}))
    with pytest.raises(ValueError, match="non-zero holdings"):
        factor_exposures(pd.Series({"A": 1.0}), pd.DataFrame({"value": [1.0]}, index=["B"]))
    with pytest.raises(ValueError, match="factor sets differ"):
        factor_exposure_drift(pd.Series({"A": 1.0}), pd.Series({"B": 1.0}))
    with pytest.raises(ValueError, match="scale"):
        factor_exposure_drift(
            pd.Series({"A": 1.0}),
            pd.Series({"A": 0.0}),
            pd.Series({"A": 0.0}),
        )


def test_factor_exposures_reject_partial_missing_but_preserve_true_zero() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5, "C": 0.0})

    with pytest.raises(ValueError, match="'B'.*'beta'"):
        factor_exposures(
            weights,
            pd.DataFrame({"beta": [1.0, np.nan]}, index=["A", "B"]),
        )

    result = factor_exposures(
        weights,
        pd.DataFrame({"beta": [1.0, 0.0]}, index=["A", "B"]),
    )
    assert result["beta"] == pytest.approx(0.5)


def test_factor_exposures_only_require_nonzero_holdings() -> None:
    result = factor_exposures(
        pd.Series({"A": 1.0, "UNHELD": 0.0}),
        pd.DataFrame({"beta": [1.25]}, index=["A"]),
    )
    assert result["beta"] == pytest.approx(1.25)


@pytest.mark.parametrize("missing_value", [np.nan, np.inf, "not-a-return"])
def test_covariance_rejects_asset_without_sufficient_returns(missing_value) -> None:
    returns = pd.DataFrame({"A": [0.01, -0.01, 0.02], "B": [missing_value] * 3})
    with pytest.raises(ValueError, match="insufficient joint return history.*B"):
        shrink_covariance(returns)


def test_covariance_requires_sufficient_joint_history() -> None:
    returns = pd.DataFrame({"A": [0.01, -0.01, np.nan], "B": [np.nan, 0.01, -0.01]})
    with pytest.raises(ValueError, match="complete_observations=1"):
        shrink_covariance(returns)


def test_factor_drift_rejects_changed_or_nonfinite_factor_set() -> None:
    with pytest.raises(ValueError, match="factor sets differ"):
        factor_exposure_drift(
            pd.Series({"value": 0.1, "momentum": 0.2}),
            pd.Series({"value": 0.1, "size": 0.2}),
        )
    with pytest.raises(ValueError, match="finite"):
        factor_exposure_drift(
            pd.Series({"value": np.inf}),
            pd.Series({"value": 0.1}),
        )
