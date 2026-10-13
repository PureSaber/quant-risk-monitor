from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_risk_monitor import ExposureSnapshot, FactorModelConfig
from quant_risk_monitor.risk_estimators import (
    estimate_covariance,
    estimate_specific_risk,
    regime_multiplier,
    solve_factor_returns,
    time_weights,
)


def snapshot():
    return ExposureSnapshot(
        effective_at="2025-01-01T00:00:00Z",
        available_at="2025-01-01T00:00:00Z",
        values=pd.DataFrame(
            {
                "market": 1.0,
                "size": [-2.0, -1.0, 0.0, 1.0, 2.0, 4.0],
                "industry:a": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                "industry:b": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            },
            index=list("ABCDEF"),
        ),
        market_caps=pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], index=list("ABCDEF")),
        industries=pd.Series(list("aaabbb"), index=list("ABCDEF")),
        industry_factors=("industry:a", "industry:b"),
    )


@pytest.mark.parametrize("weighting", ["equal", "market_cap", "sqrt_market_cap"])
def test_constrained_wls_matches_independent_kkt_solution(weighting):
    item = snapshot()
    returns = pd.Series([0.04, -0.02, 0.01, 0.08, -0.03, 0.05], index=item.values.index)
    config = FactorModelConfig(
        model_kind="fundamental_style", regression_weighting=weighting, industry_constraint=True
    )
    coefficients, residual, rank, _, diagnostic = solve_factor_returns(item, returns, config)
    x = item.values.to_numpy()
    caps = item.market_caps.to_numpy()
    w = {"equal": np.ones(6), "market_cap": caps, "sqrt_market_cap": np.sqrt(caps)}[weighting]
    constraint = np.array([0, 0, 7 / 63, 56 / 63])
    normal = x.T @ np.diag(w) @ x
    kkt = np.block([[normal, constraint[:, None]], [constraint[None, :], np.zeros((1, 1))]])
    expected = np.linalg.solve(kkt, np.r_[x.T @ (w * returns.to_numpy()), 0])[:4]
    np.testing.assert_allclose(coefficients, expected, atol=1e-14)
    np.testing.assert_allclose(x @ coefficients + residual, returns, atol=1e-14)
    assert abs(constraint @ coefficients) < 1e-14
    assert rank == 3 and diagnostic["free_coefficients"] == 3
    # Reordering assets and factors cannot change economics.
    reordered = replace(item, values=item.values.iloc[::-1, ::-1])
    other = solve_factor_returns(reordered, returns.iloc[::-1], config)[0]
    np.testing.assert_allclose(other[::-1], coefficients, atol=1e-14)


def test_wls_objective_weights_are_not_squared_again():
    item = replace(
        snapshot(), values=pd.DataFrame({"market": 1.0}, index=list("ABCDEF")), industry_factors=()
    )
    returns = pd.Series(range(6), index=list("ABCDEF"), dtype=float)
    coefficients = solve_factor_returns(
        item,
        returns,
        FactorModelConfig(model_kind="statistical_proxy", regression_weighting="market_cap"),
    )[0]
    assert coefficients[0] == pytest.approx(np.average(returns, weights=item.market_caps))


@pytest.mark.parametrize("half_life", [None, 2.5])
@pytest.mark.parametrize("lags", [0, 1, 3])
def test_newey_west_matches_independent_double_sum(half_life, lags):
    values = np.array([[1.0, 4.0], [3.0, -1.0], [2.0, 2.0], [-2.0, 3.0], [4.0, 0.0]])
    raw = np.ones(5) if half_life is None else 2.0 ** (np.arange(5) / half_life)
    w = raw / raw.sum()
    centered = values - np.sum(w[:, None] * values, axis=0)
    expected = np.zeros((2, 2))
    for i in range(5):
        for j in range(5):
            kernel = max(0.0, 1 - abs(i - j) / (lags + 1))
            expected += np.sqrt(w[i] * w[j]) * kernel * np.outer(centered[i], centered[j])
    expected /= 1 - sum(w * w)
    actual, diagnostic = estimate_covariance(values, half_life=half_life, lags=lags)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    assert np.linalg.eigvalsh(actual).min() >= -1e-13
    assert diagnostic["effective_observations"] == pytest.approx(1 / sum(w * w))


def test_equal_weight_zero_lag_is_sample_covariance_and_shrinkage_is_exact():
    values = np.array([[1, 2], [-2, 1], [4, -1], [3, 5]])
    sample = np.cov(values, rowvar=False, ddof=1)
    result, _ = estimate_covariance(values, shrinkage=0.3)
    np.testing.assert_allclose(result, 0.7 * sample + 0.3 * np.diag(np.diag(sample)))


def test_regime_uses_prior_only_forecasts_and_reports_bound():
    values = np.array([1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 8.0, -8.0])[:, None]
    config = FactorModelConfig(
        model_kind="statistical_proxy", regime_window=2, regime_min_periods=4, regime_scale_max=100
    )
    expected = np.mean(
        [(values[t, 0] - values[:t, 0].mean()) ** 2 / values[:t, 0].var(ddof=1) for t in (6, 7)]
    )
    scale, diagnostic = regime_multiplier(values, config, half_life=None)
    assert scale == pytest.approx(expected)
    bounded, diagnostic = regime_multiplier(
        values, replace(config, regime_scale_max=2), half_life=None
    )
    assert bounded == 2 and diagnostic["bound_applied"]
    with pytest.raises(ValueError, match="prior history"):
        regime_multiplier(values[:4], config, half_life=None)
    with pytest.raises(ValueError, match="zero-variance"):
        regime_multiplier(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0])[:, None], config, half_life=None)


def test_industry_bayesian_specific_variance_matches_posterior_mean():
    item = snapshot()
    residuals = {
        asset: np.array([-1.0, 0.0, 1.0, 0.0]) * (i + 1) / 100
        for i, asset in enumerate(item.values.index)
    }
    config = FactorModelConfig(
        model_kind="fundamental_style",
        specific_target="industry",
        specific_shrinkage_method="bayesian",
        specific_prior_dof=6,
        annualization=1,
    )
    result, diagnostic = estimate_specific_risk(residuals, item, config, 4)
    raw = {asset: np.var(values, ddof=1) for asset, values in residuals.items()}
    for asset in item.values.index:
        peers = [
            a
            for a in item.values.index
            if a != asset and item.industries[a] == item.industries[asset]
        ]
        prior = np.mean([raw[a] for a in peers])
        assert result[asset] == pytest.approx((3 * raw[asset] + 6 * prior) / 9)
        assert diagnostic["shrinkage_weights"][asset] == pytest.approx(2 / 3)
    thin = replace(item, industries=pd.Series(list("aabbbc"), index=list("ABCDEF")))
    with pytest.raises(ValueError, match="insufficient peer"):
        estimate_specific_risk(residuals, thin, config, 4)
    _, fallback = estimate_specific_risk(
        residuals, thin, replace(config, specific_group_fallback="global"), 4
    )
    assert fallback["group_fallback_assets"] == ["A", "B", "F"]


@pytest.mark.parametrize(
    "change",
    [
        {"regression_weighting": "bad"},
        {"industry_constraint": 1},
        {"factor_half_life": 0},
        {"specific_half_life": float("nan")},
        {"newey_west_lags": 2},
        {"risk_horizon": 5, "newey_west_lags": 2},
        {"regime_window": 1},
        {"specific_target": "bad"},
        {"specific_shrinkage_method": "bad"},
        {"regime_scale_min": 2},
        {"regime_scale_max": 0.5},
        {"specific_prior_dof": -1},
        {"specific_min_group_assets": True},
        {"specific_group_fallback": "unknown"},
    ],
)
def test_rejects_ambiguous_or_invalid_estimator_configuration(change):
    with pytest.raises((ValueError, TypeError)):
        FactorModelConfig(model_kind="statistical_proxy", **change)


def test_metadata_and_unidentified_industries_fail_closed():
    item = snapshot()
    config = FactorModelConfig(model_kind="fundamental_style", industry_constraint=True)
    returns = pd.Series(np.arange(6), index=list("ABCDEF"))
    with pytest.raises(ValueError, match="requires market"):
        solve_factor_returns(replace(item, market_caps=None), returns, config)
    with pytest.raises(ValueError, match="rank deficient"):
        solve_factor_returns(item, returns, replace(config, industry_constraint=False))
    with pytest.raises(ValueError, match="market_caps"):
        replace(item, market_caps=pd.Series({"A": -1}))
    with pytest.raises(ValueError, match="industries"):
        replace(item, industries=pd.Series({"A": ""}))
    with pytest.raises(ValueError, match="industry_factors"):
        replace(item, industry_factors=("missing",))
    with pytest.raises(ValueError, match="history"):
        estimate_covariance(np.ones((3, 2)), lags=3)
    with pytest.raises(ValueError, match="effective observations"):
        time_weights(4, 0.00001)
