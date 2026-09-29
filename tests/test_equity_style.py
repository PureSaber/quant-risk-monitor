from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_risk_monitor import (
    AssetReturnObservation,
    EquityRiskConfig,
    ExposureSnapshot,
    FactorModelConfig,
    IndustrySnapshot,
    MarketCapSnapshot,
    StyleExposureSnapshot,
    apply_volatility_regime,
    attribute_realized_return,
    blend_available_descriptors,
    blend_specific_variance,
    eigenfactor_risk_adjustment,
    estimate_constrained_factor_returns,
    exponential_weights,
    fit_barra_style_risk_model,
    fit_equity_style_risk_model,
    impute_style_exposures,
    standardize_style_cross_section,
    weighted_newey_west_covariance,
)


def test_industry_constraint_and_sqrt_cap_weights_change_the_country_return() -> None:
    assets = ["A1", "A2", "B1", "B2", "C1", "C2"]
    industries = pd.Series(
        ["Banks", "Banks", "Energy", "Energy", "Materials", "Materials"], index=assets
    )
    caps = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=assets)
    returns = pd.Series([0.03, 0.03, 0.0, 0.0, 0.0, 0.0], index=assets)
    fitted = estimate_constrained_factor_returns(
        returns, industries, caps, None, weight_kind="equal"
    )

    assert fitted.factor_returns["country"] == pytest.approx(0.01)
    assert fitted.factor_returns["Banks"] == pytest.approx(0.02)
    assert fitted.factor_returns["Energy"] == pytest.approx(-0.01)
    assert fitted.factor_returns["Materials"] == pytest.approx(-0.01)
    industry_names = ["Banks", "Energy", "Materials"]
    cap_by_industry = industries.map(caps.groupby(industries).sum())
    weighted = float(
        (fitted.factor_returns[industry_names] * cap_by_industry.groupby(industries).first()).sum()
        / caps.sum()
    )
    assert weighted == pytest.approx(0.0, abs=1e-12)
    assert fitted.residuals.abs().max() == pytest.approx(0.0, abs=1e-12)

    uneven_caps = pd.Series([100.0, 1.0, 1.0], index=["A", "B", "C"])
    uneven_returns = pd.Series([0.10, 0.00, 0.02], index=["A", "B", "C"])
    uneven_style = pd.DataFrame({"style": [1.0, -1.0, 0.25]}, index=uneven_returns.index)
    one_industry = pd.Series(["Banks", "Banks", "Banks"], index=uneven_returns.index)
    equal = estimate_constrained_factor_returns(
        uneven_returns, one_industry, uneven_caps, uneven_style, weight_kind="equal"
    )
    weighted_fit = estimate_constrained_factor_returns(
        uneven_returns, one_industry, uneven_caps, uneven_style, weight_kind="sqrt_market_cap"
    )
    assert equal.factor_returns["Banks"] == pytest.approx(0.0, abs=1e-12)
    assert weighted_fit.factor_returns["country"] != pytest.approx(equal.factor_returns["country"])


def test_covariance_adjustments_and_specific_risk_blend() -> None:
    recent = exponential_weights(6, 2.0)
    assert recent[-1] > recent[0]
    early = pd.DataFrame({"factor": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
    late = pd.DataFrame({"factor": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]})
    weights = exponential_weights(6, 2.0)
    early_variance = float(weighted_newey_west_covariance(early, weights, 0).iloc[0, 0])
    late_variance = float(weighted_newey_west_covariance(late, weights, 0).iloc[0, 0])
    assert late_variance > early_variance

    persistent = pd.DataFrame({"factor": [1.0, 1.0, -1.0, -1.0] * 5})
    base_weights = np.ones(len(persistent)) / len(persistent)
    plain = float(weighted_newey_west_covariance(persistent, base_weights, 0).iloc[0, 0])
    corrected = float(weighted_newey_west_covariance(persistent, base_weights, 1).iloc[0, 0])
    assert corrected > plain

    history = pd.DataFrame({"factor": [0.01] * 20 + [0.05, -0.05] * 5})
    baseline = pd.DataFrame([[0.0001]], index=["factor"], columns=["factor"])
    scaled = apply_volatility_regime(
        baseline, history, lookback=10, shrinkage=1.0, scale_cap=4.0, annualization=1
    )
    assert float(scaled.iloc[0, 0]) > float(baseline.iloc[0, 0])

    assert blend_specific_variance(0.4, 0, 0.2, 20) == pytest.approx(0.2)
    assert blend_specific_variance(0.4, 20, 0.2, 20) == pytest.approx(0.3)


def test_eigenfactor_adjustment_raises_small_eigenvalues_without_cutting_market_risk() -> None:
    names = ["country"] + [f"industry_{i}" for i in range(30)] + ["size", "beta"]
    variances = np.array([0.04] + [0.002] * 30 + [0.0004, 0.0009])
    covariance = pd.DataFrame(np.diag(variances), index=names, columns=names)
    adjusted = eigenfactor_risk_adjustment(covariance, 60, simulations=200, scale=1.0, seed=3)

    before = np.linalg.eigvalsh(covariance.to_numpy())
    after = np.linalg.eigvalsh(adjusted.to_numpy())
    assert after[0] > before[0]
    assert np.sqrt(adjusted.loc["country", "country"]) == pytest.approx(0.2, rel=0.03)
    again = eigenfactor_risk_adjustment(covariance, 60, simulations=200, scale=1.0, seed=3)
    pd.testing.assert_frame_equal(adjusted, again)
    unchanged = eigenfactor_risk_adjustment(covariance, 60, simulations=0, scale=1.0, seed=3)
    pd.testing.assert_frame_equal(unchanged, covariance)
    neutral = eigenfactor_risk_adjustment(covariance, 60, simulations=50, scale=0.0, seed=3)
    np.testing.assert_allclose(neutral.to_numpy(), covariance.to_numpy(), atol=1e-15)


def test_imputation_standardization_and_descriptor_blend() -> None:
    styles = pd.DataFrame({"size": [1.0, np.nan, 3.0, 10.0]}, index=["A", "B", "C", "D"])
    industries = pd.Series(["Banks", "Banks", "Banks", "Energy"], index=styles.index)
    filled = impute_style_exposures(styles, industries, method="industry_median")
    assert filled.loc["B", "size"] == pytest.approx(2.0)
    assert filled.loc["B", "size"] != pytest.approx(0.0)
    with pytest.raises(ValueError, match="missing values"):
        impute_style_exposures(styles, industries, method="reject")

    raw = pd.DataFrame({"size": [1.0, 2.0, 3.0, 4.0], "cube": [1.0, 8.0, 27.0, 64.0]})
    caps = pd.Series(1.0, index=raw.index)
    standardized = standardize_style_cross_section(
        raw, caps, winsor_z=3.0, orthogonalize=(("cube", ("size",)),)
    )
    assert standardized["size"].std(ddof=0) == pytest.approx(1.0)
    correlation = np.corrcoef(standardized["cube"], standardized["size"])[0, 1]
    assert correlation == pytest.approx(0.0, abs=1e-8)

    blended = blend_available_descriptors(
        {
            "trailing": pd.Series([1.0, np.nan], index=["A", "B"]),
            "cash": pd.Series([np.nan, 4.0], index=["A", "B"]),
            "forward": pd.Series([3.0, 4.0], index=["A", "B"]),
        },
        {"trailing": 1.0, "cash": 1.0, "forward": 3.0},
    )
    assert blended.loc["A"] == pytest.approx((1.0 + 9.0) / 4.0)
    assert blended.loc["B"] == pytest.approx(4.0)


def _snapshots():
    assets = ["A1", "A2", "A3", "B1", "B2", "B3", "C1"]
    industries = ["Banks", "Banks", "Banks", "Energy", "Energy", "Energy", "Banks"]
    caps = [10.0, 10.0, 10.0, 12.0, 12.0, 12.0, 8.0]
    style = [1.0, 0.2, -0.4, 0.5, -0.7, 0.1, 0.3]
    when = "2024-01-01T00:00:00Z"
    return (
        [
            StyleExposureSnapshot(
                effective_at=when,
                available_at=when,
                values=pd.DataFrame({"value_style": style}, index=assets),
                source="test-style",
            )
        ],
        [
            IndustrySnapshot(
                effective_at=when,
                available_at=when,
                labels=pd.Series(industries, index=assets),
                source="test-industry",
            )
        ],
        [
            MarketCapSnapshot(
                effective_at=when,
                available_at=when,
                values=pd.Series(caps, index=assets),
                source="test-cap",
            )
        ],
        assets,
    )


def _returns(assets: list[str], periods: int = 24) -> list[AssetReturnObservation]:
    base = [asset for asset in assets if asset != "C1"]
    style = {"A1": 1.0, "A2": 0.2, "A3": -0.4, "B1": 0.5, "B2": -0.7, "B3": 0.1, "C1": 0.3}
    industry = {asset: 1.0 if asset.startswith("A") or asset == "C1" else -1.0 for asset in assets}
    observations = []
    for day in range(periods):
        names = assets if day == periods - 1 else base
        values = [
            0.001 * ((-1) ** day)
            + 0.002 * industry[name] * ((-1) ** (day // 2))
            + 0.0005 * style[name]
            for name in names
        ]
        start = pd.Timestamp("2024-01-02T00:00:00Z") + pd.Timedelta(days=day)
        end = start + pd.Timedelta(days=1)
        observations.append(
            AssetReturnObservation(
                period_start=start,
                period_end=end,
                available_at=end,
                values=pd.Series(values, index=names),
                source="test-return",
            )
        )
    return observations


def test_equity_style_model_closes_the_risk_identity_and_attributes_return() -> None:
    styles, industries, caps, assets = _snapshots()
    model = fit_equity_style_risk_model(
        style_snapshots=styles,
        industry_snapshots=industries,
        market_cap_snapshots=caps,
        return_observations=_returns(assets),
        as_of="2024-01-27T00:00:00Z",
        config=EquityRiskConfig(
            model_kind="statistical_proxy",
            min_periods=20,
            min_assets_per_period=6,
            newey_west_lags=2,
            regime_lookback=8,
            eigen_simulations=50,
            specific_prior_strength=20,
        ),
    )

    expected = (
        model.exposures.to_numpy()
        @ model.factor_covariance.to_numpy()
        @ model.exposures.to_numpy().T
        + np.diag(model.specific_variances.to_numpy())
    )
    np.testing.assert_allclose(model.asset_covariance.to_numpy(), expected)
    assert model.to_dict()["vendor_model"] is False
    assert model.diagnostics.estimator == "equity_style"
    assert model.diagnostics.to_dict()["parameters_are_vendor_calibration"] is False
    assert model.diagnostics.estimation["industry_factors"] == ["Banks", "Energy"]
    assert model.specific_variances["C1"] > 0
    assert int(model.diagnostics.residual_observations["C1"]) < 2

    early_cap = {"Banks": 30.0, "Energy": 36.0}
    last_cap = {"Banks": 38.0, "Energy": 36.0}
    for position, (_, row) in enumerate(model.factor_returns.iterrows()):
        weights = last_cap if position == len(model.factor_returns) - 1 else early_cap
        total = sum(weights.values())
        constrained = sum(row[name] * weight for name, weight in weights.items()) / total
        assert constrained == pytest.approx(0.0, abs=1e-10)

    realized = pd.Series(0.01, index=model.exposures.index)
    attribution = attribute_realized_return(model, realized, caps[0].values)
    reconstructed = attribution.factor_contribution.sum(axis=1) + attribution.specific_return
    pd.testing.assert_series_equal(reconstructed, realized, check_names=False)
    json.dumps(model.to_dict(), allow_nan=False)
    json.dumps(attribution.to_dict(), allow_nan=False)
    report = model.analyze({"A1": 0.5, "B1": 0.5}, {"A1": 0.25, "A2": 0.25, "B1": 0.25, "B2": 0.25})
    assert report.tracking_risk.volatility > 0


def test_equity_style_rejects_future_inputs_and_missing_styles_when_asked() -> None:
    styles, industries, caps, assets = _snapshots()
    observations = _returns(assets)
    late = StyleExposureSnapshot(
        effective_at="2024-01-01T00:00:00Z",
        available_at="2024-01-03T00:00:00Z",
        values=styles[0].values,
    )
    with pytest.raises(ValueError, match="no style snapshot"):
        fit_equity_style_risk_model(
            style_snapshots=[late],
            industry_snapshots=industries,
            market_cap_snapshots=caps,
            return_observations=observations,
            as_of="2024-01-27T00:00:00Z",
            config=EquityRiskConfig(
                model_kind="statistical_proxy",
                min_periods=20,
                min_assets_per_period=6,
                newey_west_lags=2,
            ),
        )
    with pytest.raises(ValueError, match="weight_kind"):
        EquityRiskConfig(model_kind="statistical_proxy", weight_kind="cap")


def test_minimal_model_attribution_remains_available() -> None:
    exposures = pd.DataFrame({"market": [1.0, 1.0, 1.0, 1.0]}, index=["A", "B", "C", "D"])
    snapshot = ExposureSnapshot(
        effective_at="2025-01-01T00:00:00Z",
        available_at="2025-01-01T00:00:00Z",
        values=exposures,
    )
    observations = []
    for day in range(6):
        start = pd.Timestamp("2025-01-02T00:00:00Z") + pd.Timedelta(days=day)
        end = start + pd.Timedelta(days=1)
        observations.append(
            AssetReturnObservation(
                period_start=start,
                period_end=end,
                available_at=end,
                values=pd.Series(0.001, index=exposures.index),
            )
        )
    model = fit_barra_style_risk_model(
        exposure_snapshots=[snapshot],
        return_observations=observations,
        as_of="2025-01-20T00:00:00Z",
        config=FactorModelConfig(
            model_kind="statistical_proxy",
            min_periods=6,
            min_assets_per_period=4,
            min_asset_observations=6,
        ),
    )
    realized = pd.Series([0.01, -0.02, 0.0, 0.03], index=exposures.index)
    attribution = attribute_realized_return(model, realized)
    reconstructed = attribution.factor_contribution.sum(axis=1) + attribution.specific_return
    pd.testing.assert_series_equal(reconstructed, realized, check_names=False)
    assert model.diagnostics.estimator == "equal_weight_ols"
