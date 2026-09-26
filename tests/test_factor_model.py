from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_risk_monitor import (
    AssetReturnObservation,
    ExposureSnapshot,
    FactorModelConfig,
    fit_barra_style_risk_model,
)


def _model_inputs(periods: int = 8):
    exposures = pd.DataFrame(
        {
            "market": [1.0, 1.0, 1.0, 1.0],
            "size_proxy": [-1.0, -0.3, 0.4, 1.1],
        },
        index=["A", "B", "C", "D"],
    )
    snapshots = [
        ExposureSnapshot(
            effective_at="2025-01-01T00:00:00Z",
            available_at="2025-01-01T00:00:00Z",
            values=exposures,
            source="test-pit-descriptors",
        )
    ]
    rng = np.random.default_rng(17)
    observations = []
    for day in range(periods):
        start = pd.Timestamp("2025-01-02T00:00:00Z") + pd.Timedelta(days=day)
        end = start + pd.Timedelta(days=1)
        factor_return = np.array([0.001 * (-1) ** day, 0.0005 * (day - periods / 2)])
        returns = exposures.to_numpy() @ factor_return + rng.normal(0, 0.0007, len(exposures))
        observations.append(
            AssetReturnObservation(
                period_start=start,
                period_end=end,
                available_at=end,
                values=pd.Series(returns, index=exposures.index),
                source="matured-close-to-close",
            )
        )
    return snapshots, observations


def _fit_model():
    snapshots, observations = _model_inputs()
    return fit_barra_style_risk_model(
        exposure_snapshots=snapshots,
        return_observations=observations,
        as_of="2025-01-20T00:00:00Z",
        config=FactorModelConfig(
            model_kind="statistical_proxy",
            min_periods=6,
            min_assets_per_period=4,
            min_asset_observations=6,
            covariance_shrinkage=0.25,
            specific_variance_shrinkage=0.3,
            annualization=252,
        ),
    )


def test_barra_style_proxy_closes_x_f_d_covariance_loop() -> None:
    model = _fit_model()

    expected = (
        model.exposures.to_numpy()
        @ model.factor_covariance.to_numpy()
        @ model.exposures.to_numpy().T
        + np.diag(model.specific_variances.to_numpy())
    )
    np.testing.assert_allclose(model.asset_covariance.to_numpy(), expected)
    assert np.linalg.eigvalsh(model.factor_covariance.to_numpy()).min() >= -1e-15
    assert np.linalg.eigvalsh(model.asset_covariance.to_numpy()).min() >= -1e-15
    assert model.diagnostics.periods == 8
    assert model.diagnostics.regression_ranks == (2,) * 8
    assert model.model_kind == "statistical_proxy"
    assert model.to_dict()["vendor_model"] is False
    json.dumps(model.to_dict(), allow_nan=False)


def test_factor_model_returns_defensive_copies_and_rejects_inconsistent_internal_state() -> None:
    model = _fit_model()
    original = model.factor_covariance
    caller_copy = model.factor_covariance
    caller_copy.iloc[0, 0] *= 100

    pd.testing.assert_frame_equal(model.factor_covariance, original)
    report = model.analyze({"A": 0.5, "B": 0.5})
    assert report.portfolio_risk.total_variance == pytest.approx(
        report.portfolio_risk.factor_variance + report.portfolio_risk.specific_variance
    )

    internal = object.__getattribute__(model, "_factor_covariance")
    internal.iloc[0, 0] *= 100
    with pytest.raises(ValueError, match="inconsistent with X F X.T \\+ D"):
        model.analyze({"A": 0.5, "B": 0.5})


def test_portfolio_and_benchmark_risk_exposures_te_and_attribution() -> None:
    model = _fit_model()
    report = model.analyze(
        {"A": 0.5, "B": 0.5, "OUTSIDE_ZERO": 0.0},
        benchmark_weights={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
    )

    assert report.portfolio_exposures["market"] == pytest.approx(1.0)
    assert report.benchmark_exposures["market"] == pytest.approx(1.0)
    assert report.active_exposures["market"] == pytest.approx(0.0)
    assert report.tracking_risk.volatility > 0
    assert report.portfolio_risk.total_variance == pytest.approx(
        report.portfolio_risk.factor_variance + report.portfolio_risk.specific_variance
    )
    assert report.tracking_risk.total_variance == pytest.approx(
        report.tracking_risk.factor_variance + report.tracking_risk.specific_variance
    )
    payload = report.to_dict()
    assert payload["model_kind"] == "statistical_proxy"
    assert payload["is_proxy"] is True
    assert payload["units"]["volatility"] == "annualized_return_rate"
    json.dumps(payload, allow_nan=False)


def test_nonzero_portfolio_assets_must_be_covered() -> None:
    model = _fit_model()
    with pytest.raises(ValueError, match="non-zero assets missing.*OUTSIDE"):
        model.analyze({"A": 0.5, "OUTSIDE": 0.5})


def test_fit_requires_exposure_available_before_return_period() -> None:
    snapshots, observations = _model_inputs()
    late = ExposureSnapshot(
        effective_at="2025-01-01T00:00:00Z",
        available_at="2025-01-02T12:00:00Z",
        values=snapshots[0].values,
    )
    with pytest.raises(ValueError, match="no exposure snapshot.*2025-01-02"):
        fit_barra_style_risk_model(
            exposure_snapshots=[late],
            return_observations=observations,
            as_of="2025-01-20T00:00:00Z",
            config=FactorModelConfig(
                model_kind="statistical_proxy",
                min_periods=6,
                min_assets_per_period=4,
                min_asset_observations=6,
            ),
        )


def test_fit_rejects_unmatured_future_returns() -> None:
    snapshots, observations = _model_inputs()
    with pytest.raises(ValueError, match="complete and available by as_of"):
        fit_barra_style_risk_model(
            exposure_snapshots=snapshots,
            return_observations=observations,
            as_of="2025-01-05T12:00:00Z",
            config=FactorModelConfig(
                model_kind="statistical_proxy",
                min_periods=2,
                min_assets_per_period=4,
                min_asset_observations=2,
            ),
        )


def test_fit_rejects_insufficient_or_overlapping_return_periods() -> None:
    snapshots, observations = _model_inputs()
    with pytest.raises(ValueError, match="at least 9 matured return periods"):
        fit_barra_style_risk_model(
            exposure_snapshots=snapshots,
            return_observations=observations,
            as_of="2025-01-20T00:00:00Z",
            config=FactorModelConfig(
                model_kind="statistical_proxy",
                min_periods=9,
                min_assets_per_period=4,
                min_asset_observations=6,
            ),
        )

    second = observations[1]
    overlapping = list(observations)
    overlapping[1] = AssetReturnObservation(
        period_start=observations[0].period_start + pd.Timedelta(hours=12),
        period_end=second.period_end,
        available_at=second.available_at,
        values=second.values,
    )
    with pytest.raises(ValueError, match="periods must not overlap"):
        fit_barra_style_risk_model(
            exposure_snapshots=snapshots,
            return_observations=overlapping,
            as_of="2025-01-20T00:00:00Z",
            config=FactorModelConfig(
                model_kind="statistical_proxy",
                min_periods=6,
                min_assets_per_period=4,
                min_asset_observations=6,
            ),
        )


def test_fit_rejects_changed_factor_universe_and_ill_conditioning() -> None:
    snapshots, observations = _model_inputs()
    snapshots.append(
        ExposureSnapshot(
            effective_at="2025-01-15T00:00:00Z",
            available_at="2025-01-15T00:00:00Z",
            values=snapshots[0].values.rename(columns={"size_proxy": "value_proxy"}),
        )
    )
    with pytest.raises(ValueError, match="exactly the same factors"):
        fit_barra_style_risk_model(
            exposure_snapshots=snapshots,
            return_observations=observations,
            as_of="2025-01-20T00:00:00Z",
            config=FactorModelConfig(
                model_kind="statistical_proxy",
                min_periods=6,
                min_assets_per_period=4,
                min_asset_observations=6,
            ),
        )

    snapshots, observations = _model_inputs()
    with pytest.raises(ValueError, match="ill-conditioned"):
        fit_barra_style_risk_model(
            exposure_snapshots=snapshots,
            return_observations=observations,
            as_of="2025-01-20T00:00:00Z",
            config=FactorModelConfig(
                model_kind="statistical_proxy",
                min_periods=6,
                min_assets_per_period=4,
                min_asset_observations=6,
                max_condition_number=1.01,
            ),
        )


def test_fit_rejects_rank_deficient_factor_exposures() -> None:
    snapshots, observations = _model_inputs()
    collinear = snapshots[0].values.assign(duplicate_market=1.0)
    with pytest.raises(ValueError, match="rank deficient"):
        fit_barra_style_risk_model(
            exposure_snapshots=[
                ExposureSnapshot(
                    effective_at=snapshots[0].effective_at,
                    available_at=snapshots[0].available_at,
                    values=collinear,
                )
            ],
            return_observations=observations,
            as_of="2025-01-20T00:00:00Z",
            config=FactorModelConfig(
                model_kind="statistical_proxy",
                min_periods=6,
                min_assets_per_period=4,
                min_asset_observations=6,
            ),
        )


def test_fit_rejects_asset_coverage_change_in_regression() -> None:
    snapshots, observations = _model_inputs()
    bad = list(observations)
    first = bad[0]
    bad[0] = AssetReturnObservation(
        period_start=first.period_start,
        period_end=first.period_end,
        available_at=first.available_at,
        values=first.values.drop(index="D"),
    )
    with pytest.raises(ValueError, match="asset coverage differs.*missing_returns.*D"):
        fit_barra_style_risk_model(
            exposure_snapshots=snapshots,
            return_observations=bad,
            as_of="2025-01-20T00:00:00Z",
            config=FactorModelConfig(
                model_kind="statistical_proxy",
                min_periods=6,
                min_assets_per_period=3,
                min_asset_observations=6,
            ),
        )


def test_fit_rejects_new_current_asset_without_specific_history() -> None:
    snapshots, observations = _model_inputs()
    current = pd.concat(
        [
            snapshots[0].values,
            pd.DataFrame({"market": [1.0], "size_proxy": [0.2]}, index=["NEW"]),
        ]
    )
    snapshots.append(
        ExposureSnapshot(
            effective_at="2025-01-15T00:00:00Z",
            available_at="2025-01-15T00:00:00Z",
            values=current,
        )
    )
    with pytest.raises(ValueError, match="specific-return history.*NEW"):
        fit_barra_style_risk_model(
            exposure_snapshots=snapshots,
            return_observations=observations,
            as_of="2025-01-20T00:00:00Z",
            config=FactorModelConfig(
                model_kind="statistical_proxy",
                min_periods=6,
                min_assets_per_period=4,
                min_asset_observations=6,
            ),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_kind": "vendor_barra"},
        {"model_kind": "statistical_proxy", "min_periods": 1},
        {"model_kind": "statistical_proxy", "covariance_shrinkage": 1.1},
        {"model_kind": "statistical_proxy", "max_condition_number": float("inf")},
    ],
)
def test_factor_model_config_rejects_invalid_contract(kwargs) -> None:
    with pytest.raises(ValueError):
        FactorModelConfig(**kwargs)


def test_pit_input_contract_rejects_naive_time_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ExposureSnapshot(
            effective_at="2025-01-01",
            available_at="2025-01-01T00:00:00Z",
            values=pd.DataFrame({"market": [1.0]}, index=["A"]),
        )
    with pytest.raises(ValueError, match="finite"):
        AssetReturnObservation(
            period_start="2025-01-01T00:00:00Z",
            period_end="2025-01-02T00:00:00Z",
            available_at="2025-01-02T00:00:00Z",
            values=pd.Series({"A": np.nan}),
        )
