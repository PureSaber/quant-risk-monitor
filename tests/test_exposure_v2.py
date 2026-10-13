from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_exposure_builder import inputs
from test_factor_model import _fit_model

from quant_risk_monitor.exposure_builder import (
    ExposureBuildConfig,
    build_exposure_snapshot,
    public_equity_style_config,
)
from quant_risk_monitor.exposure_cache import DailyExposureStore
from quant_risk_monitor.style_descriptors import (
    additional_market_descriptors,
    annual_growth_descriptors,
    paired_market_descriptors,
)


def test_nonlinear_size_is_unit_invariant_and_orthogonal():
    kwargs = inputs()
    cap = np.geomspace(1e8, 1e12, 12)
    kwargs["descriptor_history"]["market_cap"] = cap
    kwargs["descriptor_history"]["size"] = np.log(cap)
    kwargs["config"] = ExposureBuildConfig(
        styles={"size": {"size": 1}},
        derived_styles={"nonlinear": "size"},
        orthogonalize={"nonlinear": ["size"]},
        industry_universe=("a", "b", "c"),
        definition_version="v2",
    )
    original = build_exposure_snapshot(**kwargs)
    kwargs["descriptor_history"]["market_cap"] /= 10000
    kwargs["descriptor_history"]["size"] -= np.log(10000)
    converted = build_exposure_snapshot(**kwargs)
    np.testing.assert_allclose(original.values, converted.values, atol=1e-11)
    w = original.market_caps / original.market_caps.sum()
    assert w @ (original.values["size"] * original.values["nonlinear"]) == pytest.approx(
        0, abs=1e-11
    )


def test_available_growth_never_imputes_raw_revenue_and_audits_fallback():
    kwargs = inputs()
    frame = kwargs["descriptor_history"]
    frame["revenue"] = [np.nan, np.nan, *range(2, 12)]
    frame["profit"] = [3, np.nan, 4, 7, 2, 9, 1, 8, 6, 12, 10, 5]
    kwargs["config"] = ExposureBuildConfig(
        styles={"growth": {"revenue": 1, "profit": 1}},
        industry_universe=("a", "b", "c"),
        definition_version="v2",
        available_styles=("growth",),
        missing_policy="industry_median",
        missing_group_fallback="global",
    )
    result = build_exposure_snapshot(**kwargs)
    audit = result.provenance
    assert audit["descriptor_processing"]["revenue"]["imputed"] == {}
    composition = audit["style_processing"]["growth"]["composition"]
    assert composition["observed_descriptors"]["S0"] == ["profit"]
    assert composition["observed_weight_fraction"]["S0"] == 0.5
    assert composition["observed_weight_fraction"]["S1"] == 0
    assert not composition["imputed"]["S1"]["global_fallback"]
    assert frame.revenue.isna().sum() == 2
    frame["revenue"] = np.nan
    # A structurally unavailable descriptor can be excluded; the observed profit
    # component is retained rather than a invented sales level.
    other = build_exposure_snapshot(**kwargs)
    assert other.provenance["descriptor_processing"]["revenue"]["observed_count"] == 0
    frame.loc[:3, "profit"] = np.nan
    with pytest.raises(ValueError, match="coverage"):
        build_exposure_snapshot(**kwargs)


def test_paired_beta_matches_independent_ols_with_missing_market_and_stock():
    rng = np.random.default_rng(19)
    dates = pd.bdate_range("2020-01-01", periods=300)
    market = pd.Series(rng.normal(0, 0.02, 300), index=dates)
    stock = 0.001 + 1.3 * market + pd.Series(rng.normal(0, 0.01, 300), index=dates)
    stock.iloc[::7] = np.nan
    market.iloc[::13] = np.nan
    result, counts = paired_market_descriptors(stock.to_frame("a"), market)
    paired = pd.concat([stock.iloc[-253:-1], market.iloc[-253:-1]], axis=1).dropna()
    design = np.column_stack([np.ones(len(paired)), paired.iloc[:, 1]])
    coef = np.linalg.lstsq(design, paired.iloc[:, 0], rcond=None)[0]
    residual = paired.iloc[:, 0].to_numpy() - design @ coef
    assert result["beta_252d"].iloc[-1, 0] == pytest.approx(coef[1])
    assert result["residual_vol_252d"].iloc[-1, 0] == pytest.approx(residual.std(ddof=1))
    assert counts.iloc[-1, 0] == len(paired)
    poisoned = stock.copy()
    poisoned.iloc[-1] = 1e10
    other, _ = paired_market_descriptors(poisoned.to_frame("a"), market)
    pd.testing.assert_frame_equal(result["beta_252d"], other["beta_252d"])


def test_growth_uses_only_consecutive_known_years_and_preserves_negative_profits():
    rows = []
    for y, revenue, profit in [(2019, 100, -10), (2020, 120, 0), (2021, 160, 20)]:
        for field, value in [("revenue_ytd", revenue), ("profit_ytd", profit)]:
            rows.append(
                {
                    "asset": "a",
                    "field": field,
                    "value": value,
                    "report_period": f"{y}1231",
                    "available_at": f"{y + 1}-04-01T00:00:00+08:00",
                }
            )
    facts = pd.DataFrame(rows)
    dates = pd.DatetimeIndex(["2022-03-31", "2022-04-01", "2023-08-01"])
    values, quality = annual_growth_descriptors(facts, dates, ["a"])
    assert np.isnan(values["revenue_growth_3y"].iloc[0, 0])
    assert values["revenue_growth_3y"].iloc[1, 0] == pytest.approx(30 / (380 / 3))
    assert values["profit_growth_3y"].iloc[1, 0] == pytest.approx(1.5)
    assert quality["revenue_growth_3y"].iloc[-1, 0] == "stale_annual_report"
    revision = facts.iloc[[-2]].copy()
    revision["available_at"] = "2022-04-02T00:00:00+08:00"
    revision["value"] = 1e30
    other, _ = annual_growth_descriptors(pd.concat([facts, revision]), dates, ["a"])
    pd.testing.assert_frame_equal(values["revenue_growth_3y"], other["revenue_growth_3y"])


def test_market_candidates_do_not_compress_missing_sessions():
    dates = pd.bdate_range("2020-01-01", periods=280)
    returns = pd.DataFrame({"a": np.linspace(-0.01, 0.02, 280)}, index=dates)
    turnover = pd.DataFrame({"a": np.linspace(1, 3, 280)}, index=dates)
    result = additional_market_descriptors(returns, turnover)
    assert result["liquidity_log_turnover_21d"].iloc[-1, 0] == pytest.approx(
        np.log(turnover.iloc[-22:-1, 0].mean() / 100)
    )
    assert np.isfinite(result["cumulative_range_252d"].iloc[-1, 0])
    returns.iloc[-10] = np.nan
    result = additional_market_descriptors(returns, turnover)
    assert np.isnan(result["cumulative_range_252d"].iloc[-1, 0])


def test_factorized_portfolio_matches_dense_without_materialization(monkeypatch):
    model = _fit_model()
    dense = model.asset_covariance
    np.testing.assert_allclose(model.asset_variances, np.diag(dense))
    weights = pd.Series([0.6, -0.2, 0.4, 0.2], index=model.exposures.index)
    expected = float(weights @ dense @ weights)
    monkeypatch.setattr(
        type(model),
        "asset_covariance",
        property(lambda _: (_ for _ in ()).throw(AssertionError("dense access"))),
    )
    assert model.analyze(weights).portfolio_risk.total_variance == pytest.approx(expected)
    assert "asset_covariance" not in model.to_dict(include_asset_covariance=False)
    assert model._asset_covariance is None


def test_daily_cache_reuses_market_snapshot_and_rejects_corruption(tmp_path):
    kwargs = inputs()
    store = DailyExposureStore(tmp_path)
    calls = []

    def build():
        calls.append(1)
        snap = build_exposure_snapshot(**kwargs)
        return replace(snap, effective_at=kwargs["as_of"])

    arguments = {
        "as_of": kwargs["as_of"],
        "configuration": {"definition_version": "v1"},
        "input_digests": {"daily": "a" * 64},
        "build": build,
    }
    first, hit = store.get_or_build(**arguments)
    assert not hit
    second, hit = store.get_or_build(**arguments)
    assert hit and len(calls) == 1
    pd.testing.assert_frame_equal(first.values, second.values)
    _, hit = store.get_or_build(**{**arguments, "input_digests": {"daily": "b" * 64}})
    assert not hit and len(calls) == 2
    file = next(tmp_path.glob("*.json"))
    file.write_text(file.read_text().replace('"S0"', '"BAD"'), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        store.get_or_build(**arguments)


def test_v2_preset_rejects_cycles_and_unknown_enhancements():
    config = public_equity_style_config(
        ("a", "b"), definition_version="v2", enhancements=("growth",), available_growth=True
    )
    assert config.derived_styles == {"nonlinear_size": "size"}
    with pytest.raises(ValueError, match="cycle"):
        replace(
            config,
            derived_styles={
                "nonlinear_size": "nonlinear_other",
                "nonlinear_other": "nonlinear_size",
            },
        )
    with pytest.raises(ValueError, match="unknown"):
        public_equity_style_config(("a",), definition_version="v2", enhancements=("made_up",))
