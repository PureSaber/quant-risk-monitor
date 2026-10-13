from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_risk_monitor import ExposureBuildConfig, build_exposure_snapshot


def inputs():
    assets = [f"S{i}" for i in range(12)]
    descriptors = pd.DataFrame(
        {
            "asset": assets,
            "effective_at": "2025-01-01T00:00:00Z",
            "available_at": "2025-01-02T00:00:00Z",
            "source": "synthetic-descriptors-v1",
            "market_cap": np.arange(1, 13) * 100.0,
            "size": np.arange(12),
            "value": [2.0, 1.0, 3.0, 7.0, 4.0, 6.0, 10.0, 9.0, 11.0, 5.0, 8.0, 50.0],
        }
    )
    classification = descriptors[["asset", "effective_at", "available_at", "source"]].copy()
    classification["industry"] = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    config = ExposureBuildConfig(
        styles={"size": {"size": 1.0}, "value": {"value": 1.0}},
        industry_universe=("a", "b", "c"),
        orthogonalize={"value": ["size"]},
    )
    return {
        "descriptor_history": descriptors,
        "classification_history": classification,
        "assets": assets,
        "as_of": "2025-01-03T00:00:00Z",
        "config": config,
    }


def test_weighted_standardization_composition_orthogonalization_and_audit():
    kwargs = inputs()
    result = build_exposure_snapshot(**kwargs)
    w = result.market_caps / result.market_caps.sum()
    for name in ("size", "value"):
        assert w @ result.values[name] == pytest.approx(0, abs=1e-14)
        assert w @ result.values[name] ** 2 == pytest.approx(1)
    assert w @ (result.values["size"] * result.values["value"]) == pytest.approx(0, abs=1e-14)
    assert result.values[list(result.industry_factors)].sum(axis=1).eq(1).all()
    assert result.provenance["descriptor_processing"]["value"]["clipped_assets"] == ["S11"]
    assert len(result.provenance["sha256"]) == 64
    reversed_styles = dict(reversed(list(kwargs["config"].styles.items())))
    other = build_exposure_snapshot(
        **{**kwargs, "config": replace(kwargs["config"], styles=reversed_styles)}
    )
    pd.testing.assert_frame_equal(result.values, other.values)
    assert result.provenance["sha256"] == other.provenance["sha256"]


def test_future_revisions_and_industry_changes_respect_dual_timestamps():
    kwargs = inputs()
    original = build_exposure_snapshot(**kwargs)
    later = kwargs["descriptor_history"].copy()
    later["available_at"] = "2025-01-05T00:00:00Z"
    later["value"] = "future-unusable"
    changed = kwargs["classification_history"].iloc[[0]].copy()
    changed["effective_at"] = "2025-01-04T00:00:00Z"
    changed["industry"] = "b"
    kwargs["descriptor_history"] = pd.concat([kwargs["descriptor_history"], later])
    kwargs["classification_history"] = pd.concat([kwargs["classification_history"], changed])
    result = build_exposure_snapshot(**kwargs)
    assert result.provenance["sha256"] == original.provenance["sha256"]
    kwargs["as_of"] = "2025-01-04T00:00:00Z"
    result = build_exposure_snapshot(**kwargs)
    assert result.industries["S0"] == "b"
    assert result.values.loc["S0", "industry:b"] == 1


def test_missing_imputation_uses_observed_peers_and_audits_fallback():
    kwargs = inputs()
    kwargs["descriptor_history"].loc[0, "value"] = np.nan
    kwargs["config"] = replace(kwargs["config"], missing_policy="industry_median")
    result = build_exposure_snapshot(**kwargs)
    audit = result.provenance["descriptor_processing"]["value"]["imputed"]["S0"]
    assert audit["value"] == 3 and audit["peer_assets"] == ["S1", "S2", "S3"]
    kwargs["descriptor_history"].loc[1, "value"] = np.nan
    with pytest.raises(ValueError, match="industry peers"):
        build_exposure_snapshot(**kwargs)
    kwargs["config"] = replace(kwargs["config"], missing_group_fallback="global")
    result = build_exposure_snapshot(**kwargs)
    assert result.provenance["descriptor_processing"]["value"]["imputed"]["S1"]["global_fallback"]
    kwargs["descriptor_history"].loc[2, "value"] = np.nan
    with pytest.raises(ValueError, match="coverage"):
        build_exposure_snapshot(**kwargs)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate", "duplicate"),
        ("missing_asset", "available row"),
        ("bad_cap", "market caps"),
        ("future_only", "available row"),
        ("stale", "stale"),
        ("unknown_industry", "universe"),
        ("infinity", "infinity"),
        ("constant", "dispersion"),
        ("missing", "missing"),
        ("source", "source"),
        ("zero_mad", "zero MAD"),
    ],
)
def test_bad_inputs_fail_closed(mutation, match):
    kwargs = inputs()
    frame = kwargs["descriptor_history"]
    if mutation == "duplicate":
        kwargs["descriptor_history"] = pd.concat([frame, frame.iloc[[0]]])
    elif mutation == "missing_asset":
        kwargs["descriptor_history"] = frame.iloc[1:]
    elif mutation == "bad_cap":
        frame.loc[0, "market_cap"] = 0
    elif mutation == "future_only":
        frame["available_at"] = "2026-01-01T00:00:00Z"
    elif mutation == "stale":
        kwargs["config"] = replace(kwargs["config"], max_descriptor_age_days=1)
    elif mutation == "unknown_industry":
        kwargs["classification_history"].loc[0, "industry"] = "unknown"
    elif mutation == "infinity":
        frame.loc[0, "value"] = np.inf
    elif mutation == "constant":
        frame["value"] = 1.0
    elif mutation == "missing":
        frame.loc[0, "value"] = np.nan
    elif mutation == "source":
        frame.loc[0, "source"] = ""
    else:
        frame["value"] = [1.0] * 11 + [2.0]
    with pytest.raises(ValueError, match=match):
        build_exposure_snapshot(**kwargs)


@pytest.mark.parametrize(
    "change",
    [
        {"styles": {}},
        {"styles": {"market": {"x": 1}}},
        {"styles": {"size": {"x": 0}}},
        {"industry_universe": ["a", "a"]},
        {"market_factor": 1},
        {"winsor_method": "unknown"},
        {"winsor_method": "quantile", "winsor_limit": 0.5},
        {"missing_policy": "zero"},
        {"max_missing_fraction": 1},
        {"min_assets": 1},
        {"orthogonalize": {"size": ["value"], "value": ["size"]}},
        {"orthogonalize": {"value": ["missing"]}},
    ],
)
def test_invalid_config_rejected(change):
    from dataclasses import asdict

    config = asdict(inputs()["config"])
    with pytest.raises((ValueError, TypeError)):
        ExposureBuildConfig(**{**config, **deepcopy(change)})


def test_public_ten_style_preset_is_buildable_and_independent():
    from quant_risk_monitor import public_equity_style_config

    kwargs = inputs()
    assets = [f"A{i:03d}" for i in range(60)]
    rng = np.random.default_rng(5)
    frame = pd.concat([kwargs["descriptor_history"].iloc[[0]]] * 60, ignore_index=True)
    frame["asset"] = assets
    frame["market_cap"] = np.arange(1, 61) * 100.0
    config = public_equity_style_config(
        ["a", "b", "c"], winsor_method="quantile", winsor_limit=0.01
    )
    for composite in config.styles.values():
        for descriptor in composite:
            frame[descriptor] = rng.normal(size=60)
    classification = frame[["asset", "effective_at", "available_at", "source"]].copy()
    classification["industry"] = ["a", "b", "c"] * 20
    result = build_exposure_snapshot(
        **{
            **kwargs,
            "descriptor_history": frame,
            "classification_history": classification,
            "assets": assets,
            "config": config,
        }
    )
    assert result.values.shape == (60, 14)
    assert np.linalg.matrix_rank(result.values.to_numpy()) == 13


def test_orthogonalization_cannot_amplify_numerical_dust():
    kwargs = inputs()
    kwargs["descriptor_history"]["value"] = kwargs["descriptor_history"]["size"]
    with pytest.raises(ValueError, match="no independent dispersion"):
        build_exposure_snapshot(**kwargs)
