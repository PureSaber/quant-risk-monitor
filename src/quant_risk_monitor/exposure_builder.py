"""Versioned, point-in-time construction of fundamental-style risk exposures."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from quant_risk_monitor.factor_model import ExposureSnapshot, _timestamp


@dataclass(frozen=True, kw_only=True)
class ExposureBuildConfig:
    """Explicit descriptor composites; negative weights reverse descriptor sign.

    Each descriptor is independently winsorized and standardized. Composite
    weights are normalized by sum(abs(weights)); composites are standardized
    again, then optionally orthogonalized in a deterministic dependency order.
    """

    styles: dict[str, dict[str, float]]
    industry_universe: tuple[str, ...]
    market_factor: bool = True
    include_industries: bool = True
    standardization_weighting: str = "market_cap"
    winsor_method: str = "mad"
    winsor_limit: float = 5.0
    missing_policy: str = "error"
    max_missing_fraction: float = 0.2
    min_industry_observations: int = 3
    missing_group_fallback: str = "error"
    min_assets: int = 6
    max_descriptor_age_days: float | None = None
    max_classification_age_days: float | None = None
    orthogonalize: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.styles, dict) or not self.styles:
            raise ValueError("styles must explicitly select at least one descriptor composite")
        for style, descriptors in self.styles.items():
            if (
                not isinstance(style, str)
                or not style.strip()
                or style == "market"
                or style.startswith("industry:")
                or not isinstance(descriptors, dict)
                or not descriptors
            ):
                raise ValueError("invalid style composite")
            for name, weight in descriptors.items():
                if (
                    not isinstance(name, str)
                    or not name.strip()
                    or isinstance(weight, bool)
                    or not isinstance(weight, (int, float))
                    or not math.isfinite(weight)
                ):
                    raise ValueError("descriptor weights must be finite")
            if sum(abs(weight) for weight in descriptors.values()) == 0:
                raise ValueError("descriptor composite has zero total weight")
        industries = tuple(self.industry_universe)
        if (
            not industries
            or len(set(industries)) != len(industries)
            or any(not isinstance(x, str) or not x.strip() for x in industries)
        ):
            raise ValueError("industry_universe must be a frozen nonempty set of names")
        for name in ("market_factor", "include_industries"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.standardization_weighting not in {"equal", "market_cap", "sqrt_market_cap"}:
            raise ValueError("unknown standardization_weighting")
        if self.winsor_method not in {"mad", "quantile", "none"}:
            raise ValueError("unknown winsor_method")
        if isinstance(self.winsor_limit, bool) or not math.isfinite(self.winsor_limit):
            raise ValueError("winsor_limit must be finite")
        if (self.winsor_method == "mad" and self.winsor_limit <= 0) or (
            self.winsor_method == "quantile" and not 0 < self.winsor_limit < 0.5
        ):
            raise ValueError("invalid winsor_limit for the chosen method")
        if self.missing_policy not in {"error", "industry_median"}:
            raise ValueError("unknown missing_policy")
        if self.missing_group_fallback not in {"error", "global"}:
            raise ValueError("unknown missing_group_fallback")
        if (
            isinstance(self.max_missing_fraction, bool)
            or not math.isfinite(self.max_missing_fraction)
            or not 0 <= self.max_missing_fraction < 1
        ):
            raise ValueError("max_missing_fraction must lie in [0,1)")
        for name in ("min_industry_observations", "min_assets"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        for name in ("max_descriptor_age_days", "max_classification_age_days"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"invalid {name}")
        if not isinstance(self.orthogonalize, dict):
            raise TypeError("orthogonalize must be a mapping")
        for name in self.styles:
            controls = self.orthogonalize.get(name, [])
            if (
                not isinstance(controls, (list, tuple))
                or len(set(controls)) != len(controls)
                or set(controls) - set(self.styles)
                or name in controls
            ):
                raise ValueError("orthogonalization controls must be distinct other styles")
        if set(self.orthogonalize) - set(self.styles):
            raise ValueError("orthogonalization references an unknown style")
        _style_order(self.styles, self.orthogonalize)
        object.__setattr__(self, "styles", deepcopy(self.styles))
        object.__setattr__(self, "industry_universe", industries)
        object.__setattr__(self, "orthogonalize", deepcopy(self.orthogonalize))


def _style_order(styles, orthogonalize):
    result, visiting = [], set()

    def visit(name):
        if name in visiting:
            raise ValueError("orthogonalization contains a dependency cycle")
        if name in result:
            return
        visiting.add(name)
        for dependency in sorted(orthogonalize.get(name, [])):
            visit(dependency)
        visiting.remove(name)
        result.append(name)

    for name in sorted(styles):
        visit(name)
    return result


def public_equity_style_config(industry_universe, **overrides):
    """Ten public single-descriptor styles matching quant-factors' catalog.

    This explicit research preset is not a vendor style definition. Descriptor
    inputs must already have been causally computed by their source producer.
    """
    descriptors = {
        "size": "size_log_mcap",
        "nonlinear_size": "nonlinear_size",
        "beta": "beta_252d",
        "momentum": "momentum_252_21",
        "residual_volatility": "residual_vol_252d",
        "liquidity": "liquidity_log_turnover_20d",
        "value": "book_to_price",
        "earnings_yield": "earnings_yield_ttm",
        "growth": "earnings_growth_yoy",
        "leverage": "book_leverage",
    }
    defaults = {
        "styles": {style: {descriptor: 1.0} for style, descriptor in descriptors.items()},
        "industry_universe": tuple(industry_universe),
        "min_assets": 30,
        "orthogonalize": {"nonlinear_size": ["size"], "residual_volatility": ["beta", "size"]},
    }
    return ExposureBuildConfig(**{**defaults, **overrides})


def _known_rows(history, assets, cutoff, columns, max_age, name):
    if not isinstance(history, pd.DataFrame):
        raise TypeError(f"{name} history must be a DataFrame")
    required = {"asset", "effective_at", "available_at", "source", *columns}
    if required - set(history):
        raise ValueError(f"{name} history lacks {sorted(required - set(history))}")
    rows = history.copy(deep=True)
    for key in ("effective_at", "available_at"):
        rows[key] = rows[key].map(lambda value, field=key: _timestamp(value, field))
    rows = rows.loc[(rows.effective_at <= cutoff) & (rows.available_at <= cutoff)].copy()
    # Values of unavailable revisions are deliberately not interpreted.
    if rows.asset.isna().any():
        raise ValueError(f"{name} asset identifier is missing")
    rows["asset"] = rows.asset.astype(str)
    rows = rows.loc[rows.asset.isin(assets)]
    if rows.duplicated(["asset", "effective_at", "available_at"]).any():
        raise ValueError(f"{name} has ambiguous duplicate PIT revisions")
    selected = rows.sort_values(["effective_at", "available_at"]).drop_duplicates(
        "asset", keep="last"
    )
    selected = selected.set_index("asset").reindex(assets)
    if selected.effective_at.isna().any():
        raise ValueError(f"{name} has no causally available row for every requested asset")
    if selected.source.isna().any() or selected.source.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} requires an explicit source for every selected row")
    if (
        max_age is not None
        and ((cutoff - selected.effective_at).dt.total_seconds() > max_age * 86400).any()
    ):
        raise ValueError(f"{name} selected row is stale")
    return selected


def _standardize(values, weights, label):
    mean = float(weights @ values)
    variance = float(weights @ (values - mean) ** 2)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError(f"{label} has no cross-sectional dispersion")
    return (values - mean) / np.sqrt(variance), {"mean": mean, "scale": float(np.sqrt(variance))}


def build_exposure_snapshot(*, descriptor_history, classification_history, assets, as_of, config):
    """Select each history independently at as_of and preserve chosen vintages.

    Descriptor history contains full per-asset descriptor snapshots and positive
    market_cap; classification history contains industry. Partial updates must
    first be materialized by the producer using its historical source contract.
    """
    if not isinstance(config, ExposureBuildConfig):
        raise TypeError("config must be ExposureBuildConfig")
    # Revalidate nested mutable configuration before use.
    config = ExposureBuildConfig(**asdict(config))
    assets = list(assets)
    if (
        len(assets) < config.min_assets
        or len(set(assets)) != len(assets)
        or any(not isinstance(a, str) or not a.strip() for a in assets)
    ):
        raise ValueError("assets must be distinct nonempty identifiers with sufficient coverage")
    cutoff = _timestamp(as_of, "as_of")
    descriptors = list(
        dict.fromkeys(d for items in config.styles.values() for d, w in items.items() if w)
    )
    selected = _known_rows(
        descriptor_history,
        assets,
        cutoff,
        ["market_cap", *descriptors],
        config.max_descriptor_age_days,
        "descriptor",
    )
    classification = _known_rows(
        classification_history,
        assets,
        cutoff,
        ["industry"],
        config.max_classification_age_days,
        "classification",
    )
    groups = classification.industry
    if groups.isna().any() or set(groups) - set(config.industry_universe):
        raise ValueError("classification is missing or outside the frozen industry universe")
    caps = pd.to_numeric(selected.market_cap, errors="raise").astype(float)
    if not np.isfinite(caps).all() or (caps <= 0).any():
        raise ValueError("positive finite market caps are required for every asset")
    weights = np.ones(len(assets))
    if config.standardization_weighting != "equal":
        weights = caps.to_numpy().copy()
        if config.standardization_weighting == "sqrt_market_cap":
            weights = np.sqrt(weights)
    weights /= weights.sum()
    processed, descriptor_audit = {}, {}
    for name in descriptors:
        values = pd.to_numeric(selected[name], errors="raise").astype(float)
        if np.isinf(values).any():
            raise ValueError(f"descriptor {name} contains infinity")
        missing = values.isna()
        if missing.mean() > config.max_missing_fraction:
            raise ValueError(f"descriptor {name} exceeds the missing-coverage limit")
        filled = {}
        observed = values.copy()
        for asset in values.index[missing]:
            if config.missing_policy == "error":
                raise ValueError(f"descriptor {name} is missing for {asset}")
            peers = observed.loc[groups == groups[asset]].dropna()
            fallback = False
            if len(peers) < config.min_industry_observations:
                if config.missing_group_fallback == "error":
                    raise ValueError(f"descriptor {name} lacks observed industry peers")
                peers = observed.dropna()
                fallback = True
            if len(peers) < config.min_industry_observations:
                raise ValueError(f"descriptor {name} lacks observed imputation peers")
            values[asset] = float(peers.median())
            filled[asset] = {
                "value": float(values[asset]),
                "global_fallback": fallback,
                "peer_assets": list(peers.index),
            }
        array = values.to_numpy()
        lower, upper = float(array.min()), float(array.max())
        if config.winsor_method == "mad":
            median = float(np.median(array))
            mad = float(np.median(np.abs(array - median)))
            if mad == 0 and lower != upper:
                raise ValueError(
                    f"descriptor {name} has zero MAD; explicitly choose quantile or none"
                )
            lower, upper = median - config.winsor_limit * mad, median + config.winsor_limit * mad
        elif config.winsor_method == "quantile":
            lower, upper = np.quantile(array, [config.winsor_limit, 1 - config.winsor_limit])
        clipped = np.clip(array, lower, upper)
        processed[name], normalization = _standardize(clipped, weights, name)
        descriptor_audit[name] = {
            "imputed": filled,
            "clipped_assets": [
                a for a, changed in zip(assets, array != clipped, strict=True) if changed
            ],
            "lower": float(lower),
            "upper": float(upper),
            **normalization,
        }
    values = pd.DataFrame(index=assets)
    style_audit = {}
    if config.market_factor:
        values["market"] = 1.0
    style_order = _style_order(config.styles, config.orthogonalize)
    for name in style_order:
        components = config.styles[name]
        denominator = sum(abs(weight) for weight in components.values())
        composite = (
            sum(
                weight * processed[descriptor]
                for descriptor, weight in components.items()
                if weight
            )
            / denominator
        )
        controls = config.orthogonalize.get(name, [])
        coefficients = None
        if controls:
            design = np.column_stack([np.ones(len(assets)), values[controls].to_numpy()])
            weighted = np.sqrt(weights[:, None]) * design
            if np.linalg.matrix_rank(weighted) != weighted.shape[1]:
                raise ValueError(f"orthogonalization controls for {name} are rank deficient")
            coefficients = np.linalg.lstsq(weighted, np.sqrt(weights) * composite, rcond=None)[0]
            residual = composite - design @ coefficients
            if np.linalg.norm(residual) <= 1e-12 * np.linalg.norm(composite):
                raise ValueError(f"orthogonalized style {name} has no independent dispersion")
            composite = residual
        values[name], normalization = _standardize(composite, weights, name)
        style_audit[name] = {
            **normalization,
            "controls": list(controls),
            "coefficients": None if coefficients is None else coefficients.tolist(),
        }
    industry_factors = []
    if config.include_industries:
        for industry in config.industry_universe:
            name = "industry:" + industry
            values[name] = groups.eq(industry).to_numpy(dtype=float)
            industry_factors.append(name)

    def records(frame):
        result = frame.reset_index().to_dict("records")
        for row in result:
            for key, value in row.items():
                if isinstance(value, pd.Timestamp):
                    row[key] = value.isoformat()
                elif pd.isna(value):
                    row[key] = None
        return result

    selected_inputs = {
        "descriptors": records(
            selected[["effective_at", "available_at", "source", "market_cap", *descriptors]]
        ),
        "classifications": records(
            classification[["effective_at", "available_at", "source", "industry"]]
        ),
    }
    config_payload = asdict(config)
    binding = {
        "config": config_payload,
        "style_order": style_order,
        "assets": assets,
        "as_of": cutoff.isoformat(),
        "selected_inputs": selected_inputs,
    }
    digest = hashlib.sha256(
        json.dumps(binding, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return ExposureSnapshot(
        effective_at=cutoff,
        available_at=cutoff,
        values=values,
        source="quant-risk.exposure-builder/v1:" + digest,
        market_caps=caps,
        industries=groups,
        industry_factors=tuple(industry_factors),
        provenance={
            "schema": "quant-risk.exposure-build/v1",
            "sha256": digest,
            **binding,
            "descriptor_processing": descriptor_audit,
            "style_processing": style_audit,
        },
    )
