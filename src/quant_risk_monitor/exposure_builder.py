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
    definition_version: str = "v1"
    derived_styles: dict[str, str] = field(default_factory=dict)
    available_styles: tuple[str, ...] = ()

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
        if self.definition_version not in {"v1", "v2"}:
            raise ValueError("unknown definition_version")
        if not isinstance(self.derived_styles, dict):
            raise TypeError("derived_styles must be a mapping")
        if self.definition_version == "v1" and (self.derived_styles or self.available_styles):
            raise ValueError("derived/available styles require definition_version v2")
        all_styles = set(self.styles) | set(self.derived_styles)
        for name, parent in self.derived_styles.items():
            if (
                not isinstance(name, str)
                or not name.strip()
                or name == "market"
                or name.startswith("industry:")
                or name in self.styles
                or parent not in all_styles
                or name == parent
            ):
                raise ValueError("invalid derived style or parent")
        if len(set(self.available_styles)) != len(self.available_styles) or set(
            self.available_styles
        ) - set(self.styles):
            raise ValueError("available_styles must name distinct descriptor composites")
        available_descriptors = {
            d for s in self.available_styles for d, w in self.styles[s].items() if w
        }
        strict_descriptors = {
            d
            for s, ds in self.styles.items()
            if s not in self.available_styles
            for d, w in ds.items()
            if w
        }
        if available_descriptors & strict_descriptors:
            raise ValueError(
                "descriptor cannot have both strict and available composition policies"
            )
        if not isinstance(self.orthogonalize, dict):
            raise TypeError("orthogonalize must be a mapping")
        for name in all_styles:
            controls = self.orthogonalize.get(name, [])
            if (
                not isinstance(controls, (list, tuple))
                or len(set(controls)) != len(controls)
                or set(controls) - all_styles
                or name in controls
            ):
                raise ValueError("orthogonalization controls must be distinct other styles")
        if set(self.orthogonalize) - all_styles:
            raise ValueError("orthogonalization references an unknown style")
        _style_order(all_styles, self.orthogonalize, self.derived_styles)
        object.__setattr__(self, "styles", deepcopy(self.styles))
        object.__setattr__(self, "industry_universe", industries)
        object.__setattr__(self, "orthogonalize", deepcopy(self.orthogonalize))
        object.__setattr__(self, "derived_styles", deepcopy(self.derived_styles))
        object.__setattr__(self, "available_styles", tuple(self.available_styles))


def _style_order(styles, orthogonalize, derived=None):
    result, visiting = [], set()

    def visit(name):
        if name in visiting:
            raise ValueError("orthogonalization contains a dependency cycle")
        if name in result:
            return
        visiting.add(name)
        dependencies = set(orthogonalize.get(name, []))
        if derived and name in derived:
            dependencies.add(derived[name])
        for dependency in sorted(dependencies):
            visit(dependency)
        visiting.remove(name)
        result.append(name)

    for name in sorted(styles):
        visit(name)
    return result


def public_equity_style_config(
    industry_universe,
    *,
    definition_version="v1",
    enhancements=(),
    available_growth=False,
    **overrides,
):
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
    enhancements = tuple(enhancements)
    if len(set(enhancements)) != len(enhancements) or set(enhancements) - {
        "liquidity",
        "growth",
        "residual_volatility",
    }:
        raise ValueError("unknown or duplicate exposure enhancement")
    if definition_version == "v1" and (enhancements or available_growth):
        raise ValueError("enhancements require v2")
    defaults["definition_version"] = definition_version
    if definition_version == "v2":
        defaults["styles"].pop("nonlinear_size")
        defaults["derived_styles"] = {"nonlinear_size": "size"}
        composites = {
            "liquidity": {f"liquidity_log_turnover_{n}d": 1.0 for n in (21, 63, 252)},
            "growth": {"revenue_growth_3y": 1.0, "profit_growth_3y": 1.0},
            "residual_volatility": {
                "daily_vol_252d": 1.0,
                "cumulative_range_252d": 1.0,
                "residual_vol_252d": 1.0,
            },
        }
        for name in enhancements:
            defaults["styles"][name] = composites[name]
        if available_growth:
            if "growth" not in enhancements:
                raise ValueError("available_growth requires growth enhancement")
            defaults["available_styles"] = ("growth",)
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


def _fill_composite(values, groups, config, label):
    """Fallback only after combining genuinely observed descriptors."""
    values = values.copy()
    observed = values.copy()
    missing = values.isna()
    if missing.mean() > config.max_missing_fraction:
        raise ValueError(f"composite {label} exceeds the missing-coverage limit")
    audit = {}
    for asset in values.index[missing]:
        if config.missing_policy == "error":
            raise ValueError(f"composite {label} is missing for {asset}")
        peers = observed[groups == groups[asset]].dropna()
        fallback = len(peers) < config.min_industry_observations
        if fallback:
            if config.missing_group_fallback == "error":
                raise ValueError(f"composite {label} lacks observed industry peers")
            peers = observed.dropna()
        if len(peers) < config.min_industry_observations:
            raise ValueError(f"composite {label} lacks observed imputation peers")
        values[asset] = float(peers.median())
        audit[asset] = {
            "value": float(values[asset]),
            "global_fallback": fallback,
            "peer_assets": list(peers.index),
        }
    return values.to_numpy(), audit


def _clip_array(array, config):
    if config.winsor_method == "mad":
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)))
        if mad == 0 and np.ptp(array) != 0:
            raise ValueError("derived style has zero MAD")
        bounds = (median - config.winsor_limit * mad, median + config.winsor_limit * mad)
    elif config.winsor_method == "quantile":
        bounds = np.quantile(array, [config.winsor_limit, 1 - config.winsor_limit])
    else:
        bounds = (float(array.min()), float(array.max()))
    return np.clip(array, *bounds), {"lower": float(bounds[0]), "upper": float(bounds[1])}


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
    available_descriptors = {
        d for s in config.available_styles for d, w in config.styles[s].items() if w
    }
    for name in descriptors:
        values = pd.to_numeric(selected[name], errors="raise").astype(float)
        if np.isinf(values).any():
            raise ValueError(f"descriptor {name} contains infinity")
        missing = values.isna()
        available = name in available_descriptors
        if not available and missing.mean() > config.max_missing_fraction:
            raise ValueError(f"descriptor {name} exceeds the missing-coverage limit")
        filled = {}
        observed = values.copy()
        if available:
            mask = ~missing.to_numpy()
            observed_array = values.to_numpy()[mask]
            array = np.full(len(assets), np.nan)
            if (
                len(observed_array) >= config.min_industry_observations
                and np.ptp(observed_array) > 0
            ):
                clipped, bounds = _clip_array(observed_array, config)
                observed_weights = weights[mask] / weights[mask].sum()
                if np.ptp(clipped) > 0:
                    array[mask], normalization = _standardize(clipped, observed_weights, name)
                    descriptor_audit[name] = {
                        "imputed": {},
                        "observed_count": int(mask.sum()),
                        "clipped_assets": [
                            a
                            for a, c in zip(
                                values.index[mask], observed_array != clipped, strict=True
                            )
                            if c
                        ],
                        **bounds,
                        **normalization,
                    }
            if name not in descriptor_audit:
                descriptor_audit[name] = {
                    "imputed": {},
                    "observed_count": int(mask.sum()),
                    "unusable_reason": "insufficient_observations_or_dispersion",
                }
            processed[name] = array
            continue
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
    style_order = _style_order(
        set(config.styles) | set(config.derived_styles), config.orthogonalize, config.derived_styles
    )
    for name in style_order:
        composition_audit = {}
        if name in config.derived_styles:
            composite = values[config.derived_styles[name]].to_numpy() ** 3
            composition_audit = {
                "transform": "standardized-parent-cube",
                "parent": config.derived_styles[name],
            }
        else:
            components = {d: w for d, w in config.styles[name].items() if w}
            denominator = sum(abs(weight) for weight in components.values())
            if name in config.available_styles:
                matrix = np.column_stack([processed[d] for d in components])
                coefficients = np.array(list(components.values()))
                available_weight = np.isfinite(matrix) @ np.abs(coefficients)
                composite = np.divide(
                    np.nan_to_num(matrix) @ coefficients,
                    available_weight,
                    out=np.full(len(assets), np.nan),
                    where=available_weight > 0,
                )
                composite, imputed = _fill_composite(
                    pd.Series(composite, index=assets), groups, config, name
                )
                composition_audit = {
                    "policy": "available",
                    "imputed": imputed,
                    "observed_weight_fraction": dict(
                        zip(assets, (available_weight / denominator).tolist(), strict=True)
                    ),
                    "observed_descriptors": {
                        a: [d for d, valid in zip(components, row, strict=True) if valid]
                        for a, row in zip(assets, np.isfinite(matrix), strict=True)
                    },
                }
            else:
                composite = (
                    sum(weight * processed[d] for d, weight in components.items()) / denominator
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
        if name in config.derived_styles:
            composite, bounds = _clip_array(composite, config)
            composition_audit["winsorization"] = bounds
            # Clipping can reintroduce the parent; remove it again so the final
            # standardized derived exposure remains orthogonal to its controls.
            if controls:
                post = np.linalg.lstsq(weighted, np.sqrt(weights) * composite, rcond=None)[0]
                composite = composite - design @ post
                composition_audit["post_clip_projection"] = post.tolist()
        values[name], normalization = _standardize(composite, weights, name)
        style_audit[name] = {
            **normalization,
            "controls": list(controls),
            "coefficients": None if coefficients is None else coefficients.tolist(),
        }
        if composition_audit:
            style_audit[name]["composition"] = composition_audit
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
    if config.definition_version == "v1":
        # Preserve legacy provenance hashes as well as numerical results.
        for key in ("definition_version", "derived_styles", "available_styles"):
            config_payload.pop(key)
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
        source=f"quant-risk.exposure-builder/{config.definition_version}:" + digest,
        market_caps=caps,
        industries=groups,
        industry_factors=tuple(industry_factors),
        provenance={
            "schema": f"quant-risk.exposure-build/{config.definition_version}",
            "sha256": digest,
            **binding,
            "descriptor_processing": descriptor_audit,
            "style_processing": style_audit,
        },
    )
