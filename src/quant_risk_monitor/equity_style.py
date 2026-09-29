"""Equity-style factor risk with the estimation steps the minimal model leaves out.

This is an independent implementation of the published linear structure used by
fundamental equity risk models: capitalization-weighted regression, a country
factor, industry factors constrained to a capitalization-weighted sum of zero,
exponentially weighted covariance, a Bartlett autocorrelation correction,
a volatility-regime scale, a simulated eigenfactor bias correction, and
Bayesian specific risk.
Half-lives, lag counts, shrinkage, and descriptor blend weights are caller
configuration. They are not MSCI Barra calibrations, and this module does not
contain vendor descriptors or a licensed factor file.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_risk_monitor.factor_model import (
    AssetReturnObservation,
    BarraStyleRiskModel,
    FactorModelDiagnostics,
    _timestamp,
)

COUNTRY_FACTOR = "country"
DEFAULT_STYLE_ORTHOGONALIZATION: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("nonlinear_size", ("size_log_mcap",)),
    ("mid_capitalization", ("size", "size_log_mcap")),
    ("residual_vol_252d", ("beta_252d", "beta_ew_252d", "size_log_mcap")),
    ("residual_volatility", ("beta", "size", "size_log_mcap")),
    ("long_term_reversal_504_252", ("momentum_252_21", "momentum")),
    ("long_term_reversal", ("momentum", "momentum_252_21")),
)


def _text(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _unique_index(values: pd.Series | pd.DataFrame, name: str) -> list[str]:
    if values.index.has_duplicates:
        raise ValueError(f"{name} contains duplicate assets")
    assets = [_text(asset, f"{name} asset") for asset in values.index]
    if len(set(assets)) != len(assets):
        raise ValueError(f"{name} asset names must be unique and non-empty")
    return assets


@dataclass(frozen=True, kw_only=True)
class StyleExposureSnapshot:
    """Style descriptors at one point in time. Missing descriptor cells may be NaN."""

    effective_at: pd.Timestamp | str
    available_at: pd.Timestamp | str
    values: pd.DataFrame
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "effective_at", _timestamp(self.effective_at, "effective_at"))
        object.__setattr__(self, "available_at", _timestamp(self.available_at, "available_at"))
        object.__setattr__(self, "values", _style_frame(self.values))
        object.__setattr__(self, "source", str(self.source))


@dataclass(frozen=True, kw_only=True)
class IndustrySnapshot:
    """Point-in-time industry label for each asset. Labels are caller-defined."""

    effective_at: pd.Timestamp | str
    available_at: pd.Timestamp | str
    labels: pd.Series
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "effective_at", _timestamp(self.effective_at, "effective_at"))
        object.__setattr__(self, "available_at", _timestamp(self.available_at, "available_at"))
        object.__setattr__(self, "labels", _industry_labels(self.labels))
        object.__setattr__(self, "source", str(self.source))


@dataclass(frozen=True, kw_only=True)
class MarketCapSnapshot:
    """Positive point-in-time capitalization used for weights and constraints."""

    effective_at: pd.Timestamp | str
    available_at: pd.Timestamp | str
    values: pd.Series
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "effective_at", _timestamp(self.effective_at, "effective_at"))
        object.__setattr__(self, "available_at", _timestamp(self.available_at, "available_at"))
        object.__setattr__(self, "values", _positive_caps(self.values))
        object.__setattr__(self, "source", str(self.source))


@dataclass(frozen=True, kw_only=True)
class EquityRiskConfig:
    """Independent estimation settings. None of these numbers is a vendor calibration."""

    model_kind: str
    min_periods: int = 20
    min_assets_per_period: int = 8
    annualization: int = 252
    max_condition_number: float = 1e8
    weight_kind: str = "sqrt_market_cap"
    covariance_half_life: float = 252.0
    volatility_half_life: float = 90.0
    newey_west_lags: int = 5
    eigen_simulations: int = 100
    eigen_scale: float = 1.0
    random_seed: int = 0
    regime_lookback: int = 42
    regime_shrinkage: float = 0.5
    regime_scale_cap: float = 2.0
    specific_half_life: float = 60.0
    specific_prior_strength: float = 20.0
    specific_variance_floor: float = 1e-8
    imputation: str = "industry_median"
    standardize_styles: bool = True
    winsor_z: float = 3.0
    orthogonalize: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_STYLE_ORTHOGONALIZATION

    def __post_init__(self) -> None:
        if self.model_kind not in ("fundamental_style", "statistical_proxy"):
            raise ValueError("model_kind must be 'fundamental_style' or 'statistical_proxy'")
        if self.weight_kind not in ("equal", "sqrt_market_cap"):
            raise ValueError("weight_kind must be 'equal' or 'sqrt_market_cap'")
        if self.imputation not in ("reject", "industry_median"):
            raise ValueError("imputation must be 'reject' or 'industry_median'")
        _positive_int(self.min_periods, "min_periods", minimum=2)
        _positive_int(self.min_assets_per_period, "min_assets_per_period", minimum=3)
        _positive_int(self.annualization, "annualization", minimum=1)
        _positive_int(self.newey_west_lags, "newey_west_lags", minimum=0)
        _positive_int(self.regime_lookback, "regime_lookback", minimum=2)
        _positive_int(self.eigen_simulations, "eigen_simulations", minimum=0)
        _positive_int(self.random_seed, "random_seed", minimum=0)
        scale = float(self.eigen_scale)
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("eigen_scale must be finite and non-negative")
        object.__setattr__(self, "eigen_scale", scale)
        _unit_interval(self.regime_shrinkage, "regime_shrinkage")
        _positive_float(self.covariance_half_life, "covariance_half_life")
        _positive_float(self.volatility_half_life, "volatility_half_life")
        _positive_float(self.specific_half_life, "specific_half_life")
        _positive_float(self.specific_prior_strength, "specific_prior_strength")
        _positive_float(self.specific_variance_floor, "specific_variance_floor")
        _positive_float(self.winsor_z, "winsor_z")
        cap = float(self.regime_scale_cap)
        if not math.isfinite(cap) or cap <= 1:
            raise ValueError("regime_scale_cap must be finite and greater than one")
        object.__setattr__(self, "regime_scale_cap", cap)
        condition = float(self.max_condition_number)
        if not math.isfinite(condition) or condition <= 1:
            raise ValueError("max_condition_number must be finite and greater than one")
        object.__setattr__(self, "max_condition_number", condition)
        object.__setattr__(self, "orthogonalize", _orthogonal_pairs(self.orthogonalize))
        if not isinstance(self.standardize_styles, bool):
            raise TypeError("standardize_styles must be a bool")


@dataclass(frozen=True)
class CrossSectionFit:
    factor_returns: pd.Series
    residuals: pd.Series
    rank: int
    condition_number: float


@dataclass(frozen=True)
class ReturnAttribution:
    """One-period split of realized return into factor contribution and specific return."""

    factor_returns: pd.Series
    factor_contribution: pd.DataFrame
    specific_return: pd.Series

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_returns": {
                "index": [str(item) for item in self.factor_returns.index],
                "data": [float(item) for item in self.factor_returns.to_numpy()],
            },
            "factor_contribution": {
                "index": [str(item) for item in self.factor_contribution.index],
                "columns": [str(item) for item in self.factor_contribution.columns],
                "data": [
                    [float(item) for item in row] for row in self.factor_contribution.to_numpy()
                ],
            },
            "specific_return": {
                "index": [str(item) for item in self.specific_return.index],
                "data": [float(item) for item in self.specific_return.to_numpy()],
            },
        }


def exponential_weights(count: int, half_life: float) -> np.ndarray:
    """Weights that sum to one. The last observation is the most recent."""
    _positive_int(count, "count", minimum=1)
    _positive_float(half_life, "half_life")
    decay = math.exp(-math.log(2.0) / float(half_life))
    ages = np.arange(count, dtype=float)[::-1]
    weights = decay**ages
    return weights / float(weights.sum())


def weighted_newey_west_covariance(
    values: pd.DataFrame, weights: np.ndarray, lags: int
) -> pd.DataFrame:
    """Bartlett HAC covariance. Lag zero is the weighted covariance around the weighted mean."""
    if not isinstance(values, pd.DataFrame) or values.empty or values.shape[1] == 0:
        raise ValueError("values must be a non-empty DataFrame")
    data = values.to_numpy(dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("values must be finite")
    if isinstance(lags, bool) or not isinstance(lags, int) or lags < 0:
        raise ValueError("lags must be a non-negative integer")
    if lags >= len(values):
        raise ValueError("lags must be shorter than the return history")
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(values):
        raise ValueError("weights must match the number of rows")
    if not np.isfinite(weights).all() or (weights < 0).any() or float(weights.sum()) <= 0:
        raise ValueError("weights must be finite, non-negative, and positive in total")
    normalized = weights / float(weights.sum())
    centered = data - normalized @ data
    accumulator = (centered * normalized[:, None]).T @ centered
    for lag in range(1, lags + 1):
        kernel = 1.0 - lag / (lags + 1.0)
        pair = np.sqrt(normalized[lag:] * normalized[:-lag])
        pair_sum = float(pair.sum())
        if pair_sum <= 0:
            continue
        pair = pair / pair_sum
        left = centered[lag:]
        right = centered[:-lag]
        gamma = (left * pair[:, None]).T @ right
        accumulator = accumulator + kernel * (gamma + gamma.T)
    accumulator = (accumulator + accumulator.T) / 2.0
    return pd.DataFrame(accumulator, index=values.columns, columns=values.columns)


def glue_volatility_and_correlation(
    volatility_covariance: pd.DataFrame, correlation_covariance: pd.DataFrame
) -> pd.DataFrame:
    """Use one covariance for volatilities and another for correlations."""
    _same_square_labels(volatility_covariance, correlation_covariance)
    variance = np.clip(np.diag(volatility_covariance.to_numpy(dtype=float)), 0.0, None)
    correlation = _correlation_from_covariance(correlation_covariance.to_numpy(dtype=float))
    volatility = np.sqrt(variance)
    glued = correlation * np.outer(volatility, volatility)
    return _frame_from_square(glued, volatility_covariance.index)


def project_to_psd(values: pd.DataFrame) -> pd.DataFrame:
    matrix = (values.to_numpy(dtype=float) + values.to_numpy(dtype=float).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    repaired = eigenvectors @ np.diag(np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
    return _frame_from_square(repaired, values.index)


def eigenfactor_risk_adjustment(
    values: pd.DataFrame,
    periods: int,
    *,
    simulations: int,
    scale: float,
    seed: int,
) -> pd.DataFrame:
    """Correct the sample bias of each eigenfactor variance by simulation.

    Each draw simulates ``periods`` returns from the matrix, re-estimates it,
    and compares the true variance of the re-estimated eigenvectors with their
    estimated variance. The average ratio, moved away from one by ``scale``,
    multiplies the eigenvalues of the same rank. Zero simulations return the
    matrix unchanged.
    """
    _positive_int(simulations, "simulations", minimum=0)
    _positive_int(periods, "periods", minimum=2)
    _positive_int(seed, "seed", minimum=0)
    scale = float(scale)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and non-negative")
    matrix = (values.to_numpy(dtype=float) + values.to_numpy(dtype=float).T) / 2.0
    if simulations == 0:
        return _frame_from_square(matrix, values.index)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    tolerance = max(float(eigenvalues.max()) * 1e-12, 1e-18)
    rng = np.random.default_rng(seed)
    ratios = np.zeros(len(eigenvalues))
    for _ in range(simulations):
        draws = rng.standard_normal((periods, len(eigenvalues))) * np.sqrt(eigenvalues)
        simulated = np.atleast_2d(np.cov(draws @ eigenvectors.T, rowvar=False))
        estimated, estimated_vectors = np.linalg.eigh(simulated)
        true = np.einsum("ik,ij,jk->k", estimated_vectors, matrix, estimated_vectors)
        ratios += np.divide(
            true,
            estimated,
            out=np.ones_like(true),
            where=(estimated > tolerance) & (true > tolerance),
        )
    bias = np.sqrt(ratios / simulations)
    multiplier = np.clip(scale * (bias - 1.0) + 1.0, 0.0, None)
    adjusted = eigenvectors @ np.diag(multiplier**2 * eigenvalues) @ eigenvectors.T
    return _frame_from_square(adjusted, values.index)


def apply_volatility_regime(
    factor_covariance: pd.DataFrame,
    factor_returns: pd.DataFrame,
    *,
    lookback: int,
    shrinkage: float,
    scale_cap: float,
    annualization: int,
) -> pd.DataFrame:
    """Scale factor variance toward the recent realized-to-model ratio, shrunk toward one."""
    _unit_interval(shrinkage, "shrinkage")
    if shrinkage == 0:
        return factor_covariance.copy(deep=True)
    _positive_int(lookback, "lookback", minimum=2)
    _positive_int(annualization, "annualization", minimum=1)
    cap = float(scale_cap)
    if not math.isfinite(cap) or cap <= 1:
        raise ValueError("scale_cap must be finite and greater than one")
    if list(factor_returns.columns) != list(factor_covariance.columns):
        raise ValueError("factor return columns must match the covariance")
    window = factor_returns.iloc[-lookback:]
    if len(window) < 2:
        return factor_covariance.copy(deep=True)
    realized = window.var(ddof=1).to_numpy(dtype=float) * annualization
    model = np.diag(factor_covariance.to_numpy(dtype=float))
    ratio = np.ones(len(model), dtype=float)
    positive = model > 0
    ratio[positive] = realized[positive] / model[positive]
    raw = np.sqrt(np.clip(ratio, 0.0, None))
    raw = np.clip(raw, 1.0 / cap, cap)
    scale = (1.0 - shrinkage) + shrinkage * raw
    scaled = factor_covariance.to_numpy(dtype=float) * np.outer(scale, scale)
    return _frame_from_square(scaled, factor_covariance.index)


def ewma_variance(values: Sequence[float], half_life: float) -> float:
    data = np.asarray(list(values), dtype=float)
    if len(data) < 2 or not np.isfinite(data).all():
        return float("nan")
    weights = exponential_weights(len(data), half_life)
    centered = data - float(weights @ data)
    return float(weights @ centered**2)


def structural_specific_variances(
    time_series: pd.Series,
    counts: pd.Series,
    styles: pd.DataFrame,
    floor: float,
) -> pd.Series:
    """Log-linear specific-variance prediction. A thin sample falls back to the median."""
    _positive_float(floor, "floor")
    if not time_series.index.equals(counts.index) or not time_series.index.equals(styles.index):
        raise ValueError("specific-risk inputs must share one asset index")
    observed = time_series.loc[counts >= 2].astype(float)
    observed = observed[np.isfinite(observed.to_numpy()) & (observed.to_numpy() > 0)]
    if observed.empty:
        raise ValueError("no asset has enough residual history to anchor specific risk")
    fallback = float(observed.median())
    eligible = observed.index
    design = styles.loc[eligible]
    if len(eligible) <= design.shape[1] + 1 or not np.isfinite(design.to_numpy(dtype=float)).all():
        return pd.Series(fallback, index=styles.index, dtype=float)
    predictors = np.column_stack([np.ones(len(eligible)), design.to_numpy(dtype=float)])
    if int(np.linalg.matrix_rank(predictors)) < predictors.shape[1]:
        return pd.Series(fallback, index=styles.index, dtype=float)
    target = np.log(np.maximum(observed.to_numpy(dtype=float), floor))
    coefficients, _, solved_rank, _ = np.linalg.lstsq(predictors, target, rcond=None)
    if int(solved_rank) < predictors.shape[1]:
        return pd.Series(fallback, index=styles.index, dtype=float)
    full = np.column_stack([np.ones(len(styles)), styles.to_numpy(dtype=float)])
    predicted = np.exp(full @ coefficients)
    lower = float(observed.min()) * 0.25
    upper = float(observed.max()) * 4.0
    return pd.Series(np.clip(predicted, lower, upper), index=styles.index, dtype=float)


def blend_specific_variance(
    time_series_variance: float,
    observations: int,
    structural_variance: float,
    prior_strength: float,
) -> float:
    """Give a short history less weight. A name with fewer than two residuals uses the prior."""
    _positive_float(prior_strength, "prior_strength")
    if isinstance(observations, bool) or not isinstance(observations, int) or observations < 0:
        raise ValueError("observations must be a non-negative integer")
    if not math.isfinite(structural_variance) or structural_variance < 0:
        raise ValueError("structural_variance must be finite and non-negative")
    if observations < 2 or not math.isfinite(time_series_variance):
        return float(structural_variance)
    weight = observations / (observations + prior_strength)
    return float(weight * time_series_variance + (1.0 - weight) * structural_variance)


def impute_style_exposures(
    styles: pd.DataFrame, industries: pd.Series, *, method: str
) -> pd.DataFrame:
    """Fill missing style cells with the industry median, or reject them. Never fill with zero."""
    if method not in ("reject", "industry_median"):
        raise ValueError("imputation must be 'reject' or 'industry_median'")
    if not isinstance(styles, pd.DataFrame) or styles.empty or styles.shape[1] == 0:
        raise ValueError("styles must be a non-empty DataFrame")
    if not styles.index.equals(industries.index):
        raise ValueError("industry labels must match the style index")
    numeric = styles.apply(pd.to_numeric, errors="coerce").astype(float)
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("style exposures must not contain infinity")
    if method == "reject":
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("style exposures contain missing values")
        return numeric
    labels = industries.astype(str)
    filled = numeric.copy(deep=True)
    for column in filled.columns:
        for members in filled.groupby(labels, sort=False).groups.values():
            member_index = pd.Index(members)
            column_values = filled.loc[member_index, column]
            observed = column_values.dropna()
            if observed.empty:
                continue
            missing = column_values.index[column_values.isna()]
            if len(missing):
                filled.loc[missing, column] = float(observed.median())
        still_missing = filled[column].isna()
        if not still_missing.any():
            continue
        universe = filled[column].dropna()
        if universe.empty:
            raise ValueError(f"style column {column} has no observed values to impute from")
        filled.loc[still_missing, column] = float(universe.median())
    if not np.isfinite(filled.to_numpy(dtype=float)).all():
        raise ValueError("style imputation left non-finite values")
    return filled


def standardize_style_cross_section(
    styles: pd.DataFrame,
    market_cap: pd.Series,
    *,
    winsor_z: float = 3.0,
    orthogonalize: Sequence[tuple[str, Sequence[str]]] = (),
) -> pd.DataFrame:
    """Winsorize, capitalization-weighted standardize, then residualize configured pairs."""
    _positive_float(winsor_z, "winsor_z")
    if not isinstance(styles, pd.DataFrame) or styles.empty or styles.shape[1] == 0:
        raise ValueError("styles must be a non-empty DataFrame")
    if not styles.index.equals(market_cap.index):
        raise ValueError("market cap index must match style exposures")
    caps = market_cap.astype(float)
    if not np.isfinite(caps.to_numpy()).all() or (caps <= 0).any():
        raise ValueError("market cap must be finite and positive")
    values = styles.apply(pd.to_numeric, errors="coerce").astype(float)
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("style standardization requires finite exposures")
    standardized = pd.DataFrame(index=values.index)
    for column in values.columns:
        standardized[column] = _standardize_column(
            values[column].to_numpy(dtype=float), caps, winsor_z
        )
    for target, controls in orthogonalize:
        if target not in standardized.columns:
            continue
        present = [name for name in controls if name in standardized.columns and name != target]
        if not present:
            continue
        residual = _weighted_residual(
            standardized[target].to_numpy(dtype=float),
            standardized[present].to_numpy(dtype=float),
            caps.to_numpy(dtype=float),
        )
        standardized[target] = _standardize_column(residual, caps, winsor_z)
    return standardized


def blend_available_descriptors(
    components: Mapping[str, pd.Series], weights: Mapping[str, float]
) -> pd.Series:
    """Weighted mean of the finite components. Each asset renormalizes over the components it has.

    Standardize the components first. Blending raw ratios mixes their units.
    """
    if not components:
        raise ValueError("at least one descriptor component is required")
    if set(weights) != set(components):
        raise ValueError("blend weights must name exactly the descriptor components")
    clean_weights = {name: float(value) for name, value in weights.items()}
    if any(not math.isfinite(value) or value < 0 for value in clean_weights.values()):
        raise ValueError("blend weights must be finite and non-negative")
    if all(value == 0 for value in clean_weights.values()):
        raise ValueError("at least one blend weight must be positive")
    frames = []
    for name, series in components.items():
        if not isinstance(series, pd.Series) or series.empty:
            raise ValueError(f"{name} must be a non-empty Series")
        numeric = pd.to_numeric(series, errors="coerce").astype(float)
        numeric.name = name
        frames.append(numeric)
    aligned = pd.concat(frames, axis=1, join="outer")
    blended = pd.Series(np.nan, index=aligned.index, dtype=float)
    weight_row = np.array([clean_weights[name] for name in aligned.columns], dtype=float)
    data = aligned.to_numpy(dtype=float)
    for position, row in enumerate(data):
        usable = np.isfinite(row) & (weight_row > 0)
        if not usable.any():
            continue
        chosen = weight_row[usable]
        blended.iloc[position] = float(np.dot(chosen, row[usable]) / chosen.sum())
    return blended


def estimate_constrained_factor_returns(
    asset_returns: pd.Series,
    industries: pd.Series,
    market_cap: pd.Series,
    styles: pd.DataFrame | None,
    *,
    weight_kind: str,
    max_condition_number: float = 1e8,
    period_label: str = "cross section",
) -> CrossSectionFit:
    """WLS factor returns with industry factors constrained to a cap-weighted sum of zero."""
    if weight_kind not in ("equal", "sqrt_market_cap"):
        raise ValueError("weight_kind must be 'equal' or 'sqrt_market_cap'")
    returns = _aligned_series(asset_returns, "asset returns")
    labels = _aligned_series(industries.astype(str), "industries")
    caps = _aligned_series(market_cap, "market cap")
    if not returns.index.equals(labels.index) or not returns.index.equals(caps.index):
        raise ValueError("returns, industries, and market cap must share one asset index")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("asset returns must be finite")
    if not np.isfinite(caps.to_numpy()).all() or (caps <= 0).any():
        raise ValueError("market cap must be finite and positive")
    style_frame = _optional_styles(styles, returns.index)
    industry_names = tuple(sorted(set(labels.tolist())))
    if not industry_names:
        raise ValueError("at least one industry is required")
    _reject_name_collisions(industry_names, style_frame.columns)
    cap_values = caps.to_numpy(dtype=float)
    label_values = labels.to_numpy()
    industry_cap = {name: float(cap_values[label_values == name].sum()) for name in industry_names}
    reference = min(industry_names, key=lambda name: (-industry_cap[name], name))
    free = [name for name in industry_names if name != reference]
    design_columns = [np.ones(len(returns))]
    design_names = [COUNTRY_FACTOR]
    for name in free:
        own = (label_values == name).astype(float)
        reference_dummy = (label_values == reference).astype(float)
        scale = industry_cap[name] / industry_cap[reference]
        design_columns.append(own - reference_dummy * scale)
        design_names.append(name)
    for column in style_frame.columns:
        design_columns.append(style_frame[column].to_numpy(dtype=float))
        design_names.append(str(column))
    design = np.column_stack(design_columns)
    if len(returns) <= design.shape[1]:
        raise ValueError(
            f"{period_label} has {len(returns)} assets and {design.shape[1]} "
            "reduced factors; more assets than factors are required"
        )
    regression_weights = np.ones(len(returns)) if weight_kind == "equal" else np.sqrt(cap_values)
    scale = np.sqrt(regression_weights)
    weighted_design = design * scale[:, None]
    weighted_target = returns.to_numpy(dtype=float) * scale
    rank = int(np.linalg.matrix_rank(weighted_design))
    condition = float(np.linalg.cond(weighted_design))
    if rank != design.shape[1]:
        raise ValueError(f"{period_label} exposure design is rank deficient: rank={rank}")
    if not math.isfinite(condition) or condition > max_condition_number:
        raise ValueError(f"{period_label} exposure design is ill-conditioned: {condition}")
    coefficients, _, solved_rank, _ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
    if int(solved_rank) != design.shape[1]:
        raise ValueError(f"{period_label} factor regression did not have full rank")
    solved = {name: float(value) for name, value in zip(design_names, coefficients, strict=True)}
    reference_value = 0.0
    for name in free:
        reference_value -= (industry_cap[name] / industry_cap[reference]) * solved[name]
    solved[reference] = reference_value
    ordered = [COUNTRY_FACTOR, *industry_names, *[str(column) for column in style_frame.columns]]
    factor_returns = pd.Series([solved[name] for name in ordered], index=ordered, dtype=float)
    predicted = _predict(labels, style_frame, factor_returns)
    residuals = returns - predicted
    return CrossSectionFit(
        factor_returns=factor_returns,
        residuals=residuals,
        rank=rank,
        condition_number=condition,
    )


def attribute_realized_return(
    model: BarraStyleRiskModel,
    asset_returns: pd.Series,
    market_cap: pd.Series | None = None,
) -> ReturnAttribution:
    """Split one realized cross-section with the same regression the estimator stored."""
    returns = _aligned_series(asset_returns, "asset returns")
    exposures = model.exposures
    if not returns.index.equals(exposures.index):
        raise ValueError("asset returns must cover exactly the model assets, in the same order")
    estimation = model.diagnostics.estimation
    if estimation.get("estimator") == "equity_style":
        if market_cap is None:
            raise ValueError("equity-style attribution requires point-in-time market cap")
        industries = pd.Series("", index=exposures.index, dtype=object)
        for name in estimation["industry_factors"]:
            industries = industries.mask(exposures[name] == 1.0, name)
        if (industries == "").any():
            raise ValueError("every asset must belong to one stored industry factor")
        styles = exposures.drop(columns=[COUNTRY_FACTOR, *estimation["industry_factors"]])
        fitted = estimate_constrained_factor_returns(
            returns,
            industries.astype(str),
            market_cap.reindex(returns.index),
            styles,
            weight_kind=str(estimation["weight_kind"]),
        )
        factor_returns = fitted.factor_returns.reindex(exposures.columns)
    else:
        design = exposures.to_numpy(dtype=float)
        coefficients, _, rank, _ = np.linalg.lstsq(
            design, returns.to_numpy(dtype=float), rcond=None
        )
        if int(rank) != design.shape[1]:
            raise ValueError("attribution regression is rank deficient")
        factor_returns = pd.Series(coefficients, index=exposures.columns, dtype=float)
    contribution = exposures.mul(factor_returns, axis=1)
    specific = returns - contribution.sum(axis=1)
    return ReturnAttribution(
        factor_returns=factor_returns,
        factor_contribution=contribution,
        specific_return=specific,
    )


def fit_equity_style_risk_model(
    *,
    style_snapshots: Sequence[StyleExposureSnapshot],
    industry_snapshots: Sequence[IndustrySnapshot],
    market_cap_snapshots: Sequence[MarketCapSnapshot],
    return_observations: Sequence[AssetReturnObservation],
    as_of: pd.Timestamp | str,
    config: EquityRiskConfig,
) -> BarraStyleRiskModel:
    """Estimate country, industry, and style risk from point-in-time inputs."""
    if not isinstance(config, EquityRiskConfig):
        raise TypeError("config must be EquityRiskConfig")
    cutoff = _timestamp(as_of, "as_of")
    styles = _typed_snapshots(style_snapshots, StyleExposureSnapshot, "style")
    industries = _typed_snapshots(industry_snapshots, IndustrySnapshot, "industry")
    caps = _typed_snapshots(market_cap_snapshots, MarketCapSnapshot, "market cap")
    observations = _prepared_returns(return_observations, cutoff, config.min_periods)
    _reject_late_snapshots(styles, cutoff, "style")
    _reject_late_snapshots(industries, cutoff, "industry")
    _reject_late_snapshots(caps, cutoff, "market cap")
    _require_same_style_columns(styles)
    current_style = _select_latest(styles, cutoff, "style snapshot")
    current_industry = _select_latest(industries, cutoff, "industry snapshot")
    current_cap = _select_latest(caps, cutoff, "market cap snapshot")
    coverage = _coverage_assets(current_style.values, current_industry.labels, current_cap.values)
    current_exposures, current_industries, _ = _process_styles(
        current_style.values.reindex(coverage),
        current_industry.labels.reindex(coverage),
        current_cap.values.reindex(coverage),
        config,
    )
    industry_names = tuple(sorted(set(current_industries.tolist())))
    factor_rows: list[np.ndarray] = []
    factor_index: list[pd.Timestamp] = []
    residuals: dict[str, list[float]] = {asset: [] for asset in coverage}
    sizes: list[int] = []
    ranks: list[int] = []
    conditions: list[float] = []
    skipped = 0
    eligible_starts: list[pd.Timestamp] = []
    for observation in observations:
        result = _estimate_period(
            observation,
            styles,
            industries,
            caps,
            industry_names,
            config,
        )
        if result is None:
            skipped += 1
            continue
        factor_returns, residual, size, rank, condition, _used_assets = result
        row = factor_returns.reindex(_factor_names(industry_names, current_style)).to_numpy(
            dtype=float
        )
        if not np.isfinite(row).all():
            raise ValueError(
                f"period ending {observation.period_end.isoformat()} did not estimate every current factor"
            )
        factor_rows.append(row)
        factor_index.append(observation.period_end)
        for asset in coverage:
            if asset in residual.index:
                residuals[asset].append(float(residual.at[asset]))
        sizes.append(size)
        ranks.append(rank)
        conditions.append(condition)
        eligible_starts.append(observation.period_start)
    if len(factor_rows) < config.min_periods:
        raise ValueError(
            f"at least {config.min_periods} return periods covering every current industry "
            f"are required; received {len(factor_rows)} and skipped {skipped}"
        )
    if config.newey_west_lags >= len(factor_rows):
        raise ValueError("newey_west_lags must be shorter than the eligible return history")
    names = _factor_names(industry_names, current_style)
    history = pd.DataFrame(factor_rows, index=pd.DatetimeIndex(factor_index), columns=names)
    covariance = _factor_covariance(history, config)
    style_block = current_exposures[list(current_style.values.columns)]
    counts = pd.Series({asset: len(residuals[asset]) for asset in coverage}, dtype=int)
    raw_specific = pd.Series(
        {
            asset: ewma_variance(residuals[asset], config.specific_half_life) * config.annualization
            for asset in coverage
        },
        dtype=float,
    )
    structural = structural_specific_variances(
        raw_specific, counts, style_block, config.specific_variance_floor
    )
    specific = pd.Series(
        {
            asset: max(
                blend_specific_variance(
                    float(raw_specific.at[asset]),
                    int(counts.at[asset]),
                    float(structural.at[asset]),
                    config.specific_prior_strength,
                ),
                config.specific_variance_floor,
            )
            for asset in coverage
        },
        dtype=float,
    )
    specific = specific.reindex(coverage)
    exposures = _model_exposures(current_industries, style_block, names)
    asset_covariance = _asset_covariance(exposures, covariance, specific)
    estimation = {
        "estimator": "equity_style",
        "vendor_model": False,
        "parameters_are_vendor_calibration": False,
        "weight_kind": config.weight_kind,
        "country_factor": COUNTRY_FACTOR,
        "industry_factors": list(industry_names),
        "style_factors": [str(column) for column in current_style.values.columns],
        "covariance_half_life": config.covariance_half_life,
        "volatility_half_life": config.volatility_half_life,
        "newey_west_lags": config.newey_west_lags,
        "eigen_simulations": config.eigen_simulations,
        "eigen_scale": config.eigen_scale,
        "random_seed": config.random_seed,
        "regime_lookback": config.regime_lookback,
        "regime_shrinkage": config.regime_shrinkage,
        "specific_half_life": config.specific_half_life,
        "specific_prior_strength": config.specific_prior_strength,
        "imputation": config.imputation,
        "standardize_styles": config.standardize_styles,
        "winsor_z": config.winsor_z,
        "skipped_periods": skipped,
        "eligible_periods": len(factor_rows),
        "style_source": current_style.source,
        "industry_source": current_industry.source,
        "market_cap_source": current_cap.source,
    }
    diagnostics = FactorModelDiagnostics(
        model_kind=config.model_kind,
        as_of=cutoff,
        factors=tuple(names),
        assets=tuple(str(asset) for asset in coverage),
        periods=len(factor_rows),
        first_period_start=eligible_starts[0],
        last_period_end=factor_index[-1],
        exposure_effective_at=current_style.effective_at,
        exposure_available_at=current_style.available_at,
        cross_section_sizes=tuple(sizes),
        regression_ranks=tuple(ranks),
        condition_numbers=tuple(conditions),
        residual_observations={str(asset): int(counts.at[asset]) for asset in coverage},
        annualization=config.annualization,
        covariance_shrinkage=0.0,
        specific_variance_shrinkage=0.0,
        estimator="equity_style",
        estimation=estimation,
    )
    return BarraStyleRiskModel(
        model_kind=config.model_kind,
        as_of=cutoff,
        annualization=config.annualization,
        exposures=exposures,
        factor_returns=history,
        factor_covariance=covariance,
        specific_variances=specific,
        asset_covariance=asset_covariance,
        diagnostics=diagnostics,
    )


def _style_frame(values: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(values, pd.DataFrame) or values.empty or values.shape[1] == 0:
        raise ValueError("style values must be a non-empty asset-by-factor DataFrame")
    assets = _unique_index(values, "style values")
    factors = [_text(factor, "style factor") for factor in values.columns]
    if len(set(factors)) != len(factors):
        raise ValueError("style factor names must be unique")
    if COUNTRY_FACTOR in factors:
        raise ValueError("style factors must not use the reserved country name")
    numeric = values.apply(pd.to_numeric, errors="coerce").astype(float)
    numeric.index = assets
    numeric.columns = factors
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("style values must not contain infinity")
    return numeric


def _industry_labels(values: pd.Series) -> pd.Series:
    if not isinstance(values, pd.Series) or values.empty:
        raise ValueError("industry labels must be a non-empty Series")
    assets = _unique_index(values, "industry labels")
    if values.isna().any():
        raise ValueError("industry labels must not be missing")
    labels = pd.Series([_text(label, "industry") for label in values.tolist()], index=assets)
    if (labels == COUNTRY_FACTOR).any():
        raise ValueError("industry labels must not use the reserved country name")
    return labels


def _positive_caps(values: pd.Series) -> pd.Series:
    if not isinstance(values, pd.Series) or values.empty:
        raise ValueError("market cap must be a non-empty Series")
    assets = _unique_index(values, "market cap")
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    numeric.index = assets
    if not np.isfinite(numeric.to_numpy()).all() or (numeric <= 0).any():
        raise ValueError("market cap must be finite and positive")
    return numeric


def _positive_int(value: object, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _positive_float(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _unit_interval(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _orthogonal_pairs(
    pairs: Sequence[tuple[str, Sequence[str]]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    resolved: list[tuple[str, tuple[str, ...]]] = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("each orthogonalization pair must contain a target and controls")
        target = _text(pair[0], "orthogonalization target")
        controls = tuple(_text(name, "orthogonalization control") for name in pair[1])
        if not controls or target in controls:
            raise ValueError(
                "orthogonalization controls must be non-empty and distinct from the target"
            )
        resolved.append((target, controls))
    return tuple(resolved)


def _same_square_labels(left: pd.DataFrame, right: pd.DataFrame) -> None:
    if list(left.index) != list(left.columns) or list(left.index) != list(right.index):
        raise ValueError("covariance frames must be square and aligned")
    if list(right.index) != list(right.columns):
        raise ValueError("covariance frames must be square and aligned")


def _frame_from_square(matrix: np.ndarray, labels: pd.Index) -> pd.DataFrame:
    repaired = (matrix + matrix.T) / 2.0
    return pd.DataFrame(repaired, index=labels, columns=labels)


def _correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denominator = np.outer(scale, scale)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.ones_like(covariance),
        where=denominator > 0,
    )
    np.fill_diagonal(correlation, 1.0)
    return np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)


def _standardize_column(values: np.ndarray, market_cap: pd.Series, winsor_z: float) -> pd.Series:
    center = float(np.mean(values))
    scale = float(np.std(values))
    if scale > 0:
        values = np.clip(values, center - winsor_z * scale, center + winsor_z * scale)
    weights = market_cap.to_numpy(dtype=float)
    weights = weights / float(weights.sum())
    demeaned = values - float(weights @ values)
    dispersion = float(np.std(demeaned))
    if dispersion == 0:
        scaled = np.zeros(len(values))
    else:
        scaled = demeaned / dispersion
    return pd.Series(scaled, index=market_cap.index, dtype=float)


def _weighted_residual(
    target: np.ndarray, controls: np.ndarray, market_cap: np.ndarray
) -> np.ndarray:
    design = np.column_stack([np.ones(len(target)), controls])
    scale = np.sqrt(market_cap)
    coefficients, _, _, _ = np.linalg.lstsq(design * scale[:, None], target * scale, rcond=None)
    return target - design @ coefficients


def _aligned_series(values: pd.Series, name: str) -> pd.Series:
    if not isinstance(values, pd.Series) or values.empty:
        raise ValueError(f"{name} must be a non-empty Series")
    assets = _unique_index(values, name)
    if name == "industries":
        return pd.Series(values.astype(str).tolist(), index=assets)
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    numeric.index = assets
    return numeric


def _optional_styles(styles: pd.DataFrame | None, index: pd.Index) -> pd.DataFrame:
    if styles is None:
        return pd.DataFrame(index=index)
    if not isinstance(styles, pd.DataFrame):
        raise TypeError("styles must be a DataFrame or None")
    if not styles.index.equals(index):
        raise ValueError("style index must match the return assets")
    if styles.shape[1] == 0:
        return pd.DataFrame(index=index)
    numeric = styles.apply(pd.to_numeric, errors="coerce").astype(float)
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("constrained regression requires finite style exposures")
    return numeric


def _reject_name_collisions(industries: Sequence[str], styles: pd.Index) -> None:
    style_names = {str(name) for name in styles}
    for name in industries:
        if name == COUNTRY_FACTOR or name in style_names:
            raise ValueError(f"industry {name} collides with the country factor or a style factor")


def _predict(industries: pd.Series, styles: pd.DataFrame, factor_returns: pd.Series) -> pd.Series:
    predicted = pd.Series(float(factor_returns.at[COUNTRY_FACTOR]), index=industries.index)
    for asset, industry in industries.items():
        predicted.at[asset] += float(factor_returns.at[str(industry)])
    for column in styles.columns:
        predicted = predicted + styles[column] * float(factor_returns.at[str(column)])
    return predicted


def _typed_snapshots(values: Sequence[Any], kind: type, name: str) -> list[Any]:
    items = list(values)
    if not items:
        raise ValueError(f"at least one {name} snapshot is required")
    if any(not isinstance(item, kind) for item in items):
        raise ValueError(f"{name} snapshots have the wrong type")
    keys = [(item.effective_at, item.available_at) for item in items]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} snapshots contain duplicate effective/available timestamps")
    return items


def _prepared_returns(
    values: Sequence[AssetReturnObservation], cutoff: pd.Timestamp, min_periods: int
) -> list[AssetReturnObservation]:
    observations = list(values)
    if not observations:
        raise ValueError("at least one return observation is required")
    if any(not isinstance(item, AssetReturnObservation) for item in observations):
        raise ValueError("return_observations must contain AssetReturnObservation values")
    observations.sort(key=lambda item: (item.period_start, item.period_end))
    keys = [(item.period_start, item.period_end) for item in observations]
    if len(set(keys)) != len(keys):
        raise ValueError("return observations contain duplicate periods")
    for position, observation in enumerate(observations):
        if observation.period_end > cutoff or observation.available_at > cutoff:
            raise ValueError("return observations must be complete and available by as_of")
        if position and observation.period_start < observations[position - 1].period_end:
            raise ValueError("return observation periods must not overlap")
    if len(observations) < min_periods:
        raise ValueError(
            f"at least {min_periods} matured return periods are required; "
            f"received {len(observations)}"
        )
    return observations


def _reject_late_snapshots(items: Sequence[Any], cutoff: pd.Timestamp, name: str) -> None:
    for item in items:
        if item.effective_at > cutoff or item.available_at > cutoff:
            raise ValueError(f"{name} snapshots must not be effective or available after as_of")


def _require_same_style_columns(items: Sequence[StyleExposureSnapshot]) -> None:
    columns = tuple(items[0].values.columns)
    for item in items[1:]:
        if tuple(item.values.columns) != columns:
            raise ValueError("every style snapshot must contain the same factors in the same order")


def _select_latest(items: Sequence[Any], cutoff: pd.Timestamp, name: str) -> Any:
    eligible = [
        item for item in items if item.effective_at <= cutoff and item.available_at <= cutoff
    ]
    if not eligible:
        raise ValueError(f"no {name} was effective and available by {cutoff.isoformat()}")
    return max(eligible, key=lambda item: (item.effective_at, item.available_at))


def _coverage_assets(
    styles: pd.DataFrame, industries: pd.Series, market_cap: pd.Series
) -> pd.Index:
    assets = styles.index.intersection(industries.index).intersection(market_cap.index)
    if len(assets) == 0:
        raise ValueError("style, industry, and market cap snapshots have no asset in common")
    caps = market_cap.reindex(assets)
    labels = industries.reindex(assets)
    keep = caps.notna() & (caps > 0) & labels.notna() & labels.astype(str).str.len().gt(0)
    covered = assets[keep.to_numpy()]
    if len(covered) == 0:
        raise ValueError("no asset has both an industry and a positive market cap")
    return covered


def _process_styles(
    styles: pd.DataFrame,
    industries: pd.Series,
    market_cap: pd.Series,
    config: EquityRiskConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    imputed = impute_style_exposures(styles, industries, method=config.imputation)
    if config.standardize_styles:
        processed = standardize_style_cross_section(
            imputed,
            market_cap,
            winsor_z=config.winsor_z,
            orthogonalize=config.orthogonalize,
        )
    else:
        processed = imputed
    return processed, industries, market_cap


def _estimate_period(
    observation: AssetReturnObservation,
    styles: Sequence[StyleExposureSnapshot],
    industries: Sequence[IndustrySnapshot],
    caps: Sequence[MarketCapSnapshot],
    required_industries: Sequence[str],
    config: EquityRiskConfig,
) -> tuple[pd.Series, pd.Series, int, int, float, pd.Index] | None:
    style = _select_latest(styles, observation.period_start, "style snapshot")
    industry = _select_latest(industries, observation.period_start, "industry snapshot")
    cap = _select_latest(caps, observation.period_start, "market cap snapshot")
    assets = _coverage_assets(style.values, industry.labels, cap.values)
    assets = assets.intersection(observation.values.index)
    labels = industry.labels.reindex(assets)
    current = set(required_industries)
    labels = labels.loc[labels.isin(current)]
    assets = labels.index
    if set(labels.astype(str).tolist()) != current:
        return None
    if len(assets) < config.min_assets_per_period:
        raise ValueError(
            f"return period ending {observation.period_end.isoformat()} has {len(assets)} assets; "
            f"at least {config.min_assets_per_period} are required"
        )
    processed, used_labels, used_caps = _process_styles(
        style.values.reindex(assets),
        labels.astype(str),
        cap.values.reindex(assets),
        config,
    )
    fitted = estimate_constrained_factor_returns(
        observation.values.reindex(assets),
        used_labels,
        used_caps,
        processed,
        weight_kind=config.weight_kind,
        max_condition_number=config.max_condition_number,
        period_label=f"period ending {observation.period_end.isoformat()}",
    )
    return (
        fitted.factor_returns,
        fitted.residuals,
        len(assets),
        fitted.rank,
        fitted.condition_number,
        assets,
    )


def _factor_names(industries: Sequence[str], style: StyleExposureSnapshot) -> list[str]:
    return [COUNTRY_FACTOR, *industries, *[str(column) for column in style.values.columns]]


def _factor_covariance(history: pd.DataFrame, config: EquityRiskConfig) -> pd.DataFrame:
    correlation_weights = exponential_weights(len(history), config.covariance_half_life)
    volatility_weights = exponential_weights(len(history), config.volatility_half_life)
    correlation = weighted_newey_west_covariance(
        history, correlation_weights, config.newey_west_lags
    )
    volatility = weighted_newey_west_covariance(history, volatility_weights, config.newey_west_lags)
    covariance = glue_volatility_and_correlation(volatility, correlation)
    covariance = project_to_psd(covariance) * config.annualization
    covariance = apply_volatility_regime(
        covariance,
        history,
        lookback=config.regime_lookback,
        shrinkage=config.regime_shrinkage,
        scale_cap=config.regime_scale_cap,
        annualization=config.annualization,
    )
    adjusted = eigenfactor_risk_adjustment(
        covariance,
        len(history),
        simulations=config.eigen_simulations,
        scale=config.eigen_scale,
        seed=config.random_seed,
    )
    return project_to_psd(adjusted)


def _model_exposures(
    industries: pd.Series, styles: pd.DataFrame, names: Sequence[str]
) -> pd.DataFrame:
    frame = pd.DataFrame(0.0, index=styles.index, columns=list(names))
    frame[COUNTRY_FACTOR] = 1.0
    for asset, industry in industries.items():
        frame.at[asset, str(industry)] = 1.0
    for column in styles.columns:
        frame[column] = styles[column].to_numpy(dtype=float)
    return frame


def _asset_covariance(
    exposures: pd.DataFrame, factor_covariance: pd.DataFrame, specific: pd.Series
) -> pd.DataFrame:
    matrix = exposures.to_numpy(dtype=float)
    covariance = factor_covariance.to_numpy(dtype=float)
    asset = matrix @ covariance @ matrix.T + np.diag(specific.to_numpy(dtype=float))
    repaired = _frame_from_square(asset, exposures.index)
    eigenvalues = np.linalg.eigvalsh(repaired.to_numpy(dtype=float))
    tolerance = max(float(np.abs(eigenvalues).max()) * 1e-12, 1e-15)
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("reconstructed asset covariance is not positive semidefinite")
    return repaired
