"""Point-in-time Barra-style factor risk model built from supplied exposures.

This module implements an auditable linear factor model. It is not an MSCI
Barra model and does not contain vendor descriptors, calibrations, or data.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

ModelKind = Literal["fundamental_style", "statistical_proxy"]


def _timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"{name} must be timestamp-compatible") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid timestamp")
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _normalize_frame(values: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(values, pd.DataFrame) or values.empty or values.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty asset-by-factor DataFrame")
    if values.index.has_duplicates:
        raise ValueError(f"{name} contains duplicate assets")
    if values.columns.has_duplicates:
        raise ValueError(f"{name} contains duplicate factors")
    assets = [str(asset).strip() for asset in values.index]
    factors = [str(factor).strip() for factor in values.columns]
    if any(not asset for asset in assets) or len(set(assets)) != len(assets):
        raise ValueError(f"{name} asset names must be unique and non-empty")
    if any(not factor for factor in factors) or len(set(factors)) != len(factors):
        raise ValueError(f"{name} factor names must be unique and non-empty")
    numeric = values.apply(pd.to_numeric, errors="coerce").astype(float)
    numeric.index = assets
    numeric.columns = factors
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{name} must be finite for every asset and factor")
    return numeric


def _normalize_series(values: pd.Series, name: str) -> pd.Series:
    if not isinstance(values, pd.Series) or values.empty:
        raise ValueError(f"{name} must be a non-empty asset Series")
    if values.index.has_duplicates:
        raise ValueError(f"{name} contains duplicate assets")
    assets = [str(asset).strip() for asset in values.index]
    if any(not asset for asset in assets) or len(set(assets)) != len(assets):
        raise ValueError(f"{name} asset names must be unique and non-empty")
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    numeric.index = assets
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{name} must be finite for every asset")
    return numeric


@dataclass(frozen=True, kw_only=True)
class ExposureSnapshot:
    """A complete exposure matrix known at a specific point in time."""

    effective_at: pd.Timestamp | str
    available_at: pd.Timestamp | str
    values: pd.DataFrame
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "effective_at", _timestamp(self.effective_at, "effective_at"))
        object.__setattr__(self, "available_at", _timestamp(self.available_at, "available_at"))
        object.__setattr__(self, "values", _normalize_frame(self.values, "exposure values"))
        object.__setattr__(self, "source", str(self.source))


@dataclass(frozen=True, kw_only=True)
class AssetReturnObservation:
    """One non-overlapping cross-section of realized asset returns."""

    period_start: pd.Timestamp | str
    period_end: pd.Timestamp | str
    available_at: pd.Timestamp | str
    values: pd.Series
    source: str = ""

    def __post_init__(self) -> None:
        start = _timestamp(self.period_start, "period_start")
        end = _timestamp(self.period_end, "period_end")
        available = _timestamp(self.available_at, "available_at")
        if start >= end:
            raise ValueError("period_start must be before period_end")
        if available < end:
            raise ValueError("return available_at cannot be before period_end")
        object.__setattr__(self, "period_start", start)
        object.__setattr__(self, "period_end", end)
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "values", _normalize_series(self.values, "return values"))
        object.__setattr__(self, "source", str(self.source))


@dataclass(frozen=True, kw_only=True)
class FactorModelConfig:
    """Estimation limits and annualization for the supplied factor definition."""

    model_kind: ModelKind
    min_periods: int = 20
    min_assets_per_period: int = 6
    min_asset_observations: int = 20
    covariance_shrinkage: float = 0.2
    specific_variance_shrinkage: float = 0.2
    annualization: int = 252
    max_condition_number: float = 1e8

    def __post_init__(self) -> None:
        if self.model_kind not in ("fundamental_style", "statistical_proxy"):
            raise ValueError("model_kind must be 'fundamental_style' or 'statistical_proxy'")
        for name in ("min_periods", "min_assets_per_period", "min_asset_observations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer of at least two")
            if value < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        if isinstance(self.annualization, bool) or not isinstance(self.annualization, int):
            raise TypeError("annualization must be a positive integer")
        if self.annualization <= 0:
            raise ValueError("annualization must be a positive integer")
        for name in ("covariance_shrinkage", "specific_variance_shrinkage"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, value)
        condition = float(self.max_condition_number)
        if not math.isfinite(condition) or condition <= 1:
            raise ValueError("max_condition_number must be finite and greater than one")
        object.__setattr__(self, "max_condition_number", condition)


def _series_payload(values: pd.Series) -> dict[str, Any]:
    return {
        "index": [str(item) for item in values.index],
        "data": [float(value) for value in values.to_numpy()],
    }


def _frame_payload(values: pd.DataFrame) -> dict[str, Any]:
    def encode_index(value: object) -> str:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return str(value)

    return {
        "index": [encode_index(item) for item in values.index],
        "columns": [str(item) for item in values.columns],
        "data": [[float(value) for value in row] for row in values.to_numpy()],
    }


@dataclass(frozen=True)
class FactorModelDiagnostics:
    model_kind: ModelKind
    as_of: pd.Timestamp
    factors: tuple[str, ...]
    assets: tuple[str, ...]
    periods: int
    first_period_start: pd.Timestamp
    last_period_end: pd.Timestamp
    exposure_effective_at: pd.Timestamp
    exposure_available_at: pd.Timestamp
    cross_section_sizes: tuple[int, ...]
    regression_ranks: tuple[int, ...]
    condition_numbers: tuple[float, ...]
    residual_observations: dict[str, int]
    annualization: int
    covariance_shrinkage: float
    specific_variance_shrinkage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_kind": self.model_kind,
            "is_proxy": self.model_kind == "statistical_proxy",
            "as_of": self.as_of.isoformat(),
            "factors": list(self.factors),
            "assets": list(self.assets),
            "periods": self.periods,
            "first_period_start": self.first_period_start.isoformat(),
            "last_period_end": self.last_period_end.isoformat(),
            "exposure_effective_at": self.exposure_effective_at.isoformat(),
            "exposure_available_at": self.exposure_available_at.isoformat(),
            "cross_section_sizes": list(self.cross_section_sizes),
            "regression_ranks": list(self.regression_ranks),
            "condition_numbers": list(self.condition_numbers),
            "residual_observations": self.residual_observations,
            "annualization": self.annualization,
            "covariance_shrinkage": self.covariance_shrinkage,
            "specific_variance_shrinkage": self.specific_variance_shrinkage,
        }


@dataclass(frozen=True)
class RiskDecomposition:
    total_variance: float
    volatility: float
    factor_variance: float
    specific_variance: float
    factor_contributions: pd.Series
    specific_contributions: pd.Series

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_variance": self.total_variance,
            "volatility": self.volatility,
            "factor_variance": self.factor_variance,
            "specific_variance": self.specific_variance,
            "factor_contributions": _series_payload(self.factor_contributions),
            "specific_contributions": _series_payload(self.specific_contributions),
        }


@dataclass(frozen=True)
class PortfolioFactorRisk:
    model_kind: ModelKind
    as_of: pd.Timestamp
    annualization: int
    portfolio_exposures: pd.Series
    portfolio_risk: RiskDecomposition
    benchmark_exposures: pd.Series | None = None
    active_exposures: pd.Series | None = None
    tracking_risk: RiskDecomposition | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_kind": self.model_kind,
            "is_proxy": self.model_kind == "statistical_proxy",
            "as_of": self.as_of.isoformat(),
            "annualization": self.annualization,
            "units": {
                "weights": "fraction_of_nav",
                "variance": "annualized_return_variance",
                "volatility": "annualized_return_rate",
                "exposure": "input_factor_unit_times_weight",
            },
            "portfolio_exposures": _series_payload(self.portfolio_exposures),
            "portfolio_risk": self.portfolio_risk.to_dict(),
        }
        if self.benchmark_exposures is not None:
            payload["benchmark_exposures"] = _series_payload(self.benchmark_exposures)
            payload["active_exposures"] = _series_payload(self.active_exposures)  # type: ignore[arg-type]
            payload["tracking_risk"] = self.tracking_risk.to_dict()  # type: ignore[union-attr]
        return payload


def _portfolio_weights(values: pd.Series | Mapping[str, object], name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        raw = values.copy()
    elif isinstance(values, Mapping):
        raw = pd.Series(dict(values), dtype=object)
    else:
        raise TypeError(f"{name} must be a Series or mapping")
    if raw.index.has_duplicates:
        raise ValueError(f"{name} contains duplicate assets")
    assets = [str(asset).strip() for asset in raw.index]
    if any(not asset for asset in assets) or len(set(assets)) != len(assets):
        raise ValueError(f"{name} asset names must be unique and non-empty")
    weights = pd.to_numeric(raw, errors="coerce").astype(float)
    weights.index = assets
    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError(f"{name} must be finite")
    return weights


@dataclass(frozen=True)
class BarraStyleRiskModel:
    """Estimated linear factor model; never represents proprietary MSCI output."""

    model_kind: ModelKind
    as_of: pd.Timestamp
    annualization: int
    exposures: pd.DataFrame
    factor_returns: pd.DataFrame
    factor_covariance: pd.DataFrame
    specific_variances: pd.Series
    asset_covariance: pd.DataFrame
    diagnostics: FactorModelDiagnostics

    def _align_weights(self, values: pd.Series | Mapping[str, object], name: str) -> pd.Series:
        weights = _portfolio_weights(values, name)
        nonzero = weights[weights != 0]
        missing = sorted(set(nonzero.index).difference(self.exposures.index))
        if missing:
            raise ValueError(f"{name} has non-zero assets missing from the risk model: {missing}")
        return weights.reindex(self.exposures.index, fill_value=0.0)

    def _decompose(self, weights: pd.Series) -> RiskDecomposition:
        vector = weights.to_numpy(dtype=float)
        exposure = self.exposures.T @ weights
        factor_marginal = self.factor_covariance @ exposure
        factor_contributions = exposure * factor_marginal
        specific_contributions = weights.pow(2) * self.specific_variances
        factor_variance = float(factor_contributions.sum())
        specific_variance = float(specific_contributions.sum())
        total_variance = float(vector @ self.asset_covariance.to_numpy() @ vector)
        scale = max(abs(factor_variance) + abs(specific_variance), 1.0)
        if total_variance < -scale * 1e-12:
            raise ValueError("portfolio variance is negative beyond numerical tolerance")
        total_variance = max(total_variance, 0.0)
        return RiskDecomposition(
            total_variance=total_variance,
            volatility=math.sqrt(total_variance),
            factor_variance=factor_variance,
            specific_variance=specific_variance,
            factor_contributions=factor_contributions,
            specific_contributions=specific_contributions,
        )

    def analyze(
        self,
        weights: pd.Series | Mapping[str, object],
        benchmark_weights: pd.Series | Mapping[str, object] | None = None,
    ) -> PortfolioFactorRisk:
        """Return absolute risk and, when supplied, benchmark-relative tracking risk."""
        portfolio = self._align_weights(weights, "weights")
        portfolio_exposures = self.exposures.T @ portfolio
        portfolio_risk = self._decompose(portfolio)
        if benchmark_weights is None:
            return PortfolioFactorRisk(
                model_kind=self.model_kind,
                as_of=self.as_of,
                annualization=self.annualization,
                portfolio_exposures=portfolio_exposures,
                portfolio_risk=portfolio_risk,
            )
        benchmark = self._align_weights(benchmark_weights, "benchmark_weights")
        benchmark_exposures = self.exposures.T @ benchmark
        active = portfolio - benchmark
        return PortfolioFactorRisk(
            model_kind=self.model_kind,
            as_of=self.as_of,
            annualization=self.annualization,
            portfolio_exposures=portfolio_exposures,
            portfolio_risk=portfolio_risk,
            benchmark_exposures=benchmark_exposures,
            active_exposures=portfolio_exposures - benchmark_exposures,
            tracking_risk=self._decompose(active),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "barra_style_factor_risk_model",
            "vendor_model": False,
            "model_kind": self.model_kind,
            "is_proxy": self.model_kind == "statistical_proxy",
            "as_of": self.as_of.isoformat(),
            "annualization": self.annualization,
            "units": {
                "factor_covariance": "annualized_factor_return_covariance",
                "specific_variances": "annualized_asset_return_variance",
                "asset_covariance": "annualized_asset_return_covariance",
            },
            "exposures": _frame_payload(self.exposures),
            "factor_returns": _frame_payload(self.factor_returns),
            "factor_covariance": _frame_payload(self.factor_covariance),
            "specific_variances": _series_payload(self.specific_variances),
            "asset_covariance": _frame_payload(self.asset_covariance),
            "diagnostics": self.diagnostics.to_dict(),
        }


def _latest_snapshot(
    snapshots: Sequence[ExposureSnapshot], cutoff: pd.Timestamp
) -> ExposureSnapshot:
    eligible = [
        snapshot
        for snapshot in snapshots
        if snapshot.effective_at <= cutoff and snapshot.available_at <= cutoff
    ]
    if not eligible:
        raise ValueError(
            f"no exposure snapshot was effective and available by {cutoff.isoformat()}"
        )
    return max(eligible, key=lambda item: (item.effective_at, item.available_at))


def _psd_shrink_covariance(
    values: pd.DataFrame, shrinkage: float, annualization: int
) -> pd.DataFrame:
    sample = values.cov().to_numpy(dtype=float) * annualization
    if not np.isfinite(sample).all():
        raise ValueError("factor covariance cannot be estimated from the supplied factor returns")
    target = np.diag(np.diag(sample))
    shrunk = (1 - shrinkage) * sample + shrinkage * target
    shrunk = (shrunk + shrunk.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    repaired = eigenvectors @ np.diag(np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
    repaired = (repaired + repaired.T) / 2
    return pd.DataFrame(repaired, index=values.columns, columns=values.columns)


def fit_barra_style_risk_model(
    *,
    exposure_snapshots: Sequence[ExposureSnapshot],
    return_observations: Sequence[AssetReturnObservation],
    as_of: pd.Timestamp | str,
    config: FactorModelConfig,
) -> BarraStyleRiskModel:
    """Estimate factor returns, F, D, and asset covariance from strictly PIT inputs."""
    if not isinstance(config, FactorModelConfig):
        raise TypeError("config must be FactorModelConfig")
    cutoff = _timestamp(as_of, "as_of")
    snapshots = list(exposure_snapshots)
    observations = list(return_observations)
    if not snapshots:
        raise ValueError("at least one exposure snapshot is required")
    if not observations:
        raise ValueError("at least one return observation is required")
    if any(not isinstance(item, ExposureSnapshot) for item in snapshots):
        raise ValueError("exposure_snapshots must contain ExposureSnapshot values")
    if any(not isinstance(item, AssetReturnObservation) for item in observations):
        raise ValueError("return_observations must contain AssetReturnObservation values")

    snapshot_keys = [(item.effective_at, item.available_at) for item in snapshots]
    if len(set(snapshot_keys)) != len(snapshot_keys):
        raise ValueError("exposure snapshots contain duplicate effective/available timestamps")
    for snapshot in snapshots:
        if snapshot.effective_at > cutoff or snapshot.available_at > cutoff:
            raise ValueError("exposure snapshots must not be effective or available after as_of")
    factor_order = tuple(snapshots[0].values.columns)
    for snapshot in snapshots[1:]:
        if set(snapshot.values.columns) != set(factor_order):
            raise ValueError("every exposure snapshot must contain exactly the same factors")

    observations.sort(key=lambda item: (item.period_start, item.period_end))
    period_keys = [(item.period_start, item.period_end) for item in observations]
    if len(set(period_keys)) != len(period_keys):
        raise ValueError("return observations contain duplicate periods")
    for position, observation in enumerate(observations):
        if observation.period_end > cutoff or observation.available_at > cutoff:
            raise ValueError("return observations must be complete and available by as_of")
        if position and observation.period_start < observations[position - 1].period_end:
            raise ValueError("return observation periods must not overlap")
    if len(observations) < config.min_periods:
        raise ValueError(
            f"at least {config.min_periods} matured return periods are required; "
            f"received {len(observations)}"
        )

    factor_rows: list[np.ndarray] = []
    residuals: dict[str, list[float]] = {}
    cross_section_sizes: list[int] = []
    ranks: list[int] = []
    conditions: list[float] = []
    for observation in observations:
        snapshot = _latest_snapshot(snapshots, observation.period_start)
        exposure = snapshot.values.reindex(columns=factor_order)
        return_assets = set(observation.values.index)
        exposure_assets = set(exposure.index)
        if return_assets != exposure_assets:
            raise ValueError(
                f"asset coverage differs for return period ending {observation.period_end.isoformat()}: "
                f"missing_exposures={sorted(return_assets - exposure_assets)}, "
                f"missing_returns={sorted(exposure_assets - return_assets)}"
            )
        exposure = exposure.reindex(observation.values.index)
        asset_count, factor_count = exposure.shape
        if asset_count < config.min_assets_per_period:
            raise ValueError(
                f"return period ending {observation.period_end.isoformat()} has {asset_count} assets; "
                f"at least {config.min_assets_per_period} are required"
            )
        if asset_count <= factor_count:
            raise ValueError("each cross-sectional regression requires more assets than factors")
        matrix = exposure.to_numpy(dtype=float)
        rank = int(np.linalg.matrix_rank(matrix))
        condition = float(np.linalg.cond(matrix))
        if rank != factor_count:
            raise ValueError(
                f"exposure matrix is rank deficient for period ending "
                f"{observation.period_end.isoformat()}: rank={rank}, factors={factor_count}"
            )
        if not math.isfinite(condition) or condition > config.max_condition_number:
            raise ValueError(
                f"exposure matrix is ill-conditioned for period ending "
                f"{observation.period_end.isoformat()}: condition_number={condition}"
            )
        coefficients, _, solved_rank, _ = np.linalg.lstsq(
            matrix, observation.values.to_numpy(dtype=float), rcond=None
        )
        if int(solved_rank) != factor_count:
            raise ValueError("cross-sectional factor-return regression did not have full rank")
        residual = observation.values.to_numpy(dtype=float) - matrix @ coefficients
        factor_rows.append(coefficients)
        for asset, value in zip(observation.values.index, residual, strict=True):
            residuals.setdefault(str(asset), []).append(float(value))
        cross_section_sizes.append(asset_count)
        ranks.append(rank)
        conditions.append(condition)

    factor_returns = pd.DataFrame(
        factor_rows,
        index=pd.DatetimeIndex([item.period_end for item in observations], name="period_end"),
        columns=factor_order,
    )
    factor_covariance = _psd_shrink_covariance(
        factor_returns, config.covariance_shrinkage, config.annualization
    )

    current_snapshot = _latest_snapshot(snapshots, cutoff)
    current_exposures = current_snapshot.values.reindex(columns=factor_order)
    residual_counts = {asset: len(values) for asset, values in sorted(residuals.items())}
    insufficient_assets = [
        asset
        for asset in current_exposures.index
        if residual_counts.get(str(asset), 0) < config.min_asset_observations
    ]
    if insufficient_assets:
        raise ValueError(
            "insufficient specific-return history for current exposure assets: "
            f"{sorted(str(asset) for asset in insufficient_assets)}"
        )
    raw_specific = pd.Series(
        {
            str(asset): float(np.var(residuals[str(asset)], ddof=1)) * config.annualization
            for asset in current_exposures.index
        },
        dtype=float,
    )
    if not np.isfinite(raw_specific.to_numpy()).all() or (raw_specific < 0).any():
        raise ValueError("specific variances could not be estimated")
    specific_target = float(raw_specific.mean())
    specific_variances = (
        (1 - config.specific_variance_shrinkage) * raw_specific
        + config.specific_variance_shrinkage * specific_target
    ).reindex(current_exposures.index)

    exposure_matrix = current_exposures.to_numpy(dtype=float)
    asset_matrix = exposure_matrix @ factor_covariance.to_numpy(
        dtype=float
    ) @ exposure_matrix.T + np.diag(specific_variances.to_numpy(dtype=float))
    asset_matrix = (asset_matrix + asset_matrix.T) / 2
    eigenvalues = np.linalg.eigvalsh(asset_matrix)
    tolerance = max(float(np.abs(eigenvalues).max()) * 1e-12, 1e-15)
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("reconstructed asset covariance is not positive semidefinite")
    asset_covariance = pd.DataFrame(
        asset_matrix,
        index=current_exposures.index,
        columns=current_exposures.index,
    )
    diagnostics = FactorModelDiagnostics(
        model_kind=config.model_kind,
        as_of=cutoff,
        factors=factor_order,
        assets=tuple(str(asset) for asset in current_exposures.index),
        periods=len(observations),
        first_period_start=observations[0].period_start,
        last_period_end=observations[-1].period_end,
        exposure_effective_at=current_snapshot.effective_at,
        exposure_available_at=current_snapshot.available_at,
        cross_section_sizes=tuple(cross_section_sizes),
        regression_ranks=tuple(ranks),
        condition_numbers=tuple(conditions),
        residual_observations=residual_counts,
        annualization=config.annualization,
        covariance_shrinkage=config.covariance_shrinkage,
        specific_variance_shrinkage=config.specific_variance_shrinkage,
    )
    return BarraStyleRiskModel(
        model_kind=config.model_kind,
        as_of=cutoff,
        annualization=config.annualization,
        exposures=current_exposures,
        factor_returns=factor_returns,
        factor_covariance=factor_covariance,
        specific_variances=specific_variances,
        asset_covariance=asset_covariance,
        diagnostics=diagnostics,
    )
