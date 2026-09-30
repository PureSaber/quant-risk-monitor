"""Portfolio risk analytics beyond threshold alerts."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi, sqrt
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TailRisk:
    confidence: float
    var: float
    cvar: float
    observations: int


@dataclass(frozen=True)
class FactorExposureCoverage:
    """Coverage of factor inputs for positions with a non-zero portfolio weight."""

    required_assets: tuple[str, ...]
    covered_assets: tuple[str, ...]
    missing_assets: tuple[str, ...]
    missing_factors: dict[str, tuple[str, ...]]

    @property
    def coverage(self) -> float:
        if not self.required_assets:
            return 1.0
        return len(self.covered_assets) / len(self.required_assets)

    @property
    def complete(self) -> bool:
        return not self.missing_assets

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_assets": list(self.required_assets),
            "covered_assets": list(self.covered_assets),
            "missing_assets": list(self.missing_assets),
            "missing_factors": {
                asset: list(factors) for asset, factors in self.missing_factors.items()
            },
            "coverage": self.coverage,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class ReturnHistoryCoverage:
    """Joint return-history coverage used by an asset covariance estimate."""

    required_assets: tuple[str, ...]
    observations: dict[str, int]
    complete_observations: int
    min_observations: int
    insufficient_assets: tuple[str, ...]

    @property
    def covered_assets(self) -> tuple[str, ...]:
        return tuple(
            asset
            for asset in self.required_assets
            if self.observations[asset] >= self.min_observations
        )

    @property
    def coverage(self) -> float:
        if not self.required_assets:
            return 1.0
        return len(self.covered_assets) / len(self.required_assets)

    @property
    def complete(self) -> bool:
        return not self.insufficient_assets and self.complete_observations >= self.min_observations

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_assets": list(self.required_assets),
            "covered_assets": list(self.covered_assets),
            "insufficient_assets": list(self.insufficient_assets),
            "observations": self.observations,
            "complete_observations": self.complete_observations,
            "min_observations": self.min_observations,
            "coverage": self.coverage,
            "complete": self.complete,
        }


def _clean_return_series(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    numeric = pd.to_numeric(returns, errors="coerce")
    invalid = numeric.isna() & ~returns.isna()
    if invalid.any() or np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("returns must contain only finite numeric values or missing observations")
    return numeric.dropna()


def historical_var_cvar(returns: pd.Series, confidence: float = 0.95) -> TailRisk:
    clean = _clean_return_series(returns)
    if len(clean) < 2:
        raise ValueError("at least two return observations are required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    cutoff = float(clean.quantile(1 - confidence))
    tail = clean[clean <= cutoff]
    return TailRisk(
        confidence=confidence,
        var=max(-cutoff, 0.0),
        cvar=max(-float(tail.mean()), 0.0),
        observations=len(clean),
    )


def parametric_var_cvar(returns: pd.Series, confidence: float = 0.95) -> TailRisk:
    """Normal-distribution VaR/CVaR using the observed sample mean and volatility."""
    clean = _clean_return_series(returns)
    if len(clean) < 2:
        raise ValueError("at least two return observations are required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    mean = float(clean.mean())
    volatility = float(clean.std(ddof=1))
    z_score = NormalDist().inv_cdf(confidence)
    density = exp(-(z_score**2) / 2) / sqrt(2 * pi)
    return TailRisk(
        confidence=confidence,
        var=max(-(mean - z_score * volatility), 0.0),
        cvar=max(-(mean - volatility * density / (1 - confidence)), 0.0),
        observations=len(clean),
    )


def shrink_covariance(
    returns: pd.DataFrame,
    shrinkage: float = 0.2,
    annualization: int = 252,
    *,
    min_observations: int = 2,
) -> pd.DataFrame:
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    coverage = return_history_coverage(returns, min_observations=min_observations)
    if not coverage.complete:
        raise ValueError(
            "insufficient joint return history for covariance: "
            f"assets={list(coverage.insufficient_assets)}, "
            f"complete_observations={coverage.complete_observations}, "
            f"required={coverage.min_observations}"
        )
    clean = _numeric_frame(returns).dropna(axis=0, how="any")
    sample = clean.cov() * annualization
    target = np.diag(np.diag(sample.to_numpy()))
    matrix = (1 - shrinkage) * sample.to_numpy() + shrinkage * target
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    tolerance = max(float(np.abs(eigenvalues).max()) * 1e-12, 1e-15)
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("shrunk covariance is not positive semidefinite")
    repaired = eigenvectors @ np.diag(np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
    return pd.DataFrame(repaired, index=sample.index, columns=sample.columns)


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def return_history_coverage(
    returns: pd.DataFrame, *, min_observations: int = 2
) -> ReturnHistoryCoverage:
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] == 0:
        raise ValueError("returns must contain at least one asset column")
    if returns.columns.has_duplicates:
        raise ValueError("returns contain duplicate asset columns")
    if isinstance(min_observations, bool) or not isinstance(min_observations, int):
        raise TypeError("min_observations must be an integer")
    if min_observations < 2:
        raise ValueError("min_observations must be at least two")
    assets = tuple(str(asset) for asset in returns.columns)
    if len(set(assets)) != len(assets) or any(not asset.strip() for asset in assets):
        raise ValueError("return asset names must be unique and non-empty")
    numeric = _numeric_frame(returns)
    observations = {
        asset: int(numeric.iloc[:, position].notna().sum()) for position, asset in enumerate(assets)
    }
    insufficient = tuple(asset for asset in assets if observations[asset] < min_observations)
    return ReturnHistoryCoverage(
        required_assets=assets,
        observations=observations,
        complete_observations=int(numeric.dropna(axis=0, how="any").shape[0]),
        min_observations=min_observations,
        insufficient_assets=insufficient,
    )


def risk_contributions(weights: pd.Series, covariance: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series")
    if weights.index.has_duplicates:
        raise ValueError("weights contain duplicate assets")
    numeric_weights = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric_weights.isna().any():
        raise ValueError("weights must be finite")
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a pandas DataFrame")
    if covariance.index.has_duplicates or covariance.columns.has_duplicates:
        raise ValueError("covariance contains duplicate asset labels")
    assets = list(weights.index)
    cov = _numeric_frame(covariance.reindex(index=assets, columns=assets))
    if cov.isna().any().any():
        raise ValueError("covariance is missing portfolio assets")
    if not np.allclose(cov.to_numpy(), cov.to_numpy().T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(cov.to_numpy(dtype=float))
    tolerance = max(float(np.abs(eigenvalues).max()) * 1e-12, 1e-15)
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("covariance must be positive semidefinite")
    vector = numeric_weights.to_numpy(dtype=float)
    marginal_variance = cov.to_numpy() @ vector
    variance = float(vector @ marginal_variance)
    if variance <= 0:
        raise ValueError("portfolio variance must be positive")
    component_variance = vector * marginal_variance
    return pd.DataFrame(
        {
            "weight": numeric_weights,
            "marginal_variance": marginal_variance,
            "component_variance": component_variance,
            "risk_contribution": component_variance / variance,
        },
        index=assets,
    )


def stress_test(weights: pd.Series, scenarios: pd.DataFrame) -> pd.DataFrame:
    """Apply asset-return scenarios and report portfolio loss."""
    active = _active_weights(weights)
    if not isinstance(scenarios, pd.DataFrame):
        raise TypeError("stress scenarios must be a pandas DataFrame")
    if scenarios.columns.has_duplicates:
        raise ValueError("stress scenarios contain duplicate asset columns")
    missing = sorted(set(active.index).difference(scenarios.columns))
    if missing:
        raise ValueError(f"stress scenarios missing assets: {missing}")
    required = _numeric_frame(scenarios.reindex(columns=active.index))
    if required.isna().any().any():
        invalid = {
            str(index): sorted(str(asset) for asset in required.columns[row.isna()])
            for index, row in required.iterrows()
            if row.isna().any()
        }
        raise ValueError(f"stress scenarios must be finite for every non-zero holding: {invalid}")
    scenario_return = required.mul(active, axis=1).sum(axis=1, skipna=False)
    return pd.DataFrame(
        {"portfolio_return": scenario_return, "loss": (-scenario_return).clip(lower=0.0)},
        index=scenarios.index,
    )


def liquidity_days_to_exit(
    market_values: pd.Series,
    average_daily_value: pd.Series,
    *,
    max_participation: float = 0.1,
) -> pd.DataFrame:
    if not np.isfinite(max_participation) or not 0 < max_participation <= 1:
        raise ValueError("max_participation must be in (0, 1]")
    for name, values in (("market values", market_values), ("ADV", average_daily_value)):
        if not isinstance(values, pd.Series) or values.empty:
            raise ValueError(f"{name} must be a non-empty Series")
        if not values.index.is_unique or any(
            pd.isna(symbol) or not str(symbol).strip() for symbol in values.index
        ):
            raise ValueError(f"{name} symbols must be unique and non-empty")
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric).all():
            raise ValueError(f"{name} must be finite")
    market_values = pd.to_numeric(market_values).astype(float)
    average_daily_value = pd.to_numeric(average_daily_value).astype(float)
    adv = average_daily_value.reindex(market_values.index)
    if adv.isna().any() or (adv <= 0).any():
        raise ValueError("ADV must be positive for every position")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        days = market_values.abs() / (adv * max_participation)
    if not np.isfinite(days).all():
        raise ValueError("liquidity days must be finite")
    return pd.DataFrame(
        {
            "market_value": market_values,
            "average_daily_value": adv,
            "days_to_exit": days,
        }
    ).sort_values("days_to_exit", ascending=False)


def _active_weights(weights: pd.Series) -> pd.Series:
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series")
    if weights.index.has_duplicates:
        raise ValueError("weights contain duplicate assets")
    numeric = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.isna().any():
        raise ValueError("weights must be finite")
    return numeric[numeric != 0]


def factor_exposure_coverage(weights: pd.Series, exposures: pd.DataFrame) -> FactorExposureCoverage:
    active = _active_weights(weights)
    if not isinstance(exposures, pd.DataFrame) or exposures.shape[1] == 0:
        raise ValueError("factor exposures must contain at least one factor column")
    if exposures.index.has_duplicates:
        raise ValueError("factor exposures contain duplicate assets")
    if exposures.columns.has_duplicates:
        raise ValueError("factor exposures contain duplicate factors")
    factors = tuple(str(factor) for factor in exposures.columns)
    if len(set(factors)) != len(factors) or any(not factor.strip() for factor in factors):
        raise ValueError("factor names must be unique and non-empty")
    required = tuple(str(asset) for asset in active.index)
    if len(set(required)) != len(required) or any(not asset.strip() for asset in required):
        raise ValueError("portfolio asset names must be unique and non-empty")
    numeric = _numeric_frame(exposures)
    missing_factors: dict[str, tuple[str, ...]] = {}
    for raw_asset, asset in zip(active.index, required, strict=True):
        if raw_asset not in numeric.index:
            missing_factors[asset] = factors
            continue
        row = numeric.loc[raw_asset]
        missing = tuple(factors[position] for position, value in enumerate(row) if pd.isna(value))
        if missing:
            missing_factors[asset] = missing
    missing_assets = tuple(asset for asset in required if asset in missing_factors)
    covered_assets = tuple(asset for asset in required if asset not in missing_factors)
    return FactorExposureCoverage(
        required_assets=required,
        covered_assets=covered_assets,
        missing_assets=missing_assets,
        missing_factors=missing_factors,
    )


def factor_exposures(weights: pd.Series, exposures: pd.DataFrame) -> pd.Series:
    active = _active_weights(weights)
    coverage = factor_exposure_coverage(weights, exposures)
    if not coverage.complete:
        raise ValueError(
            "factor exposures are missing or non-finite for non-zero holdings: "
            f"{coverage.missing_factors}"
        )
    if active.empty:
        return pd.Series(0.0, index=exposures.columns, dtype=float)
    aligned = _numeric_frame(exposures).loc[active.index]
    return aligned.mul(active, axis=0).sum(axis=0)


def factor_exposure_drift(
    current: pd.Series,
    baseline: pd.Series,
    baseline_scale: pd.Series | None = None,
) -> pd.DataFrame:
    """Return factor deltas after requiring complete, identical factor universes."""
    for name, values in (("current", current), ("baseline", baseline)):
        if not isinstance(values, pd.Series) or values.empty:
            raise ValueError(f"{name} factor exposures are empty")
        if values.index.has_duplicates:
            raise ValueError(f"{name} factor exposures contain duplicate factors")
    current_factors = set(current.index)
    baseline_factors = set(baseline.index)
    if current_factors != baseline_factors:
        raise ValueError(
            "factor sets differ: "
            f"missing_from_current={sorted(map(str, baseline_factors - current_factors))}, "
            f"missing_from_baseline={sorted(map(str, current_factors - baseline_factors))}"
        )
    aligned = pd.concat(
        [
            pd.to_numeric(current, errors="coerce").rename("current"),
            pd.to_numeric(baseline.reindex(current.index), errors="coerce").rename("baseline"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    if aligned.isna().any().any():
        invalid = sorted(str(factor) for factor in aligned.index[aligned.isna().any(axis=1)])
        raise ValueError(f"factor exposures must be finite for every factor: {invalid}")
    delta = aligned["current"] - aligned["baseline"]
    if baseline_scale is None:
        scale_value = float(delta.std(ddof=0)) or 1.0
        scale = pd.Series(scale_value, index=aligned.index)
    else:
        if not isinstance(baseline_scale, pd.Series):
            raise TypeError("baseline factor scale must be a pandas Series")
        if baseline_scale.index.has_duplicates or set(baseline_scale.index) != current_factors:
            raise ValueError("baseline factor scale must contain exactly the same factors")
        scale = pd.to_numeric(baseline_scale.reindex(aligned.index), errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if scale.isna().any() or (scale <= 0).any():
            raise ValueError("baseline factor scale must be positive for every factor")
    return aligned.assign(delta=delta, z_score=delta / scale)
