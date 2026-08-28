"""Portfolio risk analytics beyond threshold alerts."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TailRisk:
    confidence: float
    var: float
    cvar: float
    observations: int


def historical_var_cvar(returns: pd.Series, confidence: float = 0.95) -> TailRisk:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
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
    clean = pd.to_numeric(returns, errors="coerce").dropna()
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
    returns: pd.DataFrame, shrinkage: float = 0.2, annualization: int = 252
) -> pd.DataFrame:
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    sample = returns.apply(pd.to_numeric, errors="coerce").cov().fillna(0.0) * annualization
    target = np.diag(np.diag(sample.to_numpy()))
    matrix = (1 - shrinkage) * sample.to_numpy() + shrinkage * target
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    floor = max(float(eigenvalues.max()) * 1e-10, 1e-12)
    repaired = eigenvectors @ np.diag(np.clip(eigenvalues, floor, None)) @ eigenvectors.T
    return pd.DataFrame(repaired, index=sample.index, columns=sample.columns)


def risk_contributions(weights: pd.Series, covariance: pd.DataFrame) -> pd.DataFrame:
    assets = list(weights.index)
    cov = covariance.reindex(index=assets, columns=assets)
    if cov.isna().any().any():
        raise ValueError("covariance is missing portfolio assets")
    vector = weights.to_numpy(dtype=float)
    marginal_variance = cov.to_numpy() @ vector
    variance = float(vector @ marginal_variance)
    if variance <= 0:
        raise ValueError("portfolio variance must be positive")
    component_variance = vector * marginal_variance
    return pd.DataFrame(
        {
            "weight": weights,
            "marginal_variance": marginal_variance,
            "component_variance": component_variance,
            "risk_contribution": component_variance / variance,
        },
        index=assets,
    )


def stress_test(weights: pd.Series, scenarios: pd.DataFrame) -> pd.DataFrame:
    """Apply asset-return scenarios and report portfolio loss."""
    missing = sorted(set(weights.index).difference(scenarios.columns))
    if missing:
        raise ValueError(f"stress scenarios missing assets: {missing}")
    scenario_return = scenarios[weights.index].astype(float).mul(weights, axis=1).sum(axis=1)
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
    if not 0 < max_participation <= 1:
        raise ValueError("max_participation must be in (0, 1]")
    adv = average_daily_value.reindex(market_values.index)
    if adv.isna().any() or (adv <= 0).any():
        raise ValueError("ADV must be positive for every position")
    days = market_values.abs() / (adv * max_participation)
    return pd.DataFrame(
        {
            "market_value": market_values,
            "average_daily_value": adv,
            "days_to_exit": days,
        }
    ).sort_values("days_to_exit", ascending=False)


def factor_exposures(weights: pd.Series, exposures: pd.DataFrame) -> pd.Series:
    aligned = exposures.reindex(index=weights.index)
    if aligned.isna().all(axis=None):
        raise ValueError("factor exposures do not overlap portfolio assets")
    return aligned.fillna(0.0).mul(weights, axis=0).sum(axis=0)


def factor_exposure_drift(
    current: pd.Series,
    baseline: pd.Series,
    baseline_scale: pd.Series | None = None,
) -> pd.DataFrame:
    """Return aligned factor deltas and z-scores without silently filling missing factors."""
    aligned = pd.concat([current.rename("current"), baseline.rename("baseline")], axis=1).dropna()
    if aligned.empty:
        raise ValueError("factor exposures do not overlap")
    delta = aligned["current"] - aligned["baseline"]
    if baseline_scale is None:
        scale_value = float(delta.std(ddof=0)) or 1.0
        scale = pd.Series(scale_value, index=aligned.index)
    else:
        scale = pd.to_numeric(baseline_scale.reindex(aligned.index), errors="coerce")
        if scale.isna().any() or (scale <= 0).any():
            raise ValueError("baseline factor scale must be positive for every factor")
    return aligned.assign(delta=delta, z_score=delta / scale)
