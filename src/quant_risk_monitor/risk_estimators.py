"""Transparent weighted regression and horizon-aware covariance estimators.

These are public statistical constructions, not vendor calibrations. Weighting
means weights in the least-squares objective, not multipliers of the residual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_weights(count, half_life=None):
    if count < 2:
        raise ValueError("risk estimation needs at least two observations")
    weights = (
        np.ones(count)
        if half_life is None
        else np.exp2(-np.arange(count - 1, -1, -1, dtype=float) / half_life)
    )
    weights /= weights.sum()
    if 1 - weights @ weights <= np.finfo(float).eps:
        raise ValueError("time weights have insufficient effective observations")
    return weights


def estimate_covariance(values, *, half_life=None, lags=0, shrinkage=0.0):
    """Weighted, centered covariance plus Bartlett HAC; return per-period rate.

    Lag products use sqrt(w_t*w_{t-l}), a congruence of the Bartlett PSD
    kernel. The same 1-sum(w**2) denominator is used for every lag. With
    uniform weights and lag zero this equals the ordinary ddof=1 covariance.
    With L lags the kernel describes an H=L+1 period variance divided by H.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("covariance requires a finite time-by-series matrix")
    count = len(values)
    if lags >= count:
        raise ValueError("Newey-West lags must be smaller than the history")
    weights = time_weights(count, half_life)
    centered = values - weights @ values
    weighted = centered * np.sqrt(weights[:, None])
    covariance = weighted.T @ weighted
    for lag in range(1, lags + 1):
        cross = weighted[lag:].T @ weighted[:-lag]
        covariance += (1 - lag / (lags + 1)) * (cross + cross.T)
    covariance /= 1 - weights @ weights
    covariance = (1 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance))
    covariance = (covariance + covariance.T) / 2
    eigenvalues, vectors = np.linalg.eigh(covariance)
    tolerance = max(float(np.abs(eigenvalues).max()) * 1e-10, 1e-18)
    if eigenvalues.min() < -tolerance or not np.isfinite(covariance).all():
        raise ValueError("estimated covariance is not finite PSD")
    repaired = vectors @ np.diag(np.maximum(eigenvalues, 0)) @ vectors.T
    return repaired, {
        "effective_observations": float(1 / (weights @ weights)),
        "half_life": half_life,
        "newey_west_lags": lags,
        "minimum_raw_eigenvalue": float(eigenvalues.min()),
        "numerical_psd_repair": bool(eigenvalues.min() < 0),
    }


def solve_factor_returns(snapshot, returns, config):
    exposure = snapshot.values.reindex(returns.index)
    matrix = exposure.to_numpy(dtype=float)
    assets, factors = matrix.shape
    weights = np.ones(assets)
    if config.regression_weighting != "equal":
        if snapshot.market_caps is None:
            raise ValueError("weighted regression requires PIT market_caps")
        weights = snapshot.market_caps.reindex(returns.index).to_numpy(dtype=float, copy=True)
        if config.regression_weighting == "sqrt_market_cap":
            weights = np.sqrt(weights)
    weights /= weights.mean()
    transform = np.eye(factors)
    constraint = np.zeros(factors)
    if config.industry_constraint:
        names = snapshot.industry_factors
        if not names or snapshot.market_caps is None or "market" not in exposure:
            raise ValueError("industry constraint requires market, industry factors and PIT caps")
        industry = exposure[list(names)].to_numpy(dtype=float)
        if (industry < 0).any() or not np.allclose(industry.sum(axis=1), 1, atol=1e-12):
            raise ValueError("industry exposures must be nonnegative and sum to one")
        if not np.allclose(exposure["market"], 1, atol=1e-12):
            raise ValueError("industry-constrained market exposure must equal one")
        caps = snapshot.market_caps.reindex(returns.index).to_numpy(dtype=float)
        shares = caps @ industry / caps.sum()
        if (shares <= 0).any():
            raise ValueError("every declared industry needs positive estimation-universe cap")
        for name, share in zip(names, shares, strict=True):
            constraint[exposure.columns.get_loc(name)] = share
        # Orthonormal null space makes rank/condition checks invariant to the
        # industry selected as a reference. No ridge term hides rank defects.
        _, _, vt = np.linalg.svd(constraint[None, :], full_matrices=True)
        transform = vt[1:].T
    design = np.sqrt(weights[:, None]) * (matrix @ transform)
    free_factors = transform.shape[1]
    if assets <= free_factors:
        raise ValueError("regression requires more assets than free factor coefficients")
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design))
    if rank != free_factors:
        raise ValueError("exposure matrix is rank deficient after applying declared constraints")
    if not np.isfinite(condition) or condition > config.max_condition_number:
        raise ValueError("exposure matrix is ill-conditioned after weighting and constraints")
    coefficients = (
        transform
        @ np.linalg.lstsq(design, np.sqrt(weights) * returns.to_numpy(dtype=float), rcond=None)[0]
    )
    residual = returns.to_numpy(dtype=float) - matrix @ coefficients
    constraint_error = float(abs(constraint @ coefficients))
    if constraint_error > 1e-10 * max(1.0, float(np.linalg.norm(coefficients))):
        raise ValueError("industry regression constraint was not satisfied")
    return (
        coefficients,
        residual,
        rank,
        condition,
        {
            "free_coefficients": free_factors,
            "objective_weighting": config.regression_weighting,
            "industry_constraint_weights": dict(
                zip(exposure.columns, constraint.tolist(), strict=True)
            ),
            "constraint_error": constraint_error,
            "weighted_residual_ss": float(weights @ residual**2),
        },
    )


def regime_multiplier(values, config, *, half_life):
    """Common variance multiplier from strictly prior standardized innovations."""
    if config.regime_window == 0:
        return 1.0, {"enabled": False}
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("regime adjustment requires a finite time-by-series matrix")
    if len(values) < config.regime_min_periods + config.regime_window:
        raise ValueError("regime adjustment requires prior history plus the complete regime window")
    scores = []
    for position in range(len(values) - config.regime_window, len(values)):
        prior = values[:position]
        # Only marginal variances enter the regime score. Computing the full
        # covariance and its eigendecomposition costs O(N^3) for a specific-risk
        # panel, without contributing any information to this calculation.
        weights = time_weights(len(prior), half_life)
        center = weights @ prior
        variance = weights @ (prior - center) ** 2 / (1 - weights @ weights)
        innovation = values[position] - center
        if np.any((variance <= 0) & (innovation != 0)):
            raise ValueError("regime innovation is nonzero against a zero-variance history")
        positive = variance > 0
        scores.append(
            float(np.mean(innovation[positive] ** 2 / variance[positive]))
            if positive.any()
            else 1.0
        )
    weights = time_weights(len(scores), config.regime_half_life)
    raw = float(weights @ scores)
    if not np.isfinite(raw):
        raise ValueError("regime multiplier is not finite")
    multiplier = float(np.clip(raw, config.regime_scale_min, config.regime_scale_max))
    return multiplier, {
        "enabled": True,
        "variance_multiplier_raw": raw,
        "variance_multiplier": multiplier,
        "bound_applied": raw != multiplier,
        "innovation_scores": scores,
        "prior_only": True,
    }


def estimate_specific_risk(residuals, current_snapshot, config, period_count):
    assets = list(current_snapshot.values.index)
    raw, counts, effective, regimes = {}, {}, {}, {}
    for asset in assets:
        values = np.asarray(residuals[asset], dtype=float)[:, None]
        if (
            config.specific_half_life is not None
            or config.specific_newey_west_lags
            or config.regime_window
        ) and len(values) != period_count:
            raise ValueError("advanced specific risk requires contiguous complete asset history")
        covariance, diagnostic = estimate_covariance(
            values, half_life=config.specific_half_life, lags=config.specific_newey_west_lags
        )
        raw[asset] = float(covariance[0, 0])
        counts[asset] = len(values)
        effective[asset] = diagnostic["effective_observations"]
    raw = pd.Series(raw)
    target = pd.Series(float(raw.mean()), index=assets)
    fallback_assets = []
    if config.specific_target == "industry":
        if current_snapshot.industries is None:
            raise ValueError("industry specific-risk shrinkage requires PIT classifications")
        groups = current_snapshot.industries.reindex(assets)
        for asset in assets:
            peers = [a for a in assets if a != asset and groups[a] == groups[asset]]
            if len(peers) < config.specific_min_group_assets - 1:
                if config.specific_group_fallback == "error":
                    raise ValueError("specific-risk industry group has insufficient peer assets")
                peers = [a for a in assets if a != asset]
                fallback_assets.append(asset)
            if not peers:
                raise ValueError("specific-risk shrinkage has no independent peers")
            target[asset] = float(raw.loc[peers].mean())
    strengths = pd.Series(config.specific_variance_shrinkage, index=assets)
    if config.specific_shrinkage_method == "bayesian":
        strengths = pd.Series(
            {
                asset: config.specific_prior_dof
                / (config.specific_prior_dof + effective[asset] - 1)
                for asset in assets
            }
        )
    shrunk = (1 - strengths) * raw + strengths * target
    multiplier = 1.0
    if config.regime_window:
        panel = np.column_stack([residuals[asset] for asset in assets])
        multiplier, regimes = regime_multiplier(panel, config, half_life=config.specific_half_life)
    return shrunk * multiplier * config.annualization, {
        "raw_period_variances": raw.to_dict(),
        "target_period_variances": target.to_dict(),
        "shrinkage_weights": strengths.to_dict(),
        "effective_observations": effective,
        "group_fallback_assets": fallback_assets,
        "method": config.specific_shrinkage_method,
        "target": config.specific_target,
        "regime": regimes,
    }
