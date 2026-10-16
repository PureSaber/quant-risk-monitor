"""Explicit risk coverage for partial histories on an unchanged trading calendar.

This is a versioned research extension, not a change to the default estimator.
Missing observations never become zero returns. Industry proxies require a
separate explicit sensitivity choice by the caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RiskCoverageError(ValueError):
    def __init__(self, reason, diagnostics):
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics


def calendar_specific_variance(
    observations,
    calendar,
    as_of,
    *,
    half_life=None,
    lags=0,
    window=252,
    min_observations=126,
    min_effective=20,
):
    """Weighted marginal HAC variance; invalid slots have zero *weight*.

    Calendar positions determine decay and lag pairs, even across missing data.
    Complete data equals estimate_covariance (including its ddof correction).
    A caller must supply the real session calendar, not a compressed return index.
    """
    calendar = pd.DatetimeIndex(calendar)
    if not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("trading calendar must be unique and increasing")
    if not observations.index.is_unique:
        raise ValueError("duplicate residual observations")
    if (
        type(window) is not int
        or window < 2
        or type(lags) is not int
        or not 0 <= lags < window
        or not 2 <= min_observations <= window
        or not np.isfinite(min_effective)
        or min_effective < 2
    ):
        raise ValueError("invalid partial-history estimation requirements")
    if half_life is not None and (not np.isfinite(half_life) or half_life <= 0):
        raise ValueError("half life must be positive")
    selected = calendar[calendar <= pd.Timestamp(as_of)][-window:]
    values = observations.reindex(selected).to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("infinite residual is a corrupt input")
    valid = np.isfinite(values)
    diagnostic = {
        "model_version": "calendar-partial-v1",
        "calendar_sessions": len(selected),
        "valid_observations": int(valid.sum()),
        "required_observations": min_observations,
        "required_effective": min_effective,
        "half_life": half_life,
        "lags": lags,
        "as_of": str(pd.Timestamp(as_of)),
        "status": "partial_history",
        "last_observation": str(selected[valid][-1]) if valid.any() else None,
    }
    if len(selected) != window or valid.sum() < min_observations:
        raise RiskCoverageError("insufficient_calendar_observations", diagnostic)
    ages = np.arange(len(selected) - 1, -1, -1, dtype=float)
    weights = np.ones(len(selected)) if half_life is None else np.exp2(-ages / half_life)
    weights[~valid] = 0
    weights /= weights.sum()
    effective = float(1 / (weights @ weights))
    diagnostic["effective_observations"] = effective
    if effective < min_effective:
        raise RiskCoverageError("insufficient_effective_observations", diagnostic)
    centered = np.zeros(len(values))
    centered[valid] = values[valid] - weights[valid] @ values[valid]
    weighted = centered * np.sqrt(weights)
    variance = float(weighted @ weighted)
    pairs = {}
    for lag in range(1, lags + 1):
        pairs[str(lag)] = int((valid[lag:] & valid[:-lag]).sum())
        variance += 2 * (1 - lag / (lags + 1)) * float(weighted[lag:] @ weighted[:-lag])
    variance /= 1 - weights @ weights
    if not np.isfinite(variance) or variance < -1e-18:
        raise ValueError("partial-history variance is not finite PSD")
    diagnostic["valid_calendar_pairs"] = pairs
    diagnostic["status"] = "direct_complete" if valid.all() else "partial_history"
    return max(variance, 0), diagnostic


def shrink_covered_variance(raw, diagnostic, *, target, config, regime_multiplier=1.0):
    """Use the frozen estimation pool's prior and regime, without refitting it."""
    if (
        not np.isfinite([raw, target, regime_multiplier]).all()
        or min(raw, target, regime_multiplier) < 0
    ):
        raise ValueError("invalid frozen specific-risk prior or regime")
    strength = config.specific_variance_shrinkage
    if config.specific_shrinkage_method == "bayesian":
        strength = config.specific_prior_dof / (
            config.specific_prior_dof + diagnostic["effective_observations"] - 1
        )
    result = ((1 - strength) * raw + strength * target) * regime_multiplier
    return result, {
        **diagnostic,
        "raw_period_variance": raw,
        "target_period_variance": target,
        "shrinkage_weight": strength,
        "regime_multiplier": regime_multiplier,
        "proxy": False,
        "prior_source": "unchanged_complete_estimation_pool",
    }
