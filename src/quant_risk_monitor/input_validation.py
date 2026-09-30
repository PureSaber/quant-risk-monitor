"""Validate risk observations without silently dropping or inventing exposures."""

import numpy as np
import pandas as pd


def nav_observations(nav: pd.Series) -> pd.Series:
    if len(nav) < 2:
        raise ValueError("NAV risk is not evaluable: at least two observations are required")
    if not isinstance(nav.index, pd.DatetimeIndex) or nav.index.hasnans:
        raise ValueError("NAV dates must be valid timestamps")
    if not nav.index.is_unique:
        raise ValueError("NAV dates must be unique")
    values = pd.to_numeric(nav, errors="coerce").astype(float).sort_index()
    if not np.isfinite(values).all() or (values < 0).any() or values.iloc[0] <= 0:
        raise ValueError("NAV must be finite and non-negative, with a positive initial value")
    return values


def holdings_weights(weights: pd.Series) -> pd.Series:
    if weights.empty:
        raise ValueError("Holdings risk is not evaluable: no observations")
    if weights.index.hasnans:
        raise ValueError("Holdings symbols must not be missing")
    symbols = weights.index.astype(str).str.strip()
    if (symbols == "").any():
        raise ValueError("Holdings symbols must not be blank")
    values = pd.to_numeric(weights, errors="coerce").astype(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Holdings weights must be finite and non-negative")
    values.index = symbols
    # Lots/strategies for the same security share a single concentration limit.
    combined = values.groupby(level=0, sort=True).sum()
    if not np.isfinite(combined).all() or not np.isfinite(combined.sum()):
        raise ValueError("Aggregated holdings weights must be finite")
    return combined


def fraction_limit(value: float) -> float:
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Risk limits must be finite fractions in [0, 1]")
    return value
