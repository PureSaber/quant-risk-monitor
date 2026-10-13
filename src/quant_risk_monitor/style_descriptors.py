"""Public, causal descriptor definitions; these are not vendor model replicas."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd


def _daily_frame(frame, label):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{label} must be a nonempty daily frame")
    if (
        not isinstance(frame.index, pd.DatetimeIndex)
        or not frame.index.is_monotonic_increasing
        or frame.index.has_duplicates
        or frame.columns.has_duplicates
    ):
        raise ValueError(f"{label} requires sorted unique trading dates and assets")
    values = frame.astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError(f"{label} contains infinity")
    return values


def paired_market_descriptors(returns, market, *, window=252, min_observations=126):
    """Lagged OLS beta/residual SD, using exactly the same paired observations.

    The index must include every exchange session, including all-NaN sessions.
    Missing sessions remain holes in the window. The residual SD uses ddof=1.
    """
    returns = _daily_frame(returns, "returns")
    if (
        type(window) is not int
        or type(min_observations) is not int
        or not 2 <= min_observations <= window
    ):
        raise ValueError("invalid paired regression window")
    if not isinstance(market, pd.Series) or not market.index.equals(returns.index):
        raise ValueError("market must have the same complete trading calendar")
    market = market.astype(float)
    if np.isinf(market.to_numpy()).any():
        raise ValueError("market contains infinity")
    beta, residual, observations = {}, {}, {}
    for asset in returns:
        valid = returns[asset].notna() & market.notna()
        y, x = returns[asset].where(valid), market.where(valid)
        cov = y.rolling(window, min_periods=min_observations).cov(x)
        vx = x.rolling(window, min_periods=min_observations).var()
        vy = y.rolling(window, min_periods=min_observations).var()
        b = cov / vx.where(vx > 0)
        variance = vy - b * cov
        tolerance = 1e-12 * vy.abs().clip(lower=1e-12)
        if (variance < -tolerance).any():
            raise ValueError("paired residual variance is materially negative")
        beta[asset] = b.shift(1)
        residual[asset] = np.sqrt(variance.clip(lower=0)).shift(1)
        observations[asset] = valid.astype(int).rolling(window, min_periods=1).sum().shift(1)
    return {
        "beta_252d": pd.DataFrame(beta),
        "residual_vol_252d": pd.DataFrame(residual),
    }, pd.DataFrame(observations)


def additional_market_descriptors(returns, turnover):
    """Unweighted public candidates, lagged one exchange session.

    Turnover uses one consistent supplier's percentage-point convention. Its
    share denominator must be established by the caller before model admission.
    """
    returns = _daily_frame(returns, "returns")
    turnover = _daily_frame(turnover, "turnover")
    if not turnover.index.equals(returns.index) or not turnover.columns.equals(returns.columns):
        raise ValueError("turnover and returns must align exactly")
    if (turnover < 0).any().any() or (returns <= -1).any().any():
        raise ValueError("negative turnover or nonpositive gross return")
    result = {}
    for window in (21, 63, 252):
        average = turnover.rolling(window, min_periods=math.ceil(0.9 * window)).mean()
        result[f"liquidity_log_turnover_{window}d"] = np.log(
            average.where(average > 0) / 100
        ).shift(1)
    result["daily_vol_252d"] = returns.rolling(252, min_periods=227).std().shift(1)
    logs = np.log1p(returns)
    # Every cumulative horizon must be complete; no compression or zero filling.
    cumulative = np.stack(
        [logs.rolling(21 * i, min_periods=21 * i).sum().to_numpy() for i in range(1, 13)]
    )
    result["cumulative_range_252d"] = pd.DataFrame(
        np.max(cumulative, axis=0) - np.min(cumulative, axis=0),
        index=returns.index,
        columns=returns.columns,
    ).shift(1)
    return result


def annual_growth_descriptors(facts, dates, assets, *, max_age_days=550):
    """Three consecutive annual facts known at each cutoff; never impute sales.

    A trend is OLS slope / mean(abs(annual values)). All-zero series are missing;
    negative profits are retained. Revised facts enter only at available_at.
    Returns descriptor frames and equally shaped categorical quality frames.
    """
    required = {"asset", "field", "report_period", "value", "available_at"}
    if required - set(facts):
        raise ValueError("financial facts lack required fields")
    if (
        isinstance(max_age_days, bool)
        or not isinstance(max_age_days, (int, float))
        or not math.isfinite(max_age_days)
        or max_age_days < 0
    ):
        raise ValueError("max_age_days must be finite and nonnegative")
    dates = pd.DatetimeIndex(dates)
    if (
        not len(dates)
        or dates.has_duplicates
        or not dates.is_monotonic_increasing
        or len(set(assets)) != len(assets)
    ):
        raise ValueError("growth requires unique ordered dates/assets")
    cutoffs = dates.tz_localize("Asia/Shanghai") if dates.tz is None else dates
    fields = {"revenue_ytd": "revenue_growth_3y", "profit_ytd": "profit_growth_3y"}
    values = {name: pd.DataFrame(np.nan, index=dates, columns=assets) for name in fields.values()}
    quality = {
        name: pd.DataFrame("no_annual_report", index=dates, columns=assets)
        for name in fields.values()
    }
    selected = facts.loc[facts.field.isin(fields) & facts.asset.isin(assets)].copy()
    selected["report_period"] = selected.report_period.astype(str)
    selected = selected[selected.report_period.str.endswith("1231")]
    selected["available_at"] = pd.to_datetime(selected.available_at, utc=True)
    # Future revisions cannot invalidate an earlier requested build, including
    # duplicates or values which are not yet interpretable at that cutoff.
    selected = selected[selected.available_at <= cutoffs.max()]
    if selected.duplicated(["asset", "field", "report_period", "available_at"]).any():
        raise ValueError("ambiguous annual fact revisions")
    for asset in assets:
        events = list(selected[selected.asset == asset].sort_values("available_at").itertuples())
        known, cursor = defaultdict(dict), 0
        for day, cutoff in zip(dates, cutoffs, strict=True):
            while cursor < len(events) and events[cursor].available_at <= cutoff:
                row = events[cursor]
                value = float(row.value)
                if not np.isfinite(value):
                    raise ValueError("known annual facts must be finite")
                if pd.Timestamp(row.report_period).tz_localize("Asia/Shanghai") > row.available_at:
                    raise ValueError("annual financial fact available before its period end")
                known[row.field][int(row.report_period[:4])] = value
                cursor += 1
            for field, name in fields.items():
                history = known[field]
                if not history:
                    continue
                year = max(history)
                end = pd.Timestamp(f"{year}-12-31", tz="Asia/Shanghai")
                if (cutoff - end).days > max_age_days:
                    reason = "stale_annual_report"
                elif not all(y in history for y in (year - 2, year - 1, year)):
                    reason = "insufficient_consecutive_years"
                else:
                    annual = np.array([history[y] for y in (year - 2, year - 1, year)])
                    denominator = float(np.abs(annual).mean())
                    reason = "zero_scale" if denominator == 0 else "observed"
                    if denominator > 0:
                        values[name].at[day, asset] = (annual[2] - annual[0]) / (2 * denominator)
                quality[name].at[day, asset] = reason
    return values, quality
