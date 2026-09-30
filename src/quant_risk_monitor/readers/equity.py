from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_risk_monitor.input_validation import holdings_weights, nav_observations


def load_capital_curve(path: Path, strategy_col: str) -> pd.Series:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"missing date column in {path}")
    if strategy_col not in df.columns:
        raise ValueError(f"missing strategy column {strategy_col!r} in {path}")
    dates = pd.to_datetime(df["date"])
    return nav_observations(pd.Series(df[strategy_col].values, index=dates))


def load_spread_nav(path: Path, nav_col: str = "nav") -> pd.Series:
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else "trade_date"
    if date_col not in df.columns:
        raise ValueError(f"missing date column in {path}")
    if nav_col not in df.columns:
        raise ValueError(f"missing nav column {nav_col!r} in {path}")
    dates = pd.to_datetime(df[date_col])
    return nav_observations(pd.Series(df[nav_col].values, index=dates))


def load_holdings_weights(path: Path, symbol_col: str, weight_col: str) -> pd.Series:
    df = pd.read_csv(path, dtype={symbol_col: str})
    if symbol_col not in df.columns or weight_col not in df.columns:
        raise ValueError(f"missing columns in {path}")
    return holdings_weights(pd.Series(df[weight_col].values, index=df[symbol_col].values))
