from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_capital_curve(path: Path, strategy_col: str) -> pd.Series:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"missing date column in {path}")
    if strategy_col not in df.columns:
        raise ValueError(f"missing strategy column {strategy_col!r} in {path}")
    dates = pd.to_datetime(df["date"])
    values = pd.to_numeric(df[strategy_col], errors="coerce")
    return pd.Series(values.values, index=dates).sort_index().dropna()


def load_spread_nav(path: Path, nav_col: str = "nav") -> pd.Series:
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else "trade_date"
    if date_col not in df.columns:
        raise ValueError(f"missing date column in {path}")
    if nav_col not in df.columns:
        raise ValueError(f"missing nav column {nav_col!r} in {path}")
    dates = pd.to_datetime(df[date_col])
    values = pd.to_numeric(df[nav_col], errors="coerce")
    return pd.Series(values.values, index=dates).sort_index().dropna()


def load_holdings_weights(path: Path, symbol_col: str, weight_col: str) -> pd.Series:
    df = pd.read_csv(path)
    if symbol_col not in df.columns or weight_col not in df.columns:
        raise ValueError(f"missing columns in {path}")
    weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    symbols = df[symbol_col].astype(str)
    return pd.Series(weights.values, index=symbols.values)
