"""Opt-in short-lived held-risk estimate; never alters prices or investability.

A missing momentum endpoint on a documented suspension is not a missing trade
that can be downloaded. For an otherwise observed, currently trading holding,
the last complete published exposure can be a separately labelled estimate.
No such estimate is supplied for unknown status, stale fundamentals, changed
industry, an unexplained quote gap, or an expired exposure. The original daily
estimation sample and all directly covered risks remain unchanged.
"""

import numpy as np
import pandas as pd


def held_exposure_candidate(day, asset, panel, snapshots, statuses, *, max_sessions):
    if type(max_sessions) is not int or not 1 <= max_sessions <= 5:
        raise ValueError("held exposure carry must be explicitly between 1 and 5 sessions")
    dates = panel["dates"]
    index = dates.get_loc(day)
    unavailable = {"reason": "missing_reliable_current_exposure"}
    if statuses.at[day, asset] != 1:
        return None, {"reason": "unknown_or_nontrading_state_without_fresh_exposure"}
    required = [
        "size_log_mcap",
        "beta_252d",
        "residual_vol_252d",
        "liquidity_log_turnover_20d",
        "book_to_price",
        "earnings_yield_ttm",
        "earnings_growth_yoy",
        "book_leverage",
    ]
    if not all(np.isfinite(panel["descriptors"][key].at[day, asset]) for key in required):
        return None, {"reason": "missing_nonprice_or_other_required_descriptor"}
    close = panel["prices"]["adjusted_close"][asset]
    if index < 253 or not np.isfinite(close.at[day]):
        return None, unavailable
    endpoints = dates[[index - 253, index - 22]]
    missing = [d for d in endpoints if pd.isna(close.at[d])]
    if not missing or any(statuses.at[d, asset] != 0 for d in missing):
        return None, {"reason": "momentum_endpoint_not_confirmed_suspension"}
    window = close.reindex(dates[index - 253 : index + 1])
    gaps = window.index[window.isna()]
    if any(statuses.at[d, asset] != 0 for d in gaps):
        return None, {"reason": "unexplained_price_gap_in_descriptor_window"}
    for previous in reversed(dates[max(0, index - max_sessions) : index]):
        snap = snapshots.get(previous)
        if snap is None or asset not in snap.values.index:
            continue
        if panel["groups"].at[day, asset] != snap.industries.get(asset):
            return None, {"reason": "industry_changed_since_published_exposure"}
        return snap.values.loc[[asset]].copy(), {
            "exposure_date": str(previous.date()),
            "exposure_age_sessions": index - dates.get_loc(previous),
            "exposure_status": "estimated_carry_confirmed_historical_suspension",
            "coverage_policy": "held-exposure-carry-v1",
            "trading_status": "observed_trading",
            "valuation_is_trade": True,
            "estimated_exposure": True,
            "price_or_return_imputed": False,
            "missing_endpoint_dates": [str(d.date()) for d in missing],
            "source": "last published PIT exposure; archived suspension and raw-price evidence",
        }
    return None, {"reason": "published_exposure_exceeds_five_session_policy"}
