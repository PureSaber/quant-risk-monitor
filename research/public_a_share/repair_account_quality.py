"""Explain stale account marks and unchanged holdings without inventing returns."""

from decimal import Decimal

import pandas as pd


def sessions(frame):
    return pd.to_datetime(frame.event_time, utc=True).dt.tz_convert(
        "Asia/Shanghai"
    ).dt.tz_localize(None).dt.normalize()


def entitlement_stop_context(stop, positions):
    terms = stop.get("diagnostics", {}).get("terms")
    if terms is None:
        return {"status": "unavailable_in_legacy_diagnostic"}
    eligible = terms["eligible_quantity"]
    expected = Decimal(eligible["units"]) / 10 ** eligible["scale"]
    held = positions[sessions(positions).eq(pd.Timestamp(terms["record_date"])) &
                     positions.instrument_id.eq(stop["asset"])].sort_values("event_time")
    actual = Decimal(0)
    if len(held):
        row = held.iloc[-1]
        actual = Decimal(int(row.quantity_units)) / 10 ** int(row.quantity_scale)
    if actual != expected:
        raise ValueError("saved safe prefix differs from stopped entitlement quantity")
    return {"status": "passed", "asset": stop["asset"], "record_date": terms["record_date"],
            "eligible_quantity": str(actual)}


def holdings_evidence(positions, snapshots, bars, statuses):
    held = positions[positions.quantity_units.ne(0)].copy()
    held["date"] = sessions(held)
    held = held.sort_values("event_time").drop_duplicates(["date", "instrument_id"], keep="last")
    account = snapshots.copy()
    account["date"] = sessions(account)
    account = account.sort_values("event_time").drop_duplicates("date", keep="last")
    nav = (account.nav_units / 10.0 ** account.nav_scale).set_axis(account.date)
    calendar = pd.DatetimeIndex(account.date)
    if calendar.empty or not calendar.is_monotonic_increasing:
        raise ValueError("account snapshots need an ordered nonempty calendar")
    status = statuses.rename_axis(index="date", columns="instrument_id").stack(
        future_stack=True
    ).rename("trading_status").reset_index()
    quotes = bars[bars.close.gt(0)][["date", "symbol"]].rename(columns={"symbol": "instrument_id"})
    quotes = quotes.drop_duplicates().merge(status, on=["date", "instrument_id"], how="left",
                                            validate="one_to_one")
    quotes = quotes[quotes.trading_status.eq(1)]
    observed = set(zip(quotes.date, quotes.instrument_id, strict=True))
    held = held.merge(status, on=["date", "instrument_id"], how="left", validate="many_to_one")
    fresh = pd.Series([(d, a) in observed for d, a in zip(held.date, held.instrument_id, strict=True)],
                      index=held.index, dtype=bool)
    stale = held[~fresh].copy()
    stale["nav_weight"] = (
        stale.base_market_value_units / 10.0 ** stale.base_market_value_scale / stale.date.map(nav)
    )
    stale["state"] = ["trading_status_unknown" if pd.isna(state) else
                      "confirmed_suspension" if state == 0 else
                      "quote_missing" if state == 1 else "trading_status_unknown"
                      for state in stale.trading_status]
    dates_by_asset = {asset: pd.DatetimeIndex(frame.date.sort_values())
                      for asset, frame in quotes.groupby("instrument_id")}
    last_dates = []
    for row in stale.itertuples():
        dates = dates_by_asset.get(row.instrument_id, pd.DatetimeIndex([]))
        index = dates.searchsorted(row.date, side="right") - 1
        last_dates.append(None if index < 0 else str(dates[index].date()))
    stale["last_observed_trading_quote"] = last_dates
    stale = stale[["date", "instrument_id", "state", "last_observed_trading_quote", "nav_weight"]]
    signatures = {}
    for day, frame in held.groupby("date"):
        signatures[day] = tuple(sorted(
            (row.instrument_id, str((Decimal(int(row.quantity_units)) /
                                    10 ** int(row.quantity_scale)).normalize()))
            for row in frame.itertuples()
        ))
    intervals, first, previous = [], 0, signatures.get(calendar[0], ())
    for index in range(1, len(calendar) + 1):
        current = None if index == len(calendar) else signatures.get(calendar[index], ())
        if current != previous:
            intervals.append({"from_date": str(calendar[first].date()),
                              "through_date": str(calendar[index - 1].date()),
                              "sessions": index - first, "held_assets": len(previous)})
            first, previous = index, current
    endpoint = stale[stale.date.eq(calendar[-1])]
    daily_weights = stale.groupby("date").nav_weight.sum()
    summary = {
        "stale_valuation_days": int(stale.date.nunique()),
        "endpoint_stale_assets": len(endpoint),
        "endpoint_stale_nav_weight": float(endpoint.nav_weight.sum()),
        "maximum_stale_nav_weight": float(daily_weights.max()) if len(daily_weights) else 0.0,
        "longest_unchanged_nonempty_holding_sessions": max(
            (row["sessions"] for row in intervals if row["held_assets"]), default=0
        ),
    }
    return summary, stale, pd.DataFrame(intervals)
