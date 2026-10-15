"""Independent exact-money and constraint checks, without the production quoter."""

from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from repair_failure_audit import date_rows, decimal_column
from study_v3_report import read, verify_account


def exact_quote(quantity, mark, day):
    if not quantity:
        return Decimal(0), Decimal(0)
    direction = 1 if quantity > 0 else -1
    shifted = mark * (1 + Decimal(direction) * Decimal(".0005"))
    price = (shifted / Decimal(".01")).to_integral_value(
        rounding=ROUND_CEILING if direction > 0 else ROUND_FLOOR
    ) * Decimal(".01")
    notional = abs(quantity) * price
    fee = max(Decimal(5), notional * Decimal(".0003"))
    if quantity < 0:
        fee += notional * (
            Decimal(".001") if day < pd.Timestamp("2023-08-28") else Decimal(".0005")
        )
    fee = fee.quantize(Decimal(".00000001"), rounding=ROUND_HALF_UP)
    cash_change = -direction * notional - fee
    return cash_change, -cash_change - quantity * mark


def verify(folder, models, bars, statuses):
    spec = read(folder / "specification.json")
    result = verify_account(folder, models, spec)
    positions = pd.read_parquet(folder / "positions.parquet")
    snapshots = pd.read_parquet(folder / "portfolio_snapshots.parquet")
    fills = pd.read_parquet(folder / "fills.parquet")
    decisions = read(folder / "allocation-decisions.json")
    published, maximum, trace = 0, 0.0, []
    for row in decisions:
        if "weights" not in row:
            continue
        day = pd.Timestamp(row["session"])
        held = date_rows(positions, day)
        held = held[held.quantity_units.ne(0)]
        account = date_rows(snapshots, day).iloc[-1]
        nav = decimal_column(account, "nav")
        cash = decimal_column(account, "cash_value")
        current = {
            item.instrument_id: int(decimal_column(item, "quantity")) for _, item in held.iterrows()
        }
        marks = {
            item.symbol: Decimal(str(item.close)) for item in bars[bars.date.eq(day)].itertuples()
        }
        marks.update(
            {item.instrument_id: decimal_column(item, "mark_price") for _, item in held.iterrows()}
        )
        current_weights = {a: float(q * marks[a] / nav) for a, q in current.items()}
        targets = row["targets"]
        turnover_limit = 0.2 if published else 2.0
        union = sorted(set(current) | set(row["weights"]) | set(models[day]["benchmark"].index))
        covered = models[day]["models"][spec["model"]]["exposures"].index
        assert set(union) <= set(covered), "published risk union was truncated"
        w = pd.Series(row["weights"], dtype=float).reindex(union, fill_value=0)
        before = pd.Series(current_weights, dtype=float).reindex(union, fill_value=0)
        residual = max(
            abs(w.sum() - spec["budget"]),
            float(w.max() - 0.05),
            float(-w.min()),
            float((w - before).abs().sum() - turnover_limit),
        )
        change, cost = Decimal(0), Decimal(0)
        bought_today = date_rows(fills, day)
        bought_today = bought_today[bought_today.side.eq("buy")]
        acquired = bought_today.groupby("instrument_id").quantity_units.sum().to_dict()
        for asset in sorted(set(current) | set(targets)):
            target = targets.get(asset, 0)
            assert type(target) is int and target >= 0
            quantity = current.get(asset, 0)
            delta = target - quantity
            if delta > 0:
                assert delta % 100 == 0, "new buy not a board lot"
            elif delta < 0:
                sold = -delta
                assert sold % 100 == 0 or sold % 100 == quantity % 100, "partial odd remainder"
                assert sold <= quantity - acquired.get(asset, 0), (
                    "sale exceeds T+1 available quantity"
                )
            if statuses.at[day, asset] != 1:
                assert delta == 0, "trade changes holding with unavailable trading status"
            movement, friction = exact_quote(delta, marks[asset], day)
            change += movement
            cost += friction
        execution = row["diagnostics"]["execution"]
        assert abs(cash + change - Decimal(execution["cash"])) <= Decimal(".00000001")
        assert abs(cost - Decimal(execution["costs"])) <= Decimal(".00000001")
        post_nav = nav - cost
        assert abs(post_nav - Decimal(execution["post_fee_nav"])) <= Decimal(".00000001")
        assert cash + change >= 0
        actual = pd.Series(
            {a: float(q * marks[a] / post_nav) for a, q in targets.items()}, dtype=float
        ).reindex(union, fill_value=0)
        claimed = pd.Series(execution["integer_weights"], dtype=float).reindex(union, fill_value=0)
        assert np.max(np.abs(actual - claimed)) <= 1e-10
        residual = max(
            residual,
            float(actual.sum() - spec["budget"]),
            float(actual.max() - 0.05),
            float(-actual.min()),
            float((actual - before).abs().sum() - turnover_limit),
        )
        assert residual <= 1e-7, (day, residual)
        maximum = max(maximum, residual)
        published += 1
        trace.append(
            {
                "date": row["session"],
                "residual": residual,
                "independent_friction": str(cost),
                "independent_cash": str(cash + change),
                "turnover_limit": turnover_limit,
            }
        )
    return {
        **result,
        "independent_exact_quote": True,
        "independent_union_bounds_turnover": True,
        "independent_maximum_residual": maximum,
        "independent_target_checks": trace,
    }
