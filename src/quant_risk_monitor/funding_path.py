"""Portfolio funding paths over external native-account scenario valuations.

This module scales externally valued remaining positions, allocates liquidity and
records hypothetical liquidations. It never derives native lifecycle events.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import ROUND_CEILING, Decimal, InvalidOperation, localcontext
from pathlib import Path

import pandas as pd


def _amount(value):
    if not isinstance(value, str):
        raise TypeError("amounts must be finite decimal strings")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid decimal amount") from exc
    if not result.is_finite():
        raise ValueError("amounts must be finite")
    return result


def _positive(value, *, upper=None):
    result = _amount(value)
    if result < 0 or (upper is not None and result > upper):
        raise ValueError("amount outside allowed bounds")
    return result


def _time(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return result.tz_convert("UTC")


def _identifier(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("identity and source references must be nonblank strings")
    return value


def _marks(raw, names):
    if {row["id"] for row in raw} != set(names) or len(raw) != len(names):
        raise ValueError("each step must value every original position exactly once")
    result = {}
    for item in raw:
        value = _amount(item["mark_value"])
        eligible = _positive(item["eligible_collateral"])
        initial = _positive(item["initial_margin"])
        maintenance = _positive(item["maintenance_margin"])
        notional = _positive(item["liquidation_notional"])
        rate = _positive(item["liquidation_cost_rate"], upper=Decimal(1))
        if eligible > max(value, Decimal(0)) or maintenance > initial:
            raise ValueError("collateral cannot exceed positive mark; maintenance <= initial")
        lag = item["settlement_lag_steps"]
        if type(lag) is not int or lag < 0:
            raise ValueError("settlement_lag_steps must be a nonnegative integer")
        result[item["id"]] = {
            "value": value,
            "eligible": eligible,
            "initial": initial,
            "maintenance": maintenance,
            "cash_pnl": _amount(item["cash_pnl"]),
            "fee": rate * notional,
            "capacity": _positive(item["max_liquidation_fraction"], upper=Decimal(1)),
            "lag": lag,
        }
    return result


def evaluate_funding_path(spec):
    """Pure projected stress calculation with 40-digit Decimal arithmetic."""
    with localcontext() as context:
        context.prec = 40
        return _evaluate(spec)


def _state(cash, remaining, marks, reserve):
    eligible = sum((remaining[k] * marks[k]["eligible"] for k in remaining), Decimal(0))
    initial = sum((remaining[k] * marks[k]["initial"] for k in remaining), Decimal(0))
    maintenance = sum((remaining[k] * marks[k]["maintenance"] for k in remaining), Decimal(0))
    return {
        "cash": cash,
        "eligible_collateral": eligible,
        "initial_margin": initial,
        "maintenance_margin": maintenance,
        "cash_gap": max(reserve - cash, Decimal(0)),
        "initial_gap": max(initial + reserve - cash - eligible, Decimal(0)),
        "maintenance_gap": max(maintenance + reserve - cash - eligible, Decimal(0)),
    }


def _fraction_to_cover(gap, relief):
    # Round required sale size upward by at most one arithmetic ULP. Rounding
    # 1/3 down would falsely retain a 1e-40 funding deficit despite spare capacity.
    with localcontext() as context:
        context.rounding = ROUND_CEILING
        return gap / relief


def _evaluate(spec):
    if spec["schema"] != "quant-risk-monitor.funding-path/v1":
        raise ValueError("unsupported funding path schema")
    base = _identifier(spec["base_currency"])
    pool = _identifier(spec["collateral_pool"])
    as_of = _time(spec["as_of"])
    if _time(spec["snapshot_known_at"]) > as_of:
        raise ValueError("snapshot is unavailable at as_of")
    _identifier(spec["source_ref"])
    reserve = _positive(spec["cash_reserve"])
    cash = _amount(spec["opening_cash"])
    names = [_identifier(row["id"]) for row in spec["positions"]]
    if len(names) != len(set(names)) or not names:
        raise ValueError("positions must have unique identities and be nonempty")
    order = spec["liquidation_order"]
    if set(order) != set(names) or len(order) != len(names):
        raise ValueError("liquidation_order must be a permutation of positions")
    remaining = dict.fromkeys(names, Decimal(1))
    old_values = {row["id"]: _amount(row["opening_mark_value"]) for row in spec["positions"]}
    initial_equity = cash + sum(old_values.values())
    if initial_equity <= 0:
        raise ValueError("opening equity must be positive")
    pending, outputs = [], []
    last_time, equity = as_of, initial_equity
    for index, raw in enumerate(spec["steps"]):
        time = _time(raw["time"])
        if time <= last_time:
            raise ValueError("scenario steps must strictly increase after as_of")
        last_time = time
        marks = _marks(raw["positions"], names)
        external = _amount(raw["external_cash_flow"])
        financing = _amount(raw["financing_cash_flow"])
        settled = sum((row["amount"] for row in pending if row["due"] == index), Decimal(0))
        pending = [row for row in pending if row["due"] != index]
        cash_pnl = sum((remaining[k] * marks[k]["cash_pnl"] for k in names), Decimal(0))
        mark_pnl = sum(
            (remaining[k] * (marks[k]["value"] - old_values[k]) for k in names), Decimal(0)
        )
        cash += external + financing + settled + cash_pnl

        before = _state(cash, remaining, marks, reserve)
        triggered = before["maintenance_gap"] > 0 or before["cash_gap"] > 0
        trades, fees = [], Decimal(0)
        if triggered:
            for name in order:
                now = _state(cash, remaining, marks, reserve)
                if now["initial_gap"] == 0 and now["cash_gap"] == 0:
                    break
                mark = marks[name]
                available = min(remaining[name], mark["capacity"])
                immediate_cash = (mark["value"] if mark["lag"] == 0 else Decimal(0)) - mark["fee"]
                collateral_relief = immediate_cash + mark["initial"] - mark["eligible"]
                eventual_cash = mark["value"] - mark["fee"]
                needs = []
                if now["cash_gap"] > 0 and immediate_cash > 0:
                    needs.append(_fraction_to_cover(now["cash_gap"], immediate_cash))
                if now["initial_gap"] > 0 and collateral_relief > 0:
                    needs.append(_fraction_to_cover(now["initial_gap"], collateral_relief))
                # A delayed sale may queue future cash, but cannot cure today's gap.
                if not needs and now["cash_gap"] > 0 and eventual_cash > 0:
                    needs.append(_fraction_to_cover(now["cash_gap"], eventual_cash))
                if not needs or available == 0:
                    continue
                fraction = min(available, max(needs))
                proceeds, fee = fraction * mark["value"], fraction * mark["fee"]
                remaining[name] -= fraction
                cash -= fee
                fees += fee
                if mark["lag"] == 0:
                    cash += proceeds
                else:
                    pending.append({"id": name, "amount": proceeds, "due": index + mark["lag"]})
                trades.append(
                    {
                        "id": name,
                        "fraction_of_original": str(fraction),
                        "fee": str(fee),
                        "proceeds": str(proceeds),
                        "settlement_step": index + mark["lag"],
                        "remaining_fraction": str(remaining[name]),
                    }
                )
        after = _state(cash, remaining, marks, reserve)
        receivables = sum((row["amount"] for row in pending), Decimal(0))
        closing = (
            cash + receivables + sum((remaining[k] * marks[k]["value"] for k in names), Decimal(0))
        )
        expected = equity + external + financing + cash_pnl + mark_pnl - fees
        residual = closing - expected
        # Division may round the last digit. Bound only numerical residual, not funding gaps.
        if abs(residual) > Decimal("1e-28") * max(abs(closing), abs(expected), Decimal(1)):
            raise ArithmeticError("projected equity conservation failed")
        outputs.append(
            {
                "time": time.isoformat(),
                "triggered": triggered,
                "before": {k: str(v) for k, v in before.items()},
                "after": {k: str(v) for k, v in after.items()},
                "liquidations": trades,
                "cash_pnl": str(cash_pnl),
                "mark_pnl": str(mark_pnl),
                "external_cash_flow": str(external),
                "financing_cash_flow": str(financing),
                "liquidation_cost": str(fees),
                "settled_cash": str(settled),
                "pending_settlement_value": str(receivables),
                "equity": str(closing),
                "conservation_residual": str(residual),
                "remaining_fractions": {k: str(v) for k, v in remaining.items()},
                "status": (
                    "insolvent"
                    if closing <= 0
                    else "funding_gap"
                    if after["cash_gap"] > 0 or after["maintenance_gap"] > 0
                    else "below_initial_buffer"
                    if after["initial_gap"] > 0
                    else "funded"
                ),
            }
        )
        old_values = {k: marks[k]["value"] for k in names}
        equity = closing
    if not outputs:
        raise ValueError("scenario must contain at least one step")
    return {
        "schema": "quant-risk-monitor.funding-path-result/v1",
        "scope": "hypothetical_portfolio_funding_projection",
        "base_currency": base,
        "collateral_pool": pool,
        "source_ref": spec["source_ref"],
        "steps": outputs,
        "unsettled": [{**row, "amount": str(row["amount"])} for row in pending],
        "assumptions": [
            "Native producer supplies full-original-position marks, settled P&L and margins.",
            "All amounts already include scenario FX into one base-currency collateral pool.",
            "Maintenance or cash breach triggers liquidation toward initial margin plus reserve.",
            "Sale exchanges fair mark for equal settlement proceeds; costs are additional cash.",
            "Margin is encumbrance, not expense; collateral is not double-counted as cash.",
            "Settlement receivables are equity but ineligible for cash and margin collateral.",
            "Fractional liquidation scales supplied values; no lots, cross-margin netting or Greeks.",
            "Insolvency remains visible; projected paths are not certified exchange liquidation rules.",
        ],
    }


def evaluate_joint_funding(spec):
    """Aggregate segregated collateral pools without inventing transfers or netting."""
    if spec["schema"] != "quant-risk-monitor.joint-funding/v1":
        raise ValueError("unsupported joint funding schema")
    if len({_time(path["as_of"]) for path in spec["pools"]}) > 1:
        raise ValueError("joint funding pools must share the same as_of")
    pools = [evaluate_funding_path(path) for path in spec["pools"]]
    if not pools or len({p["collateral_pool"] for p in pools}) != len(pools):
        raise ValueError("joint funding needs uniquely identified collateral pools")
    times = [row["time"] for row in pools[0]["steps"]]
    if any([r["time"] for r in p["steps"]] != times for p in pools):
        raise ValueError("joint paths must have identical scenario times")
    if len({p["base_currency"] for p in pools}) != 1:
        raise ValueError("joint funding needs a common reporting currency")
    aggregate = []
    with localcontext() as context:
        context.prec = 40
        for index, time in enumerate(times):
            rows = [p["steps"][index] for p in pools]
            aggregate.append(
                {
                    "time": time,
                    "equity": str(sum((_amount(r["equity"]) for r in rows), Decimal(0))),
                    "required_external_cash": str(
                        sum(
                            (
                                max(
                                    _amount(r["after"]["cash_gap"]),
                                    _amount(r["after"]["maintenance_gap"]),
                                )
                                for r in rows
                            ),
                            Decimal(0),
                        )
                    ),
                    "initial_restoration_cash": str(
                        sum(
                            (
                                max(
                                    _amount(r["after"]["cash_gap"]),
                                    _amount(r["after"]["initial_gap"]),
                                )
                                for r in rows
                            ),
                            Decimal(0),
                        )
                    ),
                    "pool_statuses": {
                        p["collateral_pool"]: r["status"] for p, r in zip(pools, rows)
                    },
                }
            )
    return {
        "schema": "quant-risk-monitor.joint-funding-result/v1",
        "pools": pools,
        "aggregate": aggregate,
        "cross_pool_transfers": "not_assumed",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    raw = Path(args.input).read_bytes()
    spec = json.loads(raw.decode("utf-8-sig"))
    result = (
        evaluate_joint_funding(spec)
        if spec["schema"] == "quant-risk-monitor.joint-funding/v1"
        else evaluate_funding_path(spec)
    )
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
