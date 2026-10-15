"""Re-evaluate frozen failures without changing any historical account or input."""

import argparse
import importlib
import json
import os
import pickle
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import pandas as pd  # noqa: E402
from a_share_multifactor.allocation_execution import executable_allocation  # noqa: E402
from a_share_multifactor.config import CostsConfig  # noqa: E402
from a_share_multifactor.run_contract import build_instrument_master  # noqa: E402
from quant_data_kit import FixedPoint  # noqa: E402
from quant_portfolio.solver import OptimizationError  # noqa: E402

from study_v3_archive import digest, write_json  # noqa: E402
from study_v3_portfolios import scenarios, schedule  # noqa: E402


def frozen_accounts(source):
    entries = json.loads((source / "portfolio-logs-cadence-03/progress.json").read_text())
    assert len(entries) == 62
    for entry in entries:
        lines = Path(entry["log"]).read_text(encoding="utf-8").splitlines()
        last = json.loads(lines[-1])
        yield (
            entry["scenario"],
            source / "portfolios" / (f"{entry['scenario']:03d}-{last['identity'][:16]}"),
        )


def decimal_column(row, field):
    return Decimal(int(row[field + "_units"])).scaleb(-int(row[field + "_scale"]))


def date_rows(frame, day):
    dates = pd.to_datetime(frame.event_time, utc=True).dt.tz_convert("Asia/Shanghai").dt.date
    return frame[dates.eq(day.date())]


def run(source, out):
    out.mkdir(parents=True, exist_ok=True)
    modules = [
        importlib.import_module(name)
        for name in (
            "a_share_multifactor.allocation_execution",
            "quant_portfolio.allocation_v2",
            "quant_portfolio.executable",
            "quant_portfolio.feasibility",
            "quant_portfolio.solver",
            "quant_portfolio.risk_snapshot",
            "a_share_multifactor.execution_models",
        )
    ]
    code_paths = {module.__name__: Path(module.__file__) for module in modules}
    code_paths["audit"] = Path(__file__)
    provenance = {name: digest(path) for name, path in code_paths.items()}
    write_json(out / "audit-code.json", provenance)
    inventory = []
    accounts = list(frozen_accounts(source))
    for scenario, directory in accounts:
        decisions = json.loads((directory / "allocation-decisions.json").read_text())
        for decision in decisions:
            if "weights" not in decision:
                inventory.append(
                    {
                        "scenario": scenario,
                        "date": decision["session"],
                        "status": decision["status"],
                        "reason": decision["error"],
                        "source": str(directory / "allocation-decisions.json"),
                        "source_sha256": digest(directory / "allocation-decisions.json"),
                    }
                )
        if (directory / "full-horizon-stop.json").exists():
            stop = json.loads((directory / "full-horizon-stop.json").read_text())
            inventory.append(
                {
                    "scenario": scenario,
                    "date": stop["at"],
                    "status": "entitlement_unresolved",
                    "reason": stop["reason"],
                    "source": str(directory / "full-horizon-stop.json"),
                }
            )
    pd.DataFrame(inventory).to_csv(out / "frozen-failure-inventory.csv", index=False)
    with (source / "panel.pkl").open("rb") as stream:
        bundle = pickle.load(stream)
    with (source / "portfolio-risk-models.pkl").open("rb") as stream:
        models = pickle.load(stream)
    bars = pd.read_parquet(source / "execution-bars.parquet")
    states = pd.read_parquet(source / "source-trading-status.parquet")
    # Only catalog metadata is needed here, avoiding rebuilding daily event windows.
    specs, _ = build_instrument_master(
        bars.iloc[:0], catalog_path=source / "execution-catalog.csv", symbols=list(states.columns)
    )
    costs = CostsConfig(
        initial_capital=1e6,
        commission=0.0003,
        slippage=0.0005,
        min_commission=5,
        statutory_stamp_tax=True,
        max_position_weight=0.05,
        cash_buffer=0,
        max_holdings=0,
        participation_rate=0.1,
        retail_mode=False,
    )
    for scenario, directory in accounts:
        decisions = json.loads((directory / "allocation-decisions.json").read_text())
        failed = [x for x in decisions if x.get("status") == "repair_failed"]
        if not failed:
            continue
        method, model, budget, te = list(scenarios())[scenario]
        dates = {pd.Timestamp(x["session"]) for x in failed}
        plans = schedule(
            bundle,
            {d: models[d] for d in dates},
            method,
            model,
            budget,
            te,
            digest(source / "portfolio-risk-models.pkl"),
        )
        positions = pd.read_parquet(directory / "positions.parquet")
        snapshots = pd.read_parquet(directory / "portfolio_snapshots.parquet")
        fills = pd.read_parquet(directory / "fills.parquet")
        for decision in failed:
            day = pd.Timestamp(decision["session"])
            target = out / f"integer-{scenario:03d}-{day.date()}.json"
            if target.exists():
                continue
            account = date_rows(snapshots, day).iloc[-1]
            nav, cash = decimal_column(account, "nav"), decimal_column(account, "cash_value")
            assert abs(float(nav) - decision["nav"]) < 1e-7
            held = date_rows(positions, day)
            marks = {
                r.symbol: FixedPoint(int(Decimal(str(r.close)) * 100), 2)
                for r in bars[bars.date.eq(day)].itertuples()
            }
            current, value = {}, Decimal(0)
            for _, row in held.iterrows():
                asset = row.instrument_id
                q = decimal_column(row, "quantity")
                assert q == int(q)
                current[asset] = FixedPoint(int(q), 0)
                marks[asset] = FixedPoint(int(row.mark_price_units), int(row.mark_price_scale))
                value += decimal_column(row, "market_value")
            receivable = nav - cash - value
            assert receivable >= 0
            buys = date_rows(fills, day)
            acquired = {}
            for _, row in buys[buys.side.eq("buy")].iterrows():
                acquired[row.instrument_id] = acquired.get(row.instrument_id, Decimal(0)) + (
                    decimal_column(row, "quantity")
                )
            ledger = SimpleNamespace(
                acquired_today=lambda asset, date: acquired.get(asset, Decimal(0)),
                dividend_receivable_balance=lambda currency: receivable,
            )
            gate_state = {
                "cash_reservations": {},
                "position_reservations": {},
                "states": {
                    a: SimpleNamespace(
                        status="normal"
                        if states.at[day, a] == 1
                        else "suspended"
                        if states.at[day, a] == 0
                        else "unknown"
                    )
                    for a in specs
                },
            }
            gate = SimpleNamespace(
                ledger=ledger, instruments=specs, capture_state=lambda: gate_state
            )
            strategy = SimpleNamespace(risk_gate=gate, costs=costs, _closing_prices=marks)
            plan = plans[day.date()]
            if not any("weights" in x and x["session"] < str(day.date()) for x in decisions):
                plan["allocation"]["max_turnover"] = 2.0
            snapshot = SimpleNamespace(
                positions=current,
                cash_balances={"CNY": FixedPoint(int(cash.scaleb(8)), 8)},
                event_time=(day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)),
            )
            record = {
                "scenario": scenario,
                "date": str(day.date()),
                "model": model,
                "frozen_account_sha256": digest(directory / "portfolio_snapshots.parquet"),
                "source": "frozen account, independently reconstructed exact quantities",
                "source_code": provenance,
                "integer_infeasibility_proven": False,
            }
            try:
                continuous, quantities, diagnostic = executable_allocation(
                    strategy, plan, snapshot, nav
                )
                record.update(
                    status="repaired",
                    quantities=quantities,
                    continuous=continuous.to_dict(),
                    diagnostics=diagnostic,
                )
            except OptimizationError as exc:
                record.update(status=exc.status, error=str(exc), diagnostics=exc.diagnostics)
            write_json(target, record)
            print(json.dumps({k: record[k] for k in ("scenario", "date", "status")}), flush=True)
    outputs = [json.loads(path.read_text()) for path in sorted(out.glob("integer-*.json"))]
    assert len(outputs) == 102
    assert all(digest(path) == provenance[name] for name, path in code_paths.items())
    write_json(
        out / "integer-summary.json",
        {
            "total": len(outputs),
            "counts": pd.Series([r["status"] for r in outputs]).value_counts().to_dict(),
            "results": [{k: r[k] for k in ("scenario", "date", "status")} for r in outputs],
            "original_constraints_preserved": True,
            "new_odd_lot_rule": True,
            "not_a_new_account_path": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out)
