"""Native-account portfolio comparison and common-weight risk cross-predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import quant_risk_monitor  # noqa: F401 -- initialize the public API before numerical bootstrap
from study_v3_archive import digest, write_json
from study_v3_risk import factor_variance

GENERATORS = [
    "legacy_ols",
    "repaired_ols",
    "residual_ols",
    "legacy_enhanced",
    "enhanced_s0.1_f90_d60_p10",
    "residual_enhanced",
]


def scenarios():
    for budget in (0.95, 0.8):
        yield "equal", "repaired_ols", budget, None
        for model in GENERATORS:
            yield "minimum_variance", model, budget, None
            for te in (0.03, 0.06):
                yield "minimum_active_risk", model, budget, te
                yield "cost_aware", model, budget, te


def peak_memory():
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        memory_info.restype = wintypes.BOOL
        if not memory_info(process(), ctypes.byref(counters), counters.cb):
            raise OSError("cannot measure process peak memory")
        return int(counters.PeakWorkingSetSize)
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def labels(bundle):
    panel = bundle["panel"]
    dates = panel["dates"]
    score = bundle["v2"]["descriptors"]["momentum_252_21"]
    prices = panel["prices"]["adjusted_close"]
    records = []
    start = dates.searchsorted(pd.Timestamp("2022-01-01"))
    for i in range(start, len(dates) - 5, 5):
        signal, end = dates[i], dates[i + 5]
        r = prices.loc[end] / prices.loc[signal] - 1
        valid = score.loc[signal].notna() & r.notna()
        for asset in score.columns[valid]:
            records.append(
                {
                    "signal_at": signal.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15),
                    "label_end": end.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15),
                    "available_at": end.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15),
                    "asset": asset,
                    "score": float(score.at[signal, asset]),
                    "return": float(r[asset]),
                    "horizon_sessions": 5,
                }
            )
    return pd.DataFrame(records)


def schedule(bundle, models, method, model, budget, te, input_digest):
    from quant_portfolio import RiskSnapshot

    panel = bundle["panel"]
    label_history = labels(bundle) if method == "cost_aware" else None
    result = {}
    for day, entry in models.items():
        data = entry["models"][model]
        x = data["exposures"]
        at = day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
        snapshot = RiskSnapshot(
            estimated_at=at,
            available_at=at,
            horizon_sessions=5,
            annualization=252,
            model_version="barra-200-v3:" + model,
            input_digest=input_digest,
            exposures=x,
            factor_covariance=pd.DataFrame(
                data["F_daily_rate"] * 252, index=x.columns, columns=x.columns
            ),
            specific_variance=pd.Series(data["D_daily_rate"] * 252, index=x.index),
            benchmark=entry["benchmark"],
        )
        score = bundle["v2"]["descriptors"]["momentum_252_21"].loc[day, x.index]
        # Exposure admission already requires these causal price descriptors.
        if not np.isfinite(score).all():
            raise ValueError("missing fixed momentum signal; no equal-weight fallback")
        cfg = {
            "protocol_version": 2,
            "mode": "equal" if method == "equal" else "cost_aware",
            "risk_objective": "active"
            if method in {"minimum_active_risk", "cost_aware"}
            else "absolute",
            "forecast_horizon_sessions": 5,
            "expected_return_model": "none"
            if method == "equal"
            else "ic_vol"
            if method == "cost_aware"
            else "provided",
            "lookback": 252,
            "min_observations": 252,
            "risk_aversion": 5.0,
            "max_turnover": 0.2,
            "turnover_penalty": 0.0,
            "max_tracking_error": te,
            "timeout_seconds": 30.0,
            "ic_lookback": 120,
            "ic_min_periods": 20,
        }
        past = panel["prices"]["returns"].loc[:day].tail(252).copy()
        past.index = past.index.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
        result[day.date()] = {
            "scores": score,
            "signal_cross_section": score,
            "trailing_returns": past,
            "risk_snapshot": snapshot,
            "allocation": cfg,
            "invested_limit": budget,
            "max_weight": 0.05,
            "initial_max_turnover": 2.0,
            "objective_costs": method == "cost_aware",
            "linear_costs": None,
        }
        if method in {"minimum_variance", "minimum_active_risk"}:
            result[day.date()]["expected_returns"] = pd.Series(0.0, index=x.index)
        if label_history is not None:
            # All labels supplied to the existing converter are already mature;
            # it independently checks maturity, revisions and non-overlap.
            mature = label_history[label_history.available_at <= at]
            starts = mature.signal_at.drop_duplicates().sort_values().tail(120)
            result[day.date()]["ic_labels"] = mature[mature.signal_at.isin(starts)]
    return result


def cross_predictions(replay, models, bundle):
    panel = bundle["panel"]
    records = []
    last_day = (
        pd.to_datetime(replay.frames["returns"].event_time, utc=True)
        .max()
        .tz_convert("Asia/Shanghai")
        .tz_localize(None)
        .normalize()
    )
    for decision in replay.allocation_decisions:
        if "weights" not in decision:
            continue
        day = pd.Timestamp(decision["session"])
        entry = models[day]
        i = panel["dates"].get_loc(day)
        end = panel["dates"][i + 5]
        if end > last_day:
            continue
        returns = panel["prices"]["returns"].iloc[i + 1 : i + 6].sum(min_count=5)
        variants = {
            "continuous": decision["weights"],
            "integer": decision["diagnostics"]["execution"]["integer_weights"],
        }
        positions = replay.frames["positions"]
        if not positions.empty:
            dates = (
                pd.to_datetime(positions.event_time, utc=True)
                .dt.tz_convert("Asia/Shanghai")
                .dt.tz_localize(None)
                .dt.normalize()
            )
            actual = positions[dates.eq(day)]
            navs = replay.frames["portfolio_snapshots"]
            nav_day = (
                pd.to_datetime(navs.event_time, utc=True)
                .dt.tz_convert("Asia/Shanghai")
                .dt.tz_localize(None)
                .dt.normalize()
            )
            nav = navs[nav_day.eq(day)].iloc[-1]
            value = float(nav.nav_units) / 10 ** int(nav.nav_scale)
            variants["actual_close"] = {
                r.instrument_id: float(r.base_market_value_units)
                / 10 ** int(r.base_market_value_scale)
                / value
                for r in actual.itertuples()
            }
        for target_kind, mapping in variants.items():
            weights = pd.Series(mapping, dtype=float)
            nonzero = weights[weights.abs() > 1e-12]
            actual_return = (
                float(nonzero @ returns.reindex(nonzero.index))
                if returns.reindex(nonzero.index).notna().all()
                else None
            )
            identity = hashlib.sha256(
                json.dumps(
                    {"date": str(day), "kind": target_kind, "weights": mapping}, sort_keys=True
                ).encode()
            ).hexdigest()
            for name, data in entry["models"].items():
                x = data["exposures"]
                if not set(nonzero.index) <= set(x.index):
                    records.append(
                        {
                            "date": day,
                            "end": end,
                            "portfolio_id": identity,
                            "target_kind": target_kind,
                            "predictor": name,
                            "status": "risk_union_incomplete",
                        }
                    )
                    continue
                w = weights.reindex(x.index, fill_value=0).to_numpy()
                wb = entry["benchmark"].reindex(x.index).to_numpy()
                variance, factor, specific = factor_variance(
                    x.to_numpy(), data["F_daily_rate"], data["D_daily_rate"], w
                )
                active, _, _ = factor_variance(
                    x.to_numpy(), data["F_daily_rate"], data["D_daily_rate"], w - wb
                )
                benchmark_actual = (
                    float(entry["benchmark"] @ returns.reindex(x.index))
                    if returns.reindex(x.index).notna().all()
                    else None
                )
                records.append(
                    {
                        "date": day,
                        "end": end,
                        "portfolio_id": identity,
                        "target_kind": target_kind,
                        "predictor": name,
                        "status": "complete" if actual_return is not None else "outcome_missing",
                        "variance": variance * 5,
                        "factor_variance": factor * 5,
                        "specific_variance": specific * 5,
                        "annualized_predicted_te": float(np.sqrt(active * 252)),
                        "actual_frozen_weight_return": actual_return,
                        "active_frozen_weight_return": None
                        if actual_return is None or benchmark_actual is None
                        else actual_return - benchmark_actual,
                        "qlike": None
                        if actual_return is None or variance <= 0
                        else float(np.log(variance * 5) + actual_return**2 / (variance * 5)),
                        "zero_risk_cash": variance <= 0,
                    }
                )
    return pd.DataFrame(records)


def execution_metrics(replay, models, bundle, destination):
    """Reconcile actual account outcomes separately from frozen-weight forecasts."""
    frame = replay.frames["returns"]
    days = (
        pd.to_datetime(frame.event_time, utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    nav = pd.Series((frame.nav_units.astype(float) / 10.0**frame.nav_scale).to_numpy(), index=days)
    nav = nav.groupby(level=0).last()
    px = bundle["panel"]["prices"]["adjusted_close"]
    calendar = bundle["panel"]["dates"]
    outcomes = []
    for day, entry in models.items():
        end = calendar[calendar.get_loc(day) + 5]
        if day not in nav.index or end not in nav.index:
            continue
        benchmark = entry["benchmark"]
        asset_return = px.loc[end, benchmark.index] / px.loc[day, benchmark.index] - 1
        observed = asset_return.notna().all()
        actual = float(nav[end] / nav[day] - 1)
        reference = float(benchmark @ asset_return) if observed else None
        outcomes.append(
            {
                "date": day,
                "end": end,
                "account_return": actual,
                "benchmark_return": reference,
                "active_account_return": None if reference is None else actual - reference,
                "benchmark_complete": bool(observed),
            }
        )
    windows = pd.DataFrame(outcomes)
    windows.to_parquet(destination / "actual-account-windows.parquet", index=False)
    active = windows.active_account_return.dropna() if len(windows) else pd.Series(dtype=float)
    costs = replay.frames["costs"]
    fees = (
        float((costs.amount_units.astype(float) / 10.0**costs.amount_scale).sum())
        if len(costs)
        else 0.0
    )
    fills = replay.frames["fills"]
    traded = 0.0
    if len(fills):
        traded = float(
            (
                (fills.quantity_units.astype(float) / 10.0**fills.quantity_scale)
                * (fills.price_units.astype(float) / 10.0**fills.price_scale)
            ).sum()
        )
    orders = replay.frames["orders"]
    snapshots = replay.frames["portfolio_snapshots"]
    cash = snapshots.cash_value_units.astype(float) / 10.0**snapshots.cash_value_scale
    successes = [x for x in replay.allocation_decisions if "weights" in x]
    failures = [x for x in replay.allocation_decisions if "weights" not in x]
    failure_counts = {}
    for item in failures:
        status = item.get("status", "unspecified")
        failure_counts[status] = failure_counts.get(status, 0) + 1
    exposure_sizes = [len(v["models"][GENERATORS[0]]["exposures"]) for v in models.values()]
    return {
        "actual_annualized_tracking_error": float(active.std(ddof=1) * np.sqrt(252 / 5))
        if len(active) > 1
        else None,
        "actual_te_complete_windows": len(active),
        "benchmark_missing_windows": len(windows) - len(active),
        "total_fees": fees,
        "gross_traded_notional": traded,
        "gross_traded_notional_over_initial_capital": traded / 1_000_000,
        "filled_orders": int(orders.status.eq("filled").sum()) if len(orders) else 0,
        "partially_filled_orders": int(
            (
                (orders.filled_quantity_units > 0)
                & (orders.filled_quantity_units < orders.quantity_units)
            ).sum()
        )
        if len(orders)
        else 0,
        "minimum_cash": float(cash.min()),
        "failure_status_counts": failure_counts,
        "candidate_count_range": [min(exposure_sizes), max(exposure_sizes)]
        if exposure_sizes
        else [],
        "allocation_space_not_uniquely_determined_by_cap": bool(
            exposure_sizes and min(exposure_sizes) * 0.05 > 0.95
        ),
        "published_target_count": len(successes),
        "actual_return_scope": "native ledger, including fees and gross entitlement cash; missing quoted marks remain explicitly stale",
    }


def run(out, scenario_index, *, limit_dates=None):
    from a_share_multifactor.config import AppConfig, CostsConfig
    from a_share_multifactor.run_contract import ReplayInputError, _replay
    from quant_data_kit import StatusEvent

    method, model, budget, te = list(scenarios())[scenario_index]
    with (out / "panel.pkl").open("rb") as stream:
        bundle = pickle.load(stream)
    with (out / "risk-models.pkl").open("rb") as stream:
        models = pickle.load(stream)
    inputs = {
        name: digest(out / name)
        for name in (
            "panel.pkl",
            "risk-models.pkl",
            "execution-bars.parquet",
            "execution-actions.parquet",
            "corporate-action-events.pkl",
            "execution-catalog.csv",
            "protocol.json",
            "execution-input-guards.json",
            "execution-input-issues.json",
            "source-trading-status.parquet",
        )
    }
    inputs["research_code"] = digest(__file__)
    import a_share_multifactor.allocation_execution as adapter
    import a_share_multifactor.run_contract as contract
    import quant_portfolio.executable as integer

    import quant_risk_monitor.cross_asset as policy
    import quant_risk_monitor.factor_model as factor_model

    for module in (contract, adapter, integer, policy, factor_model):
        inputs[module.__name__] = digest(module.__file__)
    binding = json.loads((out / "risk-models-binding.json").read_text(encoding="utf8"))
    if (
        binding["models"] != inputs["risk-models.pkl"]
        or binding["protocol"] != inputs["protocol.json"]
    ):
        raise ValueError("risk model cache binding changed")
    specification = {
        "method": method,
        "model": model,
        "budget": budget,
        "te": te,
        "inputs": inputs,
        "limit_dates": limit_dates,
        "benchmark": "research-sample-cap-weighted",
        "generator_selection": "six frozen structural model combinations; all 22 estimators cross-predict",
        "tax_scope": "trading costs and gross distributions; no investor-specific dividend income tax",
    }
    identity = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    destination = out / "portfolios" / f"{scenario_index:03d}-{identity[:16]}"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "complete.json").exists():
        receipt = json.loads((destination / "complete.json").read_text(encoding="utf8"))
        if any(digest(destination / name) != sha for name, sha in receipt["outputs"].items()):
            raise ValueError("changed historical study outputs")
        return receipt
    write_json(destination / "specification.json", specification)
    started = perf_counter()
    try:
        raw = pd.read_parquet(out / "execution-bars.parquet")
        if limit_dates:
            final = sorted(raw.date.unique())[limit_dates - 1]
            raw = raw[raw.date <= final]
            models = {d: v for d, v in models.items() if d <= final}
        plans = schedule(bundle, models, method, model, budget, te, inputs["risk-models.pkl"])
        print(
            json.dumps(
                {
                    "scenario": scenario_index,
                    "phase": "replay",
                    "plans": len(plans),
                    "bars": len(raw),
                    "setup_seconds": perf_counter() - started,
                }
            ),
            flush=True,
        )
        with (out / "corporate-action-events.pkl").open("rb") as stream:
            events = tuple(
                e for e in pickle.load(stream) if pd.Timestamp(e.trading_day) <= raw.date.max()
            )
        unresolved = json.loads((out / "execution-input-issues.json").read_text(encoding="utf8"))
        if unresolved:
            raise ValueError(
                "unresolved execution-input discrepancies; full ledger publication blocked: "
                + str(unresolved)
            )
        config = AppConfig(
            quantiles=1,
            rebalance_freq="daily",
            start_date=str(raw.date.min().date()),
            end_date=str(raw.date.max().date()),
            costs=CostsConfig(
                initial_capital=1_000_000,
                commission=0.0003,
                min_commission=5.0,
                slippage=0.0005,
                statutory_stamp_tax=True,
                max_position_weight=0.05,
                cash_buffer=0.0,
                max_holdings=0,
                participation_rate=0.1,
                retail_mode=False,
            ),
        )
        guards = json.loads((out / "execution-input-guards.json").read_text(encoding="utf8"))
        guards = tuple(
            g
            for g in guards
            if pd.Timestamp(g["available_at"]).tz_localize(None).normalize() <= raw.date.max()
        )
        statuses = pd.read_parquet(out / "source-trading-status.parquet").loc[
            raw.date.min() : raw.date.max()
        ]
        status_events = []
        for day, row in statuses.iterrows():
            for asset in row.index[row.eq(0)]:
                at = (day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)).tz_convert("UTC")
                status_events.append(
                    StatusEvent(
                        event_id=f"observed-suspension:{asset}:{day.date()}",
                        instrument_id=asset,
                        event_time=at.to_pydatetime(),
                        available_at=at.to_pydatetime(),
                        received_at=at.to_pydatetime(),
                        source="baostock.archived_daily_status",
                        trading_day=day.date(),
                        session_id=f"CN-A-SHARE:{day.date()}",
                        sequence=0,
                        status="suspended",
                        reason="observed at decision-day close; no future trading assumption",
                    )
                )
        kwargs = dict(
            catalog_path=out / "execution-catalog.csv",
            target_schedule={},
            allocation_schedule=plans,
            corporate_actions=events,
            status_events=tuple(status_events),
            input_guards=guards,
            account_id="barra-200-v3-" + identity[:16],
            strategy_id="research-" + method,
            snapshot_frequency="daily",
            cache_risk_views=True,
        )
        stopped = None
        try:
            replay = _replay(raw, config, identity, **kwargs)
        except ReplayInputError as exc:
            # The engine atomically rolls back an unsuccessful replay. Rebuild
            # the identical safe prefix solely as diagnostic evidence. Never
            # resume beyond the bad event or call this a completed full study.
            cutoff = (
                pd.Timestamp(exc.occurred_at)
                .tz_convert("Asia/Shanghai")
                .tz_localize(None)
                .normalize()
            )
            stopped = {
                "status": "stopped_on_execution_input",
                "asset": exc.instrument_id,
                "at": exc.occurred_at.isoformat(),
                "reason": str(exc),
                "requested_end": str(raw.date.max().date()),
                "diagnostic_prefix_end": str((raw.loc[raw.date < cutoff, "date"].max()).date()),
                "subsequent_nav_and_targets": "unavailable; no continuation or substituted account",
            }
            write_json(destination / "full-horizon-stop.json", stopped)
            raw = raw[raw.date < cutoff]
            kwargs["allocation_schedule"] = {
                day: p for day, p in plans.items() if pd.Timestamp(day) < cutoff
            }
            for key in ("corporate_actions", "status_events"):
                kwargs[key] = tuple(e for e in kwargs[key] if pd.Timestamp(e.trading_day) < cutoff)
            kwargs["input_guards"] = tuple(
                g
                for g in guards
                if pd.Timestamp(g["available_at"]).tz_localize(None).normalize() < cutoff
            )
            replay = _replay(raw, config, identity, **kwargs)
        for name, frame in replay.frames.items():
            frame.to_parquet(destination / (name + ".parquet"), index=False)
        write_json(destination / "allocation-decisions.json", list(replay.allocation_decisions))
        write_json(destination / "retry-diagnostics.json", list(replay.retry_diagnostics))
        write_json(destination / "risk-events.json", list(replay.runtime_risk_events))
        cross_predictions(replay, models, bundle).to_parquet(
            destination / "cross-predictions.parquet", index=False
        )
        decisions = replay.allocation_decisions
        successes = [r for r in decisions if "weights" in r]
        returns = replay.frames["returns"]
        nav = returns.nav_units.astype(float) / 10.0**returns.nav_scale
        seconds = [r["diagnostics"]["rebalance_seconds"] for r in successes]
        metrics = {
            "status": "completed" if stopped is None else "stopped_on_execution_input",
            "full_horizon_completed": stopped is None,
            "evaluated_through": str(raw.date.max().date()),
            "decisions": len(decisions),
            "published": len(successes),
            "failure_rate": 1 - len(successes) / max(len(decisions), 1),
            "net_return": float(nav.iloc[-1] / 1_000_000 - 1),
            "maximum_drawdown": float((nav / nav.cummax() - 1).min()),
            "rebalance_p95_seconds": float(np.quantile(seconds, 0.95)) if seconds else None,
            "wall_seconds": perf_counter() - started,
            "process_peak_bytes": peak_memory(),
            "model_promotion": "not_passed",
            **execution_metrics(replay, models, bundle, destination),
        }
        write_json(destination / "metrics.json", metrics)
        outputs = {
            p.name: digest(p)
            for p in destination.iterdir()
            if p.is_file() and p.name != "complete.json"
        }
        receipt = {"scenario": scenario_index, "identity": identity, "outputs": outputs, **metrics}
        write_json(destination / "complete.json", receipt)
        print(json.dumps(receipt | {"outputs": len(outputs)}), flush=True)
        return receipt
    except Exception as exc:
        failure = {
            "scenario": scenario_index,
            "identity": identity,
            "status": "failed",
            "reason": str(exc),
            "exception": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "wall_seconds": perf_counter() - started,
            "process_peak_bytes": peak_memory(),
            "published": False,
        }
        write_json(destination / "failure.json", failure)
        print(json.dumps(failure | {"traceback": "saved in failure.json"}), flush=True)
        return failure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument("--limit-dates", type=int)
    parser.add_argument("--debug-stacks", action="store_true")
    args = parser.parse_args()
    if args.debug_stacks:
        import faulthandler

        faulthandler.dump_traceback_later(120, repeat=True)
    result = run(args.out, args.scenario, limit_dates=args.limit_dates)
    if result["status"] == "failed":
        raise SystemExit(1)
    # Resumed verified artifacts need a log identity just like newly run accounts.
    print(json.dumps({"scenario": args.scenario, "identity": result["identity"],
                      "status": result["status"], "artifact_receipt": "complete.json"}), flush=True)
