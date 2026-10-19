"""Extend frozen v3 risk coverage while preserving its estimation pool and benchmark."""

import argparse
import json
import os
import pickle
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from exposure_v2_study import subset  # noqa: E402
from quant_risk_monitor.coverage import (  # noqa: E402
    RiskCoverageError,
    calendar_specific_variance,
    shrink_covered_variance,
)
from quant_risk_monitor.risk_estimators import estimate_specific_risk  # noqa: E402
from study_v3_archive import digest, write_json  # noqa: E402
from study_v3_risk import model_definitions  # noqa: E402


def extend_day(
    day,
    entry,
    panel,
    histories,
    definitions,
    statuses,
    *,
    lifecycle=None,
    held_exposure_carry_sessions=0,
):
    assets = list(entry["assets"])
    result = {"assets": assets, "benchmark": entry["benchmark"].copy(), "models": {}}
    i = panel["dates"].get_loc(day)
    window = panel["dates"][i - 252 : i]
    records = []
    for model, data in entry["models"].items():
        variant, config, _ = definitions[model]
        snapshots = histories["snapshots"][variant]
        snapshot = snapshots[day]
        residuals = histories["histories"][variant][config.regression_weighting]["residuals"]
        core = subset(snapshot, assets)
        _, prior = estimate_specific_risk(
            {a: residuals[a].reindex(window).to_list() for a in assets},
            core,
            config,
            252,
        )
        raw = pd.Series(prior["raw_period_variances"])
        regime = prior["regime"].get("variance_multiplier", 1.0)
        rows = [data["exposures"]]
        specific = [pd.Series(data["D_daily_rate"], index=data["exposures"].index)]
        coverage = {
            a: {
                "status": "direct_complete",
                "valid_observations": 252,
                "exposure_date": str(day.date()),
                "proxy": False,
                "source": "unchanged_frozen_v3",
            }
            for a in assets
        }
        for asset in sorted(set(panel["assets"]) - set(assets)):
            info = {"status": "unavailable", "asset": asset, "proxy": False}
            state = None if lifecycle is None else lifecycle.at[day, asset]
            info["lifecycle_state"] = state
            marks = panel["prices"]["close"].loc[:day, asset].dropna()
            info["valuation_date"] = str(marks.index[-1].date()) if len(marks) else None
            info["valuation_source"] = "archived_raw_close; stale values are not returns"
            exposure = None
            if state == "exited_a_share":
                info.update(
                    reason="exited_a_share_without_current_risk_evidence",
                    recovery_condition="evidenced post-exit valuation and risk model",
                    trading_status="exited_a_share",
                )
                coverage[asset] = info
                continue
            if asset in snapshot.values.index:
                exposure = snapshot.values.loc[[asset]]
                info.update(exposure_date=str(day.date()), exposure_status="direct")
            elif statuses.at[day, asset] == 0:
                # The last *published* complete exposure is a disclosed stale
                # estimate for an evidenced suspended holding, never a return.
                for previous in sorted((d for d in snapshots if d < day), reverse=True):
                    if asset in snapshots[previous].values.index:
                        exposure = snapshots[previous].values.loc[[asset]]
                        info.update(
                            exposure_date=str(previous.date()),
                            exposure_status="stale_confirmed_suspension",
                            trading_status="suspended",
                            valuation_is_trade=False,
                            source="frozen_PIT_exposure_and_archived_daily_status",
                        )
                        break
            elif held_exposure_carry_sessions:
                from held_risk_policy import held_exposure_candidate

                exposure, extra = held_exposure_candidate(
                    day,
                    asset,
                    panel,
                    snapshots,
                    statuses,
                    max_sessions=held_exposure_carry_sessions,
                )
                info.update(extra)
                if exposure is not None:
                    previous = pd.Timestamp(info["exposure_date"])
            if exposure is None:
                info.setdefault("reason", "missing_reliable_current_exposure")
            elif set(exposure.columns) != set(data["exposures"].columns):
                info["reason"] = "factor_basis_changed"
            elif asset not in residuals:
                info["reason"] = "no_observed_residual_history"
            else:
                try:
                    variance, diagnostic = calendar_specific_variance(
                        residuals[asset],
                        panel["dates"],
                        window[-1],
                        half_life=config.specific_half_life,
                        lags=config.specific_newey_west_lags,
                    )
                    peers = list(raw.index)
                    if config.specific_target == "industry":
                        group = (
                            snapshot.industries.get(asset)
                            if asset in snapshot.values.index
                            else snapshots[previous].industries.get(asset)
                        )
                        peers = list(core.industries[core.industries.eq(group)].index)
                        if len(peers) < config.specific_min_group_assets - 1:
                            if config.specific_group_fallback == "error":
                                raise RiskCoverageError("insufficient_industry_peers", diagnostic)
                            peers = list(raw.index)
                            diagnostic["industry_fallback"] = "global_frozen_pool"
                    variance, diagnostic = shrink_covered_variance(
                        variance,
                        diagnostic,
                        target=float(raw.loc[peers].mean()),
                        config=config,
                        regime_multiplier=regime,
                    )
                    info.update(diagnostic)
                    rows.append(exposure.reindex(columns=data["exposures"].columns))
                    specific.append(pd.Series({asset: variance}))
                except RiskCoverageError as exc:
                    info.update(exc.diagnostics, status="unavailable", reason=exc.reason)
            coverage[asset] = info
        x = pd.concat(rows)
        d = pd.concat(specific).reindex(x.index)
        # No change whatsoever to the old, complete-history subproblem.
        pd.testing.assert_frame_equal(x.loc[assets], data["exposures"])
        np.testing.assert_array_equal(d.loc[assets], data["D_daily_rate"])
        result["models"][model] = {
            **data,
            "exposures": x,
            "D_daily_rate": d.to_numpy(),
            "coverage": coverage,
            "coverage_version": "held-coverage-v2"
            if lifecycle is not None or held_exposure_carry_sessions
            else "calendar-partial-v1",
        }
        records.extend(
            {"date": str(day.date()), "model": model, "asset": a, **info}
            for a, info in coverage.items()
        )
    return result, records


def run(source, out, *, lifecycle_events=None, held_exposure_carry_sessions=0):
    if type(held_exposure_carry_sessions) is not int or not 0 <= held_exposure_carry_sessions <= 5:
        raise ValueError("held exposure carry must be between 0 and 5 sessions")
    out.mkdir(parents=True, exist_ok=True)
    target = out / "portfolio-risk-models.pkl"
    if target.exists():
        raise ValueError("coverage cache is immutable; select a new output run")
    with (source / "panel.pkl").open("rb") as stream:
        panel = pickle.load(stream)["panel"]
    with (source / "histories.pkl").open("rb") as stream:
        histories = pickle.load(stream)
    with (source / "portfolio-risk-models.pkl").open("rb") as stream:
        original = pickle.load(stream)
    statuses = pd.read_parquet(source / "source-trading-status.parquet")
    lifecycle = None
    if lifecycle_events is not None:
        from recovery_lifecycle import state_overlay

        lifecycle = state_overlay(statuses, json.loads(lifecycle_events.read_text(encoding="utf8")))
    definitions = model_definitions(json.loads((source / "protocol.json").read_text()))
    models, records = {}, []
    for day, entry in original.items():
        models[day], rows = extend_day(
            day,
            entry,
            panel,
            histories,
            definitions,
            statuses,
            lifecycle=lifecycle,
            held_exposure_carry_sessions=held_exposure_carry_sessions,
        )
        records.extend(rows)
        print(
            json.dumps(
                {
                    "date": str(day.date()),
                    "phase": "coverage",
                    "original_assets": len(entry["assets"]),
                    "new_assets": len(models[day]["models"]["repaired_ols"]["exposures"]),
                }
            ),
            flush=True,
        )
    with target.open("wb") as stream:
        pickle.dump(models, stream, protocol=5)
    # JSON-lines preserves nested per-lag diagnostics without lossy stringification.
    with (out / "risk-coverage.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
    write_json(
        out / "coverage-binding.json",
        {
            "version": "held-coverage-v2"
            if lifecycle_events or held_exposure_carry_sessions
            else "calendar-partial-v1",
            "held_exposure_carry_sessions": held_exposure_carry_sessions,
            "lifecycle_events": None
            if lifecycle_events is None
            else {"path": str(lifecycle_events), "sha256": digest(lifecycle_events)},
            "source": str(source),
            "inputs": {
                name: digest(source / name)
                for name in (
                    "panel.pkl",
                    "histories.pkl",
                    "portfolio-risk-models.pkl",
                    "source-trading-status.parquet",
                )
            },
            "producer": digest(__file__),
            "policy_modules": {
                name: digest(Path(__file__).with_name(name))
                for name in ("held_risk_policy.py", "recovery_lifecycle.py")
            },
            "code_modules": {
                module.__name__: digest(module.__file__)
                for module in (
                    __import__("quant_risk_monitor.coverage", fromlist=["*"]),
                    __import__("quant_risk_monitor.risk_estimators", fromlist=["*"]),
                )
            },
            "output": digest(target),
            "benchmark_preserved": True,
            "investment_pool_preserved": True,
            "complete_history_numerics_preserved": True,
            "default_model_changed": False,
            "stale_exposures": "explicit confirmed-suspension estimate, never observed return",
            "industry_proxy_used": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lifecycle-events", type=Path)
    parser.add_argument("--held-exposure-carry-sessions", type=int, default=0)
    args = parser.parse_args()
    run(
        args.source,
        args.out,
        lifecycle_events=args.lifecycle_events,
        held_exposure_carry_sessions=args.held_exposure_carry_sessions,
    )
