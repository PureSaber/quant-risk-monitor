from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from quant_risk_monitor.analytics import (
    factor_exposure_coverage,
    factor_exposures,
    historical_var_cvar,
    liquidity_days_to_exit,
    parametric_var_cvar,
    return_history_coverage,
    risk_contributions,
    shrink_covariance,
    stress_test,
)
from quant_risk_monitor.models import Alert, CheckResult, Severity
from quant_risk_monitor.readers.equity import (
    load_capital_curve,
    load_holdings_weights,
    load_spread_nav,
)
from quant_risk_monitor.rules.concentration import check_single_name_weight
from quant_risk_monitor.rules.drawdown import run_nav_rules


def _merge_results(*results: CheckResult) -> CheckResult:
    alerts = []
    metrics = {}
    for r in results:
        alerts.extend(r.alerts)
        metrics.update(r.metrics)
    return CheckResult(alerts=alerts, metrics=metrics)


def _run_advanced_checks(config_path: Path, cfg: dict) -> CheckResult:
    advanced = cfg.get("advanced") or {}
    if not advanced:
        return CheckResult()
    alerts: list[Alert] = []
    metrics: dict = {}

    returns_cfg = advanced.get("returns") or {}
    if returns_cfg.get("path"):
        returns = pd.read_csv(_resolve_config_path(config_path, str(returns_cfg["path"])))
        column = str(returns_cfg.get("column", "net_return"))
        tail = historical_var_cvar(returns[column], float(returns_cfg.get("confidence", 0.95)))
        parametric_tail = parametric_var_cvar(returns[column], tail.confidence)
        metrics["tail_risk"] = {
            "confidence": tail.confidence,
            "var": tail.var,
            "cvar": tail.cvar,
            "observations": tail.observations,
            "parametric": {
                "var": parametric_tail.var,
                "cvar": parametric_tail.cvar,
                "observations": parametric_tail.observations,
            },
        }
        limits = cfg.get("rules") or {}
        for name, value in (("var", tail.var), ("cvar", tail.cvar)):
            limit = limits.get(f"{name}_limit")
            if limit is not None and value > float(limit):
                alerts.append(
                    Alert(
                        rule_id=f"tail_{name}",
                        severity=Severity.CRITICAL,
                        message=f"{name.upper()} {value:.2%} exceeds {float(limit):.2%}",
                        details={"value": value, "limit": float(limit)},
                    )
                )
        for name, value in (
            ("parametric_var", parametric_tail.var),
            ("parametric_cvar", parametric_tail.cvar),
        ):
            limit = limits.get(f"{name}_limit")
            if limit is not None and value > float(limit):
                alerts.append(
                    Alert(
                        rule_id=f"tail_{name}",
                        severity=Severity.CRITICAL,
                        message=f"{name.upper()} {value:.2%} exceeds {float(limit):.2%}",
                        details={"value": value, "limit": float(limit)},
                    )
                )

    positions_cfg = advanced.get("positions") or {}
    weights = pd.Series(dtype=float)
    positions = pd.DataFrame()
    if positions_cfg.get("path"):
        positions = pd.read_csv(
            _resolve_config_path(config_path, str(positions_cfg["path"])),
            dtype={"symbol": "string"},
            keep_default_na=False,
        )
        if "date" in positions.columns:
            positions = positions[positions["date"] == positions["date"].max()]
        required_columns = {"symbol", "weight"}
        missing_columns = sorted(required_columns.difference(positions.columns))
        invalid_reason = None
        if missing_columns:
            invalid_reason = f"positions are missing columns: {missing_columns}"
        else:
            symbols = positions["symbol"].astype(str).str.strip()
            parsed_weights = pd.to_numeric(positions["weight"], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            if symbols.eq("").any() or symbols.duplicated().any():
                invalid_reason = "position symbols must be unique and non-empty"
            elif parsed_weights.isna().any():
                invalid_reason = "position weights must be finite"
            else:
                positions["symbol"] = symbols
                weights = pd.Series(parsed_weights.to_numpy(), index=symbols, dtype=float)
        if invalid_reason is not None:
            if (advanced.get("liquidity") or {}).get("path"):
                raise ValueError(invalid_reason)
            metrics["positions_input"] = {
                "status": "not_evaluable",
                "reason": invalid_reason,
            }
            alerts.append(
                Alert(
                    rule_id="positions_input_not_evaluable",
                    severity=Severity.CRITICAL,
                    message="portfolio positions cannot be evaluated",
                    details={"reason": invalid_reason},
                )
            )
            return CheckResult(alerts=alerts, metrics=metrics)

    scenarios_cfg = advanced.get("scenarios") or {}
    if scenarios_cfg.get("path") and not weights.empty:
        scenarios = pd.read_csv(
            _resolve_config_path(config_path, str(scenarios_cfg["path"])), index_col=0
        )
        stressed = stress_test(weights, scenarios)
        metrics["stress"] = stressed.to_dict(orient="index")
        limit = (cfg.get("rules") or {}).get("stress_loss_limit")
        if limit is not None and float(stressed["loss"].max()) > float(limit):
            alerts.append(
                Alert(
                    rule_id="stress_loss",
                    severity=Severity.CRITICAL,
                    message="Stress loss exceeds configured limit",
                    details={
                        "worst_loss": float(stressed["loss"].max()),
                        "limit": float(limit),
                    },
                )
            )

    liquidity_cfg = advanced.get("liquidity") or {}
    if liquidity_cfg.get("path"):
        if positions.empty or "market_value" not in positions.columns:
            raise ValueError("liquidity requires non-empty positions with market_value")
        liquidity = pd.read_csv(
            _resolve_config_path(config_path, str(liquidity_cfg["path"])),
            dtype={"symbol": "string"},
            keep_default_na=False,
        )
        adv_column = str(liquidity_cfg.get("adv_column", "average_daily_value"))
        if not {"symbol", adv_column}.issubset(liquidity.columns):
            raise ValueError("liquidity requires symbol and ADV columns")
        liquidity["symbol"] = liquidity["symbol"].str.strip()
        liquidity = liquidity.set_index("symbol")
        market_values = positions.set_index("symbol")["market_value"]
        report = liquidity_days_to_exit(
            market_values,
            liquidity[adv_column],
            max_participation=float(liquidity_cfg.get("max_participation", 0.1)),
        )
        metrics["liquidity"] = report.reset_index().to_dict(orient="records")
        limit = (cfg.get("rules") or {}).get("max_days_to_exit")
        if limit is not None and (not np.isfinite(float(limit)) or float(limit) < 0):
            raise ValueError("max_days_to_exit must be finite and non-negative")
        if limit is not None and float(report["days_to_exit"].max()) > float(limit):
            alerts.append(
                Alert(
                    rule_id="liquidity_days_to_exit",
                    severity=Severity.CRITICAL,
                    message="Portfolio cannot be liquidated within configured horizon",
                    details={
                        "max_days": float(report["days_to_exit"].max()),
                        "limit": float(limit),
                    },
                )
            )

    asset_returns_cfg = advanced.get("asset_returns") or {}
    if asset_returns_cfg.get("path") and not weights.empty:
        asset_returns = pd.read_csv(
            _resolve_config_path(config_path, str(asset_returns_cfg["path"]))
        )
        date_column = str(asset_returns_cfg.get("date_column", "date"))
        if date_column in asset_returns:
            asset_returns = asset_returns.drop(columns=date_column)
        active_weights = weights[weights != 0]
        aligned_returns = asset_returns.reindex(columns=active_weights.index)
        min_observations = int(asset_returns_cfg.get("min_observations", 2))
        try:
            coverage = return_history_coverage(aligned_returns, min_observations=min_observations)
        except (TypeError, ValueError) as exc:
            coverage_details = {"complete": False, "reason": str(exc)}
        else:
            coverage_details = coverage.to_dict()
        metrics["risk_contribution_coverage"] = coverage_details
        if not coverage_details["complete"]:
            metrics["risk_contributions"] = {"status": "not_evaluable"}
            alerts.append(
                Alert(
                    rule_id="risk_contributions_not_evaluable",
                    severity=Severity.CRITICAL,
                    message="asset covariance cannot be estimated from the supplied returns",
                    details=coverage_details,
                )
            )
        else:
            covariance = shrink_covariance(
                aligned_returns,
                shrinkage=float(asset_returns_cfg.get("shrinkage", 0.2)),
                annualization=int(asset_returns_cfg.get("annualization", 252)),
                min_observations=min_observations,
            )
            contribution = risk_contributions(active_weights, covariance)
            metrics["risk_contributions"] = contribution.reset_index(names="symbol").to_dict(
                orient="records"
            )

    factor_cfg = advanced.get("factor_exposures") or {}
    if factor_cfg.get("path") and not weights.empty:
        symbol_column = str(factor_cfg.get("symbol_column", "symbol"))
        factor_frame = pd.read_csv(
            _resolve_config_path(config_path, str(factor_cfg["path"])),
            dtype={symbol_column: "string"},
            keep_default_na=False,
        )
        if symbol_column not in factor_frame.columns:
            coverage_details = {
                "complete": False,
                "reason": f"factor exposures are missing symbol column: {symbol_column}",
            }
        else:
            factor_frame = factor_frame.set_index(symbol_column)
            try:
                coverage = factor_exposure_coverage(weights, factor_frame)
            except (TypeError, ValueError) as exc:
                coverage_details = {"complete": False, "reason": str(exc)}
            else:
                coverage_details = coverage.to_dict()
        metrics["factor_exposure_coverage"] = coverage_details
        if not coverage_details["complete"]:
            metrics["factor_exposures"] = {"status": "not_evaluable"}
            alerts.append(
                Alert(
                    rule_id="factor_exposures_not_evaluable",
                    severity=Severity.CRITICAL,
                    message="factor exposures are incomplete for non-zero holdings",
                    details=coverage_details,
                )
            )
        else:
            metrics["factor_exposures"] = {
                str(name): float(value)
                for name, value in factor_exposures(weights, factor_frame).items()
            }
    return CheckResult(alerts=alerts, metrics=metrics)


def _resolve_config_path(config_path: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    config_dir = config_path.parent
    for base in (config_dir, config_dir.parent):
        candidate = (base / path).resolve()
        if candidate.is_file():
            return candidate
    return (config_dir / path).resolve()


def run_check(config_path: Path) -> CheckResult:
    config_path = config_path.resolve()
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    results: list[CheckResult] = []
    nav_cfg = cfg.get("nav") or {}
    if nav_cfg.get("path"):
        path = _resolve_config_path(config_path, str(nav_cfg["path"]))
        source = str(nav_cfg.get("source", "equity"))
        if source == "equity":
            nav = load_capital_curve(path, str(nav_cfg.get("column", "ols")))
        else:
            nav = load_spread_nav(path, str(nav_cfg.get("column", "nav")))
        results.append(run_nav_rules(nav, cfg.get("rules") or {}))

    holdings_cfg = cfg.get("holdings") or {}
    if holdings_cfg.get("path"):
        weights = load_holdings_weights(
            _resolve_config_path(config_path, str(holdings_cfg["path"])),
            str(holdings_cfg.get("symbol_col", "symbol")),
            str(holdings_cfg.get("weight_col", "weight")),
        )
        max_weight = float((cfg.get("rules") or {}).get("single_name_weight", 0.3))
        alerts = check_single_name_weight(weights, max_weight)
        results.append(CheckResult(alerts=alerts))

    results.append(_run_advanced_checks(config_path, cfg))
    return _merge_results(*results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check portfolio risk rules")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Run risk checks")
    check.add_argument("--config", required=True)
    check.add_argument("--out", required=True)
    calibration = sub.add_parser("validate-forecasts", help="Evaluate causal factor-risk forecasts")
    calibration.add_argument("--input", required=True)
    calibration.add_argument("--out", required=True)
    verify = sub.add_parser("verify-forecasts", help="Recompute a bound forecast evaluation")
    verify.add_argument("--run", required=True)
    overlap = sub.add_parser(
        "portfolio-overlap", help="Build a PIT multi-strategy look-through overlap report"
    )
    overlap.add_argument("--input", required=True)
    overlap.add_argument("--out", required=True)
    overlap.add_argument(
        "--risk-run", help="Verified forecast run to summarize without changing its scores"
    )
    args = parser.parse_args(argv)

    if args.command == "portfolio-overlap":
        from quant_risk_monitor.lookthrough import run_portfolio_overlap

        try:
            result = run_portfolio_overlap(args.input, args.out, risk_run=args.risk_run)
        except (ValueError, TypeError, KeyError, OSError, ArithmeticError) as exc:
            print(json.dumps({"status": "unavailable", "reason": str(exc)}, ensure_ascii=False))
            raise SystemExit(2) from exc
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "strategies": len(result["strategies"]),
                    "pairwise_available": sum(
                        item["status"] == "available" for item in result["pairwise"]
                    ),
                    "pairwise_unavailable": sum(
                        item["status"] == "unavailable" for item in result["pairwise"]
                    ),
                    "risk_forecast_attached": result["risk_forecast"] is not None,
                }
            )
        )
        return

    if args.command != "check":
        from quant_risk_monitor.risk_validation import (
            readable_risk_forecast_summary,
            run_risk_validation,
            verify_risk_validation,
        )

        try:
            result = (
                run_risk_validation(args.input, args.out)
                if args.command == "validate-forecasts"
                else verify_risk_validation(args.run)
            )
        except (ValueError, TypeError, KeyError, OSError, ArithmeticError) as exc:
            print(json.dumps({"status": "unavailable", "reason": str(exc)}, ensure_ascii=False))
            raise SystemExit(2) from exc
        summary = readable_risk_forecast_summary(
            result,
            verification_status=(
                "verified_by_hash_and_full_recomputation"
                if args.command == "verify-forecasts"
                else "evaluated_complete"
            ),
        )
        print(json.dumps(summary, allow_nan=False))
        return

    invalid_input = False
    try:
        result = run_check(Path(args.config))
        serialized = json.dumps(result.to_dict(), indent=2, allow_nan=False)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        invalid_input = True
        result = CheckResult(
            alerts=[
                Alert(
                    rule_id="input_data_invalid",
                    severity=Severity.CRITICAL,
                    message=f"Risk is not evaluable: {exc}",
                    details={"error_type": type(exc).__name__},
                )
            ],
            metrics={"evaluation_status": "unavailable"},
        )
        serialized = json.dumps(result.to_dict(), indent=2, allow_nan=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(serialized, encoding="utf-8")
    print(f"wrote {out_path} alerts={len(result.alerts)} critical={result.has_critical}")
    if invalid_input:
        raise SystemExit(2)
    if result.has_critical:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
