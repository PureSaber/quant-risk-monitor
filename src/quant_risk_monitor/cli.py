from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from quant_risk_monitor.models import CheckResult
from quant_risk_monitor.readers.equity import (
    load_capital_curve,
    load_holdings_weights,
    load_spread_nav,
)
from quant_risk_monitor.rules.concentration import check_single_name_weight
from quant_risk_monitor.rules.drawdown import run_nav_rules


def _merge_results(*results: CheckResult) -> CheckResult:
    alerts = []
    for r in results:
        alerts.extend(r.alerts)
    return CheckResult(alerts=alerts)


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

    return _merge_results(*results) if results else CheckResult()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check portfolio risk rules")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Run risk checks")
    check.add_argument("--config", required=True)
    check.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    result = run_check(Path(args.config))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"wrote {out_path} alerts={len(result.alerts)} critical={result.has_critical}")
    if result.has_critical:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
