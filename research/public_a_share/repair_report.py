"""Reconcile versioned failure repairs without rewriting frozen accounts."""

import argparse
import html
import json
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd
from repair_account_audit import verify
from repair_failure_audit import frozen_accounts

from study_v3_archive import digest, write_json
from study_v3_cross_audit import recompute_cross
from study_v3_report import batch_results, facts_table, page, read, table
from study_v3_risk import paired_summary


def cross_metrics(frame):
    """Keep missing forecasts in the paired date filter, not just finite rows."""
    if frame.empty:
        return {}, pd.DataFrame()
    consistency = frame.groupby("portfolio_id")[[
        "actual_frozen_weight_return", "active_frozen_weight_return"
    ]].nunique()
    if (consistency > 1).any().any():
        raise ValueError("predictors saw different realized returns for the same portfolio")
    summaries, calibrations = {}, []
    for objective, variance, outcome, loss in (
        ("absolute", "variance", "actual_frozen_weight_return", "qlike"),
        ("active", "active_variance", "active_frozen_weight_return", "active_qlike"),
    ):
        evaluation = frame.copy()
        evaluation["qlike"] = evaluation[loss]
        finite = evaluation.dropna(subset=[loss]).copy()
        finite["standardized_square"] = finite[outcome] ** 2 / finite[variance]
        calibration = finite.groupby(["generator", "predictor", "target_kind"]).agg(
            forecasts=("portfolio_id", "nunique"),
            mean_qlike=(loss, "mean"),
            mean_squared_standardized_return=("standardized_square", "mean"),
        ).reset_index()
        calibration["risk_objective"] = objective
        calibrations.append(calibration)
        summaries[objective] = {}
        for kind in ("continuous", "integer", "actual_close"):
            selected = evaluation[evaluation.target_kind.eq(kind)].copy()
            selected = selected[~selected[variance].eq(0)]
            selected = selected.rename(columns={"predictor": "model"})
            selected["family"] = selected.scenario.astype(str)
            names = sorted(selected.model.unique())
            summaries[objective][kind] = (
                paired_summary(selected, names, "legacy_ols") if "legacy_ols" in names else {}
            )
    return summaries, pd.concat(calibrations, ignore_index=True)


def old_integer_audit(source, audit, out, models, bars, statuses):
    summary = read(audit / "integer-summary.json")
    folders = dict(frozen_accounts(source))
    records = []
    for path in sorted(audit.glob("integer-*.json")):
        if path.name == "integer-summary.json":
            continue
        row = read(path)
        result = {k: row[k] for k in ("scenario", "date", "status")}
        result.update(source_file=str(path), source_sha256=digest(path),
                      integer_infeasibility_proven=False)
        if row["status"] == "repaired":
            replacement = {
                "session": row["date"], "weights": row["continuous"],
                "targets": row["quantities"], "diagnostics": row["diagnostics"],
            }
            checked = verify(folders[row["scenario"]], models, bars, statuses,
                             replacement=replacement)
            result["independent_check"] = checked["independent_target_checks"][0]
        else:
            diagnostic = row.get("diagnostics", {})
            result["failure_stage"] = diagnostic.get("failure_stage", "integer_repair")
            attempts = diagnostic.get("repair_attempts", [])
            result["search_termination"] = [a.get("termination") for a in attempts]
            result["best_unverified_residual"] = min(
                (a["residual"] for a in attempts if "residual" in a), default=None
            )
            result["explanation"] = (
                "time budget exhausted; no integer infeasibility proof"
                if row["status"] == "timeout" else
                "bounded search did not verify a feasible integer target; no infeasibility proof"
            )
        records.append(result)
    assert len(records) == summary["total"] == 102
    assert dict(Counter(row["status"] for row in records)) == summary["counts"]
    witnesses = [r for r in records if r["scenario"] in (41, 46) and r["date"] == "2024-01-05"]
    assert len(witnesses) == 2 and all(r["status"] == "repaired" for r in witnesses)
    write_json(out / "102-integer-independent-checks.json", records)
    pd.DataFrame([{k: v for k, v in r.items() if k != "independent_check"}
                  for r in records]).to_csv(out / "102-integer-results.csv", index=False)
    return summary


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "acceptance.json").exists():
        raise ValueError("completed reports are immutable; choose another output version")
    binding = read(args.coverage / "coverage-binding.json")
    if digest(args.coverage / "portfolio-risk-models.pkl") != binding["output"]:
        raise ValueError("expanded risk cache changed")
    for name, value in binding["inputs"].items():
        if digest(args.source / name) != value:
            raise ValueError("frozen input changed: " + name)
    with (args.coverage / "portfolio-risk-models.pkl").open("rb") as stream:
        models = pickle.load(stream)
    with (args.source / "portfolio-risk-models.pkl").open("rb") as stream:
        old_models = pickle.load(stream)
    with (args.source / "panel.pkl").open("rb") as stream:
        panel = pickle.load(stream)["panel"]
    bars = pd.read_parquet(args.source / "execution-bars.parquet")
    statuses = pd.read_parquet(args.source / "source-trading-status.parquet")
    old_results = batch_results(args.source, "cadence-03", allow_partial=False)
    old_by_id = {r["scenario"]: r for r in old_results}
    checks, records, failures, branches, cross_readouts = {}, [], [], {}, []
    for label, root, batch in args.branch:
        rows = batch_results(Path(root), batch, allow_partial=label != "strict" or args.partial)
        if not args.partial and any("outputs" not in r for r in rows):
            raise ValueError("worker failed before writing verifiable outputs")
        branch_cross = []
        for row in rows:
            if "outputs" not in row:
                continue
            folder = Path(row["folder"])
            spec = row["specification"]
            print(json.dumps({"branch": label, "scenario": row["scenario"], "phase": "audit"}), flush=True)
            checked = verify(folder, models, bars, statuses)
            checks[f"{label}:{row['scenario']}"] = checked
            decisions = read(folder / "allocation-decisions.json")
            decision_source = folder / "allocation-decisions.json"
            if not decision_source.exists():
                decision_source = decision_source.with_suffix(".json.gz")
            for decision in decisions:
                if "weights" not in decision:
                    failures.append({
                        "branch": label, "scenario": row["scenario"], "date": decision["session"],
                        "status": decision.get("status"), "reason": decision.get("error"),
                        "source": str(decision_source),
                    })
            if (folder / "full-horizon-stop.json").exists():
                stop = read(folder / "full-horizon-stop.json")
                failures.append({"branch": label, "scenario": row["scenario"],
                                 "date": stop["at"], "status": "execution_input_stop",
                                 "reason": json.dumps(stop, ensure_ascii=False),
                                 "source": str(folder / "full-horizon-stop.json")})
            original = old_by_id[row["scenario"]]
            record = {k: row.get(k) for k in (
                "scenario", "status", "full_horizon_completed", "evaluated_through", "published",
                "decisions", "net_return", "maximum_drawdown", "actual_annualized_tracking_error",
                "actual_te_complete_windows", "total_fees", "gross_traded_notional_over_initial_capital",
                "rebalance_p95_seconds", "wall_seconds", "process_peak_bytes", "filled_orders",
                "partially_filled_orders", "longest_no_target_rebalances", "effective_rebalance_rate",
            )}
            record.update(branch=label, model=spec["model"], method=spec["method"],
                          budget=spec["budget"], te=spec.get("te"), folder=str(folder),
                          old_published=original.get("published"), old_status=original["status"],
                          old_evaluated_through=original.get("evaluated_through"),
                          old_net_return=original.get("net_return"),
                          return_endpoints_match=row.get("evaluated_through") == original.get("evaluated_through"))
            records.append(record)
            frame = recompute_cross(decisions, pd.read_parquet(folder / "positions.parquet"),
                                    pd.read_parquet(folder / "portfolio_snapshots.parquet"), models, panel)
            if len(frame):
                frame["scenario"] = row["scenario"]
                frame["generator"] = spec["model"]
                frame["method"] = spec["method"]
                branch_cross.append(frame)
        if branch_cross:
            merged = pd.concat(branch_cross, ignore_index=True)
            merged.to_parquet(out / f"{label}-cross-predictions.parquet", index=False)
            merged.groupby(["scenario", "predictor", "target_kind", "status"]).size().rename(
                "forecasts").reset_index().to_csv(out / f"{label}-cross-coverage.csv", index=False)
            inference, calibration = cross_metrics(merged)
            write_json(out / f"{label}-cross-inference.json", inference)
            calibration.to_csv(out / f"{label}-risk-underestimation.csv", index=False)
            for objective, kinds in inference.items():
                for kind, result in kinds.items():
                    selected = result.get("comparisons", {}).get("enhanced_s0.1_f90_d60_p10", {})
                    cross_readouts.append({
                        "分支": label, "风险": objective, "组合": kind,
                        "共同完整日期": result.get("complete_dates", 0),
                        "冻结增强模型相对旧OLS平均损失差": selected.get("mean_delta"),
                        "多重校正95%区间": str(selected.get("simultaneous_95_interval")),
                        "推断状态": result.get("inference_status", "unavailable"),
                    })
        branches[label] = {
            "scenarios": sorted(row["scenario"] for row in rows),
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "complete_horizon": sum(bool(row.get("full_horizon_completed")) for row in rows),
            "zero_targets": sum(row.get("published") == 0 for row in rows),
            "published_targets": sum(row.get("published", 0) for row in rows),
            "scheduled_attempts": sum(row.get("decisions", 0) for row in rows),
        }
    accounts = pd.DataFrame(records)
    accounts.to_csv(out / "portfolio-before-after.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).to_csv(out / "new-failure-inventory.csv", index=False, encoding="utf-8-sig")
    write_json(out / "account-independent-checks.json", checks)
    required_sensitive = {r["scenario"] for r in old_results if not r.get("full_horizon_completed")}
    required_sensitive.update(accounts.loc[
        accounts.branch.eq("strict") & ~accounts.full_horizon_completed, "scenario"
    ].tolist())
    sensitivities_complete = all(required_sensitive <= set(branches.get(label, {}).get("scenarios", []))
                                 for label in ("sensitivity_floor", "sensitivity_ceil"))
    integer_summary = old_integer_audit(args.source, args.integer_audit, out, old_models, bars, statuses)
    evidence = read(args.evidence / "data-failure-evidence.json")
    certificates = read(args.certificates / "constraint-certificates.json")
    acceptance = {
        "engineering_research_delivery": "partial" if args.partial or not sensitivities_complete
        else "complete_with_explicit_unresolved_cases",
        "branches": branches, "required_sensitivity_scenarios": sorted(required_sensitive),
        "sensitivity_coverage_complete": sensitivities_complete,
        "original_integer_failures": integer_summary["counts"],
        "data_failures": evidence, "constraint_certificates": certificates,
        "all_published_targets_independently_verified": True,
        "model_promotion": "not_passed", "independent_new_windows": 0,
        "minimum_independent_windows": 80, "default_v1_changed": False,
        "frozen_accounts_changed": False, "investment_constraints_relaxed": False,
        "live_trading_connected": False, "producer_sha256": digest(__file__),
        "coverage_binding": binding,
        "report_scope": "retrospective; coverage improvement is not predictive-quality acceptance",
    }
    write_json(out / "acceptance.json", acceptance)
    display = accounts[["branch", "scenario", "method", "model", "budget", "te", "status",
                        "old_published", "published", "decisions", "effective_rebalance_rate",
                        "longest_no_target_rebalances", "evaluated_through", "net_return",
                        "actual_annualized_tracking_error", "total_fees", "rebalance_p95_seconds"]]
    cross_links = "".join(
        f'<li>{html.escape(label)}：<a href="{label}-cross-predictions.parquet">逐期预测</a> · '
        f'<a href="{label}-cross-coverage.csv">覆盖与缺失</a> · '
        f'<a href="{label}-cross-inference.json">区块推断</a> · '
        f'<a href="{label}-risk-underestimation.csv">生成模型×预测模型校准</a></li>'
        for label in branches if (out / f"{label}-cross-predictions.parquet").exists()
    )
    body = f"""<p class="note">工程研究状态：{html.escape(acceptance['engineering_research_delivery'])}。
模型推广未通过，新增独立窗口为 0；默认 v1、旧账户、基准和全部投资约束保持原样。
目标生成率与成交完成率分开；净收益含目标失败后的原持仓延续，不代表持续正常调仓。</p>
<h2>冻结失败的逐项结论</h2>{facts_table(integer_summary['counts'])}{facts_table(evidence)}
<p>原 26 个全期连续不可行场景、5,551 次结构性约束冲突与 14 次换手冲突，使用原风险输入独立复算。
诊断中的放松问题仅用于证明约束下界，不发布为交易目标。整数失败保留搜索失败或超时，没有无解证明。</p>
<h2>主线与敏感性分支</h2>{facts_table(branches)}
<p>严格主线遇到无法唯一确定的账户权益继续停止。600763 的零碎股分配依赖登记机构全体账户排序；
敏感性分支分别取公告规则允许的下、上整数结果。603007 最终方案明确舍去小数，但公告比例近似，
虚拟账户最终登记数量及实际到账仍不能凭空确认；两个敏感性分支均按该规则取下整数，到账日是假设。</p>
<h2>逐场景实施结果</h2><div class="scroll">{table(display)}</div>
<p>仅终点相同的账户可以比较同区间净收益。3%/6% 限制是预测 TE，实际 TE 是随后已实现收益的统计量。
首次目标计算全部成本，此后换手上限 20%；独立核验现金、最低佣金、印花税、整手、零股卖出及 T+1。
P95 为成功发布目标的端到端调仓时间；所有失败及超时留在逐项清单，未记录的耗时不估造。</p>
<h2>风险交叉预测与边界</h2><p>每个连续目标、整数目标及实际持仓由相同 22 个模型交叉预测，
同一组合的实际五日收益保持一致。缺报价或风险输入保留缺失，不补零、不重新归一化。
推断按日期联合区块重采样 5,000 次、每块十日并校正多重比较；少于 20 个共同日期不输出区间。
绝对风险为零的全现金组合仍保留相对完整基准的主动风险。该固定权重风险评价与含费用的真实账户收益分别展示。</p>
<div class="scroll">{table(pd.DataFrame(cross_readouts))}</div><ul>{cross_links}</ul>
<p>风险覆盖使用单独版本的 252 日历窗口、至少 126 个观测和有效样本量 20；完整历史原计算保持不变。
停牌估值及旧暴露明确标记，不当作真实成交或零收益。缺可靠当前暴露的资产继续拒绝；不以覆盖改善宣称模型更准确。</p>
<h2>可复算证据</h2><ul>
<li><a href="portfolio-before-after.csv">逐账户前后比较</a></li>
<li><a href="account-independent-checks.json">独立现金、数量与风险核验</a></li>
<li><a href="new-failure-inventory.csv">新失败逐项清单</a></li>
<li><a href="102-integer-results.csv">原 102 次修复逐例结论</a></li>
<li><a href="102-integer-independent-checks.json">整数结果独立复核</a></li>
<li><a href="acceptance.json">完整验收与输入绑定</a></li></ul>"""
    rendered = page("Barra 失败场景修复：独立验收", body)
    nav_start, nav_end = rendered.index("<nav>"), rendered.index("</nav>") + len("</nav>")
    rendered = rendered[:nav_start] + rendered[nav_end:]
    (out / "report.html").write_text(rendered, encoding="utf8")
    print(json.dumps({"status": acceptance["engineering_research_delivery"],
                      "accounts": len(accounts), "out": str(out)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "coverage", "out", "integer-audit", "evidence", "certificates"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--branch", nargs=3, action="append", required=True,
                        metavar=("LABEL", "DIRECTORY", "BATCH_ID"))
    parser.add_argument("--partial", action="store_true")
    run(parser.parse_args())
