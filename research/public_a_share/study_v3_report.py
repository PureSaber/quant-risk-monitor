"""Source-linked study reports and independent checks of native account artifacts."""

from __future__ import annotations

import argparse
import html
import json
import pickle
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pandas as pd

from study_v3_archive import digest, write_json
from study_v3_cross_audit import recompute_cross, session_dates
from study_v3_risk import factor_variance, paired_summary


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def table(frame):
    return frame.to_html(
        index=False, border=0, escape=True, na_rep="—", float_format=lambda n: f"{n:.6g}"
    )


def facts_table(mapping):
    return table(pd.DataFrame({"项目": list(mapping), "证据": [
        json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
        for v in mapping.values()
    ]}))


def page(title, body):
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font:16px/1.65 "Microsoft YaHei",system-ui,sans-serif;color:#172b3a;background:#f4f6f8;margin:0}}
main{{max-width:1280px;margin:auto;background:white;padding:36px}}h1{{font-size:30px;line-height:1.25}}h2{{margin-top:38px;font-size:22px;border-bottom:2px solid #dce6ee;padding-bottom:10px}}
a{{color:#075d98}}table{{border-collapse:collapse;font-size:13px;width:100%;margin:16px 0}}th,td{{padding:8px;border-bottom:1px solid #dce6ee;text-align:left;vertical-align:top}}th{{background:#eaf0f4;position:sticky;top:0}}tr:nth-child(even){{background:#f7f9fa}}
.scroll{{overflow:auto}}.note{{background:#fff3d8;padding:16px;border-left:4px solid #c58916}}.card{{background:#eaf5f0;padding:16px;border-left:4px solid #237b5b}}code{{overflow-wrap:anywhere}}small{{color:#536575}}nav a{{margin-right:20px}}@media(max-width:700px){{main{{padding:18px}}h1{{font-size:25px}}}}</style>
<main><nav><a href="report.html">总览</a><a href="data-quality-report.html">数据质量</a><a href="risk-estimation-report.html">风险估计</a><a href="portfolio-report.html">组合与执行</a></nav><h1>{html.escape(title)}</h1>{body}</main></html>"""


def batch_results(out, batch_id, *, allow_partial):
    logs = out / ("portfolio-logs-" + batch_id)
    progress = read(logs / "progress.json")
    indices = [item["scenario"] for item in progress]
    if len(indices) != len(set(indices)) or not set(indices) <= set(range(62)):
        raise ValueError("batch scenario identities must be unique and in the frozen specification")
    if not allow_partial and set(indices) != set(range(62)):
        raise ValueError("all 62 scenario processes must finish before final reporting")
    results = []
    for item in progress:
        lines = Path(item["log"]).read_text(encoding="utf8").splitlines()
        rows = []
        for line in lines:
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if "identity" in obj:
                rows.append(obj)
        if not rows:
            results.append({**item, "status": "worker_failed", "reason": "no scenario receipt"})
            continue
        row = rows[-1]
        folder = out / "portfolios" / f"{item['scenario']:03d}-{row['identity'][:16]}"
        if (folder / "complete.json").exists():
            row = read(folder / "complete.json")
            for name, sha in row["outputs"].items():
                if digest(folder / name) != sha:
                    raise ValueError("account artifact changed: " + str(folder / name))
        row["folder"] = str(folder)
        row["specification"] = read(folder / "specification.json")
        results.append(row)
    return results


def verify_account(folder, models=None, specification=None):
    journal = pd.read_parquet(folder / "cash_ledger.parquet")
    if len(journal):
        # The certified bridge exports every posting at one exact money scale.
        if (
            journal.amount_scale.nunique() != 1
            or (journal.groupby(["transaction_id", "currency"]).amount_units.sum() != 0).any()
        ):
            raise ValueError("exact double-entry journal does not balance")
    positions = pd.read_parquet(folder / "positions.parquet")
    if len(positions) and (positions.quantity_scale.ne(0) | positions.quantity_units.lt(0)).any():
        raise ValueError("invalid long-only exact share quantity")
    snapshots = pd.read_parquet(folder / "portfolio_snapshots.parquet")
    if snapshots.cash_value_units.lt(0).any():
        raise ValueError("negative native cash")
    decisions = read(folder / "allocation-decisions.json")
    if models is not None:
        verify_decision_calendar(decisions, session_dates(snapshots), models)
    maximum, integer_maximum, durations, statuses, errors = 0.0, 0.0, [], Counter(), Counter()
    residuals = []
    for row in decisions:
        diagnostic = row.get("diagnostics", {})
        if "weights" not in row:
            statuses[row.get("status", "unknown")] += 1
            errors[row.get("error", row.get("reason", "unspecified"))] += 1
            continue
        value = float(diagnostic["max_constraint_residual"])
        if value > 1e-7:
            raise ValueError("published continuous target violates normalized tolerance")
        maximum = max(maximum, value)
        execution = diagnostic["execution"]
        integer_residual = max(execution["constraint_residuals"].values(), default=0.0)
        if integer_residual > 1e-7 or Decimal(execution["cash"]) < 0:
            raise ValueError("published integer target violates constraints or cash")
        integer_maximum = max(integer_maximum, integer_residual)
        post_nav = Decimal(execution["post_fee_nav"])
        if abs(Decimal(str(row["nav"])) - Decimal(execution["costs"]) - post_nav) > Decimal("0.0001"):
            raise ValueError("integer fee and NAV accounting disagree")
        integer_weights = pd.Series(execution["integer_weights"], dtype=float)
        cash_weight = float((Decimal(execution["cash"]) + Decimal(execution["non_cash_assets"])) / post_nav)
        if abs(float(integer_weights.sum()) + cash_weight - 1) > 1e-10:
            raise ValueError("integer holdings, cash and receivables do not reconcile")
        if models is not None:
            entry = models[pd.Timestamp(row["session"])]
            model = entry["models"][specification["model"]]
            x = model["exposures"]
            w = integer_weights.reindex(x.index, fill_value=0).to_numpy()
            wb = entry["benchmark"].reindex(x.index, fill_value=0).to_numpy()
            active, _, _ = factor_variance(x.to_numpy(), model["F_daily_rate"], model["D_daily_rate"], w - wb)
            predicted_te = (max(active, 0) * 252) ** 0.5
            if abs(predicted_te - diagnostic["integer_annualized_tracking_error"]) > 1e-9:
                raise ValueError("integer TE differs from independent factor calculation")
            if specification.get("te") is not None and predicted_te > specification["te"] + 1e-7:
                raise ValueError("integer target exceeds declared TE limit")
        durations.append(diagnostic["rebalance_seconds"])
        residuals.append(
            {
                "date": row["session"],
                "maximum_normalized_residual": value,
                "continuous_te": diagnostic.get("annualized_tracking_error"),
                "integer_te": diagnostic.get("integer_annualized_tracking_error"),
                "turnover": diagnostic.get("turnover"),
                "execution": diagnostic.get("execution"),
            }
        )
    return {
        "exact_journal_balanced": True,
        "long_only_integer_shares": True,
        "nonnegative_cash": True,
        "maximum_continuous_residual": maximum,
        "maximum_integer_residual": integer_maximum,
        "integer_fee_cash_reconciliation": True,
        "published_targets": len(durations),
        "rebalance_seconds": durations,
        "failure_status_counts": dict(statuses),
        "failure_reason_counts": dict(errors),
        "constraints": residuals,
    }


def verify_decision_calendar(decisions, snapshot_days, model_dates):
    expected = [d for d in model_dates if snapshot_days.min() <= d <= snapshot_days.max()]
    observed = [pd.Timestamp(row["session"]) for row in decisions]
    if observed != expected:
        raise ValueError("native account omitted or duplicated a scheduled five-session decision")


def run(out, independent, batch_id, *, allow_partial=False):
    results = batch_results(out, batch_id, allow_partial=allow_partial)
    binding = read(out / "portfolio-risk-models-binding.json")
    if digest(out / "portfolio-risk-models.pkl") != binding["models"]:
        raise ValueError("risk model source changed before independent cross prediction")
    # Trusted, archived local research inputs, verified against the producer receipt.
    with (out / "portfolio-risk-models.pkl").open("rb") as stream:
        models = pickle.load(stream)
    with (out / "panel.pkl").open("rb") as stream:
        panel = pickle.load(stream)["panel"]
    quality, financial = (
        read(out / "quality-summary.json"),
        read(out / "financial-quality-summary.json"),
    )
    audit = read(out / "prior-ten-window-audit.json")
    if audit["windows"] != 10:
        raise ValueError("the prior ten-window audit is incomplete")
    recompute = read(independent / "independent-recomputation.json")
    risk = read(out / "risk-summary.json")
    comparisons = []
    for experiment, periods in risk.items():
        for period, values in periods.items():
            for model, score in values["comparisons"].items():
                lo, hi = score["simultaneous_95_interval"]
                comparisons.append(
                    {
                        "实验": experiment,
                        "区间": period,
                        "模型": model,
                        "完整日期": values["complete_dates"],
                        "ΔQLIKE": score["mean_delta"],
                        "校正区间下界": lo,
                        "校正区间上界": hi,
                        "历史改善区间上界<0": score["historical_improvement"],
                    }
                )
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(out / "risk-comparison-summary.csv", index=False, encoding="utf-8-sig")
    quality_body = f"""<p class="note">固定保留 200 家原历史样本；不以今天是否存续更换公司。数据按原公告可用时间重建，实际抓取时间单独记录。这是回顾性公开数据重建，不等于认证的历史供应商数据版本。</p>
<h2>逐日覆盖与缺口</h2>{facts_table(quality)}
<p>缺报价不推断为退市，不填零收益，也不对剩余持仓重新归一化。原十个缺失窗口的价格端点已与停牌原始记录逐项核查，保留为缺失结果。</p>
<p><a href="prior-ten-window-audit.json">原十窗口证据</a> · <a href="daily-quality.parquet">237,400 条逐资产日质量记录</a> · <a href="company-quality.csv">逐公司清单</a> · <a href="listing-exit-audit.parquet">上市和退出状态核查</a></p>
<h2>财报版本与填补</h2>{facts_table(financial)}
<p>财报过期、缺少成熟财报及已知修订逐日保留；暴露填补比例按模型版本记录，不把填补后的完整矩阵当作原始数据完整。</p>
<p><a href="financial-vintages.parquet">财报版本清单</a> · <a href="financial-daily-quality.parquet">财报逐日质量</a> · <a href="imputation-daily-quality.parquet">暴露填补与来源</a> · <a href="descriptor-coverage.parquet">描述子原始覆盖率</a></p>
<h2>执行价格与实际权益</h2><p>交换参考价与股东实际现金、股份分开。差异化分红和缺失分红由发行人公告佐证；宁科生物重整不向原股东分配，花王股份的普通股东比例不能用公司总股本转增比例替代。无法精确确认的实际持仓权益触发明确停止。</p>
<p><a href="corporate-action-repair-evidence.json">逐项修复与原公告摘要</a> · <a href="execution-input-guards.json">未解决权益保护条件</a> · <a href="execution-input-issues.json">剩余价格复权冲突</a></p>
<h2>独立复算</h2>{facts_table(recompute)}<p>全部原始价格重新解析，暴露缓存从空目录重建；财务输入使用已归档的提取事实，原 PDF 摘要另行核验，不声称本轮重新人工核读全部 PDF。</p>"""
    (out / "data-quality-report.html").write_text(
        page("200 家样本：数据质量与修复证据", quality_body), encoding="utf8"
    )
    risk_body = (
        """<p class="note">模型推广未通过：新增独立窗口为 0。2024—2026 全部属于已查看的回顾性诊断；不能将这些窗口重新计入 80 个独立成熟窗口门槛。</p>
<p>先固定 OLS 比较三种暴露，再固定修正版暴露比较原十二组估计参数及五项消融。5 日预测、252 日历史，使用相同资产、日期和评价权重。负的 ΔQLIKE 表示相对该实验控制组改善；必须同时检查经多重比较校正的区间上界。</p>
<p>按预测日期联合区块重采样 5,000 次，每块十个日期，日期内保留全部组合；先按组合族汇总，避免行业组合数量改变权重。高低波动状态只用预测日前数据。</p>
<p>v3 沿连续五日网格安排 180 个预测日期，其中 2026 年为 35 个、完整共同结果为 27 个。这与此前独立安排的 36 个窗口不是同一日期网格；原十个缺失窗口仍单独保留，不把新网格的样本数变化称为修复了缺失价格。</p>
<h2>冻结比较结果</h2><div class="scroll">"""
        + table(comparison)
        + """</div>
<h2>校准与适用边界</h2><p>波动校准采用实际五日线性复权收益平方均值与预测方差均值的比值再开方；理想值为 1。风险估计使用公开复权收益口径；原生账户收益另含真实费用及股东现金权益，两个口径分别报告。合成测试覆盖的特殊情形不能当作本样本的真实验证。</p>
<p><a href="risk-calibration.json">按模型、组合族和市场状态的校准</a> · <a href="risk-forecasts.parquet">全部 51,480 条预测及实际结果</a> · <a href="model-configurations.json">22 个冻结模型配置</a> · <a href="risk-selection.json">仅用 2023 的选择记录</a> · <a href="risk-summary.json">区块推断原始结果</a></p>"""
    )
    (out / "risk-estimation-report.html").write_text(
        page("200 家样本：暴露与风险估计比较", risk_body), encoding="utf8"
    )
    accounts, checks, cross = [], {}, []
    for row in results:
        spec = row.get("specification", {})
        record = {
            "场景": row["scenario"],
            "方法": spec.get("method"),
            "模型": spec.get("model"),
            "预算": spec.get("budget"),
            "TE上限": spec.get("te"),
            "状态": row["status"],
            "回放终点": row.get("evaluated_through"),
            "完整期间": row.get("full_horizon_completed", False),
            "目标成功数": row.get("published"),
            "失败率": row.get("failure_rate"),
            "实际净收益": row.get("net_return"),
            "最大回撤": row.get("maximum_drawdown"),
            "实际年化TE": row.get("actual_annualized_tracking_error"),
            "TE完整窗口": row.get("actual_te_complete_windows"),
            "费用": row.get("total_fees"),
            "成交额/初始资金": row.get("gross_traded_notional_over_initial_capital"),
            "成功调仓P95秒": row.get("rebalance_p95_seconds"),
            "全回放秒": row.get("wall_seconds"),
            "进程峰值MiB": row.get("process_peak_bytes", 0) / 1024**2,
        }
        accounts.append(record)
        if "outputs" not in row:
            continue
        folder = Path(row["folder"])
        checks[str(row["scenario"])] = verify_account(folder, models, spec)
        frame = recompute_cross(
            read(folder / "allocation-decisions.json"),
            pd.read_parquet(folder / "positions.parquet"),
            pd.read_parquet(folder / "portfolio_snapshots.parquet"),
            models,
            panel,
        )
        if len(frame):
            frame["generator"] = spec["model"]
            frame["method"] = spec["method"]
            frame["scenario"] = row["scenario"]
            cross.append(frame)
    account_frame = pd.DataFrame(accounts)
    account_frame.to_csv(out / "portfolio-comparison.csv", index=False, encoding="utf-8-sig")
    write_json(out / "account-independent-checks.json", checks)
    cross_summary, calibration = {}, pd.DataFrame()
    if cross:
        merged = pd.concat(cross, ignore_index=True)
        merged.to_parquet(out / "portfolio-cross-predictions.parquet", index=False)
        coverage = (
            merged.groupby(["scenario", "predictor", "target_kind", "status"])
            .size().rename("forecasts").reset_index()
        )
        coverage.to_csv(out / "portfolio-cross-coverage.csv", index=False)
        gaps = merged[merged.status.eq("risk_union_incomplete")][
            ["date", "scenario", "target_kind", "missing_risk_assets"]
        ].drop_duplicates()
        gaps = gaps.assign(asset=gaps.missing_risk_assets.str.split(",")).explode("asset")
        gaps.to_csv(out / "portfolio-risk-union-gaps.csv", index=False)
        # Shared portfolio identifiers ensure each predictor sees the same return.
        consistency = merged.groupby("portfolio_id")[["actual_frozen_weight_return", "active_frozen_weight_return"]].nunique()
        if (consistency > 1).any().any():
            raise ValueError("models were evaluated against different actual returns")
        calibrations = []
        for objective, variance_col, return_col, loss_col in (
            ("absolute", "variance", "actual_frozen_weight_return", "qlike"),
            ("active", "active_variance", "active_frozen_weight_return", "active_qlike"),
        ):
            evaluation = merged.copy()
            evaluation["qlike"] = evaluation[loss_col]
            finite = evaluation.dropna(subset=["qlike"]).copy()
            finite["variance_ratio"] = finite[return_col]**2 / finite[variance_col]
            calibration = (
                finite.groupby(["generator", "predictor", "target_kind"])
                .agg(
                    forecasts=("portfolio_id", "nunique"),
                    mean_qlike=("qlike", "mean"),
                    mean_squared_standardized_return=("variance_ratio", "mean"),
                )
                .reset_index()
            )
            calibration["risk_objective"] = objective
            calibrations.append(calibration)
            cross_summary[objective] = {}
            for kind in ("continuous", "integer", "actual_close"):
                selected = evaluation[evaluation.target_kind.eq(kind)].copy()
                # All-cash absolute risk is zero, but its active risk is the
                # entire benchmark. Keep such active forecasts explicitly.
                selected = selected[~selected[variance_col].eq(0)]
                selected = selected.rename(columns={"predictor": "model"})
                selected["family"] = selected.scenario.astype(str)
                names = sorted(selected.model.unique())
                cross_summary[objective][kind] = (
                    paired_summary(selected, names, "legacy_ols") if "legacy_ols" in names else {}
                )
        calibration = pd.concat(calibrations, ignore_index=True)
        calibration.to_csv(
            out / "optimizer-risk-underestimation.csv", index=False, encoding="utf-8-sig"
        )
    write_json(out / "optimizer-cross-inference.json", cross_summary)
    status_counts = dict(Counter(r["status"] for r in results))
    acceptance = {
        "scenario_processes": len(results),
        "expected_scenarios": 62,
        "scheduled_rebalance_dates": binding["decision_dates"],
        "study_delivery": "complete_with_recorded_limitations"
        if len(results) == 62 and all("outputs" in r for r in results)
        else "incomplete",
        "scenario_status_counts": status_counts,
        "independent_recomputation": recompute["status"],
        "model_promotion": "not_passed",
        "independent_new_windows": 0,
        "minimum_independent_windows": 80,
        "default_model_changed": False,
        "frozen_accounts_changed": False,
        "live_trading_connected": False,
        "engineering_ci": "see release-evidence.json; study completion does not certify pending CI",
    }
    write_json(out / "study-acceptance.json", acceptance)
    diagnosis = ""
    if (out / "tracking-error-floor-diagnostic.json").exists():
        floor = read(out / "tracking-error-floor-diagnostic.json")
        if floor["model_sha256"] != binding["models"]:
            raise ValueError("TE diagnosis uses a different portfolio model cache")
        low, high = floor["benchmark_max_weight_range"]
        diagnosis += f"""<h2>紧跟踪误差限制的原因诊断</h2><p>这是事后解释，不参与参数选择或账户交易。研究样本基准最大单股权重介于 {low:.2%} 和 {high:.2%}；组合单股上限为 5%。先保留确切仓位预算与单股上限，暂不加换手、费用和整数约束，计算连续最小主动风险。这个乐观下界超过 TE 上限时，更严格的账户问题也无法通过；下界低于上限不等于账户可执行。</p>
<p>诊断共 {floor['problems']} 个问题，失败 {floor['failure_count']} 个；实际账户配置和失败记录没有因此改变。</p><div class="scroll">{table(pd.DataFrame(floor['results']))}</div><p><a href="tracking-error-floor-diagnostic.csv">逐日下界</a> · <a href="tracking-error-floor-diagnostic.json">诊断摘要与模型绑定</a></p>"""
    if (out / "resource-incident.json").exists():
        diagnosis += """<h2>共享机器的资源限制</h2><p class="note">增加补充工作进程后曾触发 Windows 内存提交压力，出现 MemoryError 和进程崩溃。空闲物理内存不能单独证明可以继续扩大并行。补充批次已停止，主批次恢复四个工作进程；原始失败和取消现场保留，再从初始账户重做未完成场景。部分耗时包含临时额外进程竞争，因此不构成专用机器性能承诺。</p><p><a href="resource-incident.json">资源故障、处理及原始证据摘要</a> · <a href="failed-attempts/memory-pressure-01/receipt.json">重试前保存的失败现场</a></p>"""
    portfolio_body = f"""<p class="note">完成 {len(results)}/62 个进程。状态统计：{html.escape(str(status_counts))}。不同终点的诊断前缀不能用来排序完整期间收益；全部目标失败而留在现金的场景也不是成功投资组合。</p>
<p>每 5 个交易日调仓，初始资金 100 万元，预算 95%/80%，单股上限 5%，首次建仓完整计费，其后 L1 换手上限 20%。3%/6% 是模型预测的年化跟踪误差上限；表中实际 TE 来自账户与完整研究样本基准的五日复合收益差，二者不能互相替代。基准端点缺失的窗口不填零、不重新归一化。</p>
<h2>原生账户比较</h2><div class="scroll">{table(account_frame)}</div>
<p><a href="portfolio-comparison.csv">完整比较表</a> · <a href="account-independent-checks.json">精确账本、现金、股份与连续约束独立复核</a> · <a href="portfolios/">每个账户的原生订单、成交、费用、持仓和失败证据</a></p>
<h2>相同组合的交叉风险预测</h2><p>六组冻结模型负责生成组合，全部 22 个风险配置对相同连续、整数及实际收盘持仓交叉预测；组合身份绑定日期与完整权重。同一组合的实际收益已逐项检查一致。绝对风险与相对完整基准的主动风险分别计算损失和区块区间；全现金账户的绝对风险为零，主动风险仍包含完整基准。平方标准化收益高于 1 提示低估方向，不能仅凭这个均值认定模型失效；需结合日期区块区间、覆盖率和样本量。</p>
<p><a href="optimizer-risk-underestimation.csv">生成模型 × 预测模型的低估诊断</a> · <a href="optimizer-cross-inference.json">联合日期区块比较</a> · <a href="portfolio-cross-predictions.parquet">全部交叉预测</a> · <a href="portfolio-cross-coverage.csv">包括失败调仓日的风险覆盖</a> · <a href="portfolio-risk-union-gaps.csv">持仓风险缺口逐项清单</a></p>
<p class="note">估计比较采用 252 日残差完整的共同资产集合。停牌等缺样会使已持仓证券退出风险可用集合；全账户缺少必要风险输入时停止发布新目标。随后持仓的收益和低换手不能解释为优化器持续正常运行。表中调仓 P95 仅统计成功发布目标的完整优化与整手修复，完整账户回放耗时和失败数另列，未记录的失败输入校验耗时不作估算。</p>
<h2>约束与实际执行的边界</h2><p>连续目标残差验收上限为 1e-7；整数修复采用原执行层的现金、费用、可卖数量与整手规则，不声称整数全局最优。真实成交后，持仓可能偏离目标，实际收盘持仓的风险另行预测。原生账本逐笔复式记账复核，实际持仓是整数股、现金非负。未解释的错误使工程验收保持未完成。</p>{diagnosis}"""
    (out / "portfolio-report.html").write_text(
        page("200 家样本：优化组合、风险交叉预测与执行", portfolio_body), encoding="utf8"
    )
    overview = f"""<p class="card">研究交付状态：{acceptance["study_delivery"]}。独立重算通过，默认模型和冻结账户保持原配置。模型推广未通过，新增独立成熟窗口为 0。</p>
<p>本研究截止 2026-09-30，使用原 200 家历史公司样本及必要预热数据。工程交付、模型替换和策略推广分别验收；80 个真正独立的五日窗口门槛不降低，也不将等待前向数据包装成本轮工程完成。</p>
<h2>三项交付</h2><ul><li><a href="data-quality-report.html">暴露、逐日数据质量、原十窗口缺口和修复证据</a></li><li><a href="risk-estimation-report.html">冻结暴露与估计器比较、校准和消融</a></li><li><a href="portfolio-report.html">62 个原生账户场景、交叉风险预测和失败记录</a></li></ul>
<h2>验收与复现入口</h2>{facts_table(acceptance)}
<p><a href="protocol.json">冻结实验协议</a> · <a href="study-acceptance.json">机器可读验收</a> · <a href="release-evidence.json">代码、环境与 CI 证据</a> · <a href="../independent-01/independent-recomputation.json">独立重算收据</a> · <a href="../../supplemental-recovery-receipt.json">新增证据恢复收据</a></p>
<p>研究隔离运行，没有切换模拟服务或改写账户。恢复时使用已记录的固定环境和 SHA-256 输入清单，在新目录重放；同机跨盘副本不被描述为独立硬件备份。真实样本耗时、峰值内存和失败在组合表中，不能用合成性能替代。</p>"""
    (out / "report.html").write_text(page("Barra 200 家样本验证 · v3", overview), encoding="utf8")
    write_json(out / "report-manifest.json", {p.name: digest(p) for p in out.glob("*.html")})
    print(json.dumps(acceptance), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    run(args.out, args.independent, args.batch_id, allow_partial=args.allow_partial)
