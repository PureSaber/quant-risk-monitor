"""Recompute accepted results, make charts, and write the reviewable research report."""

import base64
import html
import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

from public_sources import ROOT
from run_study import daily_scores, digest

NAMES = {
    "baseline": "OLS / common basis",
    "legacy_ols": "Original OLS",
    "enhanced": "Enhanced",
}
COLORS = {"baseline": "#65758b", "legacy_ols": "#aa7733", "enhanced": "#127e80"}


def embedded(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    active = json.loads((ROOT / "active-run.json").read_text(encoding="utf8"))
    run = ROOT / active["run"]
    selection = json.loads((run / "selection.json").read_text(encoding="utf8"))
    result = json.loads((run / "holdout.json").read_text(encoding="utf8"))
    validation = json.loads(
        (ROOT / "clean/preselection-validation.json").read_text(encoding="utf8")
    )
    assert validation["status"] == "passed"
    records = pd.read_csv(run / "holdout-forecasts.csv")
    complete = records[records["return"].notna()].copy()
    recomputed = np.log(complete.variance) + complete["return"].pow(2) / complete.variance
    np.testing.assert_allclose(complete.qlike, recomputed, rtol=1e-12, atol=1e-12)
    independent = (
        complete.assign(recomputed=recomputed)
        .groupby(["model", "date", "family"])
        .recomputed.mean()
        .groupby(["model", "date"])
        .mean()
        .unstack("model")
    )
    # Same common complete dates as the formal three-family criterion.
    paired = daily_scores(records)
    independent = independent.loc[paired.index]
    pd.testing.assert_frame_equal(
        independent[paired.columns], paired, check_names=False, rtol=1e-12, atol=1e-12
    )
    for model in paired:
        assert abs(paired[model].mean() - result["mean_qlike"][model]) < 1e-10
    complete["year"] = pd.to_datetime(complete.date).dt.year
    complete["squared_return"] = complete["return"] ** 2
    by_year = (
        complete.groupby(["year", "model", "family"])
        .agg(
            qlike=("qlike", "mean"),
            observations=("qlike", "size"),
            squared_return=("squared_return", "sum"),
            variance=("variance", "sum"),
        )
        .reset_index()
    )
    by_year["volatility_bias"] = np.sqrt(by_year.squared_return / by_year.variance)
    by_year.to_csv(run / "holdout-by-year-and-family.csv", index=False)
    by_portfolio = (
        complete.groupby(["model", "portfolio"])
        .agg(mean_qlike=("qlike", "mean"), observations=("qlike", "size"))
        .reset_index()
    )
    by_portfolio.to_csv(run / "holdout-by-portfolio.csv", index=False)
    # Independent vectorized moving-block CI reuses the frozen design, not its implementation.
    delta = (paired.enhanced - paired.baseline).to_numpy()
    n = len(delta)
    block = 10
    rng = np.random.default_rng(20261013)
    starts = rng.integers(0, n - block + 1, size=(2000, int(np.ceil(n / block))))
    indices = (starts[:, :, None] + np.arange(block)).reshape(2000, -1)[:, :n]
    ci = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
    np.testing.assert_allclose(ci, result["paired_block_bootstrap_95_ci"], rtol=1e-12, atol=1e-12)
    original_delta = (paired.enhanced - paired.legacy_ols).to_numpy()
    original_ci = np.quantile(original_delta[indices].mean(axis=1), [0.025, 0.975])
    np.testing.assert_allclose(
        original_ci, result["original_ols_comparison_95_ci"], rtol=1e-12, atol=1e-12
    )
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 145,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    dates = pd.to_datetime(paired.index)
    axes[0].plot(
        dates,
        (paired.enhanced - paired.baseline).expanding().mean(),
        color=COLORS["enhanced"],
        label="Enhanced minus common-basis OLS",
    )
    axes[0].plot(
        dates,
        (paired.enhanced - paired.legacy_ols).expanding().mean(),
        color=COLORS["legacy_ols"],
        label="Enhanced minus original OLS",
    )
    axes[0].axhline(0, color="#999", lw=0.8)
    axes[0].set(
        title="Holdout cumulative average QLIKE difference",
        ylabel="Lower favors enhanced",
    )
    axes[0].legend(fontsize=8)
    locator = mdates.MonthLocator(bymonth=[1, 7])
    axes[0].xaxis.set_major_locator(locator)
    axes[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    families = ["equal", "capitalization", "industry"]
    positions = np.arange(3)
    for i, model in enumerate(["legacy_ols", "baseline", "enhanced"]):
        axes[1].bar(
            positions + (i - 1) * 0.24,
            [result["volatility_bias"][model][f] for f in families],
            width=0.23,
            color=COLORS[model],
            label=NAMES[model],
        )
    axes[1].axhline(1, color="#777", lw=0.8)
    axes[1].set_xticks(positions, ["Equal weight", "Cap weight", "Industry"])
    axes[1].set(title="Realized / predicted volatility", ylabel="Ideal = 1")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    chart = run / "holdout-comparison.png"
    fig.savefig(chart)
    plt.close(fig)
    selection_scores = pd.Series(selection["scores"]).sort_values()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(
        selection_scores.index,
        selection_scores.values,
        color=[
            "#127e80" if x == selection["selected_id"] else "#a1afbf"
            for x in selection_scores.index
        ],
    )
    ax.invert_yaxis()
    ax.set(
        title="2023 selection scores — parameters frozen before holdout",
        xlabel="Average QLIKE (lower is better)",
    )
    fig.tight_layout()
    selection_chart = run / "selection-comparison.png"
    fig.savefig(selection_chart)
    plt.close(fig)
    quality = validation["data_quality"]
    comparison = pd.DataFrame(
        [
            {
                "模型": NAMES[m],
                "样本外平均 QLIKE": result["mean_qlike"][m],
                "等权波动偏差": result["volatility_bias"][m]["equal"],
                "市值加权波动偏差": result["volatility_bias"][m]["capitalization"],
                "行业组合波动偏差": result["volatility_bias"][m]["industry"],
            }
            for m in ["legacy_ols", "baseline", "enhanced"]
        ]
    )
    decision = (
        "达到预先约定的研究晋级门槛"
        if result["performance_gate_passed"]
        else "未达到预先约定的研究晋级门槛，保留原模型作为默认"
    )
    missing = records[records["return"].isna()][
        ["date", "end", "model", "portfolio", "missing_assets"]
    ]
    missing.to_csv(run / "missing-endpoint-audit.csv", index=False)
    extra = json.loads((run / "selection-ablations.json").read_text(encoding="utf8"))
    rejection_reasons = []
    if ci[1] >= 0 or original_ci[1] >= 0:
        rejection_reasons.append("相对 OLS 的改进置信区间未全部低于零")
    if not result["bias_guardrail_passed"]:
        rejection_reasons.append("组合波动偏差恶化超过冻结容忍值")
    if not result["coverage_gate_passed"]:
        rejection_reasons.append(f"完整预测日期少于 {result['minimum_holdout_dates']}")
    gate_explanation = (
        "模型未晋级的原因：" + "；".join(rejection_reasons) + "。"
        if rejection_reasons
        else "模型通过冻结的研究晋级门槛。"
    )
    ablation_notes = []
    for key, label in [
        ("without_cap_weighting", "市值加权"),
        ("without_time_decay", "时间衰减"),
        ("without_serial_adjustment", "序列相关调整"),
        ("without_regime_adjustment", "波动状态缩放"),
        ("without_group_bayesian_specific", "分组特异风险"),
    ]:
        effect = (
            "有改善" if extra["mean_qlike"]["selected"] < extra["mean_qlike"][key] else "未改善"
        )
        ablation_notes.append(label + effect)
    year_display = by_year[
        ["year", "model", "family", "qlike", "observations", "volatility_bias"]
    ].copy()
    year_display["model"] = year_display.model.map(
        {"baseline": "统一口径 OLS", "legacy_ols": "原始 OLS", "enhanced": "增强版"}
    )
    year_display["family"] = year_display.family.map(
        {"equal": "等权", "capitalization": "市值加权", "industry": "行业"}
    )
    year_display.columns = ["年份", "模型", "组合类别", "平均 QLIKE", "组合观测数", "波动偏差"]
    acceptance = {
        "status": "research_completed_and_checked",
        "performance_gate_passed": result["performance_gate_passed"],
        "production_parameters_changed": False,
        "coverage_gate_passed": result["coverage_gate_passed"],
        "complete_holdout_dates": result["date_count"],
        "required_holdout_dates": result["minimum_holdout_dates"],
        "independent_score_recalculation_passed": True,
        "independent_bootstrap_recalculation_passed": True,
        "canonical_original_and_enhanced_agreement_passed": True,
        "pit_future_filing_poison_passed": True,
        "protocol_sha256": selection["protocol_sha256"],
        "selection_sha256": digest(run / "selection.json"),
        "holdout_sha256": digest(run / "holdout.json"),
        "scope": "frozen 200-name historical Shanghai/Shenzhen main-board cohort; 2023 selection; 2024-2025 historical holdout",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run / "acceptance.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2), encoding="utf8"
    )
    (run / "calibrated-research-config.json").write_text(
        json.dumps(
            {
                "config": selection["selected_config"],
                "eligible_for_promotion": result["performance_gate_passed"],
                "production_default_changed": False,
            },
            indent=2,
        ),
        encoding="utf8",
    )
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>公开数据 Barra 风格模型验证</title>
<style>body{{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#182b3c;max-width:1100px;margin:40px auto;padding:0 25px}}h1{{font-size:31px}}h2{{font-size:23px;margin-top:42px}}.lead{{background:#eaf5f3;padding:22px;border-left:5px solid #127e80}}.sub{{color:#526475}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #dce4eb;padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}pre{{font-size:12px;white-space:pre-wrap;background:#f3f6f9;padding:15px}}a{{color:#126773}}</style>
<h1>公开数据 Barra 风格模型：校准与样本外验证</h1><p class="sub">不使用汇升 / Dataway · 研究创建于 2026-10-13 · 回溯研究，未声称历史预注册</p>
<div class="lead"><b>{decision}。</b><br>真实样本外比较已完成，{result["scheduled_date_count"]} 个五交易日预测窗口中有 {result["date_count"]} 个具备完整终点价格，原定完整窗口门槛为 {result["minimum_holdout_dates"]}。增强版相对统一口径 OLS 的 QLIKE 差为 {result["enhanced_minus_baseline_qlike"]:.5f}，95% 区间 [{ci[0]:.5f}, {ci[1]:.5f}]；相对原始 OLS 的差为 {result["enhanced_minus_original_ols_qlike"]:.5f}，95% 区间 [{original_ci[0]:.5f}, {original_ci[1]:.5f}]。差值越低越好。生产参数未改动。</div>
<h2>一、直接比较结果</h2>{comparison.to_html(index=False, float_format=lambda x: f"{x:.5f}", border=0)}
<p>原始 OLS 直接对照升级前 54fcf4d 的代码：普通最小二乘、20% 对角协方差收缩、20% 全局均值特异方差收缩。统一口径 OLS 使用与增强版相同的市场/行业识别约束，帮助区分因子表示与估计方法的影响。三个模型共用股票、历史暴露、日期和组合权重。</p>
<img alt="样本外预测损失和波动偏差" src="{embedded(chart)}">
<p>波动偏差为实际收益均方根与预测波动的比值，理想值为 1。这里评估风险预测精度，没有把结果表述为策略收益或实盘盈利。</p>
<h2>二、参数怎样选出来</h2><p>2023 年仅用于选择参数，{selection["dates"]} 个完整预测窗口比较 12 组增强参数。选中 <b>{html.escape(selection["selected_id"])}</b> 后冻结数据、代码及配置；2024—2025 年只用于验收。预测窗口不重叠，选择期与样本外之间保留五个交易日间隔。</p>
<img alt="2023年参数选择结果" src="{embedded(selection_chart)}"><pre>{html.escape(json.dumps(selection["selected_config"], ensure_ascii=False, indent=2))}</pre>
<p>以下消融只查看 2023 年，不重新选择参数，也不使用样本外结果调参。本次对照中：{"；".join(ablation_notes)}。这里只诊断机制，未据此另选模型。</p>{pd.Series(extra["mean_qlike"], name="2023 QLIKE").to_frame().to_html(float_format=lambda x: f"{x:.5f}", border=0)}
<h2>三、数据清洗和 PIT 处理</h2>
<p>样本从 2021 年 11 月已公开的证监会行业名单中固定抽取 200 家沪深主板公司，随后不按存续状态换股。数据跨度 {quality["prices_begin"]}—{quality["prices_end"]}，覆盖 {quality["dates"]} 个交易日；从原始公告独立提取 {quality["original_facts"]:,} 条财务事实，覆盖 {quality["original_fact_assets"]} 家公司。</p>
<ul><li>财务数值取巨潮原始公告，按公告次日可用；同一期间的后续修订只影响其公开之后。现今供应商财报表仅作交叉核查，不决定历史值或历史缺失。</li><li>股本按公告日和生效日同时限制，九家巨潮股本接口缺失的公司用带公告日期的新浪历史股本表补齐，原始股份变动表作独立补充；行业使用当时已公开的证监会/上市公司协会文件，未用现今分类倒填。</li><li>未复权价格用于市值；股票收益与动量改用 BaoStock 乘法复权序列。独立抽查发现东方财富加法后复权会扭曲百分比收益，因此该字段已排除。八家抽查公司的未复权收盘价完全一致。</li><li>已修复总负债与流动负债混淆、负净资产漏读、千元单位、括号负数、附注编号和股份表换行等提取问题。原报表与摘要有无法解释的数值冲突时保留为缺失，并记录原因。</li><li>价格信息滞后一交易日；缺失财务描述子按当时行业中位数处理，同行不足时使用当时全体中位数，每个描述子的缺失比例上限 40%。不使用未来值填充。</li></ul>
<h2>四、稳定性与缺失处理</h2><p>下表与波动偏差统计使用各组合有终点价格的观测，行业观测数为“日期 × 组合”；主 QLIKE 比较则只使用全部组合共同完整的 {result["date_count"]} 个日期。所有模型在同一统计口径下配对。</p>{year_display.to_html(index=False, float_format=lambda x: f"{x:.5f}", border=0)}
<p>缺少区间终点价格的组合不重分配权重。主结果使用三个模型共同完整的预测日期；另外保留停牌/无报价沿用最近价格的敏感性结果，增强版相对统一口径 OLS 的 QLIKE 差为 {result["stale_mark_sensitivity_delta"]:.5f}。缺失区间明细见 <a href="missing-endpoint-audit.csv">缺失终点审计</a>。</p>
<p>置信区间按预测日期做配对区块自助抽样，区块长度 10 个预测日，重复 2,000 次；同一日所有行业组合一起保留，避免把高度相关的组合当成独立样本。完整逐组合结果见 <a href="holdout-by-portfolio.csv">逐组合表</a>。</p>
<h2>五、自验收与可复现范围</h2>
<p>通过未来财务公告污染测试、历史可用时间检查、与原始模型及增强模型正式入口的协方差一致性检查；预测损失和置信区间由独立计算再次核对。{gate_explanation}没有据此重调参数或放宽门槛。</p>
<p>结论范围为这 200 家历史主板公司的公开资料重建。未覆盖全市场、后续新股或实盘数据到达延迟；股本市值使用总股数与 A 股价格形成一致的公开代理口径。2021—2023 年行业资料发布存在间隔，期间沿用上次可用分类。完整终点筛选可能遗漏停牌或退出窗口，敏感性结果也不代表可实际成交的退市回收价格。项目采用公开因子定义，并非 MSCI 的专有 Barra 参数。</p>
<p>公开来源：<a href="https://www.csrc.gov.cn/csrc/c100103/c1558619/content.shtml">证监会历史行业名单</a>、<a href="https://www.capco.org.cn/">中国上市公司协会历史行业发布</a>、<a href="https://www.cninfo.com.cn/">巨潮原始公告</a>、<a href="http://www.baostock.com/">BaoStock 乘法复权行情</a>、<a href="https://quote.eastmoney.com/">东方财富未复权行情</a>、<a href="https://money.finance.sina.com.cn/">新浪带公告日期的历史股本表</a>。逐文件来源、捕获时间和哈希保存在本地 source-manifest.json，原始档案保留以供重放。</p>
<pre>{html.escape(json.dumps(acceptance, ensure_ascii=False, indent=2))}</pre>
<p>证据文件：<a href="selection.json">冻结参数与代码哈希</a> · <a href="holdout.json">样本外结果</a> · <a href="holdout-forecasts.csv">逐项预测</a> · <a href="paired-daily-scores.csv">配对日期损失</a> · <a href="acceptance.json">验收记录</a> · <a href="research-audit.ipynb">复算 Notebook</a></p>
</html>'''
    (run / "report.html").write_text(text, encoding="utf8")
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# 公开数据 Barra 风格风险模型：复算\n\n本 Notebook 读取已冻结的逐项预测，独立计算主要比较结果；不重新调参。原始数据和构建步骤保存在研究脚本与逐源收据中。"
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\nroot=Path.cwd()\nrecords=pd.read_csv(root/'holdout-forecasts.csv')\nselection=json.loads((root/'selection.json').read_text(encoding='utf8'))\nrecords.head()"
        ),
        nbformat.v4.new_code_cell(
            "records['recomputed_qlike']=np.log(records.variance)+records['return']**2/records.variance\nnp.testing.assert_allclose(records.qlike,records.recomputed_qlike,equal_nan=True)\nfamily=records.groupby(['model','date','family']).recomputed_qlike.mean().unstack('family')\nscores=family.dropna(subset=['equal','capitalization','industry']).mean(axis=1).unstack('model').dropna()\nscores.mean()"
        ),
        nbformat.v4.new_code_cell(
            "delta=(scores.enhanced-scores.baseline).to_numpy()\nn=len(delta);block=10\nrng=np.random.default_rng(20261013)\nstarts=rng.integers(0,n-block+1,size=(2000,int(np.ceil(n/block))))\nix=(starts[:,:,None]+np.arange(block)).reshape(2000,-1)[:,:n]\n{'mean_difference':float(delta.mean()),'ci95':np.quantile(delta[ix].mean(axis=1),[.025,.975]).tolist(),'dates':n}"
        ),
        nbformat.v4.new_code_cell(
            "import matplotlib.pyplot as plt\nimport matplotlib.dates as mdates\nfig,ax=plt.subplots(figsize=(10,4))\nfor control,label in [('baseline','Common-basis OLS'),('legacy_ols','Original OLS')]:\n    ax.plot(pd.to_datetime(scores.index),(scores.enhanced-scores[control]).expanding().mean(),label=label)\nax.axhline(0,color='gray',linewidth=.8)\nax.set(title='Enhanced minus OLS: cumulative average QLIKE',ylabel='Lower favors enhanced')\nlocator=mdates.MonthLocator(bymonth=[1,7])\nax.xaxis.set_major_locator(locator)\nax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))\nax.legend()\nfig.tight_layout()\nplt.show()\npd.read_csv('holdout-by-year-and-family.csv')"
        ),
        nbformat.v4.new_markdown_cell(
            "局限：固定的 200 家历史主板样本、公开公告的 PIT 重建；不是全市场或实盘延迟重放。缺失终点不重分配持仓；另外保留沿用最近报价的敏感性比较。"
        ),
    ]
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nbformat.write(notebook, run / "research-audit.ipynb")
    print(
        json.dumps(
            {"report": str(run / "report.html"), "acceptance": acceptance},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
