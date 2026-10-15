"""Static PDF companion using reported facts only; no browser or JavaScript."""

import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    TableStyle,
)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def run(root, target, font):
    pdfmetrics.registerFont(TTFont("StudyCJK", str(font)))
    body = ParagraphStyle(
        "Body",
        fontName="StudyCJK",
        fontSize=10,
        leading=16,
        textColor=colors.HexColor("#172b3a"),
        wordWrap="CJK",
        spaceAfter=8,
    )
    title = ParagraphStyle("Title", parent=body, fontSize=24, leading=32, spaceAfter=18)
    heading = ParagraphStyle(
        "Heading", parent=body, fontSize=15, leading=21, spaceBefore=14, spaceAfter=10
    )
    small = ParagraphStyle(
        "Small", parent=body, fontSize=8, leading=12, spaceAfter=0, alignment=TA_LEFT
    )
    story = []

    def p(text, style=body):
        return Paragraph(escape(str(text)), style)

    def section(text):
        story.append(p(text, heading))

    def grid(headers, records, widths):
        data = [[p(value, small) for value in headers]]
        data += [[p(value, small) for value in row] for row in records]
        item = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5edf2")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f5f8fa")],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#7093ab")),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(item)

    acceptance = read(root / "study-acceptance.json")
    quality = read(root / "quality-summary.json")
    financial = read(root / "financial-quality-summary.json")
    risk = read(root / "risk-summary.json")
    story.append(p("Barra 200 家历史样本验证", title))
    story.append(p("研究截止 2026-09-30 | 固定样本与有限参数网格 | v3"))
    status = f"已完成 {acceptance['scenario_processes']}/62 个场景进程；交付状态：{acceptance['study_delivery']}。"
    story.append(p(status))
    story.append(
        p(
            "模型推广未通过。新增独立成熟窗口为 0；2024-2026 全部属于已查看的回顾性诊断。80 个独立五日窗口门槛不降低。默认 v1、冻结账户和既有实验保持原配置，本轮没有接入实盘或切换模拟服务。"
        )
    )
    section("已完成的数据与数值证据")
    grid(
        ["项目", "证据", "解释边界"],
        [
            [
                "固定历史样本",
                f"200 家；{quality['asset_days']:,} 条资产日记录",
                "保留原样本，不按当前存续状态替换公司",
            ],
            [
                "财报与修订",
                f"{financial['facts']:,} 条事实；{financial['revision_rows']:,} 条修订记录",
                "公告可用时间与实际抓取时间分别留存",
            ],
            ["原十个缺失窗口", "已逐一核对原始停牌记录", "不补零收益，不重新归一化剩余持仓"],
            [
                "独立重新计算",
                "51,480 条预测文件 SHA-256 完全一致",
                "空缓存重建；价格原始响应重新解析",
            ],
            [
                "风险算术对照",
                "242 个因子/稠密案例；最大误差 3.3e-19",
                "稠密矩阵仅在独立验证路径构造",
            ],
        ],
        [125, 280, 345],
    )
    section("三项结果的阅读顺序")
    story.append(
        p(
            "先读数据质量清单和缺口证据，再看冻结风险比较及校准，最后看原生账户的实际执行和交叉预测。工程代码通过不代表模型替换通过，部分区间的历史改善也不代表前向投资有效性。"
        )
    )
    story.append(
        p(
            "详细来源：data-quality-report.html、risk-estimation-report.html、portfolio-report.html；机器可读验收：study-acceptance.json。原始预测、配置、账本和每个场景失败收据与本文件同目录归档。",
            small,
        )
    )
    story.append(PageBreak())
    story.append(p("风险估计与暴露比较", title))
    story.append(
        p(
            "5 个交易日预测、252 个交易日历史。按日期联合区块重采样 5,000 次，每块 10 个日期；日期内保留全部组合相关性。区间使用 Bonferroni 多重比较校正。ΔQLIKE 为负表示相对该实验控制组改善。"
        )
    )
    rows = []
    periods = {
        "development_2023": "2023 开发",
        "diagnostic_2024_2026": "2024-2026 诊断",
        "diagnostic_2026": "2026 诊断",
    }
    for experiment, candidate, label in (
        ("exposures", "residual_ols", "残差波动暴露 / 修正版 OLS"),
        ("estimators", "enhanced_s0.1_f90_d60_p10", "冻结增强估计 / 修正版 OLS"),
    ):
        for period, text in periods.items():
            result = risk[experiment][period]
            score = result["comparisons"][candidate]
            lo, hi = score["simultaneous_95_interval"]
            rows.append(
                [
                    label,
                    text,
                    result["complete_dates"],
                    f"{score['mean_delta']:.6f}",
                    f"[{lo:.6f}, {hi:.6f}]",
                    "是" if hi < 0 else "否",
                ]
            )
    grid(
        ["比较", "区间", "完整日期", "ΔQLIKE", "校正 95% 区间", "上界 < 0"],
        rows,
        [190, 120, 65, 90, 210, 75],
    )
    section("校准和覆盖率同样是验收条件")
    story.append(
        p(
            "不能只看点估计或历史夏普。需要同时检查波动校准、因子/特异风险贡献、原始覆盖率与失败率。完整的 12 组估计参数和 5 项消融均保留，包括无改善及变差的结果；不利用 2024-2026 结果重新挑参数。"
        )
    )
    story.append(
        p(
            "风险损失采用公开复权收益口径；原生账户另含真实费用和股东现金权益。相同组合在各模型之间使用完全相同的实际收益代理。实际账户的净收益、回撤和跟踪误差另行报告，不混为同一指标。"
        )
    )
    story.append(
        p(
            "来源：risk-summary.json、risk-calibration.json、risk-forecasts.parquet、model-configurations.json、risk-selection.json；独立复算：../independent-01/independent-recomputation.json。",
            small,
        )
    )
    story.append(PageBreak())
    story.append(p("优化组合与精确执行", title))
    story.append(
        p(
            "每 5 个交易日调仓，初始资金 100 万元；策略预算 95% / 80%，单股上限 5%，初次建仓完整计费，此后 L1 换手上限 20%。主动风险上限为年化 3% / 6%。研究样本市值基准不是市场指数。"
        )
    )
    story.append(
        p(
            "六组冻结模型负责生成组合，全部 22 个风险配置对相同连续、整数和实际收盘持仓交叉预测。不可行、风险输入缺失及整手修复失败不会自动放宽约束或切换等权。"
        )
    )
    story.append(
        p(
            "完整期间为“否”的行仅是安全区间诊断，不能与其他终点收益排序。全部调仓失败而留在现金的场景也不是成功组合。预测 TE 约束与表中的未来实际 TE 是两种量，不能相互替代。"
        )
    )
    with (root / "portfolio-comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        accounts = list(csv.DictReader(stream))
    methods = {
        "equal": "等权",
        "minimum_variance": "最小方差",
        "minimum_active_risk": "最小主动风险",
        "cost_aware": "成本感知",
    }

    def pct(value):
        return "-" if not value else f"{float(value) * 100:.2f}%"

    rows = []
    for row in accounts:
        rows.append(
            [
                row["场景"],
                methods.get(row["方法"], row["方法"]),
                row["模型"],
                pct(row["预算"]),
                pct(row["TE上限"]),
                row["回放终点"] or "-",
                "是" if row["完整期间"] == "True" else "否",
                row["目标成功数"] or "0",
                pct(row["实际净收益"]),
                pct(row["最大回撤"]),
                pct(row["实际年化TE"]),
                "-" if not row["成功调仓P95秒"] else f"{float(row['成功调仓P95秒']):.2f}",
            ]
        )
    grid(
        [
            "ID",
            "方法",
            "模型",
            "预算",
            "TE限",
            "终点",
            "完整",
            "目标数",
            "净收益",
            "回撤",
            "实际TE",
            "成功P95秒",
        ],
        rows,
        [24, 68, 120, 45, 44, 67, 32, 40, 59, 55, 55, 51],
    )
    section("执行限制与失败证据")
    story.append(
        p(
            "除权参考价与股东实际权益分开。无法精确确认的股份配送、零碎股份或其他执行输入会使完整回放回滚，另行重建的此前区间只用于诊断。不得据此虚构后续净值、股份或现金。整数修复不声称全局最优。"
        )
    )
    story.append(
        p(
            "来源：portfolio-comparison.csv、account-independent-checks.json、optimizer-risk-underestimation.csv、optimizer-cross-inference.json，以及 portfolios/<场景身份>/ 下的订单、成交、持仓、费用、求解诊断和停止收据。",
            small,
        )
    )
    story.append(PageBreak())
    story.append(p("工程、存储与恢复", title))
    section("可复现边界")
    story.append(
        p(
            "代码 PR、固定环境、实际导入路径和 CI 证据记录在 release-evidence.json。每个场景身份绑定输入摘要、配置和运行源码。源代码的精确字节保存在 producer-source 内容寻址副本中，性能改动不会隐藏为同一版本。"
        )
    )
    story.append(
        p(
            "工程验收保留原覆盖率门槛。合成边界测试覆盖停牌、不可卖、部分成交、费用、财报修订、未来数据污染、缓存失效和未确认股份权益；不声称这些特殊情况全部经过真实 200 家样本验证。"
        )
    )
    recovery = root.parents[1] / "supplemental-recovery-receipt.json"
    if recovery.exists():
        data = read(recovery)
        grid(
            ["目录/盘", "记录值", "预算或说明"],
            [
                [
                    "D 研究目录",
                    f"{data['data_bytes'] / 1024**3:.2f} GiB",
                    "预算 25 GiB；最终占用以恢复收据为准",
                ],
                [
                    "F 恢复副本",
                    f"{data['archive_bytes'] / 1024**3:.2f} GiB",
                    "预算 20 GiB；同机副本，不声称独立硬件备份",
                ],
                [
                    "C 代码与环境",
                    f"{data['code_environment_bytes'] / 1024**3:.2f} GiB",
                    "预算 5 GiB；保留至少 40 GiB 空闲",
                ],
                [
                    "C / D / E / F 空闲",
                    " / ".join(str(data["free_gib"][d]) for d in "CDEF") + " GiB",
                    "实测时点及摘要见恢复收据",
                ],
            ],
            [150, 280, 320],
        )
    section("恢复步骤")
    for text in (
        "1. 保留原始目录、旧环境、账户记录和本次失败证据，不覆盖或删除它们。",
        "2. 按 input-manifest.json 与 supplemental/manifest.json，从 F 盘对象库恢复到新的 D 盘目录，逐文件核对 SHA-256。",
        "3. 用记录的固定提交和依赖做普通安装，核对实际导入路径、版本、源码摘要及 pip check。",
        "4. 从空缓存运行独立重算，再运行固定批次；禁止将缓存损坏、输入缺失或求解失败转为等权回退。",
        "5. 本轮是隔离研究，没有切换模拟服务或冻结账户；模型替换与策略推广另行验收。",
    ):
        story.append(p(text))
    story.append(
        p(
            "恢复与可复算命令见仓库 research/public_a_share/README.md。报告数据截至 2026-09-30，工程记录以各收据实际生成时间为准。",
            small,
        )
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(target),
        pagesize=landscape(A4),
        rightMargin=40,
        leftMargin=40,
        topMargin=35,
        bottomMargin=38,
        title="Barra 200 家历史样本验证",
        author="Quant Research",
    )

    def footer(canvas, document):
        canvas.setFont("StudyCJK", 8)
        canvas.setFillColor(colors.HexColor("#536575"))
        canvas.drawString(40, 20, "固定样本研究 | 回顾性诊断 | 默认模型与冻结账户未变更")
        canvas.drawRightString(802, 20, str(document.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.pdf, args.font)
