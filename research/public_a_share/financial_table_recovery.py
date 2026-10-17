"""Opt-in, evidence-bearing annual table recovery. Never changes the v1 parser.

Cells are interpreted only in identified issuer summary/consolidated statements.
Column years, financial scope, display units and audit status stay in evidence.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

SCALE = {"元": 1, "千元": 1000, "万元": 10000, "百万元": 1000000, "亿元": 100000000}
FIELDS = {
    "revenue_ytd": r"(?:[一二三四五六七八九十]+[、.．])?(?:其中[:：])?营业收入(?:[（(]元[)）])?",
    "total_revenue": r"(?:[一二三四五六七八九十]+[、.．])?营业总收入(?:[（(]元[)）])?",
    "profit_ytd": r"(?:[一二三四五六七八九十]+[、.．])?(?:\d+[.、])?[（(]?[一二三四五六七八九十]*[)）]?归属[于於]?(?:上市公司|挂牌公司|母公司)(?:的)?(?:股东|所有者)(?:的)?净利润.*",
    "book_equity": r"归属[于於]?(?:上市公司|挂牌公司|母公司)(?:的)?(?:股东|所有者)(?:的)?(?:净资产|权益合计|所有者权益合计).*",
    "assets": r"(?:资产总计|总资产)(?:[（(]元[)）])?",
    "liabilities": r"(?:负债合计|总负债)(?:[（(]元[)）])?",
}


def compact(value):
    return re.sub(r"\s+", "", str(value or "")).replace("，", ",").replace("−", "-")


def number(cell):
    value = compact(cell).replace("﹣", "-")
    if not value or "%" in value or "％" in value:
        return None
    if value.startswith(("(", "（")) and value.endswith((")", "）")):
        value = "-" + value[1:-1]
    if not re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", value):
        return None
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None


def field_name(label):
    label = compact(label)
    # Nonrecurring, subsidiary, per-share and attribution ratios are different facts.
    if re.search(r"扣除|每股|比例|比重|同比|净利润率|增长率", label):
        return None
    return next((field for field, pattern in FIELDS.items() if re.fullmatch(pattern, label)), None)


def units(text):
    found = re.findall(
        r"(?:单位(?:为|[:：])?|金额单位(?:为|[:：])?)\s*(?:人民币)?\s*(百万元|千元|万元|亿元|元)",
        text,
    )
    return SCALE[found[-1]] if found else None


def table_facts(rows, *, page, context, scope, report_year, bbox=None, cell_boxes=None):
    """Recover annual observations from explicitly year-labelled cells.

    A merged header propagates only into columns explicitly labelled adjusted.
    Quarter tables, ambiguous year spans and unknown units are rejected.
    """
    if scope not in {"annual_summary", "consolidated_income", "consolidated_balance"}:
        raise ValueError("unapproved_financial_scope")
    width = max((len(row) for row in rows), default=0)
    if width < 3:
        return []
    rows = [list(row) + [None] * (width - len(row)) for row in rows]

    # A PDF grid may split a visually single label/amount cell into empty subcolumns.
    def row_label(row):
        return next((compact(c) for c in row if re.search(r"[\u3400-\u9fff]", compact(c))), "")

    labels = [row_label(row) for row in rows]
    first_data = next((i for i, label in enumerate(labels) if field_name(label)), None)
    if first_data is None:
        return []
    headers = rows[:first_data]
    header_text = compact(" ".join(str(c or "") for row in headers for c in row))
    if re.search(r"1月1日|年初余额|调整数", header_text):
        # Opening-balance transition reconciliations are not annual observations.
        return []
    if re.search(r"第一季度|第二季度|第三季度|第四季度|本报告期|1[—–-]3月|1[—–-]9月", header_text):
        return []
    columns = {}
    boundaries = []
    for j in range(1, width):
        cell_header = compact(" ".join(str(row[j] or "") for row in headers))
        if re.search(r"20\d{2}|增减|增长|比例|调整|追溯", cell_header):
            boundaries.append(j)
        if re.search(r"增减|增长|比例|调整前|追溯前", cell_header):
            continue
        years = set(re.findall(r"(20\d{2})年?", cell_header))
        if len(years) == 1:
            columns[j] = int(next(iter(years)))
        elif not years and re.search(r"调整后|重述后|追溯后", cell_header):
            # Recover a merged year heading, not an arbitrary unlabelled column.
            for k in range(j - 1, 0, -1):
                before = compact(" ".join(str(row[k] or "") for row in headers))
                previous = set(re.findall(r"(20\d{2})年?", before))
                if previous:
                    if len(previous) == 1:
                        columns[j] = int(next(iter(previous)))
                    break
        elif not years and scope == "annual_summary":
            if cell_header in {"本期", "本期期末", "本年", "本年末"}:
                columns[j] = report_year
            elif cell_header in {"上年同期", "上年期末", "上年", "上年末"}:
                columns[j] = report_year - 1
    # A merged year span is not an additional observation when adjusted columns
    # explicitly identify which version is required.
    adjusted = {
        year
        for j, year in columns.items()
        if re.search("调整后|重述后|追溯后", "".join(str(row[j] or "") for row in headers))
    }
    columns = {
        j: year
        for j, year in columns.items()
        if year not in adjusted
        or re.search("调整后|重述后|追溯后", "".join(str(row[j] or "") for row in headers))
    }
    result = []
    for i, row in enumerate(rows[first_data:], first_data):
        label = labels[i]
        if not re.match(
            r"(?:[一二三四五六七八九十]+[、.．])?(?:其中[:：])?(?:营业|归属|总资产|资产总|负债合计|总负债)",
            label,
        ):
            continue
        end = i + 1
        while end < len(rows) and not re.match(
            r"营业|归属|总资产|资产总|负债|总负债|经营|基本|稀释|加权|利润|主要|扣除", labels[end]
        ):
            # A new year header ends a financial row.
            if re.search(r"20\d{2}年", labels[end]):
                break
            label += labels[end]
            end += 1
        field = field_name(label)
        if not field:
            continue
        inline_unit = re.search(r"[（(](百万元|千元|万元|亿元|元)[)）]", label)
        scale = SCALE[inline_unit[1]] if inline_unit else units(context)
        if scale is None:
            continue
        for j, year in columns.items():
            if not report_year - 2 <= year <= report_year:
                continue
            stop = next((k for k in boundaries if k > j), width)
            cells = []
            if cell_boxes is not None:
                header_boxes = [
                    cell_boxes[h][j]
                    for h in range(first_data)
                    if headers[h][j] and cell_boxes[h][j] is not None
                ]
                if not header_boxes:
                    continue
                box = header_boxes[-1]
                center = (box[0] + box[2]) / 2
                pieces = []
                if i and not labels[i - 1]:
                    before = [
                        compact(rows[i - 1][k])
                        for k, b in enumerate(cell_boxes[i - 1])
                        if b is not None and b[0] < center < b[2] and rows[i - 1][k]
                    ]
                    if (
                        len(before) == 1
                        and number(before[0]) is None
                        and re.fullmatch(r"[-+0-9,.]+", before[0])
                    ):
                        pieces.extend(before)
                for t in range(i, end):
                    matches = [
                        compact(rows[t][k])
                        for k, b in enumerate(cell_boxes[t])
                        if b is not None and b[0] < center < b[2] and rows[t][k]
                    ]
                    if len(matches) > 1:
                        pieces = []
                        break
                    pieces.extend(matches)
                token = "".join(pieces)
                if number(token) is not None:
                    cells.append(token)
                indexes = []
            else:
                anchor = j
                while anchor > 0 and row[anchor] is None:
                    anchor -= 1
                indexes = [anchor] if number(row[anchor]) is not None else range(j, stop)
            for k in indexes:
                pieces = [compact(rows[t][k]) for t in range(i, end)]
                if (
                    i
                    and not labels[i - 1]
                    and compact(rows[i - 1][k])
                    and number(rows[i - 1][k]) is None
                ):
                    pieces.insert(0, compact(rows[i - 1][k]))
                token = "".join(pieces)
                if number(token) is not None:
                    cells.append(token)
            if len(cells) != 1:
                continue
            value = number(cells[0])
            if value is None:
                continue
            amount = float(value * scale)
            if not math.isfinite(amount) or (field in {"assets", "liabilities"} and amount <= 0):
                continue
            result.append(
                {
                    "field": field,
                    "report_period": f"{year}1231",
                    "value": amount,
                    "page": page,
                    "row": i,
                    "column": j,
                    "label": label,
                    "raw_cell": cells[0],
                    "scale": scale,
                    "scope": scope,
                    "column_headers": [str(h[j] or "") for h in headers],
                    "bbox": bbox,
                    "comparison_year": year != report_year,
                }
            )
    return result


def audit_opinion(text):
    """Flag problematic audit opinions; presence of a statement is not admission."""
    front = compact(text[:18000])
    if re.search(r"出具.{0,40}无法表示意见|出具.{0,40}否定意见", front):
        return "disclaimer_or_adverse"
    if re.search(r"出具.{0,60}(?<!无)保留意见", front):
        return "qualified"
    return "not_classified"


def extract_tables(pdf, report_year):
    """Single-process PDF extraction with bounded issuer statement regions."""
    import pymupdf

    output = []
    with pymupdf.open(pdf) as document:
        texts = [p.get_text(sort=False) for p in document]
        section = None
        section_start = -100
        preceding_header = None
        for i, (page, text) in enumerate(zip(document, texts, strict=True)):
            clean = compact(text)
            # Start only at actual tables, not contents listings or references.
            income = re.search(
                r"合并(?:及公司)?利润表(?:[（(][^）)]*[)）])?(?:编制|单位|20\d{2}|项目)", clean
            )
            balance = re.search(
                r"合并(?:及公司)?资产负债表(?:[（(][^）)]*[)）])?(?:编制|单位|20\d{2}|项目)", clean
            )
            if income or balance:
                section = "consolidated_income" if income else "consolidated_balance"
                section_start = i
            if i > section_start and re.search(r"母公司(?:资产负债|利润)表|合并现金流量表", clean):
                section = None
            scope = section if section and i - section_start <= 4 else None
            if (
                not scope
                and i < 30
                and re.search(r"主要会计数据|主要财务指标|盈利能力|偿债能力", clean)
            ):
                scope = "annual_summary"
            # Continuation of the preceding annual summary, with no quarter label.
            if (
                not scope
                and i < 30
                and i
                and re.search(r"主要会计数据", compact(texts[i - 1]))
                and "分季度" not in clean
            ):
                scope = "annual_summary"
            if not scope:
                continue
            tables = list(page.find_tables().tables)
            for table in tables:
                top = table.bbox[1]
                # Unit context ends at the table top; table cells supply row-level units.
                prefix = page.get_text(clip=(0, max(0, top - 180), page.rect.width, top))
                if not units(prefix) and scope == "annual_summary" and i:
                    prior = texts[i - 1]
                    # Unit line can immediately precede the summary on the previous page.
                    if "主要会计数据" in compact(prior[-1600:]):
                        prefix = prior[-1600:] + prefix
                if not units(prefix) and section and i > section_start:
                    prefix = texts[section_start][:1800] + prefix
                rows = table.extract()
                boxes = [list(row.cells) for row in table.rows]
                # Repeated physical column coordinates identify a header split
                # across adjacent pages. No guessed year from publication date.
                if preceding_header and preceding_header["page"] == i - 1 and table is tables[0]:
                    old = preceding_header
                    if (
                        rows
                        and len(rows[0]) == len(old["rows"][0])
                        and abs(table.bbox[0] - old["bbox"][0]) < 2
                        and abs(table.bbox[2] - old["bbox"][2]) < 2
                    ):
                        first = next(
                            (k for k, r in enumerate(rows) if any(field_name(c) for c in r)), 0
                        )
                        if not re.search(r"20\d{2}", str(rows[:first])):
                            rows = old["rows"] + rows
                            boxes = old["boxes"] + boxes
                output.extend(
                    table_facts(
                        rows,
                        page=i + 1,
                        context=prefix,
                        scope=scope,
                        report_year=report_year,
                        bbox=list(table.bbox),
                        cell_boxes=boxes,
                    )
                )
                if (
                    table is tables[-1]
                    and table.bbox[3] > page.rect.height - 100
                    and re.search(r"20\d{2}", str(rows))
                    and not any(field_name(c) for row in rows for c in row)
                ):
                    preceding_header = {"page": i, "rows": rows, "boxes": boxes, "bbox": table.bbox}
        return output, audit_opinion("\n".join(texts[:30]))


def agree(candidates):
    """Do not accept conflicting same-scope values by taking an arbitrary row."""
    if not candidates:
        return None
    first = candidates[0]
    for other in candidates[1:]:
        # Rounded display amounts may differ by one reported unit at most.
        tolerance = max(first["scale"], other["scale"])
        if abs(first["value"] - other["value"]) > tolerance:
            return None
    return min(candidates, key=lambda c: (c["scale"], c["page"], c["row"]))
