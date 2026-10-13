"""Extract each historical filing independently of any present-day vendor value."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pymupdf

from prepare_pit import ALIASES, period
from public_sources import ROOT

UNIT = {
    "元": 1,
    "千元": 1000,
    "万元": 10000,
    "百万元": 1000000,
    "亿元": 100000000,
    "股": 1,
    "万股": 10000,
}
NUMBER = re.compile(r"(?<![\d.])-?\d[\d,]*(?:\.\d+)?")


def capital_from_table(path, text=None):
    """Recover wrapped share-count cells, without equating yuan capital to shares."""
    candidates = None
    if text is not None:
        candidates = []
        for match in re.finditer(r"\[\[PAGE (\d+)\]\](.*?)(?=\[\[PAGE|\Z)", text, re.S):
            compact = re.sub(r"\s+", "", match[2])
            if "股份总数" in compact and re.search(r"单位[:：]股", compact):
                candidates.append((int(match[1]) - 1, match[2]))
        if not candidates:
            return None
    with pymupdf.open(path) as document:
        pages = (
            candidates
            if candidates is not None
            else [(i, document[i].get_text()) for i in range(len(document))]
        )
        for page_no, page_text in pages:
            page = document[page_no]
            compact = re.sub(r"\s+", "", page_text)
            if "股份总数" not in compact or not re.search(r"单位[:：]股", compact):
                continue
            for table in page.find_tables().tables:
                rows = table.extract()
                for i, row in enumerate(rows):
                    label = re.sub(r"\s+", "", next((str(cell) for cell in row if cell), ""))
                    if not re.fullmatch(r"(?:[一二三四五六七八九十]+[、.．]?)?股份总数", label):
                        continue
                    combined = list(row)
                    for j in range(len(row)):
                        pieces = []
                        if i and not rows[i - 1][0] and rows[i - 1][j]:
                            pieces.append(rows[i - 1][j])
                        if row[j]:
                            pieces.append(row[j])
                        if i + 1 < len(rows) and not rows[i + 1][0] and rows[i + 1][j]:
                            pieces.append(rows[i + 1][j])
                        combined[j] = re.sub(r"\s+", "", "".join(pieces))
                    candidates = []
                    for cell in combined:
                        if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.0+)?|\d{7,}(?:\.0+)?", cell):
                            value = float(cell.replace(",", ""))
                            if value >= 1e6:
                                candidates.append(value)
                    if candidates:
                        return {
                            "value": candidates[-1],
                            "matched_label": json.dumps(combined, ensure_ascii=False),
                            "display_unit_scale": 1,
                            "from_consolidated_statement": False,
                            "share_table_page": page_no + 1,
                        }
    return None


def normalize(text):
    # Join wrapped Chinese labels, while preserving separators between amounts.
    for _ in range(3):
        text = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", text)
    text = re.sub(r"(?<=\d)\s*\n\s*(?=,\d)", "", text)
    text = re.sub(
        r"(\d{1,3}(?:,\d{3})*,\d{1,2})\s*\n\s*(\d{1,2}\.\d{2})",
        lambda m: (
            m[1] + m[2] if len(m[1].rsplit(",", 1)[-1]) + len(m[2].split(".")[0]) == 3 else m[0]
        ),
        text,
    )
    text = re.sub(r"-\s+(?=\d)", "-", text)
    return re.sub(r"\s+", " ", text).replace("，", ",").replace("−", "-")


def amount_row(text, field, p, *, normalized=False, summary_only=False):
    """Return a current-period figure from a labeled financial table, with its evidence."""
    if not normalized:
        text = normalize(text)
    regions = []
    if not summary_only and field in {"profit_ytd", "revenue_ytd"}:
        # Consolidated YTD income statements avoid quarter-vs-YTD ambiguity.
        headings = list(re.finditer(r"合并(?:及公司|年初[至到]报告期末)?利润表", text))
        for h in headings:
            if re.search(r"\.{2,}|…", text[h.end() : h.end() + 40]):
                continue
            if not re.search(r"单位|项目|编制|20\d{2}年", text[h.end() : h.end() + 160]):
                continue
            if not re.match(
                r"\s*(?:[（(](?:续|除|未经)|编制|单位|项目|20\d{2}|金额|币种)",
                text[h.end() :],
            ):
                continue
            if re.match(r"\s*项目(?:及|变化|调整)", text[h.end() :]):
                continue
            end = re.search(
                r"母公司(?:年初[至到]报告期末)?利润表|合并(?:及公司)?现金流量表|企业负责人|法定代表人|后附财务报表附注",
                text[h.end() :],
            )
            section = (
                text[h.end() : h.end() + min(end.start(), 16000)]
                if end
                else text[h.end() : h.end() + 16000]
            )
            regions.append((section, True))
    elif not summary_only and field in {"assets", "book_equity", "liabilities"}:
        for h in re.finditer(r"合并(?:及公司)?资产负债表", text):
            header = text[max(0, h.start() - 100) : h.end() + 400]
            header_dates = re.findall(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", header)
            expected = (int(p[:4]), int(p[4:6]), int(p[6:]))
            if header_dates and expected not in {tuple(map(int, item)) for item in header_dates}:
                continue
            if re.search(r"\.{2,}|…", text[h.end() : h.end() + 40]):
                continue
            if not re.search(r"单位|项目|编制|20\d{2}年", text[h.end() : h.end() + 160]):
                continue
            if not re.match(
                r"\s*(?:[（(](?:续|除|未经)|编制|单位|项目|20\d{2}|金额|币种)",
                text[h.end() :],
            ):
                continue
            if re.match(r"\s*项目(?:及|变化|调整)", text[h.end() :]):
                continue
            end = re.search(
                r"母公司资产负债表|合并(?:及公司|年初[至到]报告期末)?利润表|企业负责人|法定代表人|后附财务报表附注",
                text[h.end() :],
            )
            section = (
                text[h.end() : h.end() + min(end.start(), 16000)]
                if end
                else text[h.end() : h.end() + 16000]
            )
            regions.append((section, True))
    # Summary tables are near the front. Never fall through to the financial
    # notes about an investee/subsidiary when the issuer table is scanned.
    cutoff = re.search(r"\[\[PAGE 19\]\]", text)
    fallback = text[: cutoff.start()] if cutoff and field != "shares" else text
    if field != "shares" and "[[PAGE" in text:
        for heading in re.finditer(r"主要会计数据(?:和财务指标)?|主要财务指标", fallback):
            if re.search(r"\.{2,}|…", fallback[heading.end() : heading.end() + 30]):
                continue
            region = fallback[heading.end() : heading.end() + 5500]
            region = re.split(
                r"非经常性损益项目|经营情况(?:讨论|分析)|股东(?:数量|总数)|报告期末(?:普通股|股东)",
                region,
            )[0]
            regions.append((region, False))
    else:
        regions.append((fallback, False))
    for region, statement in regions:
        for match in re.finditer(ALIASES[field], region):
            prefix = region[max(0, match.start() - 8000) : match.start()]
            tail = region[match.end() : match.end() + 650]
            if field == "shares":
                tail = re.split(r"股份变动|股东|其他说明|\[\[PAGE", tail)[0]
            # Restrict to the row, stopping at the next financial label.
            boundaries = []
            for label in list(ALIASES.values()) + [
                r"扣除非经常性",
                r"经营活动",
                r"基本每股",
                r"稀释每股",
                r"加权平均",
                r"少数股东",
                r"负债和所有者",
                r"法定代表人",
                r"公司负责人",
            ]:
                boundary = re.search(label, tail)
                if boundary:
                    boundaries.append(boundary.start())
            if boundaries:
                tail = tail[: min(boundaries)]
            # Remove note markers; they are not amounts.
            tail = re.sub(r"[（(]注\s*\d+[)）]", "", tail)
            tail = re.sub(r"[一二三四五六七八九十]+[（(]\d+[)）]", "", tail)
            units = re.findall(
                r"(?:单位\s*[为:：]?\s*|[（(]\s*)(?:人民币)?\s*(百万元|千元|万元|亿元|元|万股|股)\s*[)）]?",
                prefix + tail[:30],
            )
            scale = UNIT[units[-1]] if units else 1
            if field == "shares" and units and units[-1] not in {"股", "万股"}:
                continue
            tokens = []
            for token in NUMBER.finditer(tail):
                value = float(token[0].replace(",", ""))
                if re.search(r"[（(]\s*$", tail[: token.start()]) and re.match(
                    r"\s*[)）]", tail[token.end() :]
                ):
                    value = -abs(value)
                is_percent = bool(re.match(r"\s*[%％]", tail[token.end() :]))
                tokens.append((value, is_percent, token.start(), token.end()))
            amounts = [x for x in tokens if not x[1]]
            if not amounts:
                continue
            if field == "shares" and tokens[0][1]:
                # "占公司总股本 30%" describes a shareholder, not total shares.
                continue
            # References in prose, ratios and note indices are not financial rows.
            leading = tail[: amounts[0][2]]
            # Permit units and the standardized loss/sign annotation, not prose.
            plain_leading = re.sub(r"[（(][^()（）]*[)）]", "", leading)
            if re.search(r"[\u3400-\u9fff]", plain_leading):
                continue
            if field == "shares" and re.search(
                r"[\u3400-\u9fff]", re.sub(r"[（(](?:万)?股[)）]", "", leading)
            ):
                continue
            if re.search(
                r"同比|增长|下降|增加|减少|详见|详阅|不超过|不低于|承诺|预计|预测|目标|约为|审计|变动原因",
                leading,
            ):
                continue
            if (
                field in {"assets", "book_equity", "liabilities", "shares"}
                and len(amounts) < 2
                and not (field == "shares" and re.search(r"总股本|股份总数", match[0]))
            ):
                continue
            chosen = amounts[0]
            if field in {"profit_ytd", "revenue_ytd"} and p.endswith("0930") and statement:
                column_header = region[:1000].split("项目", 1)[-1]
                columns = re.findall(r"(20\d{2})\s*年\s*(第三季度|前三季度)", column_header)
                if columns:
                    target = next(
                        (i for i, item in enumerate(columns) if item == (p[:4], "前三季度")), None
                    )
                    if target is None or target >= len(amounts):
                        continue
                    chosen = amounts[target]
                else:
                    ranges = re.findall(r"(20\d{2})\s*年\s*([17])\s*[—–-]\s*9\s*月", column_header)
                    if ranges:
                        target = next(
                            (i for i, item in enumerate(ranges) if item == (p[:4], "1")), None
                        )
                        if target is None or target >= len(amounts):
                            continue
                        chosen = amounts[target]
            if field == "shares" and "股份总数" in match[0] and sum(t[1] for t in tokens) >= 2:
                chosen = amounts[-1]
            if field in {"profit_ytd", "revenue_ytd"} and p.endswith("0930") and not statement:
                header = prefix[-1600:]
                has_quarter = bool(re.search(r"本报告期(?!末)", header))
                has_ytd = bool(re.search(r"年初至(?:本)?报告期(?:期)?末", header))
                if has_quarter and has_ytd:
                    percent = next((x for x in tokens if x[1]), None)
                    after = [x for x in amounts if percent is not None and x[2] > percent[3]]
                    if after:
                        chosen = after[0]
                    elif len(amounts) == 2:
                        chosen = amounts[1]
                    else:
                        continue
                elif not has_ytd:
                    continue
            value = chosen[0] * scale
            if field in {"assets", "liabilities", "shares"} and value <= 0:
                continue
            if field in {"assets", "liabilities"} and abs(value) < 1e6:
                continue
            if field == "shares" and value < 1e6:
                continue
            # Header years / numbered notes are too small to be standalone yuan income.
            if field in {"profit_ytd", "revenue_ytd"} and scale == 1 and 0 < abs(value) < 10000:
                continue
            return {
                "value": value,
                "matched_label": match[0] + tail[:300],
                "display_unit_scale": scale,
                "from_consolidated_statement": statement,
            }
    return None


def disclosure_period(title, text, announcement_date):
    """The original cover controls a mislabelled directory title, never a future period."""
    cover = text.split("[[PAGE 2]]", 1)[0]
    cover = re.sub(r"\s+", "", cover)
    titles = re.findall(r"20\d{2}年(?:第一季度|第三季度|半年度|年度)报告", cover)
    periods = {period(value) for value in titles}
    if len(periods) > 1:
        return None
    result = next(iter(periods)) if periods else period(title)
    if result and pd.Timestamp(result).date() > announcement_date.date():
        return None
    return result


def extract_member(member):
    records = []
    missing = []
    code = member["code"]
    index = json.loads(
        (ROOT / "raw/announcements" / (code + "-index.json")).read_text(encoding="utf-8")
    )
    if "--partial" not in sys.argv and (
        index.get("errors") or len(index.get("reports", [])) != index["report_count"]
    ):
        raise ValueError(f"Original document archive incomplete for {code}")
    selected_ids = {str(r["id"]) for r in index.get("reports", [])}
    for ann in index["all_records"]:
        if str(ann["announcementId"]) not in selected_ids:
            continue
        p = period(ann["announcementTitle"])
        if p is None or not "20190331" <= p <= "20250930":
            continue
        path = ROOT / "raw/filings" / code / (str(ann["announcementId"]) + ".logical.txt")
        if not path.exists():
            continue
        marker = path.with_suffix(".extract.json")
        if "--partial" not in sys.argv and (
            not marker.exists()
            or not json.loads(marker.read_text(encoding="utf8")).get("isolated_process")
        ):
            raise ValueError(f"Original document extraction is not independently complete: {path}")
        text = path.read_text(encoding="utf-8")
        normalized_text = normalize(text)
        ann_date = (
            pd.Timestamp(ann["announcementTime"], unit="ms", tz="UTC")
            .tz_convert("Asia/Shanghai")
            .normalize()
        )
        p = disclosure_period(ann["announcementTitle"], text, ann_date)
        if p is None or not "20190331" <= p <= "20250930":
            missing.append(
                {
                    "asset": code,
                    "announcement_id": ann["announcementId"],
                    "reason": "ambiguous_or_future_original_report_period",
                }
            )
            continue
        for field in ALIASES:
            parsed = amount_row(normalized_text, field, p, normalized=True)
            if (
                parsed
                and parsed["from_consolidated_statement"]
                and field in {"assets", "book_equity", "profit_ytd", "revenue_ytd"}
            ):
                summary = amount_row(normalized_text, field, p, normalized=True, summary_only=True)
                if (
                    summary
                    and abs(parsed["value"] - summary["value"])
                    > max(parsed["display_unit_scale"], summary["display_unit_scale"], 1) * 1.01
                ):
                    missing.append(
                        {
                            "asset": code,
                            "report_period": p,
                            "field": field,
                            "announcement_id": ann["announcementId"],
                            "reason": "same_original_summary_statement_disagreement",
                            "statement_value": parsed["value"],
                            "summary_value": summary["value"],
                        }
                    )
                    continue
            if field == "shares" and parsed is None:
                parsed = capital_from_table(
                    ROOT / "raw/filings" / code / (str(ann["announcementId"]) + ".pdf"), text
                )
            if parsed:
                records.append(
                    {
                        "asset": code,
                        "report_period": p,
                        "field": field,
                        **parsed,
                        "directory_report_period": period(ann["announcementTitle"]),
                        "available_at": (ann_date + pd.Timedelta(days=1)).isoformat(),
                        "announcement_id": str(ann["announcementId"]),
                        "document": "raw/filings/"
                        + code
                        + "/"
                        + str(ann["announcementId"])
                        + ".pdf",
                        "text_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "source": "independent_original_cninfo_disclosure",
                        "revision_rank": int(
                            bool(re.search("更正|修订|更新", ann["announcementTitle"]))
                        ),
                    }
                )
            else:
                missing.append(
                    {
                        "asset": code,
                        "report_period": p,
                        "field": field,
                        "announcement_id": ann["announcementId"],
                        "reason": "original_table_not_unambiguously_parsed",
                    }
                )
    return records, missing


def build():
    records = []
    missing = []
    members = json.loads((ROOT / "cohort.json").read_text(encoding="utf-8"))["members"]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(extract_member, member) for member in members]
        for i, future in enumerate(as_completed(futures), 1):
            found, absent = future.result()
            records.extend(found)
            missing.extend(absent)
            if i % 25 == 0:
                print(json.dumps({"financial_issuers_parsed": i}), flush=True)
    out = ROOT / "clean"
    out.mkdir(exist_ok=True)
    data = pd.DataFrame(records).sort_values(
        [
            "asset",
            "report_period",
            "field",
            "available_at",
            "revision_rank",
            "announcement_id",
        ]
    )
    data = data.drop_duplicates(["asset", "report_period", "field", "available_at"], keep="last")
    data.to_csv(out / "financial_facts.csv", index=False, encoding="utf-8")
    pd.DataFrame(missing).to_csv(
        out / "unmatched_financial_facts.csv", index=False, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "independent_original_facts": len(data),
                "assets": data.asset.nunique(),
                "by_field": data.groupby("field").size().to_dict(),
                "unmatched": len(missing),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    build()
