"""Bounded, resumable original-PDF extraction; isolated processes only."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from financial_table_recovery import agree, compact, extract_tables
from recovery_inventory import digest, write_json
from report_discovery import original_period


def statement_sections(text, heading, period):
    """Only actual statement headings, not narrative references or old restatements."""
    prefix = (
        r"(?:\d+[、.．]\s*|[（(][一二三四五六七八九十]+[）)]\s*)?"
        r"(?:20\d{2}\s*年\s*(?:度|\d{1,2}\s*月\s*\d{1,2}\s*日)?\s*)?"
    )
    pattern = r"(?m)^\s*" + prefix + r"合并(?:及公司)?" + heading + r"\s*$"
    for match in re.finditer(pattern, text):
        tail = text[match.end() :]
        header = re.split(r"营业收入|资产总计|利润总额|负债合计", tail[:500])[0]
        first_year = re.search(r"20\d{2}", match[0] + header)
        if not first_year or first_year[0] != period[:4]:
            continue
        stop = re.search(
            r"(?m)^\s*"
            + prefix
            + r"(?:母公司|公司|合并)(?:利润表|资产负债表|现金流量表|所有者权益变动表)\s*$",
            tail,
        )
        end = match.end() + (stop.start() if stop else min(len(tail), 26000))
        pages = list(re.finditer(r"\[\[PAGE (\d+)\]\]", text[: match.start()]))
        page_marker = pages[-1][0] + "\n" if pages else ""
        section = text[match.start() : end]
        page_start = pages[-1].end() if pages else max(0, match.start() - 300)
        units_above = re.findall(
            r"单位\s*[为:：]?\s*(?:人民币)?\s*(?:百万元|千元|万元|亿元|元)",
            text[page_start:match.start()],
        )
        if units_above:
            section = match[0] + "\n" + units_above[-1] + "\n" + text[match.end():end]
        # A continuation heading may follow the currency unit on the page.
        # Keep the original statement's unit context instead of starting a new
        # amount_row region after that unit and silently interpreting yuan.
        section = re.sub(r"合并(?:及公司)?" + heading + r"[（(]续[）)]", "本表续页", section)
        # These are repeated footers inside our already bounded issuer table;
        # the legacy parser otherwise stops before its continuation page.
        section = section.replace("后附财务报表附注", "本表附注")
        section = re.sub(r"企业负责人|法定代表人", "本表签署人", section)
        yield page_marker + section


def statement_fallback(first_text, second_text, period):
    """Two PDF text engines must agree on an explicitly consolidated current year."""
    from original_financials import amount_row, normalize
    from prepare_pit import ALIASES

    aliases = {**ALIASES, "revenue_ytd": r"营业收入"}
    result = []
    for field in ["revenue_ytd", "profit_ytd", "assets", "book_equity", "liabilities"]:
        heading = "利润表" if field in {"revenue_ytd", "profit_ytd"} else "资产负债表"
        left = list(statement_sections(first_text, heading, period))
        right = list(statement_sections(second_text, heading, period))
        if len(left) != 1 or len(right) != 1:
            continue
        a = amount_row(left[0], field, period, aliases=aliases)
        b = amount_row(right[0], field, period, aliases=aliases)
        if (
            not a
            or not b
            or not a["from_consolidated_statement"]
            or not b["from_consolidated_statement"]
        ):
            continue
        if abs(a["value"] - b["value"]) > max(a["display_unit_scale"], b["display_unit_scale"]):
            continue
        # An annual consolidated statement must identify the current fiscal year.
        normalized = normalize(left[0])
        location = normalized.find(a["matched_label"])
        pages = re.findall(r"\[\[PAGE (\d+)\]\]", normalized[:location]) if location >= 0 else []
        if not pages:
            continue
        result.append(
            {
                "field": field,
                "report_period": period,
                "value": a["value"],
                "page": int(pages[-1]),
                "row": 0,
                "column": 0,
                "label": a["matched_label"],
                "raw_cell": str(a["value"]),
                "scale": a["display_unit_scale"],
                "scope": "consolidated_text",
                "column_headers": [period],
                "bbox": None,
                "comparison_year": False,
                "independent_text_match": True,
            }
        )
    return result


def one(row, cache):
    import pymupdf
    from pypdf import PdfReader

    path = Path(row["pdf"])
    sha = digest(path)
    if row["sha256"] and sha != row["sha256"]:
        raise ValueError("input_digest_mismatch")
    import hashlib

    code_sha = hashlib.sha256(
        "".join(
            digest(Path(__file__).with_name(n))
            for n in [
                "financial_table_recovery.py",
                "recovery_extract.py",
                "original_financials.py",
                "report_discovery.py",
            ]
        ).encode()
    ).hexdigest()
    target = cache / (row["announcement_id"] + ".json")
    if target.exists():
        saved = json.loads(target.read_text(encoding="utf-8"))
        if saved["sha256"] == sha and saved["parser_sha256"] == code_sha:
            return saved
        raise ValueError("cache_binding_changed_use_new_cache_version")
    with pymupdf.open(path) as doc:
        first_pages = [p.get_text() for p in doc]
        cover = "\n".join(first_pages[:2])
    period = original_period(row["title"], cover, row["published"])
    if not period.endswith("1231"):
        raise ValueError("not_annual_original")
    candidates, opinion = extract_tables(path, int(period[:4]))
    # A separate PDF engine must also recover the row label and numeric token.
    independent = PdfReader(path)
    page_text = {}
    for c in candidates:
        page = c["page"]
        if page not in page_text:
            page_text[page] = compact(independent.pages[page - 1].extract_text() or "")
        token = compact(c["raw_cell"])
        label = compact(c["label"])
        text = page_text[page]
        pos = text.find(label)
        c["independent_text_match"] = bool(pos >= 0 and token in text[pos : pos + 1200])
    first_text = "\n".join(f"[[PAGE {i + 1}]]\n{t}" for i, t in enumerate(first_pages))
    # Extract only independently identified statement pages, not every PDF note.
    selected = set()
    for heading in ("利润表", "资产负债表"):
        for section in statement_sections(first_text, heading, period):
            selected.update(int(x) for x in re.findall(r"\[\[PAGE (\d+)\]\]", section))
    second_text = "\n".join(
        f"[[PAGE {i}]]\n{independent.pages[i - 1].extract_text() or ''}" for i in sorted(selected)
    )
    candidates.extend(statement_fallback(first_text, second_text, period))
    accepted = []
    rejected = []
    frame = pd.DataFrame(candidates)
    if not frame.empty:
        for (field, p), group in frame.groupby(["field", "report_period"]):
            entries = group.to_dict("records")
            verified = [e for e in entries if e["independent_text_match"]]
            chosen = agree(verified)
            if chosen is None:
                rejected.append(
                    {
                        "field": field,
                        "report_period": p,
                        "reason": "conflicting_or_unverified_table_cells",
                        "candidates": entries,
                    }
                )
                continue
            accepted.append(
                {
                    **chosen,
                    "evidence_count": len(verified),
                    "verification": "table_geometry_and_independent_pypdf_text",
                    "comparison_kind": "summary_and_statement"
                    if len(set(e["scope"] for e in verified)) > 1
                    else "single_issuer_table_two_readers",
                }
            )
    result = {
        **row,
        "sha256": sha,
        "parser_sha256": code_sha,
        "original_report_period": period,
        "audit_opinion": opinion,
        "accepted": accepted,
        "rejected": rejected,
        "candidate_count": len(candidates),
    }
    write_json(target, result)
    return result


def run(inventory, cache):
    rows = pd.read_csv(inventory, dtype=str, keep_default_na=False)
    rows = rows[rows.selected.str.lower() == "true"]
    cache.mkdir(parents=True, exist_ok=True)
    statuses = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(one, r, cache): r for r in rows.to_dict("records") if r["pdf"]}
        for i, future in enumerate(as_completed(jobs), 1):
            row = jobs[future]
            try:
                result = future.result()
                statuses.append(
                    {
                        "id": row["announcement_id"],
                        "asset": row["asset"],
                        "status": "parsed",
                        "accepted": len(result["accepted"]),
                        "rejected": len(result["rejected"]),
                    }
                )
            except Exception as exc:
                statuses.append(
                    {
                        "id": row["announcement_id"],
                        "asset": row["asset"],
                        "status": "failed",
                        "error": str(exc),
                    }
                )
            if i % 10 == 0:
                print(
                    json.dumps({"completed": i, "total": len(jobs), "last": statuses[-1]}),
                    flush=True,
                )
    write_json(cache / "statuses.json", statuses)
    print(
        json.dumps(
            {
                "parsed": sum(x["status"] == "parsed" for x in statuses),
                "failed": [x for x in statuses if x["status"] == "failed"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    run(args.inventory, args.cache)
