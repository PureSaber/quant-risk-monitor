"""Explicit admission of recovered evidence; no writes to the frozen fact store."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

KEY = ["asset", "field", "report_period", "available_at"]
FINANCIAL_FIELDS = {"revenue_ytd", "profit_ytd", "assets", "liabilities", "book_equity"}


def opinion_evidence(text, year):
    """Read the current opinion, not a later discussion of last year's opinion.

    Unclassified is deliberately not treated as an unqualified opinion. Caller
    retains the text and page for review; a mirror alone is never sufficient.
    """
    text = re.sub(r"\s+", "", text)
    patterns = [
        r"审计意见类型[:：]?([^\n]{0,50}?(?:无保留意见|(?<!无)保留意见|无法表示意见|否定意见))",
        r"会计师事务所.{0,150}?出具.{0,65}?(?:无保留意见|(?<!无)保留意见|无法表示意见|否定意见)",
    ]
    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            excerpt = match[0]
            if "内部控制" in text[max(0, match.start() - 30) : match.end() + 30]:
                continue
            years = re.findall(r"(20\d{2})年", excerpt)
            if years and str(year) not in years:
                continue
            candidates.append((match.start(), excerpt))
    if not candidates:
        return {"status": "unclassified", "excerpt": ""}
    _, excerpt = min(candidates)
    if "无法表示意见" in excerpt or "否定意见" in excerpt:
        status = "disclaimer_or_adverse"
    elif "无保留意见" in excerpt:
        status = (
            "unqualified_with_emphasis"
            if re.search(r"强调|重大不确定性|非标准", excerpt)
            else "unqualified"
        )
    else:
        status = "qualified"
    return {"status": status, "excerpt": excerpt}


def normalize_facts(frame):
    frame = frame.copy()
    frame["asset"] = frame.asset.astype(str).str.zfill(6)
    frame["report_period"] = frame.report_period.astype(str)
    frame["available_at"] = pd.to_datetime(frame.available_at, utc=True, format="mixed")
    frame["value"] = pd.to_numeric(frame.value, errors="raise")
    if not np.isfinite(frame.value).all():
        raise ValueError("nonfinite_financial_fact")
    ends = pd.to_datetime(frame.report_period, format="%Y%m%d").dt.tz_localize("Asia/Shanghai")
    if (ends > frame.available_at).any():
        raise ValueError("fact_before_period_end")
    return frame


def merge_evidence(original, additions, *, reviewed_corrections=()):
    """Keep original vintages, quarantine disagreement, and retain every decision.

    Different publication times are separate revisions. A same-time conflict is
    never resolved by file ordering or overwriting the original numeric value.
    """
    original = normalize_facts(original)
    if original.duplicated(KEY).any():
        raise ValueError("ambiguous_original_vintages")
    if additions.empty:
        return original, pd.DataFrame()
    additions = normalize_facts(additions)
    original_keys = {tuple(r[k] for k in KEY): r for r in original.to_dict("records")}
    corrections = {}
    for record in reviewed_corrections:
        record = dict(record)
        record["available_at"] = pd.Timestamp(record["available_at"])
        key = tuple(record[k] for k in KEY)
        if key in corrections or not record.get("review_basis"):
            raise ValueError("ambiguous_or_unsupported_correction")
        corrections[key] = record
    accepted, decisions, applied = [], [], set()
    for key, group in additions.groupby(KEY, sort=True):
        rows = group.to_dict("records")
        eligible = [r for r in rows if r["admission_reason"] == "eligible"]
        chosen = None
        if eligible:
            values = np.array([r["value"] for r in eligible])
            tolerance = max(float(r["scale"]) for r in eligible)
            if np.ptp(values) > tolerance:
                reason = "same_time_source_conflict"
            elif key in original_keys:
                old = original_keys[key]
                correction = corrections.get(key)
                if correction:
                    matching = [
                        r
                        for r in eligible
                        if r["source_sha256"] == correction["source_sha256"]
                        and abs(r["value"] - correction["new_value"]) <= float(r["scale"])
                    ]
                    if old["value"] != correction["old_value"] or not matching:
                        raise ValueError("reviewed_correction_input_changed")
                    chosen = matching[0]
                    original.loc[
                        (original[KEY] == pd.Series(dict(zip(KEY, key, strict=True)))).all(axis=1),
                        "value",
                    ] = chosen["value"]
                    reason = "reviewed_correction_in_new_overlay"
                    applied.add(key)
                else:
                    reason = (
                        "agrees_with_original"
                        if abs(old["value"] - values[0]) <= tolerance
                        else "conflicts_with_original_review_required"
                    )
            else:
                chosen = min(eligible, key=lambda r: (float(r["scale"]), r["announcement_id"]))
                accepted.append(chosen)
                reason = "new_vintage_admitted"
        for row in rows:
            decision = row["admission_reason"] if row["admission_reason"] != "eligible" else reason
            decisions.append({**row, "decision": decision, "selected": row is chosen})
    if applied != set(corrections):
        raise ValueError("unapplied_reviewed_correction")
    if accepted:
        merged = pd.concat([original, pd.DataFrame(accepted)], ignore_index=True)
    else:
        merged = original.copy()
    if merged.duplicated(KEY).any():
        raise AssertionError("duplicate_after_recovery")
    return merged.sort_values(KEY), pd.DataFrame(decisions)
