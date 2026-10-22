"""Full-report discovery without relying on a specific Chinese title spelling."""

from __future__ import annotations

import re

EXCLUDED = re.compile(
    r"取消|英文|摘要|提示|董事|监事|核查|审核|专项|独立|审计|问询|回复|公告|工作报告"
)
PERIOD = re.compile(r"(20\d{2})年?(第一季度|一季度|第三季度|三季度|半年度|半年|年度|年)(?:报告|报)")


def is_full_report(title):
    title = re.sub(r"\s+|<[^>]+>", "", title)
    return bool(re.search(r"年度报告|季度报告|年报", title)) and not EXCLUDED.search(title)


def title_period(title):
    """A yearless annual title is discoverable, but has no inferred fiscal year."""
    title = re.sub(r"\s+", "", title)
    match = PERIOD.search(title)
    if not match:
        return None
    suffix = {
        "第一季度": "0331",
        "一季度": "0331",
        "第三季度": "0930",
        "三季度": "0930",
        "半年度": "0630",
        "半年": "0630",
        "年度": "1231",
        "年": "1231",
    }[match[2]]
    return match[1] + suffix


def original_period(title, cover, published):
    """The original cover controls; neither publication year nor a future date does."""
    compact = re.sub(r"\s+", "", cover)
    periods = {title_period(m[0]) for m in PERIOD.finditer(compact)} - {None}
    if len(periods) > 1:
        raise ValueError("ambiguous_original_period")
    result = next(iter(periods)) if periods else title_period(title)
    if not result or result > str(published)[:10].replace("-", ""):
        raise ValueError("unresolved_or_future_original_period")
    return result
