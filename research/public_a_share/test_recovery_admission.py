import pandas as pd
import pytest
from recovery_admission import merge_evidence, opinion_evidence


def fact(value=100, at="2023-04-01T00:00:00+08:00", **extra):
    return {
        "asset": "000001",
        "field": "profit_ytd",
        "report_period": "20221231",
        "available_at": at,
        "value": value,
        **extra,
    }


def candidate(value=100, at="2023-04-01T00:00:00+08:00", **extra):
    return fact(
        value, at, scale=0.01, announcement_id="example", admission_reason="eligible", **extra
    )


def test_current_audit_does_not_inherit_last_years_disclaimer():
    text = "会计师事务所对公司2022年度出具了带强调事项段的无保留意见。2021年会计师事务所出具无法表示意见。"
    assert opinion_evidence(text, 2022)["status"] == "unqualified_with_emphasis"
    assert (
        opinion_evidence("会计师事务所对2021年度出具了无法表示意见", 2022)["status"]
        == "unclassified"
    )


@pytest.mark.parametrize(
    "text,status",
    [
        ("审计意见类型：标准无保留意见", "unqualified"),
        ("会计师事务所出具了保留意见", "qualified"),
        ("会计师事务所对本公司出具无法表示意见", "disclaimer_or_adverse"),
        ("本报告真实完整", "unclassified"),
    ],
)
def test_audit_is_explicit(text, status):
    assert opinion_evidence(text, 2023)["status"] == status


def test_same_time_conflict_never_overwrites_original():
    original = pd.DataFrame([fact()])
    merged, decision = merge_evidence(original, pd.DataFrame([candidate(200)]))
    assert merged.value.tolist() == [100]
    assert decision.decision.tolist() == ["conflicts_with_original_review_required"]


def test_new_revision_does_not_rewrite_earlier_vintage():
    merged, decisions = merge_evidence(
        pd.DataFrame([fact()]), pd.DataFrame([candidate(120, "2024-04-02T00:00:00+08:00")])
    )
    assert merged.value.tolist() == [100, 120]
    assert (decisions.decision == "new_vintage_admitted").all()


def test_qualified_or_unverified_source_is_retained_only_as_evidence():
    row = candidate(120, "2024-04-02T00:00:00+08:00")
    row["admission_reason"] = "mirror_requires_exchange_crosscheck"
    merged, decisions = merge_evidence(pd.DataFrame([fact()]), pd.DataFrame([row]))
    assert len(merged) == 1
    assert decisions.decision.iloc[0] == row["admission_reason"]


def test_duplicate_new_sources_cannot_resolve_disagreement_by_sort_order():
    additions = pd.DataFrame(
        [candidate(120, "2024-04-02T00:00:00+08:00"), candidate(150, "2024-04-02T00:00:00+08:00")]
    )
    merged, decisions = merge_evidence(pd.DataFrame([fact()]), additions)
    assert len(merged) == 1
    assert set(decisions.decision) == {"same_time_source_conflict"}


def test_impossible_vintage_is_rejected():
    with pytest.raises(ValueError, match="before_period_end"):
        merge_evidence(pd.DataFrame([fact(at="2022-01-01T00:00:00+08:00")]), pd.DataFrame())
