import json

import pandas as pd
import pytest
from financial_table_recovery import agree, audit_opinion, field_name, number, table_facts
from report_discovery import is_full_report, original_period, title_period


@pytest.mark.parametrize(
    "title", ["2020年度报告", "2021年年报", "2024年报", "锦江酒店年报", "2025年年度报告（修订版）"]
)
def test_discovery_includes_real_full_report_title_variants(title):
    assert is_full_report(title)


@pytest.mark.parametrize(
    "title",
    [
        "2020年年度报告摘要",
        "关于2020年年度报告更正的公告",
        "2020年年报问询回复",
        "2020年年度报告（英文版）",
    ],
)
def test_discovery_excludes_supplements_and_summaries(title):
    assert not is_full_report(title)


def test_yearless_report_needs_original_cover_and_never_publication_year():
    assert title_period("锦江酒店年报") is None
    assert original_period("锦江酒店年报", "2023 年年度报告", "2024-04-02") == "20231231"
    assert original_period("2022年年度报告", "2023年年度报告", "2024-04-02") == "20231231"
    with pytest.raises(ValueError):
        original_period("锦江酒店年报", "无明确所属期", "2024-04-02")
    with pytest.raises(ValueError):
        original_period("2025年年度报告", "2025年年度报告", "2025-04-02")


def test_table_units_comparatives_wrapped_amounts_and_negative_profit():
    rows = [
        ["项目", "2023年", "2022年", "本年比上年增减", "2021年"],
        ["营业收入（元）", "1,929,441,\n918.97", "1,000,000.00", "92.94%", "900,000.00"],
        ["归属于上市公司股东的净利润", "(12.50)", "20.00", "-162.5%", "8.00"],
    ]
    facts = table_facts(
        rows, page=7, context="单位：人民币百万元", scope="annual_summary", report_year=2023
    )
    values = {(r["field"], r["report_period"]): r["value"] for r in facts}
    assert values["revenue_ytd", "20231231"] == 1929441918.97
    assert values["profit_ytd", "20231231"] == -12500000
    assert values["profit_ytd", "20211231"] == 8000000
    assert len(facts) == 6


def test_quarter_summary_and_unknown_units_are_not_annual_facts():
    rows = [
        ["项目", "第一季度", "第二季度", "第三季度", "第四季度"],
        ["营业收入", "12", "13", "14", "15"],
    ]
    assert not table_facts(
        rows, page=9, context="单位：元", scope="annual_summary", report_year=2023
    )
    rows = [["项目", "2023年", "2022年"], ["营业收入", "120000", "130000"]]
    assert not table_facts(rows, page=9, context="", scope="annual_summary", report_year=2023)


def test_adjusted_comparative_columns_are_explicit_and_do_not_shift_years():
    rows = [
        ["项目", "2023年", "2022年", None, "2021年", None],
        [None, None, "调整前", "调整后", "调整前", "调整后"],
        ["营业收入（元）", "100", "90", "85", "80", "75"],
    ]
    found = table_facts(rows, page=7, context="", scope="annual_summary", report_year=2023)
    assert {r["report_period"]: r["value"] for r in found} == {
        "20231231": 100,
        "20221231": 85,
        "20211231": 75,
    }


def test_operating_income_and_total_income_are_distinct():
    assert field_name("一、营业总收入") == "total_revenue"
    assert field_name("其中：营业收入") == "revenue_ytd"
    assert field_name("归属于上市公司股东的扣除非经常性损益的净利润（元）") is None
    assert number("1 2 3") == 123
    assert number("20.4%") is None
    assert number("五(1)") is None
    assert (
        agree(
            [
                {"value": 100, "scale": 1, "page": 1, "row": 1},
                {"value": 105, "scale": 1, "page": 2, "row": 1},
            ]
        )
        is None
    )


def test_audit_disclaimer_cannot_become_clean_by_presence_of_report():
    assert audit_opinion("本公司被出具无法表示意见的审计报告") == "disclaimer_or_adverse"
    assert audit_opinion("审计师出具带持续经营段落的保留意见") == "qualified"
    assert audit_opinion("出具无保留意见") == "not_classified"


def test_incomplete_and_repeated_pagination_is_not_a_success(monkeypatch):
    import filings

    def response(batch, total):
        return json.dumps({"announcements": batch, "totalAnnouncement": total})

    row = {"secCode": "000001", "announcementId": "a"}
    replies = iter([response([row], 2), response([], 2)])
    monkeypatch.setattr(filings, "fetch", lambda *a, **k: next(replies))
    with pytest.raises(ValueError, match="Empty"):
        filings.announcements("000001", "org")
    replies = iter([response([row], 2), response([row], 2)])
    with pytest.raises(ValueError, match="Repeated"):
        filings.announcements("000001", "org")


def test_recovered_comparative_preserves_publication_time():
    from quant_risk_monitor.style_descriptors import annual_growth_descriptors

    facts = pd.DataFrame(
        [
            {
                "asset": "a",
                "field": "revenue_ytd",
                "report_period": f"{year}1231",
                "value": value,
                "available_at": "2022-04-29T00:00:00+08:00",
            }
            for year, value in [(2019, 100), (2020, 110), (2021, 120)]
        ]
    )
    dates = pd.to_datetime(["2022-04-28", "2022-04-29"])
    values, _ = annual_growth_descriptors(facts, dates, ["a"])
    assert pd.isna(values["revenue_growth_3y"].iloc[0, 0])
    assert values["revenue_growth_3y"].iloc[1, 0] == pytest.approx(10 / 110)
    later = facts.iloc[[0]].assign(value=999999, available_at="2027-01-01T00:00:00+08:00")
    poisoned, _ = annual_growth_descriptors(pd.concat([facts, later]), dates, ["a"])
    pd.testing.assert_frame_equal(values["revenue_growth_3y"], poisoned["revenue_growth_3y"])


def test_header_and_amount_occupy_different_grid_subcolumns():
    rows = [
        ["项目", "", "2023年", "", "2022年", ""],
        ["营业收入（元）", "675.50", None, "625.97", None, None],
    ]
    boxes = [
        [(0, 0, 20, 10), None, (20, 0, 50, 10), None, (50, 0, 80, 10), None],
        [(0, 10, 20, 20), (20, 10, 50, 20), None, (50, 10, 80, 20), None, None],
    ]
    found = table_facts(
        rows, page=8, context="", scope="annual_summary", report_year=2023, cell_boxes=boxes
    )
    assert {r["report_period"]: r["value"] for r in found} == {
        "20231231": 675.50,
        "20221231": 625.97,
    }


def test_old_restatement_table_is_not_current_consolidated_statement():
    from recovery_extract import statement_sections

    text = "[[PAGE 45]]\n合并利润表\n2018年度\n营业收入 2277 1950\n[[PAGE 89]]\n合并利润表\n编制单位：公司\n2019年度\n营业收入 2059 1950\n母公司利润表\n2019年度"
    sections = list(statement_sections(text, "利润表", "20191231"))
    assert len(sections) == 1
    assert "2059" in sections[0] and "2277" not in sections[0]


def test_opening_balance_reconciliation_cannot_become_year_end():
    rows = [
        ["项目", "2020年12月31日", "2021年1月1日", "调整数"],
        ["负债合计", "30207", "30327", "120"],
    ]
    assert (
        table_facts(
            rows, page=130, context="单位：百万元", scope="consolidated_balance", report_year=2021
        )
        == []
    )


def test_year_prefixed_statement_heading_is_supported_but_supplement_is_not():
    from recovery_extract import statement_sections

    text = "[[PAGE 100]]\n2023年度合并利润表\n单位：百万元\n2023年度 2022年度\n营业收入 10 20\n2023年度母公司利润表\n项目\n2023年度新准则合并利润表\n营业收入 1 2"
    found = list(statement_sections(text, "利润表", "20231231"))
    assert len(found) == 1 and "10 20" in found[0] and "新准则" not in found[0]


def test_numeric_prefix_above_label_is_not_dropped_as_cents():
    rows = [
        ["项目", "2021年", "2020年"],
        [None, "8,881,246,764.", "8,242,208,932."],
        ["总资产（元）", "39", "64"],
    ]
    boxes = [
        [(0, 0, 10, 10), (10, 0, 20, 10), (20, 0, 30, 10)],
        [(0, 10, 10, 20), (10, 10, 20, 20), (20, 10, 30, 20)],
        [(0, 20, 10, 30), (10, 20, 20, 30), (20, 20, 30, 30)],
    ]
    found = table_facts(
        rows, page=8, context="", scope="annual_summary", report_year=2021, cell_boxes=boxes
    )
    assert {r["report_period"]: r["value"] for r in found} == {
        "20211231": 8881246764.39,
        "20201231": 8242208932.64,
    }


def test_repeated_statement_headers_do_not_mutate_heading_pattern():
    from recovery_extract import statement_sections

    text = "[[PAGE 1]]\n2023年度合并利润表\n2023年度\n营业收入 10 9\n母公司利润表\n[[PAGE 8]]\n2023年度合并利润表\n2023年度\n营业收入 20 19\n母公司利润表\n"
    found = list(statement_sections(text, "利润表", "20231231"))
    assert len(found) == 2
    assert "20 19" not in found[0] and "10 9" not in found[1]


def test_statement_unit_above_continuation_survives_signed_footer():
    from recovery_extract import statement_fallback

    text = ("[[PAGE 50]]\n（金额单位为人民币百万元）\n2020年6月30日合并资产负债表\n"
            "项目\n2020年6月30日 2019年12月31日\n资产总计 4000000 3000000\n"
            "后附财务报表附注为组成部分。\n法定代表人：甲\n[[PAGE 51]]\n"
            "（金额单位为人民币百万元）\n2020年6月30日合并资产负债表（续）\n"
            "负债：\n负债合计 3,543,856 3,317,392\n2020年6月30日公司资产负债表\n"
            "负债合计 3,000,000 2,000,000")
    second = text.replace("2020年6月", "2020年 6月")
    rows = statement_fallback(text, second, "20200630")
    assert next(r for r in rows if r["field"] == "liabilities")["value"] == 3543856e6


def test_reconciliation_with_year_below_financial_rows_is_rejected():
    from recovery_extract import statement_sections

    text = "[[PAGE 176]]\n合并利润表\n营业收入 837859 -432819 405040\n2024年政策解释"
    assert list(statement_sections(text, "利润表", "20241231")) == []


def test_reviewed_restated_columns_require_order_and_unit():
    import pytest
    from recovery_rebuild import comparative_values

    text = "2024年度合并利润表\n金额单位为人民币百万元\n2024年度 2023年度 2022年度\n营业收入 528,567 405,040 388,895\n"
    assert comparative_values(text, "营业收入", [2024, 2023, 2022]) == [528567e6, 405040e6, 388895e6]
    with pytest.raises(ValueError):
        comparative_values(text, "营业收入", [2024, 2022, 2023])
