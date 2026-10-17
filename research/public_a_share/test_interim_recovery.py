import pytest
from interim_table_recovery import section_for, target_field
from report_discovery import is_full_report, title_period


def test_half_year_short_title_is_discovered_without_becoming_annual():
    assert is_full_report("安彩高科2022年半年报")
    assert title_period("安彩高科2022年半年报") == "20220630"
    assert title_period("2022年年报") == "20221231"


def test_quarter_report_running_header_does_not_confuse_ytd_columns():
    text = "[[PAGE 8]]\n合并利润表\n2024 年1—9 月\n单位：元\n项目\n2024年前三季度\n2023年前三季度\n"
    text += "[[PAGE 9]]\n公司2024年第三季度报告\n归属于母公司股东的净利润\n123,000\n100,000"
    result = section_for(text, text.index("归属于"), "20240930", "profit_ytd")
    assert result["scale"] == 1
    assert result["scope"] == "consolidated"
    mixed = text.replace("2024年前三季度", "2024年第三季度 2024年前三季度")
    assert section_for(mixed, mixed.index("归属于"), "20240930", "profit_ytd") is None


def test_parent_and_opening_balance_cannot_substitute_for_current_group():
    text = "[[PAGE 1]]\n母公司资产负债表\n2024年9月30日\n单位：元\n负债合计\n100,000\n90,000"
    assert section_for(text, text.index("负债合计"), "20240930", "liabilities", True) is None
    opening = text.replace("母公司", "合并").replace("2024年9月30日", "2024年1月1日")
    assert section_for(opening, opening.index("负债合计"), "20240930", "liabilities") is None


def test_joint_group_parent_table_requires_explicit_group_first_columns():
    text = "[[PAGE 8]]\n2024年3月31日资产负债表(未经审计)\n单位：百万元\n"
    text += "2024年3月31日 2023年12月31日 2024年3月31日 2023年12月31日 本集团 本集团 本公司 本公司\n负债合计\n100,000\n90,000"
    result = section_for(text, text.index("负债合计"), "20240331", "liabilities")
    assert result["scope"] == "group_first_in_group_parent_table"
    reverse = text.replace("本集团 本集团 本公司 本公司", "本公司 本公司 本集团 本集团")
    assert section_for(reverse, reverse.index("负债合计"), "20240331", "liabilities") is None


def test_plain_profit_needs_explicit_single_entity_review():
    label = "四、净利润（净亏损以“-”号填列）"
    assert target_field(label) is None
    assert target_field(label, allow_single_entity=True) == "profit_ytd"
    assert target_field("归属于母公司股东的股东权益合计") == "book_equity"


def test_numbered_continued_statement_and_later_notes_boundary():
    text = "[[PAGE 11]]\n金额单位为人民币百万元\n4.1 2020年3月31日合并资产负债表（未经审计）（续）\n"
    text += "2020年3月31日 2019年12月31日\n归属于母公司股东的股东权益合计\n413,715\n403,764"
    assert section_for(text, text.index("归属于"), "20200331", "book_equity")["scale"] == 1e6
    notes = text.replace("归属于", "其他综合收益变动表\n归属于")
    # A recognized different financial statement must terminate the prior scope.
    notes = notes.replace("其他综合收益变动表", "合并现金流量表")
    assert section_for(notes, notes.index("归属于"), "20200331", "book_equity") is None


def test_reviewed_numeric_value_must_match_the_original_token():
    from recovery_supplemental import recheck_review

    with pytest.raises(ValueError, match="not_bound"):
        recheck_review({"value": 200, "raw_cell": "100.00", "scale": 1,
                        "checks": {"1": ["100.00"]}})


def test_later_comparative_cannot_be_backdated(monkeypatch):
    from types import SimpleNamespace

    import recovery_supplemental as module

    class Document:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def __getitem__(self, _):
            return SimpleNamespace(get_text=lambda: "2024 上年同期 100.00")

    monkeypatch.setattr(module, "digest", lambda _: "known")
    monkeypatch.setattr(module.pymupdf, "open", lambda _: Document())
    monkeypatch.setattr(module, "PdfReader", lambda _: SimpleNamespace(
        pages=[SimpleNamespace(extract_text=lambda: "2024 上年同期 100.00")]))
    item = {"source_pdf": "reviewed.pdf", "source_sha256": "known", "value": 100,
            "raw_cell": "100.00", "scale": 1, "checks": {"1": ["2024", "上年同期", "100.00"]},
            "available_at": "2023-04-30T00:00:00+08:00",
            "source_available_at": "2024-04-30T00:00:00+08:00"}
    with pytest.raises(ValueError, match="backdated"):
        module.recheck_review(item)
    item["available_at"] = item["source_available_at"]
    module.recheck_review(item)
