import json

import numpy as np
import pandas as pd
import pytest

import build_panel
from original_financials import amount_row
from run_study import bootstrap_delta, daily_scores


def test_original_current_and_ytd_columns_are_distinguished():
    text = "主要会计数据 本报告期 本报告期同比增减 年初至报告期末 同比增减 单位：元 归属于上市公司股东的净利润 12,000,000.00 20% 35,000,000.00 15% 基本每股收益 0.2"
    assert amount_row(text, "profit_ytd", "20230930")["value"] == 35000000
    assert amount_row(text, "profit_ytd", "20230331")["value"] == 12000000


def test_wrapped_financial_numbers_keep_column_separators():
    text = "单位：元 总资产 18,198,332\n,914.41 18,487,856\n,446.74 归属于上市公司股东的净资产 359,412,54\n9.43 373,428,19\n9.20"
    assert amount_row(text, "assets", "20230331")["value"] == 18198332914.41
    assert amount_row(text, "book_equity", "20230331")["value"] == 359412549.43


def test_current_liability_subtotal_is_not_total_liabilities():
    text = "合并资产负债表 单位：元 流动负债合计 30,000,000 20,000,000 非流动负债合计 10,000,000 10,000,000 负债合计 40,000,000 30,000,000 所有者权益 股本 10,000,000 10,000,000"
    assert amount_row(text, "liabilities", "20230331")["value"] == 40000000


def test_negative_book_equity_does_not_fall_through_to_an_associate():
    text = "合并资产负债表 单位：元 归属于母公司所有者权益合计 -37,703,287.29 136,566,569.35 少数股东权益 130,636,722.63 138,927,539.35 母公司资产负债表 归属于母公司所有者权益合计 2,052,803.35 845,356,757.99"
    assert amount_row(text, "book_equity", "20210630")["value"] == -37703287.29


def test_original_million_yuan_unit_and_negative_profit():
    text = "合并利润表 单位：人民币百万元 归属于母公司股东的净利润 -135 120 少数股东损益 3 4"
    assert amount_row(text, "profit_ytd", "20230930")["value"] == -135000000


def test_table_notes_and_thousand_yuan_units():
    text = "合并及公司利润表 (除特别注明外，金额单位为人民币千元) 项目 2023年度 2022年度 营业收入四(44) 242,774 288,849 归属于母公司股东的净利润 (18,960) 54,107 少数股东损益 3 4"
    assert amount_row(text, "revenue_ytd", "20231231")["value"] == 242774000
    assert amount_row(text, "profit_ytd", "20231231")["value"] == -18960000


def test_prose_ratio_is_not_an_income_row():
    text = "归属于上市公司股东净利润的30.20%。本次分配后剩余未分配利润 909,565,126.22元。主要会计数据 单位：元 归属于上市公司股东净利润 76,998,070.00 65,000,000.00"
    assert amount_row(text, "profit_ytd", "20191231")["value"] == 76998070


def test_share_capital_in_yuan_is_not_invented_as_share_count():
    assert (
        amount_row("单位：元 实收资本（或股本） 10,000,000 10,000,000", "shares", "20231231")
        is None
    )


def test_shareholder_percentage_cannot_become_total_share_count():
    text = "占公司总股本 30.03%。其中，通过普通证券账户持有公司股份 166,395,487 股；通过信用账户持有 81,795,900 股。"
    assert amount_row(text, "shares", "20230930") is None


def test_historical_capital_narrative_cannot_replace_current_capital_table():
    text = "公司在2018年总股本增至626,926,827股。单位：股 股份总数 877,697,557 100.00% 0 877,697,557 100.00% 股份变动的原因"
    assert amount_row(text, "shares", "20210630")["value"] == 877697557


def test_future_revision_does_not_rewrite_prior_facts(tmp_path, monkeypatch):
    clean = tmp_path / "clean"
    clean.mkdir()
    fields = [
        {
            "asset": "000001",
            "report_period": "20231231",
            "field": "book_equity",
            "value": 10000000,
            "available_at": "2024-05-01T00:00:00+08:00",
        },
        {
            "asset": "000001",
            "report_period": "20231231",
            "field": "book_equity",
            "value": 900000000,
            "available_at": "2024-08-01T00:00:00+08:00",
        },
    ]
    pd.DataFrame(fields).to_csv(clean / "financial_facts.csv", index=False)
    monkeypatch.setattr(build_panel, "ROOT", tmp_path)
    dates = pd.to_datetime(["2024-04-30", "2024-05-01", "2024-08-01"])
    data = build_panel.financial_daily(dates, ["000001"])["book_equity"]["000001"]
    assert np.isnan(data.iloc[0])
    assert data.iloc[1] == 10000000
    assert data.iloc[2] == 900000000


def test_share_announcement_and_effective_date_both_apply(tmp_path, monkeypatch):
    folder = tmp_path / "raw/shares"
    folder.mkdir(parents=True)
    pd.DataFrame(
        {
            "证券代码": ["000001"] * 3,
            "公告日期": ["2024-01-01", "2024-01-02", "2024-02-10"],
            "变动日期": ["2023-12-31", "2024-02-01", "2023-12-31"],
            "总股本": [100, 200, 101],
        }
    ).to_csv(folder / "000001.csv", index=False)
    monkeypatch.setattr(build_panel, "ROOT", tmp_path)
    dates = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-02-01", "2024-02-11"])
    result = build_panel.capital_daily(
        dates, ["000001"], pd.DataFrame(np.nan, index=dates, columns=["000001"])
    )["000001"]
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == 1000000
    assert result.iloc[2] == result.iloc[3] == 2000000


def test_family_weighting_does_not_overweight_many_industries():
    rows = [
        dict(model="a", date="2024-01-01", family="equal", qlike=1),
        dict(model="a", date="2024-01-01", family="capitalization", qlike=2),
    ]
    rows += [dict(model="a", date="2024-01-01", family="industry", qlike=3)] * 10
    assert daily_scores(pd.DataFrame(rows)).iloc[0, 0] == 2


def test_block_bootstrap_keeps_fixed_paired_difference():
    assert bootstrap_delta(np.full(100, -0.2), replications=50) == pytest.approx([-0.2, -0.2])


def test_public_collector_rejects_excluded_source():
    from public_sources import fetch

    with pytest.raises(ValueError, match="allowlist"):
        fetch("never-created.json", "http://dataway.hhhstz.com/hsic_base_fmt/cube")


def test_future_prices_cannot_change_past_exposure_descriptors():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-01", periods=400)
    prices = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0, 0.02, (400, 3)), axis=0)) * 20,
        index=dates,
        columns=["a", "b", "c"],
    )
    market = pd.Series(rng.normal(0, 0.01, 400), index=dates)
    px = {
        "close": prices,
        "adjusted_close": prices,
        "returns": prices.pct_change(fill_method=None),
        "turnover": prices * 0 + 2,
    }
    shares = prices * 0 + 1e8
    financial = {
        key: prices * 0 + value
        for key, value in [
            ("book_equity", 1e9),
            ("profit_ttm", 1e8),
            ("liabilities", 2e9),
            ("growth", 0.1),
        ]
    }
    cap, base = build_panel.descriptors_from_history(px, financial, shares, market)
    poisoned = {key: frame.copy() for key, frame in px.items()}
    for frame in poisoned.values():
        frame.iloc[330:] *= 100
    future_market = market.copy()
    future_market.iloc[330:] *= 50
    cap2, changed = build_panel.descriptors_from_history(poisoned, financial, shares, future_market)
    pd.testing.assert_frame_equal(cap.iloc[:331], cap2.iloc[:331])
    for key in base:
        pd.testing.assert_frame_equal(base[key].iloc[:331], changed[key].iloc[:331])


def test_issuer_summary_cannot_fall_through_to_subsidiary_notes():
    text = "[[PAGE 1]] 主要会计数据 总资产 10,000,000 9,000,000 [[PAGE 19]] 重要子公司的主要财务信息 单位：元 负债合计 3,000,000 2,000,000"
    assert amount_row(text, "liabilities", "20231231") is None


def test_industry_changes_wait_for_the_publication_date(tmp_path, monkeypatch):
    (tmp_path / "clean").mkdir()
    pd.DataFrame(
        [
            {
                "asset": "000001",
                "sector": "A",
                "available_at": "2021-11-11T00:00:00+08:00",
            },
            {
                "asset": "000001",
                "sector": "B",
                "available_at": "2024-04-04T00:00:00+08:00",
            },
        ]
    ).to_csv(tmp_path / "clean/industry_history.csv", index=False)
    monkeypatch.setattr(build_panel, "ROOT", tmp_path)
    dates = pd.to_datetime(["2021-11-10", "2021-11-11", "2024-04-03", "2024-04-04"])
    values = build_panel.industry_daily(dates, ["000001"], {"A": "agriculture", "B": "mining"})[
        "000001"
    ]
    assert pd.isna(values.iloc[0])
    assert values.iloc[1:3].eq("agriculture").all()
    assert values.iloc[3] == "mining"


def test_opening_transition_balance_is_not_the_current_balance():
    text = "[[PAGE 50]] 合并资产负债表 2019年1月1日 单位：元 负债合计 1,095,656,131.76 1,098,274,787.46 母公司资产负债表 单位：元 负债合计 200,000,000 190,000,000"
    assert amount_row(text, "liabilities", "20191231") is None


def test_book_equity_after_nonrecurring_profit_row_is_still_in_the_summary():
    text = "[[PAGE 5]] 主要会计数据 单位：元 归属于上市公司股东的净利润 30,000,000 20,000,000 归属于上市公司股东的扣除非经常性损益的净利润 28,000,000 19,000,000 归属于上市公司股东的净资产 100,000,000 90,000,000 总资产 300,000,000 280,000,000 非经常性损益项目和金额"
    assert amount_row(text, "book_equity", "20211231")["value"] == 100000000


def test_parenthetical_equity_label_is_part_of_the_same_field():
    text = "合并资产负债表 2021年12月31日 单位：元 归属于母公司所有者权益（或股东权益）合计 100,000,000 90,000,000 少数股东权益 3,000,000 2,000,000"
    assert amount_row(text, "book_equity", "20211231")["value"] == 100000000


def test_consolidated_four_column_quarter_statement_selects_ytd():
    text = "合并利润表 2019年1—9月 单位：元 项目 2019年第三季度（7-9月） 2018年第三季度（7-9月） 2019年前三季度（1-9月） 2018年前三季度（1-9月） 归属于母公司股东的净利润 46,604,120.30 -120,063,812.65 4,257,068.34 242,912,783.90 少数股东损益 1 1 1 1"
    assert amount_row(text, "profit_ytd", "20190930")["value"] == 4257068.34


def test_original_cover_corrects_directory_year_and_future_period_is_rejected():
    from original_financials import disclosure_period

    day = pd.Timestamp("2024-04-26", tz="Asia/Shanghai")
    text = "[[PAGE 1]] 2023 年年度报告 公司代码：601113 [[PAGE 2]]"
    assert disclosure_period("2024年年度报告", text, day) == "20231231"
    assert disclosure_period("2024年年度报告", "unreadable cover", day) is None


def test_present_day_vendor_values_do_not_control_historical_admission(tmp_path, monkeypatch):
    import original_financials as extractor

    monkeypatch.setattr(extractor, "ROOT", tmp_path)
    folder = tmp_path / "raw/filings/000001"
    folder.mkdir(parents=True)
    (tmp_path / "raw/announcements").mkdir()
    (tmp_path / "raw/shares").mkdir()
    (tmp_path / "raw/shares/000001.csv").write_text("existing public capital archive")
    ann = {
        "announcementTitle": "2023年年度报告",
        "announcementId": "123",
        "announcementTime": int(pd.Timestamp("2024-04-20", tz="Asia/Shanghai").timestamp() * 1000),
    }
    (tmp_path / "raw/announcements/000001-index.json").write_text(
        json.dumps(
            {"report_count": 1, "reports": [{"id": "123"}], "all_records": [ann], "errors": []}
        ),
        encoding="utf8",
    )
    (folder / "123.logical.txt").write_text(
        "合并资产负债表 2023年12月31日 单位：元 资产总计 300,000,000 280,000,000 负债合计 200,000,000 190,000,000 归属于母公司所有者权益合计 100,000,000 90,000,000 合并利润表 单位：元 归属于母公司股东的净利润 30,000,000 25,000,000 少数股东损益 1 1",
        encoding="utf8",
    )
    (folder / "123.logical.extract.json").write_text(json.dumps({"isolated_process": True}))
    first, _ = extractor.extract_member({"code": "000001"})
    (tmp_path / "raw/financial").mkdir()
    (tmp_path / "raw/financial/000001-fzb.json").write_text(
        json.dumps({"future_restatement": 1e30})
    )
    second, _ = extractor.extract_member({"code": "000001"})
    assert first == second
    assert next(row for row in first if row["field"] == "book_equity")["value"] == 100000000
