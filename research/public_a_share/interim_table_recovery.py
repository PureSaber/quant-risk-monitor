"""Targeted interim statement recovery, separate from frozen legacy parsing."""

import re

from financial_table_recovery import compact, number, units

HEADING = re.compile(
    r"(?m)^\s*(?:[（(]?\d+(?:\.\d+)*(?:[、.．）)]|\s+)\s*)?(?:20\d{2}\s*年\s*(?:\d{1,2}\s*月\s*\d{1,2}\s*日|度)?\s*)?"
    r"(?P<scope>合并(?:及公司|及母公司)?|母公司|公司)?(?P<ytd>年初[至到]报告期末)?"
    r"(?P<table>资产负债表|利润表|现金流量表|所有者权益变动表|股东权益变动表)"
    r"\s*(?:[（(](?:续|未经审计)[）)]\s*)*\s*$"
)


def section_for(text, offset, period, field, allow_single_entity=False):
    matches = [m for m in HEADING.finditer(text) if m.start() < offset]
    if not matches:
        return None
    heading = matches[-1]
    expected = "利润表" if field == "profit_ytd" else "资产负债表"
    if heading['table'] != expected or heading['scope'] in {'母公司', '公司'}:
        return None
    prefix = text[heading.start():offset]
    header = compact(prefix[:1200])
    subsequent_headings = re.findall(
        r'(?m)^\s*[^\n]{0,100}?(资产负债表|利润表|现金流量表|股东权益变动表|所有者权益变动表)\s*(?:[（(][^\n）)]{0,12}[）)])?\s*$',
        text[heading.end():offset],
    )
    if any(table != expected for table in subsequent_headings):
        return None
    dated_columns = re.findall(r'(20\d{2})年(\d{1,2})月(\d{1,2})日', header)
    joint_group_columns = (
        header.count('本集团') >= 1 and header.count('本公司') >= 1 and len(dated_columns) >= 4
        and header.index('本集团') < header.index('本公司')
    )
    consolidated = bool(heading['scope'] and heading['scope'].startswith('合并')) or joint_group_columns
    if not consolidated and not allow_single_entity:
        return None
    if expected == '资产负债表':
        dates = re.findall(r'(20\d{2})年(\d{1,2})月(\d{1,2})日', header)
        if dates and ''.join((dates[0][0], dates[0][1].zfill(2), dates[0][2].zfill(2))) != period:
            return None
    if field == 'profit_ytd' and period.endswith('0930'):
        if re.search(r'20\d{2}年第三季度(?!报告)', header) and '前三季度' in header:
            return None  # Four-column current-quarter/YTD layouts need explicit mapping.
        if not heading['ytd'] and not re.search(period[:4] + r'年1[—–－-]9月|'+period[:4]+r'年前三季度', header):
            return None
    # A continuation can have the currency unit immediately above its heading.
    page_markers = list(re.finditer(r'\[\[PAGE (\d+)\]\]', text[:heading.start()]))
    start = page_markers[-1].end() if page_markers else max(0, heading.start()-200)
    scale = units(text[start:offset])
    if scale is None:
        return None
    return {'heading': heading[0].strip(), 'header': prefix[:1200], 'scale': scale,
            'scope': 'group_first_in_group_parent_table' if joint_group_columns else
                     'consolidated' if consolidated else 'issuer_only_review_required'}


def target_field(label, *, allow_single_entity=False):
    text = compact(label)
    if re.search('扣除|每股|综合收益|增长率', text):
        return None
    if re.fullmatch(r'(?:[（(]?[一二三四五六七八九十\d]+[、.．）)])?归属于母公司(?:股东|所有者)(?:的)?净利润.*', text):
        return 'profit_ytd'
    if allow_single_entity and re.fullmatch(r'(?:[一二三四五六七八九十]+[、.．])?净利润[（(].*', text):
        return 'profit_ytd'
    if re.fullmatch(r'归属于母公司(?:股东|所有者)(?:的)?(?:股东权益|所有者权益|权益)(?:合计|总计)', text):
        return 'book_equity'
    if text=='负债合计':
        return 'liabilities'
    return None


def extract(path, period, requested, *, allow_single_entity=False):
    import pymupdf
    from pypdf import PdfReader

    document = pymupdf.open(path)
    pages = [p.get_text() for p in document]
    starts, parts, length = [], [], 0
    for i, page in enumerate(pages):
        part = f'[[PAGE {i+1}]]\n{page}\n'
        starts.append(length)
        parts.append(part)
        length += len(part)
    text = ''.join(parts)
    reader, other_pages, candidates = PdfReader(path), {}, []
    for i, page in enumerate(document):
        if not any(s in compact(pages[i]) for s in ('归属于母公司', '负债合计')) and not (allow_single_entity and '净利润' in pages[i]):
            continue
        for table in page.find_tables().tables:
            for row in table.extract():
                labelled = [(j, str(cell)) for j, cell in enumerate(row) if target_field(cell, allow_single_entity=allow_single_entity) in requested]
                if len(labelled)!=1:
                    continue
                j, label = labelled[0]
                field = target_field(label, allow_single_entity=allow_single_entity)
                # Locate the real row within this page, retaining preceding headings.
                tokens = re.findall(r'[\u3400-\u9fff]+', label)
                if not tokens:
                    continue
                located = re.search(r'\s*'.join(re.escape(c) for c in tokens[0]), pages[i])
                if located is None:
                    continue
                pos = located.start()
                offset = starts[i] + len(f'[[PAGE {i+1}]]\n') + pos
                section = section_for(text, offset, period, field, allow_single_entity)
                if section is None:
                    continue
                amounts = [(column, cell, number(cell)) for column, cell in enumerate(row[j+1:], j+1)]
                amounts = [(column, cell, value) for column, cell, value in amounts if value is not None]
                # Footnote indexes are not monetary values; ambiguous small values stay missing.
                amounts = [(column, cell, value) for column, cell, value in amounts if abs(value)*section['scale'] >= 10000]
                if len(amounts)<2:
                    continue
                column, raw, value = amounts[0]
                if i not in other_pages:
                    other_pages[i] = compact(reader.pages[i].extract_text() or '')
                label_match = re.search(r'归属于母公司|负债合计|净利润', compact(label))
                independent = other_pages[i]
                loc = independent.find(label_match[0])
                if loc<0 or compact(raw) not in independent[loc:loc+1500]:
                    continue
                candidates.append({'field':field,'report_period':period,'value':float(value)*section['scale'],
                    'page':i+1,'column':column,'label':label,'raw_row':row,'raw_cell':raw,
                    'verification':'statement_geometry_and_independent_pdf_text',**section})
        # China Life's borderless statements have explicit labelled amounts and units.
        if 'book_equity' in requested:
            pattern = r'归属于母公司股东的股东权益合计\s*([\d,]+)\s+([\d,]+)'
            compact_labels = re.sub(r'(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])','',pages[i])
            match = re.search(pattern, compact_labels)
            if match:
                pos = pages[i].find('归属于母公司')
                section = section_for(text, starts[i]+len(f'[[PAGE {i+1}]]\n')+pos,period,'book_equity')
                if section and section['scale']==1e6:
                    other = re.sub(r'(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])','',reader.pages[i].extract_text() or '')
                    check = re.search(pattern,other)
                    if check and check.groups()==match.groups():
                        candidates.append({'field':'book_equity','report_period':period,
                            'value':float(match[1].replace(',',''))*1e6,'page':i+1,'column':0,
                            'label':'归属于母公司股东的股东权益合计','raw_row':list(match.groups()),
                            'raw_cell':match[1],'verification':'two_pdf_readers_explicit_borderless_statement',**section})
    accepted=[]
    for field in requested:
        group=[r for r in candidates if r['field']==field]
        if group and max(r['value'] for r in group)-min(r['value'] for r in group)<=max(r['scale'] for r in group):
            accepted.append(group[0])
    return {'accepted':accepted,'candidates':candidates}


def extract_record(row):
    from recovery_inventory import digest

    if digest(row['pdf']) != row['sha256']:
        raise ValueError('interim_original_digest_changed')
    result = extract(row['pdf'], row['report_period'], row['requested_fields'],
                     allow_single_entity=row['asset'] == '600992')
    # Single-entity admission still requires dated, source-bound scope evidence.
    return {**row, **result, 'parser_sha256': digest(__file__)}


if __name__ == '__main__':
    import argparse
    import json
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    from recovery_inventory import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('use_new_interim_extraction_version')
    records = json.loads(args.inventory.read_text(encoding='utf-8'))
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(extract_record, records))
    write_json(args.out, results)
