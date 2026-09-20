"""Record checks after human/agent review of every rendered page."""
import hashlib
import argparse
import json
import re
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E

ROOT=Path(__file__).resolve().parents[1]
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--all-pages-reviewed',action='store_true')
    assert parser.parse_args().all_pages_reviewed, 'Actual visual review must precede this command'
    config=json.loads((ROOT/'submission.json').read_text(encoding='utf-8'))
    qa={'date':'2026-09-20','all_pages_visually_reviewed':True,
        'render_method':'Microsoft Word temporary PDF export + Poppler PNG; temporary PDFs deleted; reviewed page numbers recorded per report',
        'review_scope':'layout, headings, native equations, figures, table headers, references, page numbers, current-literature positioning, practical significance, figure/table interpretation and post-review attribution section; retained v2/v3 and completed 120-task-fold v4 historical rolling backtest; no experiment results changed and no full ETT retraining',
        'pdf_status':'deferred until explicit user instruction after DOCX finalization',
        'reports':{}}
    for r in config['reports']:
        pages=r['rendered_docx_pages']
        render_dir=ROOT/'tmp'/f"qa_current_{r['role']}"
        rendered=list(render_dir.glob('page-*.png'))
        assert len(rendered)==pages,(r['role'],len(rendered),pages)
        assert not (render_dir/'_render_intermediate.pdf').exists()
        if r['role']=='final':assert pages<=30,'User-requested 30-page limit; reduce repeated exposition, not evidence'
        with ZipFile(ROOT/r['docx']) as z:tree=E.fromstring(z.read('word/document.xml'))
        paragraphs=tree.findall('.//w:p',NS)
        # Citations: bibliographic numbering and in-text numbering agree.
        texts=[''.join(p.xpath('.//w:t/text()',namespaces=NS)) for p in paragraphs]
        refs={int(m.group(1)) for t in texts if (m:=re.match(r'^\[(\d+)\]',t))}
        cited=set()
        for t in texts:
            if re.match(r'^\[\d+\]',t):continue
            for bracket in re.findall(r'\[([0-9,\-]+)\]',t):
                for token in bracket.split(','):
                    nums=[int(x) for x in token.split('-')]
                    cited.update(range(nums[0],nums[-1]+1))
        assert refs==cited,(r['role'],refs-cited,cited-refs)
        qa['reports'][r['role']]={'rendered_pages':pages,'reviewed_pages':list(range(1,pages+1)),
            'render_directory':str(render_dir.relative_to(ROOT)).replace('\\','/'),
            'docx_sha256':sha(ROOT/r['docx']),
            'native_equations':len(tree.findall('.//m:oMath',NS)),
            'text_paragraphs_checked':len(paragraphs),
            'references':len(refs),'cross_references':'checked by verify_project.py'}
    protected=json.loads((ROOT/'tmp/revision_20260919/protected_hashes.json').read_text())
    changed=[name for name,digest in protected.items() if sha(ROOT/name)!=digest]
    assert not changed,changed
    qa['protected_data_result_model_files_unchanged']=len(protected)
    (ROOT/'docs/submission_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('QA recorded; protected files unchanged:',len(protected))

if __name__=='__main__':main()
