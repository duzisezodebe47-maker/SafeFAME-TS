"""Read-only checks for DOCX structure and previously rendered QA pages."""
import json
from pathlib import Path
from docx import Document
from PIL import Image, ImageStat

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tmp/revision_20260919/v4_pages'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'submission.json').read_text(encoding='utf-8'))
    for report in config['reports']:
        doc=Document(ROOT/report['docx'])
        render_dir=ROOT/'tmp'/f"qa_current_{report['role']}"
        pages=sorted(render_dir.glob('page-*.png'))
        assert len(pages)==report['rendered_docx_pages'],(report['role'],len(pages),report['rendered_docx_pages'])
        for page in pages:
            image=Image.open(page).convert('L')
            stat=ImageStat.Stat(image)
            assert stat.var[0]>1.0,(report['role'],page.name,'visually blank render')
        print(report['role'],len(pages),'rendered DOCX pages',len(doc.tables),'tables')
        tables=[[[c.text for c in row.cells] for row in table.rows] for table in doc.tables]
        (OUT/(report['role']+'_tables.json')).write_text(json.dumps(tables,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
