"""检查现有官方档案及六次重复抓取，不将修改时间冒充首次公开时间。"""
import xml.etree.ElementTree as ET
from common16 import *

def audit():
    rows=[];tag_counts={};fields={'pubdate','published','issued','datepublished','publicationdate','firstpublished'}
    for path in sorted((PREVIOUS/'raw/narratives').glob('*.xml')):
        m=read(path.with_suffix('.xml.meta.json'));require(sha(path)==m['sha256'],'old source bytes changed')
        date=pd.to_datetime(path.stem.split('_')[1],format='%Y%m%d');scheduled=(date+pd.Timedelta(days=2,hours=8,minutes=30)).tz_localize('America/New_York').tz_convert('UTC');modified=pd.to_datetime(m.get('last_modified'),utc=True,errors='coerce')
        record={'file':path.name,'url':m['url'],'sha256':m['sha256'],'map_date':str(date.date()),'last_modified':m.get('last_modified'),'scheduled_release_utc':scheduled.isoformat(),'modified_before_scheduled_release':None if pd.isna(modified) else bool(modified<scheduled),'parse_ok':False,'publication_fields':'','first_publication_verified':False}
        try:
            root=ET.fromstring(path.read_bytes());tags=sorted({e.tag.split('}')[-1] for e in root.iter()});record['parse_ok']=True;record['publication_fields']=';'.join(t for t in tags if t.lower() in fields)
            for tag in tags:tag_counts[tag]=tag_counts.get(tag,0)+1
        except ET.ParseError as e:record['error']=str(e)
        rows.append(record)
    repeated=[]
    for path in sorted((LOCAL/'raw/repeated_xml').glob('*.xml')):
        m=read(path.with_suffix('.xml.meta.json'));old=PREVIOUS/'raw/narratives'/path.name;oldm=read(old.with_suffix('.xml.meta.json'));require(sha(path)==m['sha256'],'repeat bytes changed')
        repeated.append({'file':path.name,'url':m['url'],'old_sha256':sha(old),'new_sha256':sha(path),'same_bytes':sha(old)==sha(path),'old_fetched_utc':oldm['fetched_utc'],'new_fetched_utc':m['fetched_utc'],'new_last_modified':m.get('last_modified'),'new_etag':m.get('etag')})
    csv(HERE/'results/source/document_audit.csv',rows);csv(HERE/'results/source/repeat_snapshot_check.csv',repeated)
    summary={'archived_xml_checked':len(rows),'parsed':sum(r['parse_ok'] for r in rows),'parse_failures':sum(not r['parse_ok'] for r in rows),'xml_tag_counts':tag_counts,'documents_with_candidate_publication_fields':sum(bool(r['publication_fields']) for r in rows),'modification_before_schedule':sum(r['modified_before_scheduled_release'] is True for r in rows),'repeat_files':len(repeated),'repeat_identical':sum(r['same_bytes'] for r in repeated),'first_publication_verified':0,'strict_text_enabled':False,'conclusion':'Inspected official XML/header fields and current repeat snapshots do not establish per-document first public contents. Absence claim limited to inspected evidence; same bytes today do not establish immutable history.','official_sources':['https://drought.unl.edu/Education/Tutorials/Usdm/Section-Six.aspx','https://droughtmonitor.unl.edu/Data/narrative/','https://droughtmonitor.unl.edu/DmData/Metadata.aspx']}
    dump(HERE/'results/source/summary.json',summary);print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':audit()
