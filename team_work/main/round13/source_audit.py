"""逐文本来源审计：不猜网址、不把事件日期当发布时间。"""
from __future__ import annotations
import re
from common import *

def main():
    out=HERE/'results/source';out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(LINEAGE,keep_default_na=False);domains=[task_spec(t)['domain'] for t in protocol()['tasks']];frame=frame[frame.domain.isin(domains)].copy()
    frame['empty_information_flag']=frame.fact.astype(str).str.contains(r'no mention|does not (?:mention|contain)|not (?:mentioned|provided)|no (?:objective )?facts|cannot (?:determine|extract)',case=False,regex=True)
    frame['original_url_present']=frame.source_url.astype(str).str.match(r'^https?://')
    frame['true_publication_verified']=frame.publication_verified.astype(str).str.lower().eq('true') & frame.original_url_present
    frame['strict_source_eligible']=frame.original_url_present & frame.true_publication_verified & ~frame.empty_information_flag
    frame['year_after_interval_flag']=[any(int(y)>pd.Timestamp(end).year for y in re.findall(r'\b(?:19|20)\d{2}\b',str(text))) if end else False for text,end in zip(frame.fact,frame.end_date)]
    frame['audit_note']='Interval end is an availability proxy; year-after-interval flag requires human review, not automatic proof of leakage.'
    csv(LOCAL/'source_catalog_audited.csv',frame.to_dict('records'))
    csv(out/'flagged_text_examples.csv',frame[frame.empty_information_flag | frame.year_after_interval_flag].groupby('domain',sort=False).head(10).to_dict('records'))
    rows=[]
    for (domain,source),g in frame.groupby(['domain','source']):
        rows.append({'domain':domain,'source':source,'raw_rows':len(g),'unique_text_ids':g.text_id.nunique(),'original_url_rows':int(g.original_url_present.sum()),'publication_verified_rows':int(g.true_publication_verified.sum()),'empty_information_rows':int(g.empty_information_flag.sum()),'year_after_interval_review_rows':int(g.year_after_interval_flag.sum()),'strict_source_eligible_rows':int(g.strict_source_eligible.sum())})
    csv(out/'source_audit_summary.csv',rows)
    cases=[{'text_id':'3c27aa9b4f93ce10','domain':'Environment','primary_url':'https://extapps.dec.ny.gov/docs/administration_pdf/pvscandd051421.pdf','document_date':'2021-05-14','public_release_date':None,'status':'RELATED_PRIMARY_DOCUMENT_CONFIRMS_EVENT; PUBLICATION_TIME_UNVERIFIED','finding':'DEC official letter supports cease-and-desist event; letter date does not establish when the corpus extraction was publicly available.','checked_at':'2026-10-07'},
           {'text_id':'0d4bb1db45b579d4','domain':'SocialGood','primary_url':'https://fraser.stlouisfed.org/title/commissioners-statement-employment-situation-9634/statement-joint-economic-committee-706260/content/fulltext/jec_19940204','document_date':'1994-02-04','public_release_date':'1994-02-04','status':'RELATED_BLS_RELEASE_FOUND; EXACT_CORPUS_LINEAGE_UNVERIFIED','finding':'BLS statement reports January unemployment on February 4, later than corpus interval end January 31. Exact original extraction/8.7-million statement not fully matched; keep strict eligibility false.','checked_at':'2026-10-07'},
           {'text_id':'9f276efce62fdbea','domain':'Climate','primary_url':'https://www.ncei.noaa.gov/access/monitoring/monthly-report/drought/199901','document_date':None,'public_release_date':None,'status':'CANDIDATE_URL_INACCESSIBLE; NOT_VERIFIED','finding':'No accessible original source retrieved; do not treat the candidate URL as verified provenance.','checked_at':'2026-10-07'},
           {'text_id':'38d16528a0b7bcd2','domain':'Agriculture','primary_url':None,'document_date':None,'public_release_date':None,'status':'ORIGINAL_REPORT_NOT_RECOVERED','finding':'USDA related poultry releases are not sufficient to identify the exact quoted market report.','checked_at':'2026-10-07'}]
    csv(out/'primary_source_case_checks.csv',cases)
    dump(out/'source_acceptance.json',{'status':'AUDIT_COMPLETE_SOURCE_GAPS_REMAIN','lineage_sha256':sha(LINEAGE),'audited_rows':len(frame),'unique_text_ids':int(frame.text_id.nunique()),'strict_source_eligible_rows':int(frame.strict_source_eligible.sum()),'primary_cases':cases,'raw_lineage_modified':False,'strict_scenario':'All unverifiable original publications are disabled. Related documents do not repair corpus lineage automatically.','catalog_sha256':sha(LOCAL/'source_catalog_audited.csv')})
    print(json.dumps({'rows':len(frame),'strict_eligible':int(frame.strict_source_eligible.sum()),'empty_information':int(frame.empty_information_flag.sum()),'year_after_interval_review':int(frame.year_after_interval_flag.sum())},ensure_ascii=False))

if __name__=='__main__':main()
