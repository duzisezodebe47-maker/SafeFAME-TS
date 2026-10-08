"""核验官方快照、清理周报、按发布时间构造时间隔离数据。"""
from __future__ import annotations
import re,html,xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import timedelta
from zoneinfo import ZoneInfo
from r14_common import *

def clean_assessment(value):
    value=re.sub(r'(?is)The discussion in the Looking Ahead section.*?(?:<br\s*/?>\s*){2}',' ',value)
    value=re.sub(r'(?i)<(?:br\s*/?|/?p|/?h[1-6]|/?div)[^>]*>','\n',value)
    value=html.unescape(re.sub(r'<[^>]+>',' ',value))
    lines=[re.sub(r'\s+',' ',line).strip() for line in value.splitlines()]
    kept=[]
    for line in lines:
        if re.match(r'(?i)^looking (ahead|forward)\b',line):break
        if line.lower() in ('national drought summary','weather summary:'):continue
        kept.append(line)
    original=' '.join(kept).strip();text=re.sub(r'\d+(?:[.,]\d+)*',' ',original);text=re.sub(r'\s+',' ',text).strip()
    return original,text

def release_times(date,explicit=None):
    # ZoneInfo handles daylight saving. Date-only explicit evidence is usable
    # from the following midnight, not an invented publication hour.
    if explicit is not None:return (pd.Timestamp(explicit)+pd.Timedelta(days=1)).tz_localize('America/New_York').tz_convert('UTC'),'document_explicit'
    return (pd.Timestamp(date)+pd.Timedelta(days=2,hours=8,minutes=30)).tz_localize('America/New_York').tz_convert('UTC'),'official_schedule_inferred'

def parse_narrative(path):
    meta=read(path.with_suffix('.xml.meta.json'));require(meta['sha256']==sha(path),'raw text changed')
    date=pd.to_datetime(path.stem.split('_')[1],format='%Y%m%d');root=ET.fromstring(path.read_bytes())
    period=root.findtext('.//forecastPeriod') or '';period_date=pd.to_datetime(period,errors='coerce');assessment=root.findtext('.//assessment') or ''
    original,text=clean_assessment(assessment);valid=not pd.isna(period_date) and period_date.normalize()==date and len(text.split())>=100
    # Most XML files contain a map date only. Never promote that to an explicit release date.
    released=re.search(r'(?i)released\s+(?:on\s+)?(?:Thursday,?\s+)?([A-Za-z]+\.?\s+\d{1,2},?\s+20\d{2})',original)
    explicit=pd.to_datetime(released.group(1),errors='coerce') if released else None
    if explicit is not None and (pd.isna(explicit) or not date<=explicit<=date+pd.Timedelta(days=7)):explicit=None
    available,grade=release_times(date,explicit)
    modified=pd.to_datetime(meta.get('last_modified'),utc=True,errors='coerce')
    conservative=max(available,modified+pd.Timedelta(days=1)) if not pd.isna(modified) else available+pd.Timedelta(days=7)
    return {'map_date':date.strftime('%Y-%m-%d'),'url':meta['url'],'raw_sha256':meta['sha256'],'fetched_utc':meta['fetched_utc'],'forecast_period':period,'period_matches':not pd.isna(period_date) and period_date.normalize()==date,'words':len(text.split()),'usable_content':valid,'publication_grade':grade,'available_utc':available.isoformat(),'conservative_available_utc':conservative.isoformat(),'last_modified':meta.get('last_modified'),'original_text':original,'model_text':text,'strict_eligible':bool(valid and grade=='document_explicit')}

def main():
    require(not (LOCAL/'prepared/MANIFEST.json').exists(),'prepared data sealed; refuse overwrite')
    require(read(LOCAL/'protocol_registration.json')['protocol_sha256']==sha(HERE/'protocol.json'),'protocol registration')
    numeric=pd.DataFrame(read(RAW/'conus_dsci.json'));numeric['map_date']=pd.to_datetime(numeric.mapDate.astype(str),format='ISO8601');numeric=numeric.sort_values('map_date').reset_index(drop=True)
    require(numeric.name.nunique()==1 and 'con' in numeric.name.iloc[0].lower(),'wrong region')
    require(not numeric.map_date.duplicated().any(),'duplicate numeric week');require(numeric.map_date.diff().dropna().eq(pd.Timedelta(days=7)).all(),'missing numeric weeks')
    require(np.isfinite(numeric.dsci).all() and numeric.dsci.between(0,500).all(),'DSCI values outside official range')
    p=cfg();require(numeric.map_date.iloc[0]==pd.Timestamp(p['target_start']) and numeric.map_date.iloc[-1]==pd.Timestamp(p['segments']['test_end']),'numeric coverage differs from registered boundary')
    entries=[];errors=[]
    for path in sorted((RAW/'narratives').glob('*.xml')):
        try:entries.append(parse_narrative(path))
        except Exception as e:errors.append({'file':path.name,'error':str(e)})
    require(entries,'no parsed texts');catalog=pd.DataFrame(entries).sort_values('map_date').reset_index(drop=True)
    out=LOCAL/'prepared';out.mkdir(parents=True,exist_ok=True);csv(out/'source_catalog.csv',entries);dump(out/'parse_errors.json',errors)
    valid=catalog[catalog.usable_content].copy();valid['date']=pd.to_datetime(valid.map_date);valid['available']=pd.to_datetime(valid.available_utc,utc=True);valid['conservative']=pd.to_datetime(valid.conservative_available_utc,utc=True)
    raw=numeric.dsci.to_numpy(float);train_rows=numeric.map_date.le(p['segments']['train_end']).to_numpy();mu=raw[train_rows].mean();std=raw[train_rows].std();values=(raw-mu)/std
    csv(out/'numeric.csv',numeric[['map_date','dsci']].assign(map_date=numeric.map_date.dt.strftime('%Y-%m-%d')).to_dict('records'))
    old=pd.read_csv(DATA/'SafeFAME_TS_第4轮_2d530ec7/Climate_numerical.csv');matched=old.merge(numeric[['map_date','dsci']],left_on=pd.to_datetime(old.start_date),right_on='map_date');difference=abs(matched.OT-matched.dsci)
    identity={'matched_weeks':len(matched),'exact_matches_at_0_01':int((difference<=.01000001).sum()),'max_abs_difference':float(difference.max()),'mean_abs_difference':float(difference.mean()),'same_target_confirmed':bool((difference<=.01000001).all()),'conclusion':'Only exact numeric matching can establish historical target identity. Otherwise treat official DSCI as a new explicitly defined task; do not splice old values.'}
    dump(HERE/'results/data/historical_target_identity.json',identity)
    for h in p['horizons']:
        rows=[];x=[];targets=[];texts=[]
        for i in range(p['input_length'],len(raw)-h+1):
            start=numeric.map_date.iloc[i];end=numeric.map_date.iloc[i+h-1];segment=None
            for name,limit in p['segments'].items():
                if start<=pd.Timestamp(limit):
                    if end<=pd.Timestamp(limit):segment=name.replace('_end','')
                    break
            if segment is None:continue
            now=(numeric.map_date.iloc[i-1]+pd.Timedelta(days=3)).tz_localize('America/New_York').tz_convert('UTC')
            eligible=valid[(valid.date<=numeric.map_date.iloc[i-1])&(valid.available<=now)&(valid.date>=numeric.map_date.iloc[i-1]-pd.Timedelta(weeks=8))]
            chosen=eligible.iloc[-1] if len(eligible) else None
            texts.append('' if chosen is None else chosen.model_text)
            rows.append({'origin_index':i,'target_date':start.strftime('%Y-%m-%d'),'target_end':end.strftime('%Y-%m-%d'),'asof_utc':now.isoformat(),'segment':segment,'text_available':chosen is not None,'source_map_date':None if chosen is None else chosen.map_date,'source_url':None if chosen is None else chosen.url,'source_sha256':None if chosen is None else chosen.raw_sha256,'publication_grade':'none' if chosen is None else chosen.publication_grade,'source_available_utc':None if chosen is None else chosen.available_utc,'strict_eligible':False if chosen is None else bool(chosen.strict_eligible),'conservative_eligible':False if chosen is None else bool(chosen.conservative<=now)})
            x.append(values[i-p['input_length']:i]);targets.append(values[i:i+h])
        frame=pd.DataFrame(rows);x=np.array(x);y=np.array(targets);folder=out/f'h{h}';folder.mkdir(exist_ok=True)
        non=frame.segment.ne('test').to_numpy();test=~non
        for side,mask in [('selection',non),('test_features',test)]:
            target=folder/side;target.mkdir(exist_ok=True);csv(target/'metadata.csv',frame[mask].to_dict('records'));np.save(target/'x.npy',x[mask]);dump(target/'texts.json',[t for t,keep in zip(texts,mask) if keep])
            if side=='selection':np.save(target/'y.npy',y[mask])
        np.save(folder/'test_truth.npy',y[test]);dump(folder/'scaler.json',{'mean':mu,'std':std,'fit_through':p['segments']['train_end']})
        require(frame[non].target_end.max()<=p['segments']['decision_end'],'boundary crossed')
        csv(HERE/f'results/data/h{h}_coverage.csv',[{'segment':name,'origins':len(g),'usable_text':int(g.text_available.sum()),'strict_explicit_text':int(g.strict_eligible.sum()),'archive_conservative_text':int(g.conservative_eligible.sum())} for name,g in frame.groupby('segment',sort=False)])
    dump(out/'MANIFEST.json',{'protocol_sha256':sha(HERE/'protocol.json'),'raw_numeric_sha256':sha(RAW/'conus_dsci.json'),'files':inventory(out)})
    summary={'numeric_weeks':len(numeric),'first_week':str(numeric.map_date.iloc[0].date()),'last_week':str(numeric.map_date.iloc[-1].date()),'raw_text_files':len(catalog)+len(errors),'parsed_text_files':len(catalog),'usable_text_files':int(catalog.usable_content.sum()),'explicit_release_eligible':int(catalog.strict_eligible.sum()),'schedule_inferred_files':int(catalog.publication_grade.eq('official_schedule_inferred').sum()),'parse_failures':len(errors),'protocol_sha256':sha(HERE/'protocol.json'),'prepared_manifest_sha256':sha(out/'MANIFEST.json'),'limits':['Official URLs and raw bytes verified; XML map dates are not first-publication timestamps.','Numeric API is a retrieval-date archive snapshot, not authenticated vintage history.','Schedule-based replay is an assumption-sensitive experiment; exact-publication mode is separately reported.']}
    dump(HERE/'results/data/data_acceptance.json',summary);print(json.dumps(summary,ensure_ascii=False));print(json.dumps(identity))

if __name__=='__main__':main()
