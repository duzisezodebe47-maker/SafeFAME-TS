"""下载官方原始快照，有限重试、错误日志、URL和采集时间；不输出测试指标。"""
from __future__ import annotations
import argparse,re,time,urllib.request,urllib.parse
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from r14_common import *
URL='https://droughtmonitor.unl.edu/Data/narrative/'
API='https://usdmdataservices.unl.edu/api/USStatistics/GetDSCI?aoi=conus&startdate=1/4/2000&enddate=9/22/2026&statisticsType=1'

def fetch(url,path):
    path=Path(path);meta=path.with_suffix(path.suffix+'.meta.json')
    if path.exists() and meta.exists():
        record=read(meta);require(record['sha256']==sha(path),'cached bytes changed');return record
    path.parent.mkdir(parents=True,exist_ok=True);error=None
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'SafeFAME-TS academic reproducibility study','Accept':'application/json' if '/api/' in url else '*/*'})
            with urllib.request.urlopen(req,timeout=45) as response:
                payload=response.read();headers=dict(response.headers);final=response.url
            require(len(payload)>20,'empty response');path.write_bytes(payload)
            record={'url':url,'final_url':final,'fetched_utc':datetime.now(timezone.utc).isoformat(),'bytes':len(payload),'sha256':sha(path),'last_modified':headers.get('Last-Modified'),'content_type':headers.get('Content-Type'),'attempts':attempt+1}
            dump(meta,record);return record
        except Exception as e:error=f'{type(e).__name__}: {e}';time.sleep(.5*(attempt+1))
    return {'url':url,'error':error,'path':str(path)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['probe','all'],default='probe');a=ap.parse_args();RAW.mkdir(parents=True,exist_ok=True)
    # Registration predates downloading the new numerical targets.
    seal=LOCAL/'protocol_registration.json'
    if not seal.exists():dump(seal,{'protocol_sha256':sha(HERE/'protocol.json'),'registered_utc':datetime.now(timezone.utc).isoformat(),'test_metrics_seen':False})
    require(read(seal)['protocol_sha256']==sha(HERE/'protocol.json'),'protocol changed after registration')
    specs=[(URL,RAW/'narrative_index.html'),(API,RAW/'conus_dsci.json'),('https://drought.unl.edu/Education/Tutorials/Usdm/Section-Six.aspx',RAW/'release_schedule.html')]
    records=[fetch(url,path) for url,path in specs];dump(LOCAL/'download_core_log.json',records)
    require(all('error' not in r for r in records),'core download failed; see log')
    index=(RAW/'narrative_index.html').read_text(encoding='utf-8-sig');links=sorted(set(re.findall(r'href="([^"]*USDM_(\d{8})\.xml)"',index,re.I)))
    selected=[(urllib.parse.urljoin(URL,href),date) for href,date in links if '20200107'<=date<='20260922']
    require(len(selected)>150,'too few archive entries')
    if a.phase=='probe':selected=[selected[0],selected[len(selected)//2],selected[-1]]
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(lambda item:fetch(item[0],RAW/'narratives'/f'USDM_{item[1]}.xml'),selected))
    dump(LOCAL/f'download_{a.phase}_log.json',results)
    print(json.dumps({'phase':a.phase,'files':len(results),'ok':sum('error' not in r for r in results),'failed':sum('error' in r for r in results)},ensure_ascii=False))
    if a.phase=='probe':
        values=read(RAW/'conus_dsci.json');print('NUMERIC_SCHEMA',type(values).__name__,list(values[0]) if isinstance(values,list) else list(values));print('TEXT_SCHEMA', (RAW/'narratives'/f'USDM_{selected[0][1]}.xml').read_text(encoding='utf-8-sig')[:2200])

if __name__=='__main__':main()
