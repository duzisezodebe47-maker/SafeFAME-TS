"""先登记协议，再下载固定三州数据和官方来源说明；失败留日志。"""
import urllib.request,time
from concurrent.futures import ThreadPoolExecutor
from common16 import *

def fetch(url,path):
    path=Path(path);side=path.with_suffix(path.suffix+'.meta.json')
    if path.exists() and side.exists():
        record=read(side);require(record['sha256']==sha(path),'cache hash mismatch');return record
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'Accept':'application/json' if '/api/' in url else '*/*','User-Agent':'SafeFAME-TS academic reproducibility study'})
            with urllib.request.urlopen(req,timeout=25) as r:body=r.read();headers=dict(r.headers);final=r.url
            require(len(body)>20,'empty response');path.write_bytes(body)
            record={'url':url,'final_url':final,'fetched_utc':pd.Timestamp.now(tz='UTC').isoformat(),'sha256':sha(path),'bytes':len(body),'last_modified':headers.get('Last-Modified'),'etag':headers.get('ETag'),'content_type':headers.get('Content-Type'),'attempt':attempt+1};dump(side,record);return record
        except Exception as e:error=type(e).__name__+': '+str(e);time.sleep(.5*(attempt+1))
    return {'url':url,'error':error,'path':str(path)}

def main():
    LOCAL.mkdir(parents=True,exist_ok=True);registration=LOCAL/'protocol_registration.json'
    if not registration.exists():dump(registration,{'protocol_sha256':sha(HERE/'protocol.json'),'registered_utc':pd.Timestamp.now(tz='UTC').isoformat(),'state_data_downloaded':False,'research_role':cfg()['role']})
    require(read(registration)['protocol_sha256']==sha(HERE/'protocol.json'),'protocol changed')
    specs=[]
    for state in cfg()['states']:
        url=f'https://usdmdataservices.unl.edu/api/StateStatistics/GetDSCI?aoi={state["fips"]}&startdate=1/4/2000&enddate=9/22/2026&statisticsType=1'
        specs.append((url,LOCAL/f'raw/{state["code"]}.json'))
    specs.extend((url,LOCAL/'raw'/name) for name,url in [('service_info.html','https://droughtmonitor.unl.edu/DmData/DataDownload/WebServiceInfo.aspx'),('release_schedule.html','https://drought.unl.edu/Education/Tutorials/Usdm/Section-Six.aspx'),('dsci_definition.html','https://droughtmonitor.unl.edu/About/AbouttheData/DSCI.aspx')])
    old=sorted((PREVIOUS/'raw/narratives').glob('*.xml'));require(len(old)==351,'previous archive count differs')
    for idx in np.linspace(0,len(old)-1,6,dtype=int):
        path=old[idx];specs.append((read(path.with_suffix('.xml.meta.json'))['url'],LOCAL/'raw/repeated_xml'/path.name))
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(lambda v:fetch(*v),specs))
    dump(LOCAL/'download_log.json',results);print(json.dumps({'requests':len(results),'success':sum('error' not in r for r in results),'errors':[r for r in results if 'error' in r]},ensure_ascii=False))
    require(all((LOCAL/f'raw/{s["code"]}.json').exists() for s in cfg()['states']),'state download incomplete')
    for s in cfg()['states']:
        rows=read(LOCAL/f'raw/{s["code"]}.json');print(s['code'],len(rows),'fields',list(rows[0]),'identity',{k:v for k,v in rows[0].items() if k not in ('dsci','mapDate')})

if __name__=='__main__':main()
