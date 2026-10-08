"""验证州别、周序列和取值范围，按完整目标期隔离选择与评价数据。"""
from common16 import *

def validate_series(rows,name):
    frame=pd.DataFrame(rows)
    require(set(('name','mapDate','dsci'))<=set(frame),'missing fields')
    require(frame.name.eq(name).all(),'wrong state identity')
    frame['date']=pd.to_datetime(frame.mapDate.astype(str),format='ISO8601');frame=frame.sort_values('date').reset_index(drop=True)
    require(not frame.date.duplicated().any(),'duplicate dates')
    require(frame.date.diff().dropna().eq(pd.Timedelta(days=7)).all(),'weekly gap')
    require(np.isfinite(frame.dsci).all() and frame.dsci.between(0,500).all(),'invalid DSCI')
    return frame

def windows(values,dates,h,p):
    rows=[];xs=[];ys=[]
    for i in range(p['input_length'],len(values)-h+1):
        start=dates.iloc[i];end=dates.iloc[i+h-1];segment=None
        for name,boundary in p['segments'].items():
            if start<=pd.Timestamp(boundary):
                if end<=pd.Timestamp(boundary):segment=name
                break
        if segment is None:continue
        rows.append({'origin_index':i,'target_date':str(start.date()),'target_end':str(end.date()),'segment':segment});xs.append(values[i-p['input_length']:i]);ys.append(values[i:i+h])
    return pd.DataFrame(rows),np.asarray(xs),np.asarray(ys)

def main():
    p=cfg();out=LOCAL/'prepared';require(not out.exists(),'prepared output exists')
    require(read(LOCAL/'protocol_registration.json')['protocol_sha256']==sha(HERE/'protocol.json'),'protocol registration mismatch')
    summaries=[]
    for state in p['states']:
        code=state['code'];raw=LOCAL/f'raw/{code}.json';require(read(raw.with_suffix('.json.meta.json'))['sha256']==sha(raw),'raw changed')
        frame=validate_series(read(raw),state['name']);require(frame.date.iloc[0]==pd.Timestamp(p['start']) and frame.date.iloc[-1]==pd.Timestamp(p['segments']['test']),'coverage differs')
        fit=frame.date.le(p['segments']['train']);mu=frame.loc[fit,'dsci'].mean();std=frame.loc[fit,'dsci'].std(ddof=0);require(std>0,'constant training data');values=(frame.dsci.to_numpy(float)-mu)/std
        csv(out/f'{code}/numeric.csv',frame[['date','dsci']].assign(date=frame.date.dt.strftime('%Y-%m-%d')).to_dict('records'));dump(out/f'{code}/scaler.json',{'mean':mu,'std':std,'fit_end':p['segments']['train'],'fit_rows':int(fit.sum())})
        for h in p['horizons']:
            meta,x,y=windows(values,frame.date,h,p);selection=meta.segment.ne('test').to_numpy();folder=out/f'{code}_h{h}';folder.mkdir(parents=True)
            np.savez_compressed(folder/'selection.npz',x=x[selection],y=y[selection],origins=meta.loc[selection,'origin_index'].to_numpy());csv(folder/'selection_metadata.csv',meta[selection].to_dict('records'))
            np.savez_compressed(folder/'test_features.npz',x=x[~selection],origins=meta.loc[~selection,'origin_index'].to_numpy());csv(folder/'test_metadata.csv',meta[~selection].to_dict('records'));np.save(folder/'test_truth.npy',y[~selection])
            summaries.extend({'state':code,'horizon':h,'segment':name,'origins':len(g),'first_target':g.target_date.min(),'last_target_end':g.target_end.max()} for name,g in meta.groupby('segment',sort=False))
    csv(HERE/'results/data_coverage.csv',summaries);dump(out/'MANIFEST.json',{'protocol_sha256':sha(HERE/'protocol.json'),'raw_files':{s['code']:sha(LOCAL/f'raw/{s["code"]}.json') for s in p['states']},'files':inventory(out)})
    print('PREPARED',len(summaries),'state-horizon-segment rows')

if __name__=='__main__':main()
