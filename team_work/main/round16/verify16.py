"""独立检查原始身份、分段、选型、保存权重和在线成熟反馈。"""
from common16 import *
from model16 import *
from prepare16 import validate_series,windows
from threadpoolctl import threadpool_limits

def verify():
    if not LOCAL.exists():print('PARTIAL: 第16轮本地原始数据和权重未随Git分发');return
    p=cfg();registration=read(LOCAL/'protocol_registration.json');require(registration['protocol_sha256']==sha(HERE/'protocol.json'),'protocol changed')
    raw_count=0
    for side in (LOCAL/'raw').rglob('*.meta.json'):
        record=read(side);require(sha(Path(str(side)[:-10]))==record['sha256'],'raw snapshot mismatch');raw_count+=1
    prepared=read(LOCAL/'prepared/MANIFEST.json')
    for rel,digest in prepared['files'].items():require(sha(LOCAL/'prepared'/rel)==digest,'prepared changed')
    commitment=read(LOCAL/'evaluation/fixed_commitment.json');require(commitment['scorer_sha256']==sha(HERE/'evaluate16.py'),'scorer changed');require(commitment['selection_seals_sha256']==sha(LOCAL/'all_selection_seals.json'),'global selection changed')
    for rel,digest in commitment['prediction_files'].items():require(sha(LOCAL/'evaluation'/rel)==digest,'fixed prediction bytes changed')
    for rel,digest in read(LOCAL/'evaluation/MANIFEST.json')['files'].items():require(sha(LOCAL/'evaluation'/rel)==digest,'evaluation changed')
    counts={'input_origins':0,'metrics':0,'seed_metrics':0,'online_predictions':0,'tasks':0};allmetrics=pd.read_csv(HERE/'results/all_metrics.csv',float_precision='round_trip');seedmetrics=pd.read_csv(HERE/'results/dlinear_seed_metrics.csv',float_precision='round_trip')
    for state in p['states']:
        code=state['code'];source=LOCAL/f'raw/{code}.json';frame=validate_series(read(source),state['name']);require(pd.Timestamp(registration['registered_utc'])<pd.Timestamp(read(source.with_suffix('.json.meta.json'))['fetched_utc']),'registration after download')
        scale=read(LOCAL/f'prepared/{code}/scaler.json');fit=frame.date.le(p['segments']['train']);np.testing.assert_allclose([scale['mean'],scale['std']],[frame.loc[fit,'dsci'].mean(),frame.loc[fit,'dsci'].std(ddof=0)]);values=(frame.dsci.to_numpy()-scale['mean'])/scale['std']
        for h in p['horizons']:
            task=f'{code}_h{h}';counts['tasks']+=1;folder=LOCAL/f'prepared/{task}';sm=pd.read_csv(folder/'selection_metadata.csv');tm=pd.read_csv(folder/'test_metadata.csv');selection=np.load(folder/'selection.npz');features=np.load(folder/'test_features.npz');truth=np.load(folder/'test_truth.npy');meta=pd.concat([sm,tm],ignore_index=True);origins=meta.origin_index.to_numpy();x=np.r_[selection['x'],features['x']];y=np.r_[selection['y'],truth]
            np.testing.assert_array_equal(x,np.stack([values[i-52:i] for i in origins]));np.testing.assert_array_equal(y,np.stack([values[i:i+h] for i in origins]));counts['input_origins']+=len(x)
            for name,g in meta.groupby('segment'):require(g.target_end.le(p['segments'][name]).all(),'segment crosses boundary')
            seal=read(LOCAL/f'models/{task}/selection_seal.json');require(not seal['test_metrics_seen'],'selection not frozen');require(seal['code']==code_anchors(),'code anchor mismatch');fixed=np.load(LOCAL/f'evaluation/{task}/fixed_predictions.npz');fresh,seeds,_=load_predictions(task,features['x']);detail=read(LOCAL/f'models/{task}/details.json');cal=sm.segment.eq('calibration').to_numpy();dec=sm.segment.eq('decision').to_numpy();tr=sm.segment.eq('train').to_numpy();sy=selection['y'];sx=selection['x']
            for name,arr in fresh.items():np.testing.assert_allclose(fixed[name],arr,rtol=1e-7,atol=1e-7)
            scores={a:mse(ridge_predict(ridge_fit(sx[tr],sy[tr],a),sx[cal]),sy[cal]) for a in p['ridge_alphas']};require(detail['ridge_alpha']==min(scores,key=lambda a:(scores[a],a)),'ridge alpha differs')
            ridge=ridge_fit(sx[tr|cal],sy[tr|cal],detail['ridge_alpha']);np.testing.assert_allclose(fresh['Ridge'],ridge_predict(ridge,features['x']),atol=1e-10)
            cp=np.load(LOCAL/f'models/{task}/calibration_predictions.npz');cal_scores={name:mse(cp[name],sy[cal]) for name in p['candidates']};winner=min(cal_scores,key=lambda n:(cal_scores[n],n));require(winner==seal['calibration_choice'],'calibration choice differs');np.testing.assert_array_equal(fixed['CalChoice'],fixed[winner])
            for i,seed in enumerate(p['seeds']):
                model=build(h);model.load_state_dict(torch.load(LOCAL/f'models/{task}/DLinear_{seed}_train.pt',weights_only=True,map_location='cpu'));stored=np.load(LOCAL/f'models/{task}/DLinear_{seed}_calibration.npy');np.testing.assert_allclose(predict(model,sx[cal]),stored,atol=1e-7)
                row=seedmetrics[(seedmetrics.state==code)&(seedmetrics.horizon==h)&(seedmetrics.seed==seed)].iloc[0];require(np.isclose(mse(seeds[i],truth),row.mse,rtol=1e-12),'seed score differs');counts['seed_metrics']+=1
            sp=np.load(LOCAL/f'models/{task}/selection_predictions.npz');names=p['candidates'];losses=np.stack([((sp[n][dec]-sy[dec])**2).mean(axis=1) for n in names],axis=1)
            for rule,name in seal['fixed_choices'].items():
                if rule=='decision_mean':score=losses.mean(axis=0)
                elif rule=='recent_26_origins':score=losses[-26:].mean(axis=0)
                else:
                    blocks=np.array([z.mean(axis=0) for z in np.array_split(losses,3)]);score=(blocks/np.maximum(blocks.min(axis=1)[:,None],1e-12)-1).max(axis=0)
                idx=min(range(len(names)),key=lambda j:(score[j],names[j]));require(name==names[idx],'fixed choice differs');np.testing.assert_array_equal(fixed[rule],fixed[name])
            prediction={n:fixed[n] for n in fixed.files if n!='origins'};combined_orig=np.r_[selection['origins'][dec],features['origins']];combined_y=np.r_[sy[dec],truth];combined=np.stack([np.r_[sp[n][dec],fixed[n]] for n in names],axis=1);warm=int(dec.sum())
            for lookback in p['online_lookbacks']:
                key=f'Online{lookback}';z=np.load(LOCAL/f'evaluation/{task}/{key}.npy');prediction[key]=z;trace=pd.read_csv(LOCAL/f'evaluation/{task}/{key}_trace.csv')
                for row,now in enumerate(features['origins']):
                    mature=np.flatnonzero(combined_orig+h<=now)[-lookback:];score=((combined[mature]-combined_y[mature,None,:])**2).mean(axis=(0,2));idx=min(range(len(names)),key=lambda j:(score[j],names[j]));require(trace.iloc[row].candidate==names[idx],'online choice differs');require(int(trace.iloc[row].last_feedback_origin)==combined_orig[mature[-1]],'feedback identity differs');np.testing.assert_array_equal(z[row],combined[warm+row,idx]);counts['online_predictions']+=1
            rows=allmetrics[(allmetrics.state==code)&(allmetrics.horizon==h)];require(set(rows.candidate)==set(prediction),'candidate set mismatch')
            for row in rows.itertuples():
                value=mse(prediction[row.candidate],truth);require(np.isclose(value,row.mse,rtol=1e-12,atol=1e-14),'metric mismatch');require(np.isclose(np.sqrt(value)*scale['std'],row.raw_rmse_dsci,rtol=1e-12),'raw units mismatch');counts['metrics']+=1
    result={'status':'PASS_SAME_DOMAIN_HISTORICAL_VALIDATION','raw_snapshots_checked':raw_count,**counts,'strict_text_enabled':False,'limits':['Three related states, not independent domains','Current archive snapshot, not immutable vintage','Original first-publication content still unproven','18 seeded training routes, not 18 independent datasets'],'verifier_sha256':sha(Path(__file__))};dump(HERE/'results/verification.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=1):verify()
