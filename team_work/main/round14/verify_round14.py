"""从原始数值复算切片、检查封存字节、独立复算测试及因果回放。"""
from __future__ import annotations
import joblib,subprocess
from r14_common import *
from train_models import code_anchors,numeric_predict,residual_predict
from adaptive import causal_replay
from threadpoolctl import threadpool_limits

def check_files(folder,mapping):
    for rel,digest in mapping.items():require(sha(folder/rel)==digest,f'byte mismatch: {folder/rel}')

def verify():
    if not LOCAL.exists():print('PARTIAL: local raw/model evidence intentionally absent from Git');return
    prepared=LOCAL/'prepared';m=read(prepared/'MANIFEST.json');check_files(prepared,m['files']);require(m['protocol_sha256']==sha(HERE/'protocol.json'),'protocol anchor');require(m['raw_numeric_sha256']==sha(RAW/'conus_dsci.json'),'numeric snapshot')
    raw_checks=0
    for meta in RAW.rglob('*.meta.json'):
        record=read(meta);original=Path(str(meta)[:-10]);require(sha(original)==record['sha256'],'download bytes changed');raw_checks+=1
    numeric=pd.read_csv(prepared/'numeric.csv');raw=numeric.dsci.to_numpy(float);data_rows=0;model_runs=0;causal_checks=0
    for h in cfg()['horizons']:
        folder=prepared/f'h{h}';scale=read(folder/'scaler.json');dates=pd.to_datetime(numeric.map_date);tr=dates.le(cfg()['segments']['train_end']).to_numpy();require(np.isclose(raw[tr].mean(),scale['mean']) and np.isclose(raw[tr].std(),scale['std']),'train scaler');values=(raw-scale['mean'])/scale['std']
        for side in ('selection','test_features'):
            meta=pd.read_csv(folder/side/'metadata.csv');x=np.load(folder/side/'x.npy');expected=np.stack([values[i-52:i] for i in meta.origin_index]);require(np.array_equal(x,expected),'history slice mismatch');require(len(read(folder/side/'texts.json'))==len(meta),'text rows')
            y=np.stack([values[i:i+h] for i in meta.origin_index]);stored_y=np.load(folder/('test_truth.npy' if side=='test_features' else 'selection/y.npy'));require(np.array_equal(y,stored_y),'target slice mismatch')
            for name,group in meta.groupby('segment'):
                limit=cfg()['segments'][name+'_end'];require(group.target_end.le(limit).all(),'target boundary crosses segment')
                previous={'calibration':'train_end','decision':'calibration_end','test':'decision_end'}.get(name)
                if previous:require(group.target_date.gt(cfg()['segments'][previous]).all(),'segment starts too early')
            with_text=meta[meta.text_available];require((pd.to_datetime(with_text.source_available_utc,utc=True)<=pd.to_datetime(with_text.asof_utc,utc=True)).all(),'future narrative')
            require((pd.to_datetime(with_text.source_map_date)<pd.to_datetime(with_text.target_date)).all(),'same/future target map leaked');data_rows+=len(meta)
        require(not (folder/'test_features/y.npy').exists(),'test labels in feature package')
        model_folder=LOCAL/'models'/f'h{h}';seal=read(model_folder/'selection_seal.json');check_files(model_folder,seal['files']);require(seal['code']==code_anchors(),'training code');require(seal['prepared_manifest_sha256']==sha(prepared/'MANIFEST.json'),'training input anchor');detail=read(model_folder/'selection_details.json')
        require(seal['selected_numeric']==min(detail['numeric_calibration_scores'],key=lambda n:(detail['numeric_calibration_scores'][n],n)),'numeric selection mismatch')
        for fold in detail['oof_folds']:
            require(fold['fit_last_origin']+h<=fold['validation_first_origin'],'OOF purge failure');stop=fold['scaler_fit_raw_rows'];require(stop==fold['validation_first_origin'],'OOF scaler future rows');require(np.isclose(raw[:stop].mean(),fold['scaler_mean']) and np.isclose(raw[:stop].std(),fold['scaler_std']),'OOF fold scaler')
        model_runs+=sum(len(v) for v in detail['deep_training'].values())
        out=LOCAL/'evaluation'/f'h{h}';commit=read(out/'fixed_commitment.json');require(commit['prediction_sha256']==sha(out/'fixed_predictions.npz'),'prediction commitment');require(commit['selection_seal_sha256']==sha(model_folder/'selection_seal.json'),'selection commitment');require(commit['scorer_sha256']==sha(HERE/'evaluate_test.py'),'scorer changed')
        fixed=np.load(out/'fixed_predictions.npz');meta=pd.read_csv(folder/'test_features/metadata.csv');require(np.array_equal(fixed['origins'],meta.origin_index),'prediction row alignment');y=np.load(folder/'test_truth.npy');pred={k:fixed[k] for k in fixed.files if k!='origins'};pred['AdaptiveFallback']=np.load(out/'adaptive_predictions.npy')
        ad=read(out/'adaptive_commitment.json');require(ad['prediction_sha256']==sha(out/'adaptive_predictions.npy') and ad['trace_sha256']==sha(out/'adaptive_trace.csv'),'adaptive bytes changed')
        objects=joblib.load(model_folder/'models.joblib');x=np.load(folder/'test_features/x.npy');texts=read(folder/'test_features/texts.json');available=meta.text_available.to_numpy(bool)
        for name,obj in objects['numeric'].items():require(np.allclose(numeric_predict(name,obj,x,h),pred[name],rtol=1e-7,atol=1e-7),'model replay mismatch')
        direct=pred['SelectedNumeric']+residual_predict(objects['text'],texts,available);require(np.allclose(direct,pred['DirectFusion'],rtol=1e-12,atol=1e-12),'text replay mismatch')
        require(np.array_equal(pred['StrictSourceFallback'],pred['SelectedNumeric']),'strict fallback differs')
        warm=np.load(model_folder/'causal_warmup.npz');origins=np.r_[warm['origins'],meta.origin_index];b=np.r_[warm['base'],pred['SelectedNumeric']];d=np.r_[warm['direct'],pred['DirectFusion']];ay=np.r_[warm['truth'],y];av=np.r_[warm['available'],available];r,trace=causal_replay(origins,b,d,av,h,cfg()['adaptive'],lambda i:ay[i]);require(np.allclose(r[len(warm['origins']):],pred['AdaptiveFallback'],rtol=1e-12,atol=1e-12),'adaptive replay differs')
        saved=pd.read_csv(out/'adaptive_trace.csv');require(np.array_equal(saved.weight,[v['weight'] for v in trace[len(warm['origins']):]]),'weight trace differs');require(all(v['last_feedback_origin'] is None or v['last_feedback_origin']+h<=v['origin_index'] for v in trace),'future feedback');causal_checks+=1
        metric=pd.read_csv(HERE/f'results/h{h}/test_metrics.csv',float_precision='round_trip');require(set(metric.candidate)==set(pred),'metric candidates')
        for row in metric.to_dict('records'):require(np.isclose(row['mse'],mse(pred[row['candidate']],y),rtol=1e-12,atol=1e-14),'metric mismatch')
    diff=subprocess.check_output(['git','diff','HEAD','--name-only','--','team_work/main/round13'],cwd=ROOT).decode();require(not diff.strip(),'round13 altered')
    result={'status':'PASS_WITH_SOURCE_TIME_LIMITATIONS','raw_snapshot_files_verified':raw_checks,'prediction_origins_checked':data_rows,'main_deep_runs':model_runs,'causal_replays':causal_checks,'test_horizons':cfg()['horizons'],'checks':['original snapshot hashes','train-only and OOF-fold-only scalers','target-contained temporal boundaries','numeric windows reconstructed from raw series','publication cutoff checks','numeric model selection and weight hashes','fixed predictions and independent MSE recalculation','model inference reproduced from saved objects','causal delayed feedback replay','strict fallback equality','previous round13 unchanged'],'limitations':['Strict exact-first-publication corpus remains empty.','Official DSCI differs from the previous Climate target; this is a distinct explicitly defined task.','Data downloaded today may contain historical revisions; no immutable vintage archive established.','Later-period historical evaluation is not prospective or cross-domain validation.'],'verifier_sha256':sha(Path(__file__))}
    dump(HERE/'results/final/verification.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    import torch
    torch.set_num_threads(2)
    with threadpool_limits(limits=1):verify()
