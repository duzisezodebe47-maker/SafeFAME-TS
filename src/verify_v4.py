"""Independently recompute v4 evidence; optionally replay saved checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from run_safefame_v4 import ROOT, DATA, digest, arrays_for, interval, write_json, VARIANTS, fit_path, refit_path, reference_null
from run_baselines import metrics, fit_ridge, predict_ridge, seasonal_naive
from train_dlinear import DLinearTarget, predict
from train_patchtst import PatchTSTTarget

def check(a,b,label):
    assert np.allclose(a,b,rtol=1e-6,atol=1e-7,equal_nan=True),(label,a,b)

def verify(folder,replay=False,partial=False):
    folder=folder.resolve()
    manifest=json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
    cfg=manifest['config']
    text_index=pd.read_csv(DATA/'text/sample_text_index.csv').set_index(['domain','origin_index'])
    for p,h in manifest['hashes'].items():assert digest(ROOT/p)==h,('source changed',p)
    expected={(d,h,f+1) for d,s in cfg['domains'].items() for h in s['horizons'] for f in range(3)}
    completed=list((folder/'tasks').glob('*/completed.json'))
    if not partial: assert len(completed)==len(expected)==120
    count=0;checkpoints=0;observed=set();testsets={};full_null_replays=[];coverage_records={}
    allhashes={ (folder/'run_manifest.json').relative_to(ROOT).as_posix():digest(folder/'run_manifest.json') }
    execution=folder/'worker_execution.json'
    if execution.exists():
        provenance=json.loads(execution.read_text(encoding='utf-8'))
        names=[]
        for run in provenance['executions']:
            assert run['manifest']['signature']==manifest['signature']
            assert run['manifest']['config']==cfg and run['manifest']['hashes']==manifest['hashes']
            names.extend(run['task_names'])
        assert sorted(names)==provenance['merged_tasks'] and len(set(names))==114
        allhashes[execution.relative_to(ROOT).as_posix()]=digest(execution)
    for done in sorted(completed):
        task=done.parent
        record=json.loads(done.read_text(encoding='utf-8'))
        assert record['signature']==manifest['signature']
        for p,h in record['hashes'].items(): assert digest(task/p)==h,('evidence changed',task.name,p)
        a=json.loads((task/'audit.json').read_text(encoding='utf-8'))
        frozen=json.loads((task/'frozen_selection.json').read_text(encoding='utf-8'))
        assert frozen['frozen_utc']<a['completed_utc']<=record['completed_utc']
        d,h,f=a['domain'],a['horizon'],a['fold'];observed.add((d,h,f))
        z=np.load(task/'predictions.npz')
        for key in z.files:assert np.isfinite(z[key]).all(),(task.name,key,'nonfinite evidence')
        rows=pd.read_csv(task/'metrics.csv').set_index('variant')
        assert set(rows.index)==set(VARIANTS)
        frame=pd.read_csv(DATA/'numerical'/d/f'{d}.csv')
        raw=frame.OT.to_numpy(float)
        bounds=[int(len(frame)*v) for v in cfg['folds'][f-1]]
        assert bounds==a['bounds']==frozen['bounds']
        truth=(raw-np.nanmean(raw[:bounds[0]]))/np.nanstd(raw[:bounds[0]])
        for name,lo,hi in [('train',cfg['domains'][d]['input_len'],bounds[0]),('cal',bounds[0],bounds[1]),
                           ('dec',bounds[1],bounds[2]),('test',bounds[2],bounds[3]),('fit',cfg['domains'][d]['input_len'],bounds[2])]:
            origins=np.array([t for t in range(lo,hi-h+1) if np.isfinite(truth[t:t+h]).all()])
            assert np.array_equal(z[f'{name}_origins'],origins),(task.name,name)
            check(z[f'{name}_actual'],np.stack([truth[t:t+h] for t in origins]),'ground truth')
        coverage_records[task.name]={segment:float((text_index.loc[[(d,int(o)) for o in z[segment+'_origins']],'total_selected_count']>0).mean()*100) for segment in ('train','test')}
        # Folds share training history; only held-out target ranges must not overlap.
        test_targets=set(j for t in z['test_origins'] for j in range(int(t),int(t)+h))
        previous=testsets.setdefault((d,h),set())
        assert not previous.intersection(test_targets),(d,h,'test folds overlap')
        previous.update(test_targets)
        middle=(bounds[1]+bounds[2])//2
        masks=(z['dec_origins']+h<=middle,z['dec_origins']>=middle)
        for i,m in enumerate(masks,1):assert np.array_equal(z[f'segment{i}_mask'],m)
        scores={name:metrics(z['cal_actual'],z[f'{name}_cal'])['mse'] for name in frozen['numeric_calibration_scores']}
        for name,v in scores.items():check(v,frozen['numeric_calibration_scores'][name],'expert score')
        fallback=min(scores,key=scores.get);assert fallback==frozen['fallback']
        for name in ('DLinear-M','PatchTST'):
            for segment in ('cal','dec'):
                check(z[f'{name}_{segment}'],np.mean([z[f'{name}_s{s}_{segment}'] for s in cfg['seeds']],axis=0),'seed ensemble')
            for seed in cfg['seeds']:
                history=pd.read_csv(task/'models'/f'{name}_cal_s{seed}.csv')
                assert 1<=len(history)<=cfg['training']['max_epochs']
        allowed=[]
        for vi,v in enumerate(VARIANTS):
            row=rows.loc[v]
            score=metrics(z['dec_actual'],z[f'{v}_dec'])['mse']
            check(score,row.decision_mse,'decision mse')
            null=z[f'{v}_row_null'];shift=z[f'{v}_circular_null']
            assert null.shape==(cfg['row_permutations'],) and shift.shape==(cfg['circular_shifts'],)
            assert np.isfinite(null).all() and np.isfinite(shift).all()
            p=(1+np.count_nonzero(null<=score))/(len(null)+1)
            check(p,row.p_value,'row p')
            check((1+np.count_nonzero(shift<=score))/(len(shift)+1),row.circular_p_value,'shift p')
            wins=all(m.any() and metrics(z['dec_actual'][m],z[f'{v}_dec'][m])['mse']<
                     metrics(z['dec_actual'][m],z[f'{fallback}_dec'][m])['mse'] for m in masks)
            gate=score<metrics(z['dec_actual'],z[f'{fallback}_dec'])['mse'] and wins and p<=cfg['p_threshold']
            assert gate==row.eligible==frozen['candidates'][v]['eligible']
            assert wins==row.segment_wins
            if gate:allowed.append(v)
            for k,value in metrics(z['test_actual'],z[f'{v}_test']).items():check(value,row[k],k)
            assert z[f'{v}_test'].shape==z['test_actual'].shape
            check(100*(1-row.mse/metrics(z['test_actual'],z['fallback_test'])['mse']),row.improvement_vs_fallback_pct,'gain')
            for control,keys in [('fallback_test',('delta','ci_low','ci_high')),(f'{v}_control_test',('matched_delta','matched_ci_low','matched_ci_high'))]:
                calculated=interval(z['test_actual'],z[control],z[f'{v}_test'],int(row.block_length),int(row.bootstrap_seed),cfg['bootstrap_repeats'])
                check(calculated,[row[k] for k in keys],'5000 block CI')
            assert int(row.test_windows)==len(z['test_origins'])
            assert int(row.calibration_windows)==len(z['cal_origins'])
            assert int(row.decision_windows)==len(z['dec_origins'])
        selected=min(allowed,key=lambda v:rows.loc[v,'decision_mse']) if allowed else 'numeric_fallback'
        assert selected==frozen['selected']
        check(z['selected_test'],z['fallback_test'] if selected=='numeric_fallback' else z[f'{selected}_test'],'route')
        for k,value in metrics(z['test_actual'],z['selected_test']).items():check(value,a['selected_metrics'][k],'selected metric')
        for k,value in metrics(z['test_actual'],z['fallback_test']).items():check(value,a['fallback_metrics'][k],'fallback metric')
        check(interval(z['test_actual'],z['fallback_test'],z['selected_test'],a['block_length'],2026,cfg['bootstrap_repeats']),a['selected_delta'],'selected 5000 block CI')
        if replay:
            arrays,origins,tensors,prep=arrays_for(d,cfg['domains'][d],h,bounds)
            for vi,v in enumerate(VARIANTS):
                b,model,alpha,_=fit_path(v,arrays['train'],arrays['cal'])
                check(alpha,rows.loc[v,'alpha'],'alpha replay')
                check(arrays['dec'][0][:,-1,None]+model.predict(b.transform(*arrays['dec'][:3])).reshape(len(arrays['dec'][0]),-1),z[f'{v}_dec'],'candidate replay')
                check(refit_path(v,alpha,arrays['fit'],arrays['test']),z[f'{v}_test'],'test replay')
                check(refit_path(v,rows.loc[v,'matched_control_alpha'],arrays['fit'],arrays['test'],control=True),z[f'{v}_control_test'],'control replay')
                # float32 PCA row-order rounding can exceed the tight scalar audit
                # tolerance. Then replay ALL draws and require the exact same p,
                # as well as bounded score error; never relax metric/route checks.
                for mode,seed,pkey in [('row',2026,'p_value'),('circular',4100+100*(f-1)+10*h+vi,'circular_p_value')]:
                    saved=z[f'{v}_{mode}_null'];p=float(rows.loc[v,pkey])
                    direct=reference_null(b,arrays['train'],arrays['cal'],arrays['dec'],2,mode,alpha,a['block_length'],seed)
                    if not np.allclose(direct,saved[:2],rtol=1e-6,atol=1e-7) or .02<=p<=.03:
                        direct=reference_null(b,arrays['train'],arrays['cal'],arrays['dec'],len(saved),mode,alpha,a['block_length'],seed)
                        assert np.allclose(direct,saved,rtol=1e-4,atol=1e-10),(task.name,v,mode,'null rounding exceeds bound')
                        direct_p=(1+int((direct<=rows.loc[v,'decision_mse']).sum()))/(len(saved)+1)
                        check(direct_p,p,'full reference null p')
                        full_null_replays.append(dict(task=task.name,variant=v,mode=mode,draws=len(saved),p=p,
                            max_absolute_error=float(np.max(abs(direct-saved))),
                            max_relative_error=float(np.max(abs(direct-saved)/np.maximum(abs(direct),1e-12)))))
            if fallback in ('DLinear-M','PatchTST'):
                predictions=[]
                for seed in cfg['seeds']:
                    payload=torch.load(task/'models'/f'{fallback}_refit_s{seed}.pt',map_location='cpu',weights_only=True)
                    cls=DLinearTarget if fallback=='DLinear-M' else PatchTSTTarget
                    model=cls(**payload['model_args']);model.load_state_dict(payload['state_dict'])
                    xp=tensors['test'][0 if fallback=='DLinear-M' else 1]
                    predictions.append(predict(model,xp,torch.device('cpu'),64));checkpoints+=1
                assert np.allclose(np.mean(predictions,axis=0),z['fallback_test'],rtol=2e-5,atol=2e-5),'deep checkpoint replay'
            else:
                x=arrays['test'][0]
                p=(np.repeat(x[:,-1:],h,axis=1) if fallback=='Last' else
                   seasonal_naive(x,h,cfg['domains'][d]['seasonal_period']) if fallback=='SeasonalNaive' else
                   predict_ridge(x,fit_ridge(arrays['fit'][0],arrays['fit'][3],frozen['ridge_alpha'])))
                check(p,z['fallback_test'],'numerical fallback replay')
        allhashes.update({(task/p).relative_to(ROOT).as_posix():h for p,h in record['hashes'].items()})
        allhashes[done.relative_to(ROOT).as_posix()]=digest(done)
        count+=1;print('VERIFIED',count,task.name,flush=True)
    if not partial:assert observed==expected
    return dict(status='PASS',task_folds=count,candidate_paths=2*count,replay_models=replay,
                replayed_refit_checkpoints=checkpoints,sha256=allhashes,
                full_null_replays=full_null_replays,
                text_coverage_pct=coverage_records,
                scope='saved predictions, targets, target boundaries, gates, both null distributions and all CIs; reference null draws and model replay when requested',
                limitation='two direct null draws per candidate/mode; all draws for rounding discrepancies or p in [0.02,0.03]; not prospective unseen data; publication timestamps remain proxies')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'outputs/safefame_v4')
    parser.add_argument('--replay-models',action='store_true');parser.add_argument('--partial',action='store_true')
    parser.add_argument('--record',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(1):result=verify(args.output,args.replay_models,args.partial)
    if args.record:
        assert not args.partial,'Partial checks cannot authorize final output'
        write_json(args.output/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='sha256'},ensure_ascii=False))
