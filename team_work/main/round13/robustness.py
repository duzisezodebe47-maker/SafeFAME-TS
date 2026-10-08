"""多折数值验证、文本错位压力测试和块长敏感性；不回写任何路线。"""
from __future__ import annotations
import joblib
from common import *
from threadpoolctl import threadpool_limits

def delayed_features(f,delay):
    if delay<0:raise ValueError('negative delay')
    indices=np.searchsorted(f.origin_index,f.origin_index-delay,side='right')-1
    indices=np.minimum(indices,np.arange(len(indices)));valid=indices>=0;safe=np.maximum(indices,0)
    sem=f.semantic[safe].copy();q=f.quality[safe].copy();avail=f.text_available[safe].copy() & valid
    sem[~valid]=0;q[~valid]=0
    return FeatureBundle(f.numeric_history,sem,q,avail,f.origin_index),indices

def block_ci(delta,block,draws=5000):
    a=np.asarray(delta,float);n=len(a);rng=np.random.default_rng(2026)
    if not 1<=block<n:raise ValueError('block must be shorter than series; otherwise interval degenerates')
    starts=rng.integers(0,n,size=(draws,int(np.ceil(n/block))));idx=(starts[:,:,None]+np.arange(block))%n
    vals=a[idx.reshape(draws,-1)[:,:n]].mean(axis=1)
    return np.quantile(vals,[.025,.975]).tolist()

def rolling_numeric():
    rows=[];manifest=[];cfg=protocol()
    for original in cfg['tasks']:
        t=task_spec(original);domain=t['domain'];file=RAW/f'{domain}_numerical.csv';require(sha(file)==t['numerical_sha256'],'numerical source changed')
        df=pd.read_csv(file);raw=df.OT.to_numpy(float);require(len(raw)==t['n_rows_expected'] and np.isfinite(raw).all(),'raw values')
        require(pd.to_datetime(df.start_date).is_monotonic_increasing,'source date order')
        period=read(SPEC)['seasonal_periods'][domain]
        for horizon in cfg['numeric_rolling_horizons'][domain]:
            for fold,ratios in enumerate(cfg['numeric_rolling_folds'],1):
                bounds=[int(len(raw)*r) for r in ratios];mu=raw[:bounds[0]].mean();sd=raw[:bounds[0]].std();require(sd>0,'scale zero');values=(raw-mu)/sd
                origins=[];segs=[]
                for name,lo,hi in zip(('train','calibration','decision','test'),[0]+bounds[:-1],bounds):
                    o=np.arange(max(t['input_len'],lo),hi-horizon+1);origins.extend(o);segs.extend([name]*len(o))
                origins=np.asarray(origins);segs=np.asarray(segs);x=np.stack([values[i-t['input_len']:i] for i in origins]);y=np.stack([values[i:i+horizon] for i in origins]);tr=segs=='train';cal=segs=='calibration';test=segs=='test'
                f=FeatureBundle(x,np.zeros((len(x),1)),np.zeros((len(x),1)),np.zeros(len(x),bool),origins,targets=y.copy(),targets_standardized=y.copy())
                fit_y=y.copy();fit_y[~(tr|cal)]=np.nan;baseline=numeric_baselines(infer(f),segs,fit_y,seasonal_period=period)
                m=BranchResidualCandidate('N');m.fit_design(f.slice(tr));m.select_alphas(f.slice(tr),f.slice(cal));m.refit(f.slice(tr|cal))
                preds=dict(baseline['predictions']);preds['N']=m.predict(infer(f))
                fit=tr|cal;xm=x[fit].mean(axis=0);ym=y[fit].mean(axis=0);xc=x[fit]-xm;w=np.linalg.solve(xc.T@xc+baseline['ridge_alpha']*np.eye(x.shape[1]),xc.T@(y[fit]-ym));preds['AR-Ridge-refit']=(x-xm)@w+ym
                case=f'{domain}_h{horizon}_f{fold}';out=LOCAL/'rolling_numeric'/case;out.mkdir(parents=True,exist_ok=True)
                # Save before scoring. These are retrospective windows, not new blind data.
                np.savez_compressed(out/'predictions.npz',origins=origins[test],**{n:p[test] for n,p in preds.items()})
                last_mse=mse(preds['Last'][test],y[test])
                for name,p in preds.items():
                    rows.append({'domain':domain,'horizon':horizon,'fold':fold,'candidate':name,'test_origins':int(test.sum()),'mse':mse(p[test],y[test]),'mae':float(abs(p[test]-y[test]).mean()),'gain_vs_Last_pct':100*(1-mse(p[test],y[test])/last_mse),'training_scope':'train+cal for N and AR-Ridge-refit; frozen AR-Ridge train-only','role':'retrospective numerical-only rolling diagnostic'})
                manifest.append({'case':case,'source_sha256':sha(file),'bounds':bounds,'train_mean':float(mu),'train_std':float(sd),'alpha_N':m.alpha_by_group['N'],'alpha_AR':baseline['ridge_alpha'],'prediction_sha256':sha(out/'predictions.npz')})
                print('ROLLING',case,'origins',int(test.sum()),flush=True)
    csv(HERE/'results/robustness/rolling_numeric_metrics.csv',rows);dump(HERE/'results/robustness/rolling_numeric_manifest.json',{'cases':manifest,'case_count':len(manifest),'scope':'Numerical-only 4 domains x 2 horizons x 3 folds; deep/text are not claimed trained across these 24 cases.'})

def text_stress():
    rows=[]
    for task in protocol()['tasks']:
        for scenario in protocol()['scenarios']:
            folder=LOCAL/'comparison'/task/scenario;objects=joblib.load(folder/'models.joblib');meta,f=features(task,scenario);yy=truth(task);model=objects['models']['N+S+Q+SF'];h=task_spec(task)['horizon']
            original=model.predict(infer(f))
            for delay in (0,h,2*h):
                altered,indices=delayed_features(f,delay);p=model.predict(altered)
                rows.append({'task':task,'scenario':scenario,'stress':f'text_delay_{delay}_observations','mse':mse(p,yy),'change_from_aligned_pct':100*(mse(p,yy)/mse(original,yy)-1),'text_available_origins':int(altered.text_available.sum()),'future_text_indices_used':0,'scope':'Frozen weights, inference perturbation only; no null refits or p-values'})
            rng=np.random.default_rng(2026);indices=np.array([rng.integers(0,i) if i else 0 for i in range(len(meta))]);altered=FeatureBundle(f.numeric_history,f.semantic[indices],f.quality[indices],f.text_available[indices],f.origin_index);p=model.predict(altered)
            rows.append({'task':task,'scenario':scenario,'stress':'past_mismatched_text','mse':mse(p,yy),'change_from_aligned_pct':100*(mse(p,yy)/mse(original,yy)-1),'text_available_origins':int(altered.text_available.sum()),'future_text_indices_used':0,'scope':'Diagnostic only; samples prior text with replacement; not a valid permutation significance test'})
    csv(HERE/'results/robustness/text_stress_metrics.csv',rows)

def interval_sensitivity():
    rows=[]
    for task in protocol()['tasks']:
        h=task_spec(task)['horizon'];df=pd.read_csv(LOCAL/'comparison'/task/'proxy/static_test_predictions.csv',float_precision='round_trip');ad=pd.read_csv(LOCAL/'comparison'/task/'proxy/adaptive_test_predictions.csv',float_precision='round_trip');df=pd.concat([df,ad]);y=truth(task);fallback=old_fallback(task)
        loss={n:np.mean((g.sort_values(['origin_index','step']).y_pred.to_numpy().reshape(y.shape)-y)**2,axis=1) for n,g in df.groupby('candidate')}
        for candidate in ('StaticGateReplay','AdaptiveRisk','N+S+Q+SF'):
            delta=loss[fallback]-loss[candidate]
            for multiple in protocol()['diagnostics']['block_multipliers']:
                block=h*multiple;ci=block_ci(delta,block) if block<len(delta) else [None,None]
                rows.append({'task':task,'candidate':candidate,'comparison':fallback,'block':block,'origins':len(y),'approx_nonoverlap_blocks':len(y)//block,'mean_delta':float(delta.mean()),'ci95_low':ci[0],'ci95_high':ci[1],'support':'NOT_ESTIMABLE_BLOCK_GE_N' if block>=len(delta) else ('LOW_BLOCK_SUPPORT' if len(y)//block<5 else 'DIAGNOSTIC'),'interpretation':'Pointwise diagnostic only; overlapping origins and adaptive feedback remain dependent. No interval is reported when block >= sample count.'})
    csv(HERE/'results/robustness/block_sensitivity.csv',rows)

def old_fallback(task):return 'AR-Ridge' if task.startswith('Agriculture') else 'N'
if __name__=='__main__':
    with threadpool_limits(limits=1):rolling_numeric();text_stress();interval_sensitivity()
