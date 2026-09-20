"""Configuration-driven historical rolling evaluation with auditable resume."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import run_safefame_v2 as base
from run_baselines import ALPHAS, make_windows, metrics as raw_metrics, seasonal_naive, fit_ridge, predict_ridge
from run_famets_selective import semantic_rows, frequency_statistics
from train_dlinear import load_domain, make_windows as torch_windows, TrainConfig, DLinearTarget
from train_patchtst import PatchTSTTarget
from run_reviewer_sensitivity import NumericControlBuilder, shifted_order, block_power_audit
from validation_boundaries import decision_segments

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data_processed/v4'
VARIANTS=base.VARIANTS

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def metrics(actual,predicted):
    assert actual.ndim==2 and actual.shape==predicted.shape,('forecast shape',actual.shape,predicted.shape)
    return raw_metrics(actual,predicted)
def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)

def fit_path(variant,train,cal,control=False):
    # sklearn squeezes a one-column target prediction to (n,). Keep (n, H)
    # explicitly: otherwise adding the Last anchor broadcasts to (n, n).
    builder=NumericControlBuilder(variant).fit(train[0]) if control else base.ResidualBuilder(variant).fit(*train[:3])
    xt=builder.transform(train[0]) if control else builder.transform(*train[:3])
    xc=builder.transform(cal[0]) if control else builder.transform(*cal[:3])
    choices=[]
    for alpha in base.RIDGE_ALPHAS:
        model=Ridge(alpha=alpha,solver='cholesky').fit(xt,train[3]-train[0][:,-1,None])
        prediction=cal[0][:,-1,None]+model.predict(xc).reshape(len(cal[0]),-1)
        choices.append((metrics(cal[3],prediction)['mse'],alpha,model,prediction))
    _,alpha,model,prediction=min(choices,key=lambda v:v[0])
    return builder,model,alpha,prediction

def refit_path(variant,alpha,fit,test,control=False):
    builder=NumericControlBuilder(variant).fit(fit[0]) if control else base.ResidualBuilder(variant).fit(*fit[:3])
    xf=builder.transform(fit[0]) if control else builder.transform(*fit[:3])
    xt=builder.transform(test[0]) if control else builder.transform(*test[:3])
    model=Ridge(alpha=alpha,solver='cholesky').fit(xf,fit[3]-fit[0][:,-1,None])
    return test[0][:,-1,None]+model.predict(xt).reshape(len(test[0]),-1)

def reference_null(builder,train,cal,dec,count,mode,alpha,block,seed):
    """Direct sklearn fits, independently checking optimized sufficient statistics."""
    out=[];rng=np.random.default_rng(seed)
    for k in range(count):
        if mode=='row':
            local=np.random.default_rng(1001+k)
            orders=[local.permutation(len(d[0])) for d in (train,cal,dec)]
        else:orders=[shifted_order(len(train[0]),rng,block),np.arange(len(cal[0])),shifted_order(len(dec[0]),rng,block)]
        xt,xc,xd=[builder.transform(d[0],d[1][o],d[2][o]) for d,o in zip((train,cal,dec),orders)]
        choices=[]
        for candidate_alpha in base.RIDGE_ALPHAS if mode=='row' else [alpha]:
            model=Ridge(alpha=candidate_alpha,solver='cholesky').fit(xt,train[3]-train[0][:,-1,None])
            cp=cal[0][:,-1,None]+model.predict(xc).reshape(len(cal[0]),-1)
            choices.append((metrics(cal[3],cp)['mse'],model))
        _,model=min(choices,key=lambda v:v[0])
        dp=dec[0][:,-1,None]+model.predict(xd).reshape(len(dec[0]),-1)
        out.append(metrics(dec[3],dp)['mse'])
    return np.asarray(out)

def interval(actual,control,candidate,block,seed=2026,repeats=5000):
    """Same moving-block statistic as v3, using cumulative sums for efficiency."""
    assert actual.shape==control.shape==candidate.shape
    losses=np.mean((control-actual)**2-(candidate-actual)**2,axis=1)
    n=len(losses);length=min(max(2,block),n);count=int(np.ceil(n/length))
    prefix=np.r_[0.,np.cumsum(losses)]
    rng=np.random.default_rng(seed)
    starts=rng.choice(np.arange(n-length+1),size=(repeats,count))
    sums=prefix[starts+length]-prefix[starts]
    final_length=n-(count-1)*length
    sums[:,-1]=prefix[starts[:,-1]+final_length]-prefix[starts[:,-1]]
    low,high=np.quantile(sums.sum(axis=1)/n,[.025,.975])
    return float(losses.mean()),float(low),float(high)

class CachedDesign:
    """Reuse fitted PCA/scalers; row permutations still recompute interactions."""
    def __init__(self,builder,data):
        self.builder=builder
        x,e,q,_=data
        self.numeric=builder.numeric_scaler.transform(x)
        self.reduced=builder.pca.transform(e)
        self.quality=builder.quality_scaler.transform(q)
        self.frequency=(builder.frequency_scaler.transform(frequency_statistics(x))
                        if builder.variant=='frequency_residual' else None)
    def transform(self,order):
        reduced=self.reduced[order]
        chunks=[self.numeric,reduced,self.quality[order]]
        if self.frequency is not None:
            interaction=(reduced[:,:,None]*self.frequency[:,None,:8]).reshape(len(order),-1)
            chunks.extend([self.frequency,self.builder.interaction_scaler.transform(interaction)])
        return np.c_[tuple(chunks)]

def ridge_grid(x,y,xcal,actual_cal,last_cal,xdec,last_dec,alphas):
    """Shared sufficient statistics; each alpha solves the same Ridge objective."""
    mean=x.mean(axis=0);ym=y.mean(axis=0)
    centered=x-mean
    gram=centered.T@centered;rhs=centered.T@(y-ym)
    cal=xcal-mean;dec=xdec-mean
    best=None
    for alpha in alphas:
        regularized=gram.copy();regularized.flat[::len(gram)+1]+=alpha
        weights=cho_solve(cho_factor(regularized,lower=True,check_finite=False),rhs,check_finite=False)
        pred=last_cal+ym+cal@weights
        score=metrics(actual_cal,pred)['mse']
        if best is None or score<best[0]:best=(score,alpha,weights)
    return best[1],last_dec+ym+dec@best[2]

def null_scores(builder,train,cal,dec,count,mode,alpha,block,seed):
    designs=[CachedDesign(builder,d) for d in (train,cal,dec)]
    residual=train[3]-train[0][:,-1,None]
    out=[]
    rng=np.random.default_rng(seed)
    for k in range(count):
        if mode=='row':
            local=np.random.default_rng(1001+k)
            orders=[local.permutation(len(d[0])) for d in (train,cal,dec)]
        else:
            orders=[shifted_order(len(train[0]),rng,block),np.arange(len(cal[0])),
                    shifted_order(len(dec[0]),rng,block)]
        xt,xc,xd=[d.transform(o) for d,o in zip(designs,orders)]
        _,pred=ridge_grid(xt,residual,xc,cal[3],cal[0][:,-1,None],xd,dec[0][:,-1,None],
                         base.RIDGE_ALPHAS if mode=='row' else [alpha])
        out.append(metrics(dec[3],pred)['mse'])
    return np.asarray(out)

def arrays_for(domain,settings,horizon,bounds):
    path=DATA/'numerical'/domain/f'{domain}.csv'
    frame=pd.read_csv(path)
    raw=frame.OT.to_numpy(float)
    train_end,cal_end,dec_end,test_end=bounds
    mean=np.nanmean(raw[:train_end]);std=np.nanstd(raw[:train_end])
    assert np.isfinite(std) and std>1e-8
    # Unknown target truth remains NaN; only input history is imputed.
    filled=pd.Series(raw).ffill().fillna(np.nanmedian(raw[:train_end])).to_numpy(float)
    inputs=(filled-mean)/std;truth=(raw-mean)/std
    multi,features,target_index,preprocessing=load_domain(path,train_end,'multivariate')
    starts=pd.to_datetime(frame.start_date);ends=pd.to_datetime(frame.end_date)
    cache=np.load(DATA/'semantic_features'/f'{domain}.npz')
    arrays={};origins={};tensors={};excluded={}
    for name,lo,hi in [('train',settings['input_len'],train_end),('cal',train_end,cal_end),
                       ('dec',cal_end,dec_end),('test',dec_end,test_end),('fit',settings['input_len'],dec_end)]:
        proposed=np.arange(lo,hi-horizon+1)
        valid=np.array([np.isfinite(truth[t:t+horizon]).all() for t in proposed])
        origin=proposed[valid];assert len(origin)>0,(domain,horizon,bounds,name)
        assert (ends.iloc[origin-1].to_numpy()<starts.iloc[origin].to_numpy()).all()
        x,_,o=make_windows(inputs,settings['input_len'],horizon,origin)
        y=np.stack([truth[t:t+horizon] for t in o])
        e,q=semantic_rows(cache,o)
        arrays[name]=(x,e,q,y);origins[name]=o;excluded[name]=int((~valid).sum())
        xm,_,_=torch_windows(multi,target_index,settings['input_len'],horizon,origin)
        tensors[name]=(xm,torch.from_numpy(x[:,:,None].astype(np.float32)),torch.from_numpy(y.astype(np.float32)))
    return arrays,origins,tensors,dict(target_mean=float(mean),target_std=float(std),features=features,
                                     preprocessing=preprocessing,excluded_unknown_targets=excluded)

def run_task(domain,horizon,fold,config,directory,device):
    started=time.perf_counter();settings=config['domains'][domain]
    n=len(pd.read_csv(DATA/'numerical'/domain/f'{domain}.csv'))
    bounds=[int(n*p) for p in config['folds'][fold]]
    a,o,t,prep=arrays_for(domain,settings,horizon,bounds)
    train,cal,dec,fit,test=[a[k] for k in ('train','cal','dec','fit','test')]
    masks=decision_segments(o['dec'],horizon,bounds[1],bounds[2])[:2]
    block=max(horizon,min(settings['seasonal_period'],24))
    evidence={**{f'{k}_origins':v for k,v in o.items()},**{f'{k}_actual':v[3] for k,v in a.items()},
              'segment1_mask':masks[0],'segment2_mask':masks[1]}
    numeric={k:{'Last':np.repeat(a[k][0][:,-1:],horizon,axis=1),
                'SeasonalNaive':seasonal_naive(a[k][0],horizon,settings['seasonal_period'])}
             for k in ('cal','dec')}
    ridge_choices=[]
    for alpha in ALPHAS:
        model=fit_ridge(train[0],train[3],alpha)
        ridge_choices.append((metrics(cal[3],predict_ridge(cal[0],model))['mse'],alpha,model))
    _,ridge_alpha,ridge=min(ridge_choices,key=lambda x:x[0])
    for k in numeric:numeric[k]['AR-Ridge']=predict_ridge(a[k][0],ridge)
    specs={'DLinear-M':(DLinearTarget,dict(input_len=settings['input_len'],channels=len(prep['features']),
                    horizon=horizon,kernel_size=7 if settings['seasonal_period']==52 else 5),0),
           'PatchTST':(PatchTSTTarget,dict(input_len=settings['input_len'],channels=1,horizon=horizon,
                    patch_len=8 if settings['input_len']>=52 else 4,stride=4 if settings['input_len']>=52 else 2,
                    d_model=32,n_heads=4,e_layers=2,d_ff=64,dropout=.1),1)}
    training=TrainConfig(**config['training']);deep_epochs={}
    for name,(cls,args,channel) in specs.items():
        forecasts=[];epochs=[]
        for seed in config['seeds']:
            cp,dp,epoch,_,_=base.deep_model(cls,args,t['train'][channel],t['train'][2],t['cal'][channel],
                t['cal'][2],t['dec'][channel],training,seed,device,directory/'models'/f'{name}_cal_s{seed}')
            forecasts.append((cp,dp));epochs.append(epoch)
            evidence[f'{name}_s{seed}_cal']=cp;evidence[f'{name}_s{seed}_dec']=dp
        numeric['cal'][name]=np.mean([p[0] for p in forecasts],axis=0)
        numeric['dec'][name]=np.mean([p[1] for p in forecasts],axis=0)
        deep_epochs[name]=epochs
    fallback,cal_scores=base.select_numeric(numeric['cal'],cal[3])
    for k in numeric:
        for model,pred in numeric[k].items():evidence[f'{model}_{k}']=pred
    audit={};models={}
    for vi,variant in enumerate(VARIANTS):
        builder,model,alpha,cp=fit_path(variant,train,cal)
        dp=dec[0][:,-1,None]+model.predict(builder.transform(dec[0],dec[1],dec[2])).reshape(len(dec[0]),-1)
        shuffled=null_scores(builder,train,cal,dec,config['row_permutations'],'row',alpha,block,4200+vi)
        score=metrics(dec[3],dp)['mse'];p=(1+int((shuffled<=score).sum()))/(len(shuffled)+1)
        wins=all(mask.any() and metrics(dec[3][mask],dp[mask])['mse']<
                 metrics(dec[3][mask],numeric['dec'][fallback][mask])['mse'] for mask in masks)
        eligible=score<metrics(dec[3],numeric['dec'][fallback])['mse'] and wins and p<=config['p_threshold']
        audit[variant]=dict(alpha=float(alpha),decision_mse=score,p_value=p,segment_wins=bool(wins),eligible=bool(eligible))
        evidence[f'{variant}_cal']=cp;evidence[f'{variant}_dec']=dp;evidence[f'{variant}_row_null']=shuffled
        models[variant]=(builder,alpha)
    eligible=[v for v in VARIANTS if audit[v]['eligible']]
    selected=min(eligible,key=lambda v:audit[v]['decision_mse']) if eligible else 'numeric_fallback'
    frozen=dict(frozen_utc=utc(),fallback=fallback,selected=selected,numeric_calibration_scores=cal_scores,
                candidates=audit,deep_epochs=deep_epochs,ridge_alpha=float(ridge_alpha),bounds=bounds,
                role=config['role'])
    write_json(directory/'frozen_selection.json',frozen)
    # All route/hyperparameter decisions have now been persisted.
    if fallback in specs:
        cls,args,channel=specs[fallback]
        base.SEEDS=tuple(config['seeds'])
        fallback_test=base.deep_refit(cls,args,t['fit'][channel],t['fit'][2],t['test'][channel],
            deep_epochs[fallback],training,device,directory/'models'/f'{fallback}_refit')
    elif fallback=='Last':fallback_test=np.repeat(test[0][:,-1:],horizon,axis=1)
    elif fallback=='SeasonalNaive':fallback_test=seasonal_naive(test[0],horizon,settings['seasonal_period'])
    else:fallback_test=predict_ridge(test[0],fit_ridge(fit[0],fit[3],ridge_alpha))
    evidence['fallback_test']=fallback_test
    result_rows=[];candidate_predictions={}
    for vi,variant in enumerate(VARIANTS):
        builder,alpha=models[variant]
        pred=refit_path(variant,alpha,fit,test)
        candidate_predictions[variant]=pred;evidence[f'{variant}_test']=pred
        cb,cm,ca,_=fit_path(variant,train,cal,control=True)
        control_dec=dec[0][:,-1,None]+cm.predict(cb.transform(dec[0])).reshape(len(dec[0]),-1)
        control=refit_path(variant,ca,fit,test,control=True)
        evidence[f'{variant}_control_dec']=control_dec;evidence[f'{variant}_control_test']=control
        shifted=null_scores(builder,train,cal,dec,config['circular_shifts'],'circular',alpha,block,4100+100*fold+10*horizon+vi)
        evidence[f'{variant}_circular_null']=shifted
        pshift=(1+int((shifted<=audit[variant]['decision_mse']).sum()))/(len(shifted)+1)
        seed=9000+100*fold+10*horizon+vi
        delta,lo,hi=interval(test[3],fallback_test,pred,block,seed,config['bootstrap_repeats'])
        matched,mlo,mhi=interval(test[3],control,pred,block,seed,config['bootstrap_repeats'])
        count,mde,mdep=block_power_audit(dec[3],control_dec,evidence[f'{variant}_dec'],block)
        result_rows.append(dict(domain=domain,horizon=horizon,fold=fold+1,variant=variant,**metrics(test[3],pred),
            fallback=fallback,selected_path=selected,improvement_vs_fallback_pct=100*(1-metrics(test[3],pred)['mse']/metrics(test[3],fallback_test)['mse']),
            delta=delta,ci_low=lo,ci_high=hi,matched_delta=matched,matched_ci_low=mlo,matched_ci_high=mhi,
            matched_control_alpha=float(ca),circular_p_value=pshift,**audit[variant],test_windows=len(test[3]),
            calibration_windows=len(cal[3]),decision_windows=len(dec[3]),block_length=block,
            power_nonoverlapping_blocks=count,approx_mde=float(mde) if np.isfinite(mde) else None,
            approx_mde_pct=float(mdep) if np.isfinite(mdep) else None,bootstrap_seed=seed))
    selected_test=fallback_test if selected=='numeric_fallback' else candidate_predictions[selected]
    evidence['selected_test']=selected_test
    np.savez_compressed(directory/'predictions.npz',**evidence)
    pd.DataFrame(result_rows).to_csv(directory/'metrics.csv',index=False)
    write_json(directory/'audit.json',dict(domain=domain,horizon=horizon,fold=fold+1,bounds=bounds,
        rows=n,settings=settings,preprocessing=prep,block_length=block,
        selected_metrics=metrics(test[3],selected_test),fallback_metrics=metrics(test[3],fallback_test),
        selected_delta=interval(test[3],fallback_test,selected_test,block,2026,config['bootstrap_repeats']),
        windows={k:len(v) for k,v in o.items()},excluded=prep['excluded_unknown_targets'],
        segment_windows=[int(m.sum()) for m in masks],seconds=time.perf_counter()-started,completed_utc=utc()))
    return frozen

def fingerprint(config):
    paths=[ROOT/'configs/safefame_v4.json',*DATA.rglob('*.csv'),*DATA.rglob('*.npz'),*DATA.rglob('*.npy'),
           *[ROOT/'src'/name for name in ('run_safefame_v4.py','prepare_v4.py','run_safefame_v2.py','run_reviewer_sensitivity.py',
             'train_dlinear.py','train_patchtst.py','data_utils.py','run_baselines.py','run_famets_selective.py','validation_boundaries.py')]]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(paths)}
    signature=hashlib.sha256(json.dumps(dict(config=config,hashes=hashes),sort_keys=True).encode()).hexdigest()
    return signature,hashes

def self_check():
    rng=np.random.default_rng(55)
    x=rng.normal(size=(70,12));e=rng.normal(size=(70,32));q=rng.normal(size=(70,10));y=rng.normal(size=(70,3))
    train=(x[:40],e[:40],q[:40],y[:40]);cal=(x[40:55],e[40:55],q[40:55],y[40:55]);dec=(x[55:],e[55:],q[55:],y[55:])
    for variant in VARIANTS:
        b,_,alpha,_=base.fit_residual(variant,train,cal,solver='cholesky')
        for data in (train,cal,dec):
            order=rng.permutation(len(data[0]));fast=CachedDesign(b,data).transform(order)
            slow=b.transform(data[0],data[1][order],data[2][order])
            np.testing.assert_allclose(fast,slow,rtol=2e-6,atol=2e-6)
        slow=base.permutation_scores(variant,b,alpha,train,cal,dec,5,solver='cholesky')
        fast=null_scores(b,train,cal,dec,5,'row',alpha,4,2026)
        np.testing.assert_allclose(fast,slow,rtol=1e-6,atol=1e-7)
    truth=rng.normal(size=(43,3));control=truth+rng.normal(size=(43,3));candidate=truth+rng.normal(size=(43,3))
    np.testing.assert_allclose(interval(truth,control,candidate,7,repeats=200),
        base.moving_block_interval(truth,control,candidate,7,repeats=200),rtol=1e-10,atol=1e-10)
    for variant in VARIANTS:
        single=[(d[0],d[1],d[2],d[3][:,:1]) for d in (train,cal,dec)]
        b,_,alpha,p=fit_path(variant,single[0],single[1])
        assert p.shape==(len(single[1][0]),1)
        assert refit_path(variant,alpha,single[0],single[2]).shape==(len(single[2][0]),1)
        for mode in ('row','circular'):
            np.testing.assert_allclose(null_scores(b,*single,5,mode,alpha,4,2026),
                reference_null(b,*single,5,mode,alpha,4,2026),rtol=1e-6,atol=1e-7)
    print('V4_SELF_CHECK_PASS: optimized nulls and block intervals match reference')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/safefame_v4.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/safefame_v4')
    parser.add_argument('--domains',nargs='+');parser.add_argument('--horizons',nargs='+',type=int)
    parser.add_argument('--folds',nargs='+',type=int);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if args.self_check:
        with threadpool_limits(1):self_check()
        return
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if args.preflight:
        torch.set_num_threads(config['threads'])
        count=0
        for d,s in config['domains'].items():
            frame=pd.read_csv(DATA/'numerical'/d/f'{d}.csv')
            assert not np.isinf(frame.select_dtypes(include='number').to_numpy()).any(),(d,'infinite input')
            for h in s['horizons']:
                for fold in config['folds']:
                    arrays,origins,tensors,_=arrays_for(d,s,h,[int(len(frame)*p) for p in fold])
                    for name,values in arrays.items():
                        assert all(np.isfinite(v).all() for v in values),(d,h,name,'nonfinite array')
                        assert all(torch.isfinite(v).all() for v in tensors[name]),(d,h,name,'nonfinite tensor')
                    count+=1
        print('V4_PREFLIGHT_PASS',count,'task-folds; finite inputs/targets and valid boundaries')
        return
    if args.smoke:
        assert args.output.resolve()!=(ROOT/'outputs/safefame_v4').resolve()
        config={**config,'seeds':[2026],'row_permutations':3,'circular_shifts':3,'bootstrap_repeats':100,
                'training':{**config['training'],'max_epochs':2,'patience':1},'role':'engineering smoke; excluded from results'}
    args.output.mkdir(parents=True,exist_ok=True)
    signature,hashes=fingerprint(config)
    manifest_path=args.output/'run_manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text(encoding='utf-8'))
        assert args.resume and old['signature']==signature,'Existing run signature mismatch; use a new output directory'
    else:write_json(manifest_path,dict(signature=signature,config=config,hashes=hashes,started_utc=utc(),
            python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None))
    torch.set_num_threads(config['threads'])
    tasks=[(d,h,f) for d,s in config['domains'].items() for h in s['horizons'] for f in range(len(config['folds']))
           if (not args.domains or d in args.domains) and (not args.horizons or h in args.horizons)
           and (not args.folds or f+1 in args.folds)]
    assert tasks,'No requested task is configured'
    with threadpool_limits(config['threads']):
        for index,(domain,horizon,fold) in enumerate(tasks,1):
            directory=args.output/'tasks'/f'{domain}_h{horizon}_f{fold+1}'
            directory.mkdir(parents=True,exist_ok=True)
            done=directory/'completed.json'
            if done.exists():
                record=json.loads(done.read_text(encoding='utf-8'))
                assert args.resume and record['signature']==signature
                assert all(digest(directory/p)==v for p,v in record['hashes'].items()),'Completed task evidence changed'
                print('RESUME',index,len(tasks),directory.name,flush=True);continue
            print('START',index,len(tasks),directory.name,utc(),flush=True)
            write_json(directory/'status.json',dict(state='running',started_utc=utc(),signature=signature))
            try:
                frozen=run_task(domain,horizon,fold,config,directory,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                write_json(directory/'status.json',dict(state='completed',completed_utc=utc(),signature=signature))
                write_json(done,dict(signature=signature,completed_utc=utc(),hashes={p.relative_to(directory).as_posix():digest(p)
                    for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='completed.json'}))
                print('DONE',index,len(tasks),directory.name,frozen['fallback'],frozen['selected'],flush=True)
            except Exception as error:
                write_json(directory/'status.json',dict(state='failed',error=repr(error),failed_utc=utc(),signature=signature))
                raise
    print('REQUESTED_TASKS_COMPLETE',len(tasks),flush=True)

if __name__=='__main__':main()
