"""主控接管入口：物理隔离数据、选择区门控、冻结后测试，禁止覆盖。

准备、选择、测试分为不同进程。选择进程只有非测试真值和历史。
不满足必要条件的候选直接拒绝；未执行置换的 p 必须为空。
只有通过改善/双半段/可训练条件的候选才执行 999 次全重拟合置换。
这项短路只减少计算，不改变原门控的可通过集合。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'model_engine'), str(HERE.parent / 'src')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from branches import FeatureBundle
from candidates import BranchResidualCandidate
from numeric_fallbacks import numeric_baselines
from permutation import _fit, decision_loss, permute_rows
from team_eval.v2 import verify_bundle

SIGNATURE = 'a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38'
SPEC_SHA = 'a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031'
SELECTION = ('train','calibration','decision')
SCENARIOS = ('proxy','conservative_lag')
NUMERIC = ('Last','SeasonalNaive','AR-Ridge','N')
GATES = ('N+S+Q','N+S+Q+SF')
ALL = NUMERIC + ('N+Q',) + GATES
TEST_ALLOWED = {'metadata.csv','origin_index.npy','numeric_missing.npy','input_time.npy','target_time.npy'} | {
    f'{s}/{k}.npy' for s in SCENARIOS for k in ('semantic','quality','text_available','source_available')}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_json(p, value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def write_csv(p, rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    rows=list(rows)
    if not rows: return
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader(); w.writerows(rows)

def inventory(root):
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file() and p.name!='MANIFEST.json'}

def task_spec(spec, task):
    return next(t for t in spec['tasks'] if f"{t['domain']}_h{t['horizon']}_f{t['fold_id']}"==task)

def prepare(bundle, spec_path, task, package, results):
    verify_bundle(bundle,SIGNATURE,spec_path)
    if sha(spec_path)!=SPEC_SHA: raise ValueError('frozen spec hash mismatch')
    if package.exists(): raise ValueError('refuse package overwrite')
    spec=json.loads(spec_path.read_text(encoding='utf-8')); t=task_spec(spec,task)
    td=bundle/task; samples=pd.read_csv(td/'samples.csv'); sel=samples.segment.isin(SELECTION).to_numpy(); test=samples.segment.eq('test').to_numpy()
    package.mkdir(parents=True); results.mkdir(parents=True,exist_ok=True)
    bounds=[0]+t['bounds']; count={}
    for i,seg in enumerate(SELECTION+('test',)):
        mask=samples.segment.eq(seg).to_numpy(); expected=np.arange(max(t['input_len'],bounds[i]),bounds[i+1]-t['horizon']+1)
        actual=samples.loc[mask,'origin_index'].to_numpy()
        if not np.array_equal(expected,actual): raise ValueError(f'{seg} boundary/order mismatch')
        count[seg]=len(actual)
    commitments={}; coverage=[]
    for side,mask in [('selection_fit',sel),('test_features',test)]:
        out=package/side; out.mkdir(); samples.loc[mask].to_csv(out/'metadata.csv',index=False,lineterminator='\n')
        names=['origin_index','numeric_missing','input_time','target_time']
        if side=='selection_fit': names+=['numeric_history','numeric_history_raw','frequency','targets','targets_raw','targets_standardized']
        for name in names:
            src=td/f'{name}.npy'
            if not src.exists(): continue
            arr=np.load(src,mmap_mode='r',allow_pickle=False); sliced=np.asarray(arr[mask]); np.save(out/f'{name}.npy',sliced,allow_pickle=False)
            commitments[f'{side}/{name}.npy']={'source_sha256':sha(src),'slice_sha256':sha(out/f'{name}.npy'),'shape':list(sliced.shape),'dtype':str(sliced.dtype)}
        for scen in SCENARIOS:
            (out/scen).mkdir()
            for name in ('semantic','quality','text_available','source_available'):
                src=td/scen/f'{name}.npy'; arr=np.load(src,mmap_mode='r',allow_pickle=False); np.save(out/scen/f'{name}.npy',np.asarray(arr[mask]),allow_pickle=False)
                commitments[f'{side}/{scen}/{name}.npy']={'source_sha256':sha(src),'slice_sha256':sha(out/scen/f'{name}.npy'),'shape':list(arr[mask].shape),'dtype':str(arr.dtype)}
    for scen in SCENARIOS:
        avail=np.load(td/scen/'text_available.npy'); q=np.load(td/scen/'quality.npy'); sem=np.load(td/scen/'semantic.npy')
        for seg in SELECTION+('test',):
            mask=samples.segment.eq(seg).to_numpy()
            coverage.append({'task':task,'scenario':scen,'segment':seg,'origins':int(mask.sum()),'text_available':int(avail[mask].sum()),'coverage':float(avail[mask].mean()),'quality_varying_columns':int((q[mask].std(axis=0)>1e-8).sum()),'no_text_nonzero_semantic_rows':int(np.any(sem[mask & ~avail.astype(bool)]!=0,axis=1).sum())})
    a=np.load(td/'proxy'/'text_available.npy').astype(bool); b=np.load(td/'conservative_lag'/'text_available.npy').astype(bool)
    flips=[]
    for i in np.flatnonzero(a!=b):
        flips.append({'origin_id':samples.iloc[i].origin_id,'origin_index':int(samples.iloc[i].origin_index),'segment':samples.iloc[i].segment,'proxy':bool(a[i]),'conservative_lag':bool(b[i])})
    write_csv(results/'coverage.csv',coverage); write_json(results/'scenario_flips.json',flips)
    write_json(package/'task_spec.json',t); write_json(package/'slice_commitments.json',commitments)
    write_json(package/'MANIFEST.json',{'task':task,'bundle_signature':SIGNATURE,'split_spec_sha256':SPEC_SHA,'segment_counts':count,'files':inventory(package),'test_forbidden':['targets*','numeric_history*','frequency']})
    write_json(results/'data_acceptance.json',verify_package(package,spec_path,bundle))
    print(json.dumps({'prepared':task,'counts':count,'coverage':coverage},ensure_ascii=False),flush=True)

def verify_package(package,spec_path,bundle=None):
    m=json.loads((package/'MANIFEST.json').read_text(encoding='utf-8'))
    if m['bundle_signature']!=SIGNATURE or m['split_spec_sha256']!=SPEC_SHA or sha(spec_path)!=SPEC_SHA: raise ValueError('anchor mismatch')
    actual=inventory(package)
    if actual!=m['files']: raise ValueError('file inventory/hash mismatch')
    testfiles={p.relative_to(package/'test_features').as_posix() for p in (package/'test_features').rglob('*') if p.is_file()}
    if not testfiles<=TEST_ALLOWED: raise ValueError('forbidden test file')
    commitments=json.loads((package/'slice_commitments.json').read_text(encoding='utf-8'))
    for rel,c in commitments.items():
        a=np.load(package/rel,allow_pickle=False)
        if sha(package/rel)!=c['slice_sha256'] or list(a.shape)!=c['shape'] or str(a.dtype)!=c['dtype']: raise ValueError('slice invalid')
        if bundle:
            side,name=rel.split('/',1); td=bundle/m['task']; source=td/name; samples=pd.read_csv(td/'samples.csv'); mask=samples.segment.isin(SELECTION).to_numpy() if side=='selection_fit' else samples.segment.eq('test').to_numpy()
            if sha(source)!=c['source_sha256'] or not np.array_equal(a,np.load(source,mmap_mode='r')[mask]): raise ValueError('source slice differs')
    meta=pd.read_csv(package/'selection_fit'/'metadata.csv'); tmeta=pd.read_csv(package/'test_features'/'metadata.csv')
    if not meta.segment.isin(SELECTION).all() or not tmeta.segment.eq('test').all(): raise ValueError('metadata segments invalid')
    if meta.origin_id.duplicated().any() or tmeta.origin_id.duplicated().any() or set(meta.origin_id)&set(tmeta.origin_id): raise ValueError('duplicate origin')
    spec=json.loads(spec_path.read_text(encoding='utf-8')); t=task_spec(spec,m['task']); bounds=[0]+t['bounds']
    for i,seg in enumerate(SELECTION+('test',)):
        df=meta if seg!='test' else tmeta; got=df.loc[df.segment.eq(seg),'origin_index'].to_numpy(); expected=np.arange(max(t['input_len'],bounds[i]),bounds[i+1]-t['horizon']+1)
        if not np.array_equal(got,expected): raise ValueError('mapping boundary invalid')
    return {'status':'PASS','task':m['task'],'package_manifest_sha256':sha(package/'MANIFEST.json'),'files_checked':len(actual),'source_bitwise_checked':bundle is not None,'counts':m['segment_counts'],'test_truth_or_history_files':0}

def load_selection(package,scenario):
    root=package/'selection_fit'; meta=pd.read_csv(root/'metadata.csv')
    return meta,FeatureBundle(numeric_history=np.load(root/'numeric_history.npy'),semantic=np.load(root/scenario/'semantic.npy'),quality=np.load(root/scenario/'quality.npy'),text_available=np.load(root/scenario/'text_available.npy'),origin_index=np.load(root/'origin_index.npy'),targets=np.load(root/'targets_raw.npy'),targets_standardized=np.load(root/'targets_standardized.npy'))

def infer_view(f):
    return FeatureBundle(f.numeric_history,f.semantic,f.quality,f.text_available,f.origin_index)

def mse(pred,truth): return float(np.mean((np.asarray(pred)-np.asarray(truth))**2))

def model_metadata(model):
    return {'requested':model.name,'effective':'+'.join(b.name for b in model.branches),'semantic_degraded':model.s_degraded,'quality_degraded':model.q_degraded,'alphas':model.alpha_by_group,'parameters':model.n_parameters,'weights_sha256':hashlib.sha256(np.ascontiguousarray(model.weights).tobytes()).hexdigest()}

_WORKER=None
def worker_init(package,scenario):
    global _WORKER
    threadpool_limits(limits=1)
    meta,f=load_selection(Path(package),scenario)
    _WORKER=tuple(f.slice(meta.segment.eq(s).to_numpy()) for s in SELECTION)

def null_iteration(args):
    name,k=args; train,cal,dec=_WORKER; rng=np.random.default_rng(2026*1000+k)
    model=_fit(name,permute_rows(train,rng),permute_rows(cal,rng))
    loss=decision_loss(model,permute_rows(dec,rng))
    return {'iteration':k,'seed':2026*1000+k,'status':'ok','loss':loss}

def select(package,spec_path,results,scenario,workers):
    verify_package(package,spec_path)
    out=results/scenario; out.mkdir(parents=True,exist_ok=True)
    if (out/'sealed_route.json').exists(): raise ValueError('route already sealed')
    meta,f=load_selection(package,scenario); masks={s:meta.segment.eq(s).to_numpy() for s in SELECTION}
    train,cal,dec=[f.slice(masks[s]) for s in SELECTION]; t=json.loads((package/'task_spec.json').read_text(encoding='utf-8')); period=json.loads(spec_path.read_text(encoding='utf-8'))['seasonal_periods'][t['domain']]
    fy=f.targets_standardized.copy(); fy[masks['decision']]=np.nan
    baselines=numeric_baselines(infer_view(f),meta.segment.to_numpy(),fy,seasonal_period=period)
    pred=dict(baselines['predictions']); models={}; stats={}; calibration_pred={c:p[masks['calibration']] for c,p in pred.items()}
    for name in ('N','N+Q')+GATES:
        model=BranchResidualCandidate(name); model.fit_design(train); model.select_alphas(train,cal); model.refit(train); calibration_pred[name]=model.predict(infer_view(cal)); model=_fit(name,train,cal); pred[name]=model.predict(infer_view(f)); models[name]=model; stats[name]=model_metadata(model)
        print(f'{t["domain"]} {scenario} {name} -> {stats[name]["effective"]}',flush=True)
    cal_mse={c:mse(calibration_pred[c],cal.targets_standardized) for c in NUMERIC}
    fallback=min(NUMERIC,key=lambda c:(cal_mse[c],c)); dmask=masks['decision']; dy=dec.targets_standardized; fp=pred[fallback][dmask]; middle=(t['bounds'][1]+t['bounds'][2])//2
    halves=[dec.origin_index+t['horizon']<=middle,dec.origin_index>=middle]
    diagnostics={}; rows=[]
    for c in ALL:
        rows.append({'candidate':c,'effective':stats[c]['effective'] if c in stats else c,'calibration_mse':mse(calibration_pred[c],cal.targets_standardized),'decision_mse':mse(pred[c][dmask],dy),'first_half_mse':mse(pred[c][dmask][halves[0]],dy[halves[0]]),'second_half_mse':mse(pred[c][dmask][halves[1]],dy[halves[1]])})
    write_csv(out/'candidate_metrics.csv',rows); write_json(out/'models.json',stats)
    for c in GATES:
        cp=pred[c][dmask]; score=mse(cp,dy); wins=all(m.any() and mse(cp[m],dy[m])<mse(fp[m],dy[m]) for m in halves)
        reason=[]
        if stats[c]['semantic_degraded']: reason.append('semantic_untrainable')
        if not score<mse(fp,dy): reason.append('no_decision_improvement')
        if not wins: reason.append('not_both_halves_better')
        p=None; runs=[]
        if not reason:
            started=time.perf_counter()
            with ProcessPoolExecutor(max_workers=workers,initializer=worker_init,initargs=(str(package),scenario)) as pool:
                for row in pool.map(null_iteration,[(c,k) for k in range(999)],chunksize=1):
                    runs.append(row)
                    if len(runs)%50==0: print(f'{t["domain"]} {scenario} {c} null {len(runs)}/999 {time.perf_counter()-started:.1f}s',flush=True)
            write_csv(out/f'{c.replace("+","_")}_null.csv',runs)
            p=float((1+sum(r['loss']<=score for r in runs))/1000)
            if p>0.025: reason.append('permutation_not_passed')
        diagnostics[c]={'decision_mse':score,'row_p':p,'null_refits':len(runs),'segment_wins':wins,'eligible':not reason,'rejected_by':reason,'effective':stats[c]['effective']}
    eligible=[c for c in GATES if diagnostics[c]['eligible']]; selected=min(eligible,key=lambda c:(diagnostics[c]['decision_mse'],c)) if eligible else fallback
    selrows=[]
    for c in ALL:
        for i,row in meta.iterrows():
            if row.segment=='train': continue
            for j,v in enumerate(calibration_pred[c][int(masks['calibration'][:i].sum())] if row.segment=='calibration' else pred[c][i]): selrows.append({'candidate':c,'effective':stats[c]['effective'] if c in stats else c,'origin_id':row.origin_id,'origin_index':int(row.origin_index),'segment':row.segment,'step':j+1,'y_pred':float(v)})
    write_csv(out/'selection_predictions.csv',selrows)
    # Write route before any test features/history or truth can enter this process.
    route={'status':'frozen','task':f"{t['domain']}_h{t['horizon']}_f{t['fold_id']}",'scenario':scenario,'selected':selected,'effective':stats[selected]['effective'] if selected in stats else selected,'fallback':fallback,'calibration_mse':cal_mse,'decision_fallback_mse':mse(fp,dy),'candidates':diagnostics,'package_manifest_sha256':sha(package/'MANIFEST.json'),'split_spec_sha256':SPEC_SHA,'code_sha256':{p.relative_to(HERE).as_posix():sha(p) for p in sorted(HERE.rglob('*.py'))},'evidence_sha256':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},'calibration_policy':'Train-only fit predicts calibration before train+cal refit. Alpha tuning uses calibration; this is a selection score, not an unbiased generalization score.','protocol_note':'Fail necessary conditions => reject without null; p=null when not run; semantics unavailable => numeric-only, not multimodal. Test never used for selection.'}
    write_json(out/'sealed_route.json',route)
    (out/'sealed_route.json.sha256').write_text(sha(out/'sealed_route.json')+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(route,ensure_ascii=False),flush=True)

def block_ci(delta,block,draws=5000):
    delta=np.asarray(delta); n=len(delta); rng=np.random.default_rng(2026); starts=rng.integers(0,n,size=(draws,int(np.ceil(n/block)))); indices=(starts[:,:,None]+np.arange(block))%n; vals=delta[indices.reshape(draws,-1)[:,:n]].mean(axis=1)
    return np.quantile(vals,[.025,.975]).tolist()

def test(package,bundle,spec_path,results,scenario):
    verify_package(package,spec_path,bundle)
    out=results/scenario; route=json.loads((out/'sealed_route.json').read_text(encoding='utf-8')); route_sha=sha(out/'sealed_route.json')
    if route_sha!=(out/'sealed_route.json.sha256').read_text().strip(): raise ValueError('route changed')
    for rel,h in route['code_sha256'].items():
        if sha(HERE/rel)!=h: raise ValueError('code changed after selection')
    for rel,h in route['evidence_sha256'].items():
        if sha(out/rel)!=h: raise ValueError('selection evidence changed')
    if (out/'test_predictions.csv').exists(): raise ValueError('refuse second test prediction')
    meta,f=load_selection(package,scenario); masks={s:meta.segment.eq(s).to_numpy() for s in SELECTION}; train,cal,dec=[f.slice(masks[s]) for s in SELECTION]
    t=json.loads((package/'task_spec.json').read_text(encoding='utf-8')); task=route['task']; td=bundle/task; allmeta=pd.read_csv(td/'samples.csv'); tm=allmeta.segment.eq('test').to_numpy(); testmeta=allmeta[tm]
    # Rolling-origin evaluation: only history up to each origin, with no future targets passed to model.
    x=np.asarray(np.load(td/'numeric_history.npy',mmap_mode='r')[tm]); tr=package/'test_features'
    tf=FeatureBundle(x,np.load(tr/scenario/'semantic.npy'),np.load(tr/scenario/'quality.npy'),np.load(tr/scenario/'text_available.npy'),np.load(tr/'origin_index.npy'))
    period=json.loads(spec_path.read_text(encoding='utf-8'))['seasonal_periods'][t['domain']]
    xall=np.r_[f.numeric_history,x]; yfit=np.r_[f.targets_standardized,np.full((len(x),t['horizon']),np.nan)]; segments=np.r_[meta.segment.to_numpy(),np.repeat('test',len(x))]; yfit[segments=='decision']=np.nan
    dummy=FeatureBundle(xall,np.zeros((len(xall),1)),np.zeros((len(xall),1)),np.zeros(len(xall),bool),np.arange(len(xall)))
    baseline=numeric_baselines(dummy,segments,yfit,seasonal_period=period)['predictions']; predictions={c:p[-len(x):] for c,p in baseline.items()}
    names=set(('N',route['selected']))-set(predictions)
    for name in sorted(names):
        model=_fit(name,train,cal)
        if model_metadata(model)!=json.loads((out/'models.json').read_text(encoding='utf-8'))[name]: raise ValueError('weights/alpha replay mismatch')
        predictions[name]=model.predict(tf)
    rows=[]
    for c,p in predictions.items():
        if p.shape!=(len(x),t['horizon']) or not np.isfinite(p).all(): raise ValueError('test prediction invalid')
        for i,(_,r) in enumerate(testmeta.iterrows()):
            for j,v in enumerate(p[i]): rows.append({'candidate':c,'origin_id':r.origin_id,'origin_index':int(r.origin_index),'step':j+1,'y_pred':float(v)})
    write_csv(out/'test_predictions.csv',rows)
    # Only after prediction bytes are fixed, open test truth for scoring.
    truth=np.asarray(np.load(td/'targets_standardized.npy',mmap_mode='r')[tm]); fit=json.loads((td/'numeric_fit.json').read_text(encoding='utf-8')); scale=fit['std']; metrics={}; loss={}
    for c,p in predictions.items():
        e=p-truth; loss[c]=(e*e).mean(axis=1); metrics[c]={'mse':float((e*e).mean()),'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt((e*e).mean())),'raw_rmse':float(np.sqrt((e*e).mean())*scale)}
    chosen=route['selected']; comparisons={}
    for c in predictions:
        comparisons[c]={'gain_pct':100*(1-metrics[chosen]['mse']/metrics[c]['mse']),'ci95_origin_delta':block_ci(loss[c]-loss[chosen],t['horizon']),'origin_win_rate':float((loss[chosen]<loss[c]).mean())}
    score={'status':'PASS','task':task,'scenario':scenario,'selected':chosen,'effective':route['effective'],'n_origins':len(x),'horizon':t['horizon'],'route_sha256':route_sha,'prediction_sha256':sha(out/'test_predictions.csv'),'metrics':metrics,'comparisons':comparisons,'seed':2026,'bootstrap_draws':5000,'block':t['horizon'],'interpretation':'Rolling-origin test with observed history up to origin; selected route frozen before test target access. No reselection.'}
    write_json(out/'test_score.json',score)
    write_csv(out/'test_origin_losses.csv',[{'origin_index':int(r.origin_index),**{c:float(loss[c][i]) for c in loss}} for i,(_,r) in enumerate(testmeta.iterrows())])
    write_csv(out/'test_horizon_metrics.csv',[{'step':j+1,**{c:float(((p[:,j]-truth[:,j])**2).mean()) for c,p in predictions.items()}} for j in range(t['horizon'])])
    print(json.dumps(score,ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',required=True,choices=['prepare','select','test','verify']); ap.add_argument('--bundle',type=Path,required=True); ap.add_argument('--split-spec',type=Path,required=True); ap.add_argument('--task',required=True); ap.add_argument('--package',type=Path,required=True); ap.add_argument('--results',type=Path,required=True); ap.add_argument('--scenario',choices=SCENARIOS,default='proxy'); ap.add_argument('--workers',type=int,default=4); a=ap.parse_args()
    with threadpool_limits(limits=1):
        if a.phase=='prepare': prepare(a.bundle,a.split_spec,a.task,a.package,a.results)
        elif a.phase=='select': select(a.package,a.split_spec,a.results,a.scenario,a.workers)
        elif a.phase=='test': test(a.package,a.bundle,a.split_spec,a.results,a.scenario)
        else: print(json.dumps(verify_package(a.package,a.split_spec,a.bundle),ensure_ascii=False))

if __name__=='__main__': main()
