"""四领域统一对照与自适应融合；旧路线不变，新结果明确为历史探索。"""
from __future__ import annotations
import argparse
import joblib
import time
from common import *
from common import _fit
from adaptive import causal_replay
from threadpoolctl import threadpool_limits

def old_selected(task):
    if task.startswith('Agriculture'):return 'N+S+Q+SF','AR-Ridge'
    return 'N','N'
def refit_ar(x,y,tr,cal,alpha):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    fit=tr|cal;xm=x[fit].mean(axis=0);ym=y[fit].mean(axis=0);xc=x[fit]-xm;w=np.linalg.solve(xc.T@xc+alpha*np.eye(x.shape[1]),xc.T@(y[fit]-ym))
    return {'x_mean':xm,'y_mean':ym,'weights':w}
def ar_predict(state,x):return (np.asarray(x)-state['x_mean'])@state['weights']+state['y_mean']
def code_anchors():
    paths=[HERE/'common.py',HERE/'adaptive.py',HERE/'experiments.py',HERE/'protocol.json']+sorted((ROOT/'team_work/main/round12/model_engine').glob('*.py'))
    return {str(p.relative_to(PROJECT)):sha(p) for p in paths}
def selection_phase(task,scenario):
    out=LOCAL/'comparison'/task/scenario;require(not out.exists(),'selection output exists');out.mkdir(parents=True)
    meta,f=selection(task,scenario)
    # Legacy refit adapter concatenates raw targets as well; provide genuine
    # non-test raw targets, although the regression uses standardized targets.
    f.targets=np.load(LOCAL/'isolated'/task/'selection_fit/targets_raw.npy')
    tr=meta.segment.eq('train').to_numpy();cal=meta.segment.eq('calibration').to_numpy();dec=meta.segment.eq('decision').to_numpy();t=task_spec(task);y=f.targets_standardized
    fit_y=y.copy();fit_y[dec]=np.nan;period=read(SPEC)['seasonal_periods'][t['domain']]
    bases=numeric_baselines(infer(f),meta.segment.to_numpy(),fit_y,seasonal_period=period);preds=dict(bases['predictions']);cal_pred={n:p[cal] for n,p in preds.items()};models={};stats={}
    for name in ('N','N+Q','N+S+Q','N+S+Q+SF'):
        m=BranchResidualCandidate(name);m.fit_design(f.slice(tr));m.select_alphas(f.slice(tr),f.slice(cal));m.refit(f.slice(tr));cal_pred[name]=m.predict(infer(f.slice(cal)))
        m=_fit(name,f.slice(tr),f.slice(cal));models[name]=m;preds[name]=m.predict(infer(f));stats[name]={'effective':'+'.join(z.name for z in m.branches),'alphas':m.alpha_by_group,'weights_sha256':weights_hash(m),'semantic_untrainable':m.s_degraded,'quality_degraded':m.q_degraded}
    # Additional fair refit comparison; frozen AR-Ridge remains unchanged.
    ar=refit_ar(f.numeric_history,y,tr,cal,bases['ridge_alpha']);preds['AR-Ridge-refit']=ar_predict(ar,f.numeric_history);cal_pred['AR-Ridge-refit']=cal_pred['AR-Ridge']
    selected,fallback=old_selected(task);scores={n:mse(cal_pred[n],y[cal]) for n in ('Last','SeasonalNaive','AR-Ridge','N')};require(min(scores,key=lambda n:(scores[n],n))==fallback,'original fallback would change; stop for review')
    preds['StaticGateReplay']=preds[selected];decmeta=meta[dec].reset_index(drop=True);predict_grid(out/'selection_predictions.csv',decmeta,{n:p[dec] for n,p in preds.items()})
    np.savez_compressed(out/'selection_state.npz',origins=f.origin_index[dec],base=preds[fallback][dec],text=preds['N+S+Q+SF'][dec],truth=y[dec],available=f.text_available[dec] & (not stats['N+S+Q+SF']['semantic_untrainable']))
    x=np.asarray(f.numeric_history[tr],dtype=float);yy=np.asarray(y[tr],dtype=float);xm=x.mean(axis=0);ym=yy.mean(axis=0);xc=x-xm;old_ar={'x_mean':xm,'y_mean':ym,'weights':np.linalg.solve(xc.T@xc+bases['ridge_alpha']*np.eye(x.shape[1]),xc.T@(yy-ym))}
    require(np.allclose(ar_predict(old_ar,f.numeric_history),preds['AR-Ridge'],rtol=1e-12,atol=1e-12),'serialized AR differs from selection baseline')
    joblib.dump({'models':models,'AR-Ridge':old_ar,'AR-Ridge-refit':ar,'period':period,'fallback':fallback,'static_selected':selected,'numeric_fit':read(BUNDLE/task/'numeric_fit.json')},out/'models.joblib')
    csv(out/'calibration_metrics.csv',[{'candidate':n,'train_only_calibration_mse':mse(p,y[cal])} for n,p in cal_pred.items()])
    dump(out/'selection_seal.json',{'status':'FROZEN_EXPLORATORY_PROTOCOL','task':task,'scenario':scenario,'fallback':fallback,'static_selected':selected,'model_metadata':stats,'protocol_sha256':sha(HERE/'protocol.json'),'code_sha256':code_anchors(),
       'package_manifest_sha256':sha(LOCAL/'isolated'/task/'MANIFEST.json'),'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'test_labels_accessed':False,'role':'retrospective extension; old route not replaced'})
    print('SELECTED',task,scenario,flush=True)

def test_phase(task,scenario):
    out=LOCAL/'comparison'/task/scenario;seal=read(out/'selection_seal.json');require(seal['code_sha256']==code_anchors(),'selection code changed')
    for rel,h in seal['files'].items():require(sha(out/rel)==h,'selection file changed')
    require(not (out/'static_test_predictions.csv').exists(),'refuse test overwrite')
    objects=joblib.load(out/'models.joblib');meta,f=features(task,scenario);t=task_spec(task);preds={n:m.predict(infer(f)) for n,m in objects['models'].items()}
    preds.update({'Last':np.repeat(f.numeric_history[:,-1:],t['horizon'],axis=1),'SeasonalNaive':np.stack([f.numeric_history[:,-objects['period']+(h%objects['period'])] for h in range(t['horizon'])],axis=1),
                  'AR-Ridge':ar_predict(objects['AR-Ridge'],f.numeric_history),'AR-Ridge-refit':ar_predict(objects['AR-Ridge-refit'],f.numeric_history)})
    preds['StaticGateReplay']=preds[objects['static_selected']]
    predict_grid(out/'static_test_predictions.csv',meta,preds)
    dump(out/'static_prediction_commitment.json',{'prediction_sha256':sha(out/'static_test_predictions.csv'),'test_truth_read_before_static_prediction':False,'selection_seal_sha256':sha(out/'selection_seal.json')})
    # Environment releases only mature labels. Mixer receives no current/future label.
    stored=np.load(out/'selection_state.npz');di=stored['origins'];ti=f.origin_index;origins=np.r_[di,ti];base=np.r_[stored['base'],preds[objects['fallback']]];text=np.r_[stored['text'],preds['N+S+Q+SF']]
    available=np.r_[stored['available'],f.text_available & (not seal['model_metadata']['N+S+Q+SF']['semantic_untrainable'])]
    test_y=None;reads=[]
    def release(index):
        nonlocal test_y
        if index<len(di):return stored['truth'][index]
        if test_y is None:test_y=truth(task)
        reads.append(int(origins[index]));return test_y[index-len(di)]
    started=time.perf_counter();adaptive,traces=causal_replay(origins,base,text,available,t['horizon'],protocol()['adaptive'],release)
    # Strict original-source audit has no verified eligible corpus rows. Disable all text.
    source=read(HERE/'results/source/source_acceptance.json');require(source['strict_source_eligible_rows']==0,'strict source population changed; regenerate eligibility at origin level')
    strict=base.copy()
    strict_traces=[{'origin_index':int(o),'weight':0.,'reason':'publication_unverified','mature_origins':0,'last_feedback_origin':None} for o in origins]
    preds['AdaptiveRisk']=adaptive[len(di):];preds['StrictSourceFallback']=strict[len(di):]
    traces=traces[len(di):];csv(out/'adaptive_trace.csv',traces);csv(out/'strict_trace.csv',strict_traces[len(di):]);predict_grid(out/'adaptive_test_predictions.csv',meta,{n:preds[n] for n in ('AdaptiveRisk','StrictSourceFallback')})
    dump(out/'causal_feedback_audit.json',{'status':'PASS','horizon':t['horizon'],'test_truth_read_before_static_prediction':False,'test_labels_released_only_after_maturity':True,'last_test_feedback_origin':max(reads) if reads else None,
       'test_feedback_origins':len(set(reads)),'minimum_origin_gap':t['horizon'],'current_target_used_to_choose_weight':False,'policy_fitting':'fixed protocol; loss history updates causally, not new model fitting','adaptive_seconds':time.perf_counter()-started,'guard':'heuristic; not a statistical safety guarantee'})
    # Scores are read only after fixed-model bytes and causal adaptive trace exist.
    yy=truth(task);summary=[]
    for name,p in preds.items():
        e=p-yy;ols=np.mean(e*e,axis=1)
        quarters=np.array_split(ols,4)
        summary.append({'task':task,'scenario':scenario,'candidate':name,'origins':len(meta),'horizon':t['horizon'],'mse':float(ols.mean()),'mae':float(np.mean(abs(e))),'rmse':float(np.sqrt(ols.mean())),
                        'raw_rmse':float(np.sqrt(ols.mean())*objects['numeric_fit']['std']),'worst_quarter_mse':float(max(z.mean() for z in quarters)),
                        'numeric_fallback':objects['fallback'],'gain_vs_fallback_pct':float(100*(1-ols.mean()/mse(preds[objects['fallback']],yy)))})
    csv(HERE/'results'/task/scenario/'comparison_metrics.csv',summary)
    dump(HERE/'results'/task/scenario/'adaptive_summary.json',{'task':task,'scenario':scenario,'text_weight_positive_origins':sum(r['weight']>0 for r in traces),'fallback_origins':sum(r['weight']==0 for r in traces),'origins':len(traces),
        'mean_weight':float(np.mean([r['weight'] for r in traces])),'final_weight':traces[-1]['weight'],'strict_source_weights_all_zero':True,'prediction_sha256':sha(out/'adaptive_test_predictions.csv'),'old_routes_changed':False,'role':'retrospective exploratory extension; no independent blind confirmation'})
    print('SCORED',task,scenario,'AdaptiveRisk',round(mse(preds['AdaptiveRisk'],yy),6),'positive_weights',sum(r['weight']>0 for r in traces),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['select','test'],required=True);a=ap.parse_args()
    with threadpool_limits(limits=1):
        for task in protocol()['tasks']:
            for scenario in protocol()['scenarios']:
                (selection_phase if a.phase=='select' else test_phase)(task,scenario)

if __name__=='__main__':main()
