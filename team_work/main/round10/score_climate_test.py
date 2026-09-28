from __future__ import annotations
import argparse, csv, hashlib, json, math, random, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve()
sys.path.insert(0,str(HERE.parents[1]/"src"))
from team_eval.v2 import load_bundle, load_prediction_grid

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def q(a,x): return float(np.quantile(np.asarray(a,float),x))
def moving_block_ci(delta, draws=5000, block=4, seed=2026):
    d=np.asarray(delta,float); n=len(d); rng=np.random.default_rng(seed); vals=[]
    for _ in range(draws):
        out=[]
        while len(out)<n:
            s=int(rng.integers(0,n))
            out.extend(d[(s+np.arange(block))%n].tolist())
        vals.append(float(np.mean(out[:n])))
    return vals,[q(vals,.025),q(vals,.975)]

def main():
    ap=argparse.ArgumentParser();
    ap.add_argument('--bundle',type=Path,required=True); ap.add_argument('--signature',required=True)
    ap.add_argument('--split-spec',type=Path,required=True); ap.add_argument('--route',type=Path,required=True)
    ap.add_argument('--baseline-csv',type=Path,required=True); ap.add_argument('--selected-csv',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); fig=a.out/'figures'; fig.mkdir(exist_ok=True)
    route=json.loads(a.route.read_text(encoding='utf-8'))
    if route.get('selected')!='numeric_fallback' or route.get('fallback')!='N': raise RuntimeError('sealed route is not numeric_fallback -> N')
    if sha(a.route)!='b3bce4713ff44485c9aa0712c5774ffe449045fb8186b39ef592298e0051e36a': raise RuntimeError('route SHA mismatch')
    bundle=load_bundle(a.bundle,a.signature,a.split_spec,'Climate_h4_f2','proxy')
    grid=load_prediction_grid([a.baseline_csv,a.selected_csv],bundle,('test',))
    keys=grid['values']; candidates=['Last','SeasonalNaive','AR-Ridge','N']; seed=2026
    samples=[(i,s) for i,s in enumerate(bundle['samples']) if s['segment']=='test']
    if len(samples)!=124: raise RuntimeError(f'expected 124 origins, got {len(samples)}')
    truth=np.vstack([bundle['arrays']['targets_standardized'][i] for i,_ in samples]).astype(float)
    pred={c:np.asarray([[keys[(c,seed)][('test',s['origin_id'],int(s['origin_index']),step)] for step in range(1,5)] for _,s in samples],float) for c in candidates}
    if any(x.shape!=(124,4) or not np.isfinite(x).all() for x in pred.values()): raise RuntimeError('invalid prediction grid')
    fit=json.loads((a.bundle/'Climate_h4_f2'/'numeric_fit.json').read_text(encoding='utf-8')); scale=float(fit['std'])
    metrics={}; origin_losses={}
    for c,x in pred.items():
        e=x-truth; ol=np.mean(e*e,axis=1); origin_losses[c]=ol
        metrics[c]={'mse':float(np.mean(e*e)),'mae':float(np.mean(np.abs(e))),'rmse':float(np.sqrt(np.mean(e*e))),
                    'raw_mse':float(np.mean(e*e)*scale*scale),'raw_mae':float(np.mean(np.abs(e))*scale),'raw_rmse':float(np.sqrt(np.mean(e*e))*scale)}
    comparisons={}; bootstrap={}
    for b in candidates[:-1]:
        delta=origin_losses[b]-origin_losses['N']; draws,ci=moving_block_ci(delta)
        comparisons[b]={'baseline':b,'gain_pct':float(100*(1-metrics['N']['mse']/metrics[b]['mse'])),'mean_origin_delta':float(np.mean(delta)),
                        'block_bootstrap_ci95':ci,'origin_win_rate':float(np.mean(origin_losses['N']<origin_losses[b]))}
        bootstrap[b]=draws
    half=[]
    for name,sl in [('first_half',slice(0,62)),('second_half',slice(62,None))]:
        row={'half':name,'origins':len(origin_losses['N'][sl])}
        for c in candidates: row[c+'_mse']=float(np.mean(origin_losses[c][sl]))
        row['N_vs_Last_gain_pct']=float(100*(1-row['N_mse']/row['Last_mse']))
        half.append(row)
    horizons=[]
    for k in range(4):
        row={'step':k+1}
        for c in candidates: row[c+'_mse']=float(np.mean((pred[c][:,k]-truth[:,k])**2))
        horizons.append(row)
    bound=[]
    for j,(_,s) in enumerate(samples):
        for k in range(4):
            row={'origin_id':s['origin_id'],'origin_index':int(s['origin_index']),'step':k+1,'y_true_standardized':float(truth[j,k])}
            for c in candidates: row[c+'_pred_standardized']=float(pred[c][j,k])
            bound.append(row)
    out={'status':'PASS','task':'Climate_h4_f2','scenario':'proxy','route_selected':'numeric_fallback','route_fallback':'N','n_origins':124,'horizon':4,
         'metrics_standardized_and_raw':metrics,'comparisons_vs_N':comparisons,'half_metrics':half,'horizon_metrics':horizons,
         'route_sha256':sha(a.route),'selected_prediction_sha256':sha(a.selected_csv),'baseline_prediction_sha256':sha(a.baseline_csv),
         'bundle_signature':bundle['signature'],'split_spec_sha256':sha(a.split_spec),'bootstrap':{'method':'circular moving-block bootstrap over ordered origins','block':4,'draws':5000,'seed':2026},
         'interpretation_boundary':'Test results describe the already sealed N route. They are not used to reopen candidate selection.'}
    dump(a.out/'Climate_test_score.json',out)
    for name,rows in [('Climate_test_predictions_bound.csv',bound),('Climate_half_metrics.csv',half),('Climate_horizon_metrics.csv',horizons)]:
        with (a.out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (a.out/'Climate_origin_losses.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['origin_id','origin_index']+[c+'_mse' for c in candidates]; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for j,(_,s) in enumerate(samples): w.writerow({'origin_id':s['origin_id'],'origin_index':int(s['origin_index']),**{c+'_mse':float(origin_losses[c][j]) for c in candidates}})
    manifest={'status':'PASS','inputs':{'route':sha(a.route),'baseline_csv':sha(a.baseline_csv),'selected_csv':sha(a.selected_csv)},'files':{}}
    for p in sorted(a.out.rglob('*')):
        if p.is_file() and p.name!='scoring_manifest.json': manifest['files'][p.relative_to(a.out).as_posix()]=sha(p)
    dump(a.out/'scoring_manifest.json',manifest)
    print(json.dumps({'status':'PASS','metrics':metrics,'N_vs_Last':comparisons['Last']},ensure_ascii=False))
if __name__=='__main__': main()
