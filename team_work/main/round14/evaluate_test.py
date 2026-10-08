"""先保存固定模型预测，再按到期时间释放标签，最后一次性报告测试指标。"""
from __future__ import annotations
import joblib
from r14_common import *
from train_models import numeric_predict,residual_predict,code_anchors
from adaptive import causal_replay
from threadpoolctl import threadpool_limits

def load_trusted(h):
    folder=LOCAL/'models'/f'h{h}';seal=read(folder/'selection_seal.json');require(seal['code']==code_anchors(),'model code changed');require(seal['protocol_sha256']==sha(HERE/'protocol.json'),'protocol changed')
    for rel,value in seal['files'].items():require(sha(folder/rel)==value,'model evidence changed')
    require(seal['prepared_manifest_sha256']==sha(LOCAL/'prepared/MANIFEST.json'),'prepared manifest changed')
    return joblib.load(folder/'models.joblib'),seal

def block_ci(delta,block):
    n=len(delta)
    if block>=n:return None,None
    rng=np.random.default_rng(2026);starts=rng.integers(0,n,(cfg()['diagnostics']['bootstrap_draws'],int(np.ceil(n/block))));idx=(starts[:,:,None]+np.arange(block))%n;means=np.asarray(delta)[idx.reshape(len(starts),-1)[:,:n]].mean(axis=1);return [float(v) for v in np.quantile(means,[.025,.975])]

def evaluate(h):
    out=LOCAL/'evaluation'/f'h{h}';require(not out.exists(),'test already evaluated; refuse overwrite');out.mkdir(parents=True)
    objects,seal=load_trusted(h);folder=LOCAL/'prepared'/f'h{h}/test_features';meta=pd.read_csv(folder/'metadata.csv');x=np.load(folder/'x.npy');texts=read(folder/'texts.json');available=meta.text_available.to_numpy(bool);n=len(meta);preds={name:numeric_predict(name,obj,x,h) for name,obj in objects['numeric'].items()};base=preds[objects['selected']];correction=residual_predict(objects['text'],texts,available)
    preds.update(SelectedNumeric=base,DirectFusion=base+correction,StaticFallback=base+objects['weight']*correction,StrictSourceFallback=base+objects['weight']*correction*meta.strict_eligible.to_numpy(bool)[:,None],ArchiveConservative=base+objects['weight']*correction*meta.conservative_eligible.to_numpy(bool)[:,None])
    # All input perturbations are fixed before any test-label score is computed.
    selection_folder=LOCAL/'prepared'/f'h{h}/selection';sm=pd.read_csv(selection_folder/'metadata.csv');st=read(selection_folder/'texts.json');all_orig=np.r_[sm.origin_index,meta.origin_index];all_texts=st+texts;all_avail=np.r_[sm.text_available.to_numpy(bool),available];offset=len(sm);stress={}
    for delay in cfg()['diagnostics']['text_delays_weeks']:
        indices=np.searchsorted(all_orig,meta.origin_index.to_numpy()-delay,side='right')-1;ok=indices>=0;indices=np.maximum(indices,0);altered=[all_texts[i] for i in indices];mask=all_avail[indices]&ok;stress[f'Delay_{delay}w']=base+residual_predict(objects['text'],altered,mask)
    rng=np.random.default_rng(2026);mask=available&(rng.random(n)>=cfg()['diagnostics']['missing_fraction']);stress['Missing_50pct']=base+residual_predict(objects['text'],texts,mask)
    indices=np.array([rng.integers(0,offset+i) for i in range(n)]);stress['PastMismatch']=base+residual_predict(objects['text'],[all_texts[i] for i in indices],all_avail[indices])
    np.savez_compressed(out/'fixed_predictions.npz',origins=meta.origin_index.to_numpy(),**preds,**stress)
    dump(out/'fixed_commitment.json',{'prediction_sha256':sha(out/'fixed_predictions.npz'),'selection_seal_sha256':sha(LOCAL/'models'/f'h{h}/selection_seal.json'),'scorer_sha256':sha(Path(__file__)),'test_labels_read':False})
    warm=np.load(LOCAL/'models'/f'h{h}/causal_warmup.npz');origins=np.r_[warm['origins'],meta.origin_index.to_numpy()];bases=np.r_[warm['base'],base];directs=np.r_[warm['direct'],preds['DirectFusion']];av=np.r_[warm['available'],available];test_y=None;reads=[]
    def release(i):
        nonlocal test_y
        if i<len(warm['origins']):return warm['truth'][i]
        if test_y is None:test_y=np.load(LOCAL/'prepared'/f'h{h}/test_truth.npy')
        reads.append(int(origins[i]));return test_y[i-len(warm['origins'])]
    adapt,trace=causal_replay(origins,bases,directs,av,h,cfg()['adaptive'],release);trace=trace[len(warm['origins']):];preds['AdaptiveFallback']=adapt[len(warm['origins']):]
    np.save(out/'adaptive_predictions.npy',preds['AdaptiveFallback']);csv(out/'adaptive_trace.csv',trace);dump(out/'adaptive_commitment.json',{'prediction_sha256':sha(out/'adaptive_predictions.npy'),'trace_sha256':sha(out/'adaptive_trace.csv'),'mature_feedback_count':len(set(reads)),'max_feedback_origin':max(reads) if reads else None,'policy':'Complete H-step target released only when past origin + H <= current origin'})
    y=np.load(LOCAL/'prepared'/f'h{h}/test_truth.npy');rows=[];base_loss=np.mean((base-y)**2,axis=1);preds.update(stress)
    for name,pred in preds.items():
        loss=np.mean((pred-y)**2,axis=1);rows.append({'horizon':h,'candidate':name,'origins':n,'mse':float(loss.mean()),'mae':float(abs(pred-y).mean()),'raw_rmse_dsci_points':float(np.sqrt(loss.mean())*objects['scaler']['std']),'gain_vs_selected_numeric_pct':float(100*(1-loss.mean()/base_loss.mean())),'worst_quarter_mse':float(max(t.mean() for t in np.array_split(loss,4))),'origin_degradation_fraction':float(np.mean(loss>base_loss+1e-12))})
    csv(HERE/f'results/h{h}/test_metrics.csv',rows)
    intervals=[]
    for name in ('DirectFusion','StaticFallback','AdaptiveFallback','ArchiveConservative'):
        delta=base_loss-np.mean((preds[name]-y)**2,axis=1)
        for multiple in cfg()['diagnostics']['block_multipliers']:
            block=h*multiple;low,high=block_ci(delta,block);intervals.append({'candidate':name,'block':block,'mean_benefit':float(delta.mean()),'ci_low':low,'ci_high':high,'support':'LOW_BLOCK_SUPPORT' if n//block<5 else 'POINTWISE_DIAGNOSTIC','approx_blocks':n//block})
    csv(HERE/f'results/h{h}/block_diagnostics.csv',intervals)
    dump(HERE/f'results/h{h}/test_summary.json',{'horizon':h,'selected_numeric':objects['selected'],'static_weight':objects['weight'],'test_origins':n,'text_available':int(available.sum()),'strict_eligible':int(meta.strict_eligible.sum()),'adaptive_text_enabled_origins':sum(t['weight']>0 for t in trace),'adaptive_mean_weight':float(np.mean([t['weight'] for t in trace])),'fixed_commitment_sha256':sha(out/'fixed_commitment.json'),'adaptive_commitment_sha256':sha(out/'adaptive_commitment.json'),'strict_mode':'Fallback because exact original release time unverified','role':'Temporal historical holdout with new target definition; not prospective or different-domain validation'})
    dump(HERE/f'results/h{h}/demo.json',{'dates':meta.target_date.tolist(),'actual':(y[:,0]*objects['scaler']['std']+objects['scaler']['mean']).tolist(),'curves':{name:(pred[:,0]*objects['scaler']['std']+objects['scaler']['mean']).tolist() for name,pred in preds.items() if name in ('SelectedNumeric','DirectFusion','StaticFallback','AdaptiveFallback','StrictSourceFallback')},'sources':meta[['target_date','asof_utc','source_map_date','source_url','publication_grade','text_available','strict_eligible','conservative_eligible']].fillna('').to_dict('records'),'weights':[t['weight'] for t in trace],'note':'Plot first forecast step only; metrics cover all H steps.'})
    print(pd.DataFrame(rows)[['horizon','candidate','mse','gain_vs_selected_numeric_pct']].to_string(index=False),flush=True)

if __name__=='__main__':
    import torch
    torch.set_num_threads(2)
    with threadpool_limits(limits=1):
        for h in cfg()['horizons']:evaluate(h)
