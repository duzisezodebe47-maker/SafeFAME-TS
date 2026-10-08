"""第15轮：封存方案，执行选择稳定性和配对残差归因分析。"""
from common15 import *
from analysis15 import *
from evaluate_test import load_trusted
from train_models import ridge_fit,ridge_predict,residual_predict
from threadpoolctl import threadpool_limits

def run_horizon(h):
    cfg=config();out=LOCAL/f'h{h}';out.mkdir();objects,_=load_trusted(h)
    data=PREVIOUS/f'prepared/h{h}';sm=pd.read_csv(data/'selection/metadata.csv');tm=pd.read_csv(data/'test_features/metadata.csv')
    sx=np.load(data/'selection/x.npy');tx=np.load(data/'test_features/x.npy');sy=np.load(data/'selection/y.npy');ty=np.load(data/'test_truth.npy')
    sp=np.load(PREVIOUS/f'models/h{h}/selection_predictions.npz');tp=np.load(PREVIOUS/f'evaluation/h{h}/fixed_predictions.npz')
    dec=sm.segment.eq('decision').to_numpy();cal=sm.segment.eq('calibration').to_numpy();names=cfg['numeric_candidates']
    selected=objects['selected'];base=tp['SelectedNumeric'];decbase=sp['SelectedNumeric'][dec]
    losses=np.column_stack([np.mean((sp[n][dec]-sy[dec])**2,axis=1) for n in names]);preds={'OriginalCalibrationChoice':base};selection=[];rules={}
    for rule in cfg['fixed_numeric_rules']:
        idx,score=choose_numeric(losses,names,rule);name=names[idx];preds[rule]=tp[name];rules[rule]=name
        selection.extend({'horizon':h,'rule':rule,'candidate':n,'score':float(score[i]),'selected':i==idx} for i,n in enumerate(names))
    # Selection stability across contiguous decision blocks, all outside numeric weight fitting.
    block_rows=[]
    for i,ids in enumerate(np.array_split(np.arange(len(losses)),3)):
        best,_=choose_numeric(losses[ids],names,'decision_mean')
        block_rows.extend({'horizon':h,'block':i+1,'candidate':n,'mse':float(losses[ids,j].mean()),'winner':j==best,'first_target':sm[dec].iloc[ids[0]].target_date,'last_target_end':sm[dec].iloc[ids[-1]].target_end} for j,n in enumerate(names))
    origins=np.r_[sm.loc[dec,'origin_index'],tm.origin_index];allpred=np.concatenate([np.stack([sp[n][dec] for n in names],axis=1),np.stack([tp[n] for n in names],axis=1)])
    truth=np.concatenate([sy[dec],ty]);warm=int(dec.sum());traces={}
    for lookback in cfg['online_lookbacks']:
        p,t=online_select(origins,allpred,h,names,lookback,lambda i:truth[i]);key=f'OnlineMatured{lookback}';preds[key]=p[warm:];traces[key]=t[warm:];csv(out/f'{key}_trace.csv',t[warm:])
    for name in names:preds['Fixed_'+name]=tp[name]
    np.savez_compressed(out/'numeric_predictions.npz',origins=tm.origin_index.to_numpy(),**preds)
    csv(HERE/f'results/h{h}/selection_scores.csv',selection);csv(HERE/f'results/h{h}/decision_block_scores.csv',block_rows)

    # All attribution controls share the same base, OOF residual rows and availability mask.
    oof=np.load(PREVIOUS/f'models/h{h}/oof_residuals.npz');indices=sm.set_index('origin_index').index.get_indexer(oof['origins']);require((indices>=0).all(),'OOF identity missing')
    residual=oof['residual'];mean=residual.mean(axis=0);sa=sm.text_available.to_numpy(bool);ta=tm.text_available.to_numpy(bool)
    scorr=sp['DirectFusion']-sp['SelectedNumeric'];tcorr=tp['DirectFusion']-base
    features=state_features(sx,pd.to_datetime(sm.target_date));testfeat=state_features(tx,pd.to_datetime(tm.target_date))
    mu=features[indices].mean(axis=0);std=features[indices].std(axis=0);std[std<1e-12]=1;z=(features-mu)/std;zt=(testfeat-mu)/std
    if selected in ('Last','SeasonalNaive'):calbase=sp[selected][cal]
    else:
        require(selected=='PatchTST','This frozen experiment expects PatchTST for h12')
        calbase=np.mean([np.load(PREVIOUS/f'models/h{h}/{selected}_{seed}_calibration.npy') for seed in (2026,2027,2028)],axis=0)
    state_scores={};state_objects={}
    for alpha in cfg['state_alphas']:
        fitted=ridge_fit(z[indices],residual,alpha);v=ridge_predict(fitted,z[cal]);v[~sa[cal]]=0
        state_scores[alpha]=mse(calbase+v,sy[cal]);state_objects[alpha]=fitted
    alpha=min(state_scores,key=lambda a:(state_scores[a],a));fitted=state_objects[alpha]
    sv=ridge_predict(fitted,z)*sa[:,None];tv=ridge_predict(fitted,zt)*ta[:,None]
    corrections={'SelectedNumeric':(np.zeros_like(scorr),np.zeros_like(tcorr)), 'OriginalText':(scorr,tcorr), 'ConstantResidual':(sa[:,None]*mean,ta[:,None]*mean), 'NumericStateResidual':(sv,tv), 'CenteredText':(scorr-sa[:,None]*mean,tcorr-ta[:,None]*mean)}
    for weeks in cfg['age_weeks']:corrections[f'TextAge{weeks}w']=(scorr*age_mask(sm,weeks)[:,None],tcorr*age_mask(tm,weeks)[:,None])
    require(np.allclose(corrections['CenteredText'][1]+corrections['ConstantResidual'][1],tcorr),'decomposition identity')
    attribution={};gated={};gates={};gate_rows=[];coverage=[]
    for name,(s,t) in corrections.items():
        attribution[name]=base+t;weight,table=gate(decbase,s[dec],sy[dec],cfg['gate_weights']);gates[name]=weight;gated[name]=base+weight*t
        gate_rows.extend({'horizon':h,'candidate':name,**r} for r in table)
    for weeks in cfg['age_weeks']:coverage.append({'age_limit_weeks':weeks,'decision_origins':int(age_mask(sm[dec],weeks).sum()),'test_origins':int(age_mask(tm,weeks).sum())})
    np.savez_compressed(out/'attribution_predictions.npz',origins=tm.origin_index.to_numpy(),**attribution)
    np.savez_compressed(out/'gated_predictions.npz',origins=tm.origin_index.to_numpy(),**gated)
    dump(out/'state_model.json',{'mean':mu.tolist(),'std':std.tolist(),'ridge':{k:v.tolist() for k,v in fitted.items()},'alpha':alpha,'calibration_scores':state_scores,'oof_origins':oof['origins'].tolist(),'constant_residual':mean.tolist()})
    dump(out/'policy_choices.json',{'fixed_numeric':rules,'gates':gates,'original_numeric':selected,'role':cfg['role']})
    csv(HERE/f'results/h{h}/gate_selection.csv',gate_rows);csv(HERE/f'results/h{h}/text_age_coverage.csv',coverage)
    metrics=[];intervals=[]
    for group,predictions in [('numeric_selection',preds),('attribution',attribution),('decision_gated_controls',gated)]:
        for name,p in predictions.items():
            loss=np.mean((p-ty)**2,axis=1);base_loss=np.mean((base-ty)**2,axis=1)
            metrics.append({'horizon':h,'group':group,'candidate':name,'origins':len(ty),'mse':float(loss.mean()),'mae':float(abs(p-ty).mean()),'gain_vs_round14_selected_pct':100*(1-loss.mean()/base_loss.mean()),'worst_quarter_mse':float(max(v.mean() for v in np.array_split(loss,4)))})
            if group=='attribution' and name!='OriginalText':
                # Positive difference means original text is better than this control.
                delta=loss-np.mean((attribution['OriginalText']-ty)**2,axis=1)
                for mult in cfg['bootstrap']['block_multipliers']:
                    block=h*mult;lo,hi=block_interval(delta,block,cfg['bootstrap']['draws'],cfg['bootstrap']['seed'])
                    intervals.append({'horizon':h,'control':name,'block':block,'text_benefit_over_control':float(delta.mean()),'ci_low':float(lo),'ci_high':float(hi),'approx_blocks':len(delta)//block,'role':'exploratory pointwise; no multiplicity correction'})
    csv(HERE/f'results/h{h}/metrics.csv',metrics);csv(HERE/f'results/h{h}/text_control_intervals.csv',intervals)
    dump(HERE/f'results/h{h}/summary.json',{'horizon':h,'fixed_choices':rules,'gate_weights':gates,'state_alpha':alpha,'original_numeric':selected,'online':{name:{'choices':pd.Series([r['candidate'] for r in trace]).value_counts().to_dict(),'switches':sum(a['candidate']!=b['candidate'] for a,b in zip(trace,trace[1:]))} for name,trace in traces.items()},'role':cfg['role']})
    print(pd.DataFrame(metrics)[['horizon','group','candidate','mse']].to_string(index=False),flush=True)

def main():
    require(not LOCAL.exists(),'Round15 evidence already exists; refuse overwrite')
    LOCAL.mkdir(parents=True);dump(LOCAL/'input_registration.json',{'registered_utc':pd.Timestamp.now(tz='UTC').isoformat(),'role':config()['role'],'files':seal_inputs()})
    with threadpool_limits(limits=1):
        for h in config()['horizons']:run_horizon(h)
    dump(LOCAL/'MANIFEST.json',{'files':inventory(LOCAL)})

if __name__=='__main__':main()
