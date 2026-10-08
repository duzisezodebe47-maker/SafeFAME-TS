"""全部选择封存后先保存六任务固定预测，再做延迟反馈回放与评分。"""
from common16 import *
from model16 import *
from threadpoolctl import threadpool_limits

def evaluate():
    p=cfg();all_seals=read(LOCAL/'all_selection_seals.json');require(all_seals['code']==code_anchors(),'source mismatch')
    for task,digest in all_seals['tasks'].items():require(sha(LOCAL/f'models/{task}/selection_seal.json')==digest,'selection changed')
    out=LOCAL/'evaluation';require(not out.exists(),'evaluation exists; refuse overwrite');out.mkdir()
    for task in all_seals['tasks']:
        folder=out/task;folder.mkdir();data=np.load(LOCAL/f'prepared/{task}/test_features.npz');preds,seeds,seal=load_predictions(task,data['x']);preds['CalChoice']=preds[seal['calibration_choice']]
        for rule,name in seal['fixed_choices'].items():preds[rule]=preds[name]
        np.savez_compressed(folder/'fixed_predictions.npz',origins=data['origins'],**preds);np.save(folder/'dlinear_seed_predictions.npy',np.asarray(seeds))
    dump(out/'fixed_commitment.json',{'prediction_files':inventory(out),'selection_seals_sha256':sha(LOCAL/'all_selection_seals.json'),'scorer_sha256':sha(Path(__file__)),'test_truth_loaded':False})
    metrics=[];seedrows=[];online_summary=[]
    for task in all_seals['tasks']:
        code,hs=task.split('_h');h=int(hs);folder=out/task;fixed=np.load(folder/'fixed_predictions.npz');preds={n:fixed[n] for n in fixed.files if n!='origins'};sm=pd.read_csv(LOCAL/f'prepared/{task}/selection_metadata.csv');dec=sm.segment.eq('decision').to_numpy();selection=np.load(LOCAL/f'prepared/{task}/selection.npz');sp=np.load(LOCAL/f'models/{task}/selection_predictions.npz');names=p['candidates'];warm=int(dec.sum());origins=np.r_[selection['origins'][dec],fixed['origins']];both=np.stack([np.r_[sp[n][dec],preds[n]] for n in names],axis=1)
        truth=None
        def release(i):
            nonlocal truth
            if i<warm:return selection['y'][dec][i]
            if truth is None:truth=np.load(LOCAL/f'prepared/{task}/test_truth.npy')
            return truth[i-warm]
        for lookback in p['online_lookbacks']:
            z,trace=online_select(origins,both,h,names,lookback,release);key=f'Online{lookback}';preds[key]=z[warm:];np.save(folder/f'{key}.npy',preds[key]);csv(folder/f'{key}_trace.csv',trace[warm:])
            current=trace[warm:];online_summary.append({'task':task,'candidate':key,'switches':sum(a['candidate']!=b['candidate'] for a,b in zip(current,current[1:])),'choices':pd.Series([r['candidate'] for r in current]).value_counts().to_dict()})
        y=np.load(LOCAL/f'prepared/{task}/test_truth.npy');scale=read(LOCAL/f'prepared/{code}/scaler.json');base_loss=mse(preds['Last'],y)
        rows=[]
        for name,pred in preds.items():
            loss=np.mean((pred-y)**2,axis=1);rows.append({'state':code,'horizon':h,'candidate':name,'origins':len(y),'mse':float(loss.mean()),'mae':float(abs(pred-y).mean()),'raw_rmse_dsci':float(np.sqrt(loss.mean())*scale['std']),'gain_vs_last_pct':100*(1-loss.mean()/base_loss),'worst_quarter_mse':float(max(v.mean() for v in np.array_split(loss,4)))})
        seedp=np.load(folder/'dlinear_seed_predictions.npy')
        for seed,pred in zip(p['seeds'],seedp):seedrows.append({'state':code,'horizon':h,'seed':seed,'mse':mse(pred,y)})
        metrics.extend(rows);csv(HERE/f'results/{task}/metrics.csv',rows);print(task,pd.DataFrame(rows)[['candidate','mse','gain_vs_last_pct']].to_string(index=False),flush=True)
    csv(HERE/'results/all_metrics.csv',metrics);csv(HERE/'results/dlinear_seed_metrics.csv',seedrows);dump(HERE/'results/online_choices.json',online_summary)
    frame=pd.DataFrame(metrics);aggregate=[]
    for name,g in frame.groupby('candidate',sort=False):aggregate.append({'candidate':name,'tasks':len(g),'wins_vs_last':int((g.gain_vs_last_pct>1e-9).sum()),'ties_vs_last':int((g.gain_vs_last_pct.abs()<=1e-9).sum()),'losses_vs_last':int((g.gain_vs_last_pct< -1e-9).sum()),'equal_task_mean_gain_pct':float(g.gain_vs_last_pct.mean()),'worst_task_gain_pct':float(g.gain_vs_last_pct.min())})
    csv(HERE/'results/aggregate.csv',aggregate);dump(out/'MANIFEST.json',{'files':inventory(out)})

if __name__=='__main__':
    torch.set_num_threads(cfg()['training']['threads'])
    with threadpool_limits(limits=1):evaluate()
