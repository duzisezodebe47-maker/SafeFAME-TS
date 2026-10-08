"""六个任务全部封存后才允许评分；每州每步长三个DLinear种子。"""
import copy,time
from common16 import *
from model16 import *
from threadpoolctl import threadpool_limits

def train(task):
    folder=LOCAL/'models'/task
    if (folder/'selection_seal.json').exists():
        seal=read(folder/'selection_seal.json');require(seal['code']==code_anchors(),'resume source mismatch')
        for rel,digest in seal['files'].items():require(sha(folder/rel)==digest,'resume model mismatch')
        print('SEALED SKIP',task,flush=True);return
    require(not folder.exists(),'partial model output must be archived');folder.mkdir(parents=True)
    code,hs=task.split('_h');h=int(hs);p=cfg();t=p['training'];data=np.load(LOCAL/f'prepared/{task}/selection.npz');meta=pd.read_csv(LOCAL/f'prepared/{task}/selection_metadata.csv');x=data['x'];y=data['y'];tr=meta.segment.eq('train').to_numpy();cal=meta.segment.eq('calibration').to_numpy();dec=meta.segment.eq('decision').to_numpy();fit=tr|cal
    alpha_scores={a:mse(ridge_predict(ridge_fit(x[tr],y[tr],a),x[cal]),y[cal]) for a in p['ridge_alphas']};alpha=min(alpha_scores,key=lambda a:(alpha_scores[a],a));ridge=ridge_fit(x[fit],y[fit],alpha);dump(folder/'ridge.json',{k:v.tolist() for k,v in ridge.items()})
    final={'Last':np.repeat(x[:,-1:],h,axis=1),'Ridge':ridge_predict(ridge,x)};calpred={'Last':final['Last'][cal],'Ridge':ridge_predict(ridge_fit(x[tr],y[tr],alpha),x[cal])};logs=[];details=[];seedp=[];seedcal=[]
    for seed in p['seeds']:
        start=time.perf_counter();seed_all(seed);m=build(h);opt=torch.optim.AdamW(m.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay']);dl=loader(x[tr],y[tr],seed);best=float('inf');state=None;best_epoch=0;stale=0
        for epoch in range(1,t['max_epochs']+1):
            loss=step(m,dl,opt);val=mse(predict(m,x[cal]),y[cal]);logs.append({'seed':seed,'phase':'calibration_selection','epoch':epoch,'train_mse':loss,'calibration_mse':val})
            if val<best-t['min_delta']:best=val;state=copy.deepcopy(m.state_dict());best_epoch=epoch;stale=0
            else:stale+=1
            if stale>=t['patience']:break
        m.load_state_dict(state);cp=predict(m,x[cal]);seedcal.append(cp);np.save(folder/f'DLinear_{seed}_calibration.npy',cp);torch.save(state,folder/f'DLinear_{seed}_train.pt')
        seed_all(seed);m=build(h);opt=torch.optim.AdamW(m.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay']);dl=loader(x[fit],y[fit],seed)
        for epoch in range(1,best_epoch+1):logs.append({'seed':seed,'phase':'train_cal_refit','epoch':epoch,'train_mse':step(m,dl,opt),'calibration_mse':None})
        torch.save(m.state_dict(),folder/f'DLinear_{seed}_refit.pt');seedp.append(predict(m,x));details.append({'seed':seed,'best_epoch':best_epoch,'calibration_mse':mse(cp,y[cal]),'parameters':sum(v.numel() for v in m.parameters()),'seconds':time.perf_counter()-start});print('TRAIN',task,seed,'epoch',best_epoch,flush=True)
    final['DLinear']=np.mean(seedp,axis=0);calpred['DLinear']=np.mean(seedcal,axis=0);cal_scores={n:mse(v,y[cal]) for n,v in calpred.items()};winner=min(cal_scores,key=lambda n:(cal_scores[n],n));choices={};choice_rows=[];losses=np.column_stack([np.mean((final[n][dec]-y[dec])**2,axis=1) for n in p['candidates']])
    for rule in p['fixed_rules']:
        idx,score=choose_numeric(losses,p['candidates'],rule);choices[rule]=p['candidates'][idx];choice_rows.extend({'rule':rule,'candidate':name,'score':float(score[i]),'selected':i==idx} for i,name in enumerate(p['candidates']))
    np.savez_compressed(folder/'selection_predictions.npz',**final);np.savez_compressed(folder/'calibration_predictions.npz',**calpred);csv(folder/'training_log.csv',logs)
    dump(folder/'details.json',{'task':task,'ridge_alpha':alpha,'ridge_calibration_scores':alpha_scores,'calibration_scores':cal_scores,'dlinear':details,'training_origins':int(tr.sum()),'refit_origins':int(fit.sum())})
    dump(folder/'selection_seal.json',{'task':task,'calibration_choice':winner,'fixed_choices':choices,'code':code_anchors(),'prepared_manifest':sha(LOCAL/'prepared/MANIFEST.json'),'files':inventory(folder),'test_metrics_seen':False})
    csv(HERE/f'results/{task}/choice_scores.csv',choice_rows);dump(HERE/f'results/{task}/selection.json',{'calibration_choice':winner,'fixed_choices':choices,'ridge_alpha':alpha,'dlinear_best_epochs':[d['best_epoch'] for d in details]});print('SEALED',task,winner,choices,flush=True)

def main():
    torch.set_num_threads(cfg()['training']['threads']);manifest=read(LOCAL/'prepared/MANIFEST.json')
    for rel,digest in manifest['files'].items():require(sha(LOCAL/'prepared'/rel)==digest,'prepared file changed')
    with threadpool_limits(limits=1):
        for state in cfg()['states']:
            for h in cfg()['horizons']:train(f'{state["code"]}_h{h}')
    dump(LOCAL/'all_selection_seals.json',{'code':code_anchors(),'tasks':{f'{s["code"]}_h{h}':sha(LOCAL/f'models/{s["code"]}_h{h}/selection_seal.json') for s in cfg()['states'] for h in cfg()['horizons']},'test_metrics_seen':False})

if __name__=='__main__':main()
