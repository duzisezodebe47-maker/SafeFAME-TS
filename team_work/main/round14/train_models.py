"""仅使用选择区：比较数值模型、用时序OOF误差训练文本修正、冻结路线。"""
from __future__ import annotations
import copy,time,joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from r14_common import *
sys.path.insert(0,str(ROOT/'team_work/main/round13'))
import deep_baselines as deep
from threadpoolctl import threadpool_limits
import torch

def ridge_fit(x,y,alpha):
    x=np.asarray(x,float);y=np.asarray(y,float);xm=x.mean(axis=0);ym=y.mean(axis=0);xc=x-xm
    return {'xm':xm,'ym':ym,'w':np.linalg.solve(xc.T@xc+alpha*np.eye(x.shape[1]),xc.T@(y-ym))}
def ridge_predict(m,x):return (np.asarray(x)-m['xm'])@m['w']+m['ym']
def train_deep(name,x,y,epochs,seed,h):
    deep.seed_all(seed);m=deep.build(name,x.shape[1],h);p=cfg()['training'];opt=torch.optim.AdamW(m.parameters(),lr=p[name+'_lr'],weight_decay=p['weight_decay']);loader=deep.loader(x,y,p['batch_size'],seed)
    for _ in range(epochs):deep.step(m,loader,opt)
    return m
def numeric_predict(name,objects,x,h):
    if name=='Last':return np.repeat(x[:,-1:],h,axis=1)
    if name=='SeasonalNaive':return np.stack([x[:,-52+i%52] for i in range(h)],axis=1)
    if name=='Ridge':return ridge_predict(objects,x)
    return np.mean([deep.predict(model,x) for model in objects],axis=0)
def text_transform(objects,texts,available):
    z=objects['svd'].transform(objects['tfidf'].transform(texts));z=objects['scale'].transform(z);z[~np.asarray(available,bool)]=0;return z
def residual_predict(objects,texts,available):
    z=text_transform(objects,texts,available);p=ridge_predict(objects['correction'],z);p[~np.asarray(available,bool)]=0;return p
def code_anchors():
    own=[HERE/n for n in ('protocol.json','r14_common.py','prepare_data.py','train_models.py')];paths=own+[ROOT/'team_work/main/round13/adaptive.py']
    return {str(p.relative_to(PROJECT)):sha(p) for p in paths}|deep.code_anchors()|{str(ROOT.relative_to(PROJECT)/'team_work/main/round13/protocol.json'):sha(ROOT/'team_work/main/round13/protocol.json')}

def train_one(h):
    prepared=LOCAL/'prepared';seal=read(prepared/'MANIFEST.json');require(seal['protocol_sha256']==sha(HERE/'protocol.json'),'protocol not same as input')
    for rel,value in seal['files'].items():require(sha(prepared/rel)==value,'prepared file changed')
    out=LOCAL/'models'/f'h{h}';require(not out.exists(),'model output exists');out.mkdir(parents=True)
    folder=prepared/f'h{h}/selection';meta=pd.read_csv(folder/'metadata.csv');x=np.load(folder/'x.npy');y=np.load(folder/'y.npy');texts=read(folder/'texts.json');tr=meta.segment.eq('train').to_numpy();cal=meta.segment.eq('calibration').to_numpy();dec=meta.segment.eq('decision').to_numpy();available=meta.text_available.to_numpy(bool)
    p=cfg();fit=tr|cal;pred_train={};pred_final={};final_objects={};scores={};deep_meta={};train_only_models={};logs=[]
    for name in ('Last','SeasonalNaive'):
        pred_train[name]=numeric_predict(name,None,x,h);pred_final[name]=pred_train[name];final_objects[name]=None
    alphas={a:mse(ridge_predict(ridge_fit(x[tr],y[tr],a),x[cal]),y[cal]) for a in p['ridge_alphas']};alpha=min(alphas,key=lambda a:(alphas[a],a));train_only_models['Ridge']=ridge_fit(x[tr],y[tr],alpha);final_objects['Ridge']=ridge_fit(x[fit],y[fit],alpha);pred_train['Ridge']=ridge_predict(train_only_models['Ridge'],x);pred_final['Ridge']=ridge_predict(final_objects['Ridge'],x)
    for name in ('DLinear','PatchTST'):
        seed_predictions=[];refit_predictions=[];final_objects[name]=[];train_only_models[name]=[];deep_meta[name]=[]
        for seed in p['seeds']:
            start=time.perf_counter();deep.seed_all(seed);model=deep.build(name,x.shape[1],h);c=p['training'];opt=torch.optim.AdamW(model.parameters(),lr=c[name+'_lr'],weight_decay=c['weight_decay']);loader=deep.loader(x[tr],y[tr],c['batch_size'],seed);best=float('inf');state=None;epoch_best=0;stale=0
            for epoch in range(1,c['max_epochs']+1):
                loss=deep.step(model,loader,opt);val=mse(deep.predict(model,x[cal]),y[cal]);logs.append({'model':name,'seed':seed,'epoch':epoch,'train_mse':loss,'calibration_mse':val})
                if val<best-c['min_delta']:best=val;epoch_best=epoch;state=copy.deepcopy(model.state_dict());stale=0
                else:stale+=1
                if stale>=c['patience']:break
            model.load_state_dict(state);seed_predictions.append(deep.predict(model,x));train_only_models[name].append(model)
            final=train_deep(name,x[fit],y[fit],epoch_best,seed,h);final_objects[name].append(final);refit_predictions.append(deep.predict(final,x));torch.save(final.state_dict(),out/f'{name}_{seed}.pt');np.save(out/f'{name}_{seed}_calibration.npy',seed_predictions[-1][cal]);deep_meta[name].append({'seed':seed,'best_epoch':epoch_best,'train_only_cal_mse':best,'parameters':sum(v.numel() for v in model.parameters()),'seconds':time.perf_counter()-start})
            print('TRAIN',h,name,seed,'epochs',epoch_best,flush=True)
        pred_train[name]=np.mean(seed_predictions,axis=0);pred_final[name]=np.mean(refit_predictions,axis=0)
    scores={name:mse(pred[cal],y[cal]) for name,pred in pred_train.items()};selected=min(scores,key=lambda n:(scores[n],n));print('NUMERIC_SELECTED',h,selected,flush=True)
    # OOF residual target: each fold's fit windows end before its validation origin.
    train_ids=np.flatnonzero(tr);n=len(train_ids);cuts=[int(n*z) for z in (.4,.6,.8,1.)];oof=np.full_like(y,np.nan,dtype=float);oof_audit=[]
    global_scale=read(prepared/f'h{h}/scaler.json');raw_numeric=pd.read_csv(prepared/'numeric.csv').dsci.to_numpy(float)
    for j in range(3):
        val_ids=train_ids[cuts[j]:cuts[j+1]];first_origin=int(meta.iloc[val_ids[0]].origin_index);fit_ids=train_ids[meta.iloc[train_ids].origin_index.to_numpy()+h<=first_origin]
        fold_mu=raw_numeric[:first_origin].mean();fold_std=raw_numeric[:first_origin].std()
        fold_x=(x*global_scale['std']+global_scale['mean']-fold_mu)/fold_std
        fold_y=(y[fit_ids]*global_scale['std']+global_scale['mean']-fold_mu)/fold_std
        if selected in ('Last','SeasonalNaive'):model=None
        elif selected=='Ridge':model=ridge_fit(fold_x[fit_ids],fold_y,alpha)
        else:model=[train_deep(selected,fold_x[fit_ids],fold_y,m['best_epoch'],m['seed'],h) for m in deep_meta[selected]]
        pred=numeric_predict(selected,model,fold_x[val_ids],h)
        oof[val_ids]=(pred*fold_std+fold_mu-global_scale['mean'])/global_scale['std'];oof_audit.append({'fold':j+1,'fit_origins':len(fit_ids),'fit_last_origin':int(meta.iloc[fit_ids[-1]].origin_index),'validation_first_origin':first_origin,'validation_origins':len(val_ids),'target_purge_ok':int(meta.iloc[fit_ids[-1]].origin_index)+h<=first_origin,'scaler_fit_raw_rows':first_origin,'scaler_mean':float(fold_mu),'scaler_std':float(fold_std)})
    ok=tr&available&np.isfinite(oof).all(axis=1);require(ok.sum()>=30,'insufficient true-text OOF residual samples')
    text_train=[t for t,k in zip(texts,tr&available) if k];tfidf=TfidfVectorizer(max_features=p['text']['max_features'],min_df=p['text']['min_df'],max_df=p['text']['max_df'],stop_words='english');matrix=tfidf.fit_transform(text_train);svd=TruncatedSVD(n_components=min(p['text']['svd_components'],min(matrix.shape)-1),random_state=2026);zt=svd.fit_transform(matrix);scale=StandardScaler().fit(zt);text_objects={'tfidf':tfidf,'svd':svd,'scale':scale};z=text_transform(text_objects,texts,available);residual=y[ok]-oof[ok];losses={}
    for a in p['text']['residual_alphas']:
        correction=ridge_fit(z[ok],residual,a);v=ridge_predict(correction,z[cal]);v[~available[cal]]=0;losses[a]=mse(pred_train[selected][cal]+v,y[cal])
    residual_alpha=min(losses,key=lambda a:(losses[a],a));text_objects['correction']=ridge_fit(z[ok],residual,residual_alpha);base=pred_final[selected];direct=base+residual_predict(text_objects,texts,available)
    gate=[]
    for weight in p['weights']:
        loss0=np.mean((base[dec]-y[dec])**2,axis=1);loss=np.mean(((1-weight)*base[dec]+weight*direct[dec]-y[dec])**2,axis=1);delta=loss0-loss;halves=[float(v.mean()) for v in np.array_split(delta,2)];gate.append({'weight':weight,'mse':float(loss.mean()),'benefit':float(delta.mean()),'half1_benefit':halves[0],'half2_benefit':halves[1],'eligible':bool(weight==0 or min(halves)>0)})
    w=min([v for v in gate if v['eligible']],key=lambda v:(v['mse'],v['weight']))['weight']
    # joblib contains only our own locally trained trusted objects, bound by seal.
    joblib.dump({'numeric':final_objects,'text':text_objects,'selected':selected,'weight':w,'horizon':h,'input_length':52,'scaler':read(prepared/f'h{h}/scaler.json')},out/'models.joblib')
    np.savez_compressed(out/'selection_predictions.npz',origins=meta.origin_index.to_numpy(),**pred_final,SelectedNumeric=base,DirectFusion=direct,StaticFallback=base+w*(direct-base))
    np.savez_compressed(out/'causal_warmup.npz',origins=meta.loc[dec,'origin_index'].to_numpy(),base=base[dec],direct=direct[dec],truth=y[dec],available=available[dec])
    np.savez_compressed(out/'oof_residuals.npz',origins=meta.loc[ok,'origin_index'].to_numpy(),prediction=oof[ok],target=y[ok],residual=residual)
    csv(out/'training_logs.csv',logs);dump(out/'selection_details.json',{'numeric_calibration_scores':scores,'ridge_alpha':alpha,'deep_training':deep_meta,'residual_alpha':residual_alpha,'residual_calibration_scores':losses,'oof_rows':int(ok.sum()),'oof_folds':oof_audit,'text_vocabulary':len(tfidf.vocabulary_),'static_gate':gate})
    dump(out/'selection_seal.json',{'horizon':h,'selected_numeric':selected,'static_weight':w,'test_metrics_seen':False,'protocol_sha256':sha(HERE/'protocol.json'),'prepared_manifest_sha256':sha(prepared/'MANIFEST.json'),'code':code_anchors(),'files':inventory(out),'role':'later-period historical temporal evaluation, not prospective validation'})
    csv(HERE/f'results/h{h}/calibration_comparison.csv',[{'candidate':name,'calibration_mse':value,'selected':name==selected} for name,value in scores.items()]);dump(HERE/f'results/h{h}/selection_summary.json',{'selected_numeric':selected,'static_weight':w,'oof_text_rows':int(ok.sum()),'test_metrics_seen':False})
    print('SEALED',h,'weight',w,'OOF_rows',int(ok.sum()),flush=True)

if __name__=='__main__':
    torch.set_num_threads(cfg()['training']['threads'])
    with threadpool_limits(limits=1):
        for h in cfg()['horizons']:train_one(h)
