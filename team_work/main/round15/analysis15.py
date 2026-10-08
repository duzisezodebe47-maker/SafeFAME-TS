"""透明的模型选择规则与残差归因；各函数不隐式读取目标。"""
import numpy as np

def choose_numeric(losses,names,rule):
    losses=np.asarray(losses,float)
    if losses.ndim!=2 or losses.shape[1]!=len(names) or not len(losses) or not np.isfinite(losses).all():
        raise ValueError('invalid selection losses')
    if rule=='decision_mean':score=losses.mean(axis=0)
    elif rule=='recent_26_origins':score=losses[-26:].mean(axis=0)
    elif rule=='three_block_minimax_relative_regret':
        if len(losses)<3:raise ValueError('three nonempty blocks required')
        blocks=np.stack([block.mean(axis=0) for block in np.array_split(losses,3)])
        score=(blocks/np.maximum(blocks.min(axis=1,keepdims=True),1e-12)-1).max(axis=0)
    else:raise ValueError('unknown rule')
    index=min(range(len(names)),key=lambda i:(float(score[i]),names[i]))
    return index,score

def online_select(origins,predictions,horizon,names,lookback,target_reader):
    """只在完整H步标签成熟后调用外部读取器，允许起点有间隔。"""
    origins=np.asarray(origins,int);predictions=np.asarray(predictions,float)
    if horizon<1 or lookback<1 or len(origins)!=len(predictions) or np.any(np.diff(origins)<=0):raise ValueError('invalid origin sequence')
    if predictions.shape[1:]!=(len(names),horizon) or not np.isfinite(predictions).all():raise ValueError('invalid predictions')
    past=[];released=0;outputs=[];trace=[]
    for i,now in enumerate(origins):
        while released<i and origins[released]+horizon<=now:
            target=np.asarray(target_reader(released),float)
            if target.shape!=(horizon,) or not np.isfinite(target).all():raise ValueError('invalid mature target')
            past.append((int(origins[released]),np.mean((predictions[released]-target)**2,axis=1)));released+=1
        history=past[-lookback:]
        if history:index,scores=choose_numeric(np.stack([v[1] for v in history]),names,'decision_mean')
        else:index=names.index('Last');scores=np.zeros(len(names))
        outputs.append(predictions[i,index]);trace.append({'origin_index':int(now),'candidate':names[index],'mature_history':len(history),'last_feedback_origin':history[-1][0] if history else None})
    return np.asarray(outputs),trace

def state_features(x,dates):
    """仅使用已观测数值及目标周日历；不读取周报正文。"""
    x=np.asarray(x,float);days=np.asarray([d.dayofyear for d in dates]);phase=2*np.pi*days/365.25
    return np.column_stack([x[:,-1],x[:,-1]-x[:,-5],x[:,-1]-x[:,-14],x[:,-13:].std(axis=1),np.sin(phase),np.cos(phase)])

def age_mask(meta,weeks):
    import pandas as pd
    last_map=pd.to_datetime(meta.target_date)-pd.Timedelta(days=7)
    age=(last_map-pd.to_datetime(meta.source_map_date)).dt.days/7
    return meta.text_available.to_numpy(bool)&age.between(0,weeks).to_numpy()

def gate(base,correction,y,weights):
    rows=[];loss0=np.mean((base-y)**2,axis=1)
    for w in weights:
        loss=np.mean((base+w*correction-y)**2,axis=1);halves=[float(v.mean()) for v in np.array_split(loss0-loss,2)]
        rows.append({'weight':w,'mse':float(loss.mean()),'half1_benefit':halves[0],'half2_benefit':halves[1],'eligible':bool(w==0 or min(halves)>0)})
    chosen=min((r for r in rows if r['eligible']),key=lambda r:(r['mse'],r['weight']))
    return chosen['weight'],rows

def block_interval(delta,block,draws=2000,seed=2026):
    delta=np.asarray(delta,float);n=len(delta);rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,(draws,int(np.ceil(n/block))))
    indices=((starts[:,:,None]+np.arange(block))%n).reshape(draws,-1)[:,:n]
    return np.quantile(delta[indices].mean(axis=1),[.025,.975])
