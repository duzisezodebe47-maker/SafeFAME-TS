"""延迟反馈安全融合：只消费完整成熟的历史标签；风险门槛是启发式。"""
from __future__ import annotations
from collections import deque
import numpy as np

class DelayedRiskMixer:
    def __init__(self,horizon,config):
        if horizon<1:raise ValueError('horizon must be positive')
        self.horizon=horizon;self.config=dict(config);self.history=deque(maxlen=config['lookback_mature_origins']);self.last_origin=-1;self.read_origins=[]
    def observe(self,origin,base,text,target,now):
        if origin+self.horizon>now:raise ValueError('future target feedback rejected')
        if self.read_origins and origin<=self.read_origins[-1]:raise ValueError('duplicate or unordered feedback')
        arrays=[np.asarray(v,float) for v in (base,text,target)]
        if any(v.shape!=(self.horizon,) or not np.isfinite(v).all() for v in arrays):raise ValueError('invalid feedback')
        self.history.append((origin,*arrays));self.read_origins.append(origin)
    def choose(self,origin,base,text,available=True,source_verified=True):
        if origin<=self.last_origin:raise ValueError('prediction origin must increase')
        self.last_origin=origin
        a,b=np.asarray(base,float),np.asarray(text,float)
        if a.shape!=(self.horizon,) or b.shape!=a.shape or not np.isfinite(a).all():raise ValueError('invalid prediction')
        detail={'origin_index':int(origin),'weight':0.,'mature_origins':len(self.history),'last_feedback_origin':self.read_origins[-1] if self.read_origins else None,'reason':'insufficient_mature_history','guard_kind':'heuristic block mean lower benefit; not a coverage guarantee'}
        if not available or not np.isfinite(b).all():detail['reason']='text_unavailable_or_untrainable';return a.copy(),detail
        if not source_verified:detail['reason']='publication_unverified';return a.copy(),detail
        if len(self.history)<self.config['minimum_mature_origins']:return a.copy(),detail
        hist=list(self.history);p0=np.stack([z[1] for z in hist]);p1=np.stack([z[2] for z in hist]);y=np.stack([z[3] for z in hist]);base_loss=np.mean((p0-y)**2,axis=1)
        block=self.horizon;count=len(hist)//block
        if count<3:detail['reason']='insufficient_mature_blocks';return a.copy(),detail
        valid=[]
        for w in self.config['weight_grid']:
            if w<=0:continue
            gain=base_loss-np.mean(((1-w)*p0+w*p1-y)**2,axis=1)
            groups=gain[-count*block:].reshape(count,block).mean(axis=1)
            lower=float(groups.mean()-self.config['risk_z']*groups.std(ddof=1)/np.sqrt(count))
            if lower>self.config['epsilon']:valid.append((float(gain.mean()),float(w),lower))
        if valid:
            benefit,w,lower=max(valid,key=lambda z:(z[0],-z[1]));detail.update(weight=w,reason='past_block_benefit_positive',past_mean_benefit=benefit,past_lower_benefit=lower)
        else:detail['reason']='past_benefit_not_stable'
        return (1-detail['weight'])*a+detail['weight']*b,detail

def causal_replay(origins,base,text,available,horizon,config,target_reader,source_verified=None):
    """target_reader只在目标已完整成熟时调用，评分与标签释放由环境管理。"""
    origins=np.asarray(origins,int)
    if len(origins)!=len(base) or np.any(np.diff(origins)<=0):raise ValueError('origin order/length')
    mixer=DelayedRiskMixer(horizon,config);predictions=[];traces=[];released=0
    for i,origin in enumerate(origins):
        while released<i and origins[released]+horizon<=origin:
            y=target_reader(released)
            mixer.observe(int(origins[released]),base[released],text[released],y,int(origin));released+=1
        value,trace=mixer.choose(int(origin),base[i],text[i],bool(available[i]),True if source_verified is None else bool(source_verified[i]))
        predictions.append(value);traces.append(trace)
    return np.asarray(predictions),traces
