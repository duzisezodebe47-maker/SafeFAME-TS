"""本地推理：固定模型只接收历史数值，无标签接口；输入保持原始OT单位。"""
from __future__ import annotations
import argparse
from functools import lru_cache
import joblib
from common import *

MODELS=('StrictSourceFallback','Last','N','AR-Ridge-refit','DLinear','PatchTST')

def validate_task(task):
    if not isinstance(task,str) or task not in protocol()['tasks']:raise ValueError('请选择已登记的四个任务之一')
    return task_spec(task)

def validate_request(payload):
    if not isinstance(payload,dict):raise ValueError('请求必须是JSON对象')
    if set(payload)-{'task','model','history'}:raise ValueError('只接受task、model、history，不接受未来真值或外部模型路径')
    task=payload.get('task');spec=validate_task(task);model=payload.get('model','StrictSourceFallback')
    if model not in MODELS:raise ValueError('不支持该模型')
    values=payload.get('history')
    if not isinstance(values,list) or len(values)!=spec['input_len']:raise ValueError(f"历史数据必须包含{spec['input_len']}个数值")
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in values):raise ValueError('历史数据只能包含数值')
    x=np.asarray(values,dtype=float)
    if not np.isfinite(x).all() or np.max(abs(x))>1e12:raise ValueError('输入包含非有限值或超过数值计算范围')
    return task,spec,model,x

@lru_cache(maxsize=4)
def trusted_objects(task):
    validate_task(task);folder=LOCAL/'comparison'/task/'proxy';seal=read(folder/'selection_seal.json')
    # Never deserialize user-supplied files. Check local experiment seal first.
    require(sha(folder/'models.joblib')==seal['files']['models.joblib'],'模型校验失败')
    from experiments import code_anchors
    require(seal['code_sha256']==code_anchors(),'模型代码与封存记录不一致')
    return joblib.load(folder/'models.joblib')

@lru_cache(maxsize=8)
def trusted_deep(task,name):
    import torch
    from deep_baselines import build,code_anchors
    spec=validate_task(task);folder=LOCAL/'deep'/task;seal=read(folder/'selection_seal.json');require(seal['code_sha256']==code_anchors(),'深度模型代码校验失败');models=[]
    torch.set_num_threads(2)
    for seed in protocol()['seeds']:
        sub=folder/name/str(seed);m=read(sub/'manifest.json');require(sha(sub/'manifest.json')==seal['model_manifests'][f'{name}/{seed}'],'深度模型清单校验失败');require(sha(sub/'refit.pt')==m['files']['refit.pt'],'深度模型权重校验失败')
        model=build(name,spec['input_len'],spec['horizon']);model.load_state_dict(torch.load(sub/'refit.pt',map_location='cpu',weights_only=True));model.eval();models.append(model)
    return models

def forecast(payload):
    task,spec,name,raw=validate_request(payload);objects=trusted_objects(task);fit=objects['numeric_fit'];x=((raw-fit['mean'])/fit['std'])[None,:];actual=objects['fallback'] if name=='StrictSourceFallback' else name
    if actual in ('DLinear','PatchTST'):
        from deep_baselines import predict
        pred=np.mean([predict(m,x) for m in trusted_deep(task,actual)],axis=0)[0]
    elif actual=='Last':pred=np.repeat(x[0,-1],spec['horizon'])
    elif actual in ('AR-Ridge','AR-Ridge-refit'):
        from experiments import ar_predict
        pred=ar_predict(objects[actual],x)[0]
    else:
        model=objects['models']['N'];f=FeatureBundle(x,np.zeros((1,1)),np.zeros((1,1)),np.zeros(1,bool),np.array([0]));pred=model.predict(f)[0]
    result=pred*fit['std']+fit['mean'];require(np.isfinite(result).all(),'预测数值异常')
    return {'task':task,'requested_model':name,'effective_model':actual,'horizon':spec['horizon'],'forecast':result.tolist(),'scale':'原始OT数值尺度；请以数据集字段定义理解单位','text_weight':0,'source_verified':False,'reason':'原始文本发布时间未核实，严格模式使用数值回退' if name=='StrictSourceFallback' else '用户指定的数值模型','scope':'已训练模型的本地推理演示；没有实时数据接入，也未证明外部数据泛化性能'}

def example(task):
    spec=validate_task(task);meta,f=features(task,'proxy');fit=read(BUNDLE/task/'numeric_fit.json');history=f.numeric_history[-1].astype(float)*fit['std']+fit['mean']
    return {'task':task,'history':history.tolist(),'origin_index':int(meta.iloc[-1].origin_index),'input_len':spec['input_len'],'horizon':spec['horizon'],'role':'历史测试起点前的已知输入窗口，不包含预测目标'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();dump(a.output,forecast(read(a.input)))
