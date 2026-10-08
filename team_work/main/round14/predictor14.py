"""DSCI原始尺度推理；未经证实的用户文本不进入严格预测。"""
from functools import lru_cache
from r14_common import *
from evaluate_test import load_trusted
from train_models import numeric_predict
@lru_cache(maxsize=2)
def models(h):return load_trusted(h)[0]
def validate(payload):
    if not isinstance(payload,dict) or set(payload)-{'horizon','history'}:raise ValueError('只接受horizon和history两个字段')
    h=payload.get('horizon');history=payload.get('history')
    if type(h) is not int or h not in cfg()['horizons']:raise ValueError('预测步长只能是4或12周')
    if not isinstance(history,list) or len(history)!=52:raise ValueError('需要52个历史周的DSCI数值')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in history):raise ValueError('输入必须为数值')
    x=np.asarray(history,float)
    if not np.isfinite(x).all() or (x<0).any() or (x>500).any():raise ValueError('DSCI必须位于0至500之间')
    return h,x
def forecast(payload):
    h,raw=validate(payload);m=models(h);s=m['scaler'];x=((raw-s['mean'])/s['std'])[None,:];name=m['selected'];z=numeric_predict(name,m['numeric'][name],x,h)[0];result=z*s['std']+s['mean']
    require(np.isfinite(result).all(),'预测出现非有限值')
    return {'horizon':h,'model':name,'forecast_dsci':result.tolist(),'text_weight':0,'reason':'严格模式：缺少逐篇首次公开时间证明，使用验证集选定的数值模型','out_of_range':bool(((result<0)|(result>500)).any()),'note':'研究模型输出不做事后裁剪；若超出DSCI范围会明确提示，不据此作实际决策。'}
def example(h):
    if h not in cfg()['horizons']:raise ValueError('步长无效')
    folder=LOCAL/'prepared'/f'h{h}';x=np.load(folder/'test_features/x.npy')[-1];s=read(folder/'scaler.json');meta=pd.read_csv(folder/'test_features/metadata.csv');return {'history':(x*s['std']+s['mean']).tolist(),'asof':meta.iloc[-1].asof_utc}
