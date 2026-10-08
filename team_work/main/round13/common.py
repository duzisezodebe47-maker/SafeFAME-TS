"""共同路径、冻结输入和证据读写；原始成果只读。"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PROJECT=ROOT.parent
DATA=PROJECT/'主控接收数据'
BUNDLE=DATA/'SafeFAME_TS_第7轮数据_02_a06dfe1/release_unpacked/inputs/formal_bundle'
RAW=DATA/'SafeFAME_TS_第4轮_2d530ec7'
LINEAGE=DATA/'SafeFAME_TS_第7轮数据_02_a06dfe1/release_unpacked/inputs/text_lineage.csv'
OFFICIAL=DATA/'PatchTST_official_204c21e/PatchTST_supervised'
LOCAL=DATA/'主控加强_20261007'
SPEC=ROOT/'team_work/main/round2/split_spec_v2.json'
sys.path[:0]=[str(ROOT/'team_work/main/round12/model_engine'),str(ROOT/'team_work/main/src')]
from branches import FeatureBundle
from candidates import BranchResidualCandidate,PROTOCOL_ALPHAS
from permutation import _fit
from numeric_fallbacks import numeric_baselines
from team_eval.v2 import verify_bundle

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False,lineterminator='\n')
def require(ok,message):
    if not ok: raise ValueError(message)
def protocol():
    p=read(HERE/'protocol.json');require(sha(SPEC)==p['split_spec_sha256'],'spec changed');return p
def task_spec(task):return next(t for t in read(SPEC)['tasks'] if f"{t['domain']}_h{t['horizon']}_f{t['fold_id']}"==task)
def selection(task,scenario):
    """只读隔离包中非测试标签。"""
    folder=LOCAL/'isolated'/task/'selection_fit';meta=pd.read_csv(folder/'metadata.csv')
    f=FeatureBundle(np.load(folder/'numeric_history.npy'),np.load(folder/scenario/'semantic.npy'),np.load(folder/scenario/'quality.npy'),np.load(folder/scenario/'text_available.npy'),np.load(folder/'origin_index.npy'),targets_standardized=np.load(folder/'targets_standardized.npy'))
    return meta,f
def features(task,scenario):
    """测试特征接口中不存在目标字段；历史只能来自起点之前。"""
    folder=LOCAL/'isolated'/task/'test_features';meta=pd.read_csv(folder/'metadata.csv')
    original=pd.read_csv(BUNDLE/task/'samples.csv');mask=original.segment.eq('test').to_numpy()
    x=np.asarray(np.load(BUNDLE/task/'numeric_history.npy',mmap_mode='r')[mask])
    f=FeatureBundle(x,np.load(folder/scenario/'semantic.npy'),np.load(folder/scenario/'quality.npy'),np.load(folder/scenario/'text_available.npy'),np.load(folder/'origin_index.npy'))
    return meta,f
def truth(task):
    meta=pd.read_csv(BUNDLE/task/'samples.csv');mask=meta.segment.eq('test').to_numpy()
    return np.asarray(np.load(BUNDLE/task/'targets_standardized.npy',mmap_mode='r')[mask],float)
def infer(f):return FeatureBundle(f.numeric_history,f.semantic,f.quality,f.text_available,f.origin_index)
def predict_grid(path,meta,predictions):
    rows=[]
    for name,p in predictions.items():
        require(p.ndim==2 and p.shape[0]==len(meta) and np.isfinite(p).all(),'bad prediction shape')
        for i,r in enumerate(meta.itertuples()):
            for h,value in enumerate(p[i],1):rows.append({'candidate':name,'origin_id':r.origin_id,'origin_index':int(r.origin_index),'step':h,'y_pred':float(value)})
    csv(path,rows)
def mse(p,y):return float(np.mean((np.asarray(p)-np.asarray(y))**2))
def weights_hash(model):return hashlib.sha256(np.ascontiguousarray(model.weights).tobytes()).hexdigest()
