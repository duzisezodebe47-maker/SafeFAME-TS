"""第16轮独立入口：官方州级数据、多序列评估与冻结证据。"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PROJECT=ROOT.parent
LOCAL=PROJECT/'主控接收数据/主控第16轮_三州验证_20261008'
PREVIOUS=PROJECT/'主控接收数据/主控第14轮_USDM_20261008'
sys.path.insert(0,str(ROOT/'team_work/main/round15'))
from analysis15 import choose_numeric,online_select
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def csv(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(v).to_csv(p,index=False,lineterminator='\n')
def require(ok,message):
    if not ok:raise ValueError(message)
def cfg():return read(HERE/'protocol.json')
def mse(a,b):return float(np.mean((np.asarray(a)-np.asarray(b))**2))
def inventory(folder):return {p.relative_to(folder).as_posix():sha(p) for p in sorted(Path(folder).rglob('*')) if p.is_file() and p.name!='MANIFEST.json'}
def code_anchors():
    paths=[HERE/n for n in ('protocol.json','common16.py','prepare16.py','model16.py','train16.py')]+[ROOT/'team_work/main/round15/analysis15.py',ROOT/'src/train_dlinear.py',ROOT/'src/data_utils.py',ROOT/'src/run_baselines.py']
    return {str(p.relative_to(PROJECT)):sha(p) for p in paths}
