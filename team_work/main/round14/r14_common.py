"""第14轮独立数据和文件入口；与已封存的第13轮隔离。"""
from pathlib import Path
import hashlib,json,os,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PROJECT=ROOT.parent
DATA=PROJECT/'主控接收数据'
LOCAL=DATA/'主控第14轮_USDM_20261008'
RAW=LOCAL/'raw'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def dump(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def csv(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(data).to_csv(p,index=False,lineterminator='\n')
def require(ok,message):
    if not ok:raise ValueError(message)
def cfg():return read(HERE/'protocol.json')
def mse(a,b):return float(np.mean((np.asarray(a)-np.asarray(b))**2))
def inventory(folder):return {p.relative_to(folder).as_posix():sha(p) for p in sorted(Path(folder).rglob('*')) if p.is_file() and p.name!='MANIFEST.json'}
