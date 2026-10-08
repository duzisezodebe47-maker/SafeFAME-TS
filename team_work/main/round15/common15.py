"""第15轮路径、哈希及只读上轮数据入口。"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PROJECT=ROOT.parent
LOCAL=PROJECT/'主控接收数据/主控第15轮_选择稳定性与归因_20261008'
PREVIOUS=PROJECT/'主控接收数据/主控第14轮_USDM_20261008'
sys.path.insert(0,str(ROOT/'team_work/main/round14'))
from r14_common import sha,read,dump,csv,require

def config():return read(HERE/'protocol.json')
def mse(a,b):return float(np.mean((np.asarray(a)-np.asarray(b))**2))
def inventory(folder):return {p.relative_to(folder).as_posix():sha(p) for p in sorted(Path(folder).rglob('*')) if p.is_file() and p.name!='MANIFEST.json'}
def seal_inputs():
    paths=[HERE/'protocol.json',HERE/'common15.py',HERE/'analysis15.py',HERE/'run15.py']
    for folder in ('prepared','models','evaluation'):
        paths.extend(p for p in (PREVIOUS/folder).rglob('*') if p.is_file())
    return {str(p.relative_to(PROJECT)):sha(p) for p in paths}
