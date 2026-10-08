"""已有DLinear及官方PatchTST的统一适配；校准先预测、再train+cal重拟合。"""
from __future__ import annotations
import argparse
import copy
import random
import sys
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from common import *
sys.path.insert(0,str(ROOT/'src'))
from train_dlinear import DLinearTarget
sys.path.insert(0,str(OFFICIAL))
from models.PatchTST import Model as OfficialPatchTST

class PatchTarget(nn.Module):
    """复用原train_patchtst.py的官方骨干配置；只改变外部源码路径。"""
    def __init__(self,input_len,horizon):
        super().__init__();p=protocol()['PatchTST'];patch=8 if input_len>=52 else 4
        args=SimpleNamespace(enc_in=1,seq_len=input_len,pred_len=horizon,e_layers=p['e_layers'],n_heads=p['n_heads'],d_model=p['d_model'],d_ff=p['d_ff'],dropout=p['dropout'],fc_dropout=p['dropout'],head_dropout=0.,individual=False,patch_len=patch,stride=patch//2,padding_patch='end',revin=True,affine=True,subtract_last=False,decomposition=False,kernel_size=25)
        self.model=OfficialPatchTST(args)
    def forward(self,x):return self.model(x)[:,:,0]

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
def build(name,length,horizon):
    return DLinearTarget(length,1,horizon,min(25,length if length%2 else length-1)) if name=='DLinear' else PatchTarget(length,horizon)
def loader(x,y,batch,seed):
    return DataLoader(TensorDataset(torch.as_tensor(x,dtype=torch.float32).unsqueeze(-1),torch.as_tensor(y,dtype=torch.float32)),batch_size=batch,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
@torch.inference_mode()
def predict(model,x,batch=512):
    model.eval();return np.concatenate([model(torch.as_tensor(x[i:i+batch],dtype=torch.float32).unsqueeze(-1)).numpy() for i in range(0,len(x),batch)])
def step(model,data,opt):
    model.train();total=0.;n=0
    for x,y in data:
        opt.zero_grad(set_to_none=True);loss=torch.mean((model(x)-y)**2)
        require(bool(torch.isfinite(loss)),'nonfinite deep loss');loss.backward();opt.step();total+=float(loss.detach())*len(x);n+=len(x)
    return total/n

def train_one(task,name,seed,force=False):
    out=LOCAL/'deep'/task/name/str(seed)
    if (out/'manifest.json').exists():
        manifest=read(out/'manifest.json')
        for rel,h in manifest['files'].items():require(sha(out/rel)==h,'resume file changed')
        require(manifest['protocol_sha256']==sha(HERE/'protocol.json'),'resume protocol changed')
        require(manifest['code_sha256']==code_anchors(),'resume code changed');print('RESUME',task,name,seed,flush=True);return
    require(not out.exists(),'incomplete output exists; archive it before retry')
    out.mkdir(parents=True)
    meta,f=selection(task,'proxy');tr=meta.segment.eq('train').to_numpy();ca=meta.segment.eq('calibration').to_numpy()
    x,y=f.numeric_history,f.targets_standardized;t=task_spec(task);p=protocol()['deep_training']
    seed_all(seed);model=build(name,t['input_len'],t['horizon']);lr=p[name+'_learning_rate'];opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=p['weight_decay'])
    data=loader(x[tr],y[tr],p['batch_size'],seed);best=float('inf');best_epoch=0;best_state=None;stale=0;logs=[];start=time.perf_counter()
    for epoch in range(1,p['max_epochs']+1):
        loss=step(model,data,opt);val=mse(predict(model,x[ca]),y[ca]);logs.append({'phase':'train_only_selection','epoch':epoch,'train_mse':loss,'calibration_mse':val})
        if val<best-p['min_delta']:best=val;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if epoch%10==0:print(task,name,seed,'epoch',epoch,'cal',round(val,5),flush=True)
        if stale>=p['patience']:break
    require(best_state is not None,'no checkpoint');model.load_state_dict(best_state)
    cal_pred=predict(model,x[ca]);np.save(out/'calibration_train_only.npy',cal_pred);torch.save(best_state,out/'train_only.pt')
    seed_all(seed);model=build(name,t['input_len'],t['horizon']);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=p['weight_decay']);fit=tr|ca;data=loader(x[fit],y[fit],p['batch_size'],seed)
    for epoch in range(1,best_epoch+1):logs.append({'phase':'refit_train_plus_cal','epoch':epoch,'train_mse':step(model,data,opt),'calibration_mse':None})
    torch.save(model.state_dict(),out/'refit.pt');np.save(out/'selection_refit.npy',predict(model,x));csv(out/'training.csv',logs)
    dump(out/'manifest.json',{'task':task,'model':name,'seed':seed,'best_epoch':best_epoch,'calibration_train_only_mse':mse(cal_pred,y[ca]),'seconds':time.perf_counter()-start,'parameters':sum(z.numel() for z in model.parameters()),'torch':torch.__version__,
       'train_origins':int(tr.sum()),'calibration_origins':int(ca.sum()),'test_labels_accessed':False,'protocol_sha256':sha(HERE/'protocol.json'),'package_manifest_sha256':sha(LOCAL/'isolated'/task/'MANIFEST.json'),'code_sha256':code_anchors(),
       'files':{z.name:sha(z) for z in out.iterdir() if z.is_file()}})
    print('DONE',task,name,seed,'best_epoch',best_epoch,'seconds',round(time.perf_counter()-start,1),flush=True)

def code_anchors():
    paths=[HERE/'common.py',HERE/'deep_baselines.py',ROOT/'src/train_dlinear.py',ROOT/'src/data_utils.py',ROOT/'src/run_baselines.py']+sorted((OFFICIAL/'layers').glob('*.py'))+[OFFICIAL/'models/PatchTST.py']
    return {str(p.relative_to(PROJECT)):sha(p) for p in paths}

def infer_test(task):
    out=LOCAL/'deep'/task;seal=read(out/'selection_seal.json');require(seal['protocol_sha256']==sha(HERE/'protocol.json'),'deep protocol changed')
    require(seal['code_sha256']==code_anchors(),'deep code changed');target=out/'test_predictions.csv';require(not target.exists(),'refuse test prediction overwrite')
    meta,f=features(task,'proxy');t=task_spec(task);preds={}
    for name in protocol()['deep_models']:
        seed_preds=[]
        for seed in protocol()['seeds']:
            folder=out/name/str(seed);m=read(folder/'manifest.json')
            for rel,h in m['files'].items():require(sha(folder/rel)==h,'model bytes changed')
            model=build(name,t['input_len'],t['horizon']);model.load_state_dict(torch.load(folder/'refit.pt',map_location='cpu',weights_only=True));p=predict(model,f.numeric_history);preds[f'{name}_s{seed}']=p;seed_preds.append(p)
        preds[name]=np.mean(seed_preds,axis=0)
    predict_grid(target,meta,preds)
    dump(out/'test_prediction_commitment.json',{'status':'PREDICTIONS_FROZEN','prediction_sha256':sha(target),'selection_seal_sha256':sha(out/'selection_seal.json'),'test_labels_accessed':False,'role':'retrospective baseline addition; not a new blind holdout'})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['train','test'],required=True);ap.add_argument('--task');a=ap.parse_args();torch.set_num_threads(protocol()['deep_training']['threads']);torch.set_num_interop_threads(1)
    tasks=[a.task] if a.task else protocol()['tasks']
    for task in tasks:
        if a.phase=='train':
            for name in protocol()['deep_models']:
                for seed in protocol()['seeds']:train_one(task,name,seed)
            dump(LOCAL/'deep'/task/'selection_seal.json',{'status':'SELECTION_FROZEN','protocol_sha256':sha(HERE/'protocol.json'),'code_sha256':code_anchors(),'model_manifests':{f'{n}/{s}':sha(LOCAL/'deep'/task/n/str(s)/'manifest.json') for n in protocol()['deep_models'] for s in protocol()['seeds']},'test_labels_accessed':False})
        else:infer_test(task)

if __name__=='__main__':main()
