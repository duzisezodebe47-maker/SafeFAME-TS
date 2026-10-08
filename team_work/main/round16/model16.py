"""复用项目DLinear，新增轻量、可独立复推的州级训练适配。"""
import random,torch
from torch.utils.data import DataLoader,TensorDataset
from common16 import *
sys.path.insert(0,str(ROOT/'src'))
from train_dlinear import DLinearTarget

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
def build(h):return DLinearTarget(cfg()['input_length'],1,h,cfg()['training']['kernel_size'])
def loader(x,y,seed):return DataLoader(TensorDataset(torch.tensor(x,dtype=torch.float32).unsqueeze(-1),torch.tensor(y,dtype=torch.float32)),batch_size=cfg()['training']['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
def step(model,data,opt):
    model.train();total=0;n=0
    for x,y in data:
        opt.zero_grad(set_to_none=True);loss=((model(x)-y)**2).mean();require(bool(torch.isfinite(loss)),'nonfinite loss');loss.backward();opt.step();total+=float(loss.detach())*len(x);n+=len(x)
    return total/n
@torch.inference_mode()
def predict(model,x):
    model.eval();return np.concatenate([model(torch.tensor(x[i:i+512],dtype=torch.float32).unsqueeze(-1)).numpy() for i in range(0,len(x),512)])
def ridge_fit(x,y,alpha):
    xm=x.mean(axis=0);ym=y.mean(axis=0);z=x-xm;w=np.linalg.solve(z.T@z+alpha*np.eye(x.shape[1]),z.T@(y-ym));return {'xm':xm,'ym':ym,'w':w}
def ridge_predict(m,x):return (x-np.asarray(m['xm']))@np.asarray(m['w'])+np.asarray(m['ym'])
def load_predictions(task,x):
    code,hs=task.split('_h');h=int(hs);folder=LOCAL/'models'/task;seal=read(folder/'selection_seal.json');require(seal['code']==code_anchors(),'model source changed')
    for rel,digest in seal['files'].items():require(sha(folder/rel)==digest,'model bytes changed')
    require(seal['prepared_manifest']==sha(LOCAL/'prepared/MANIFEST.json'),'prepared manifest changed')
    result={'Last':np.repeat(x[:,-1:],h,axis=1),'Ridge':ridge_predict(read(folder/'ridge.json'),x)};seeds=[]
    for seed in cfg()['seeds']:
        m=build(h);m.load_state_dict(torch.load(folder/f'DLinear_{seed}_refit.pt',map_location='cpu',weights_only=True));seeds.append(predict(m,x))
    result['DLinear']=np.mean(seeds,axis=0);return result,seeds,seal
