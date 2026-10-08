"""多时段数值选择的严格回退入口；明确保留探索性标签。"""
import argparse
from common15 import *
from predictor14 import validate,example
from evaluate_test import load_trusted
from train_models import numeric_predict

def forecast(payload):
    h,raw=validate(payload)
    registration=read(LOCAL/'input_registration.json')
    for rel,digest in registration['files'].items():require(sha(PROJECT/rel)==digest,'Round15 source/input changed')
    folder=LOCAL/f'h{h}';manifest=read(LOCAL/'MANIFEST.json')['files']
    require(sha(folder/'policy_choices.json')==manifest[f'h{h}/policy_choices.json'],'policy changed')
    choice=read(folder/'policy_choices.json')['fixed_numeric']['three_block_minimax_relative_regret'];objects,_=load_trusted(h);scale=objects['scaler'];x=((raw-scale['mean'])/scale['std'])[None,:]
    pred=numeric_predict(choice,objects['numeric'][choice],x,h)[0]*scale['std']+scale['mean']
    require(np.isfinite(pred).all(),'nonfinite forecast')
    return {'horizon':h,'model':choice,'selection':'three_block_minimax_relative_regret','forecast_dsci':pred.tolist(),'text_weight':0,'out_of_range':bool(((pred<0)|(pred>500)).any()),'role':config()['role'],'source_limit':'首次公开时间证据仍不充分，严格模式不启用文本'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--horizon',type=int,choices=(4,12),required=True);parser.add_argument('--history-json',type=Path);parser.add_argument('--example',action='store_true');args=parser.parse_args()
    require(bool(args.history_json)^args.example,'Specify exactly one of --history-json / --example')
    history=read(args.history_json) if args.history_json else example(args.horizon)['history']
    print(json.dumps(forecast({'horizon':args.horizon,'history':history}),ensure_ascii=False,indent=2))
