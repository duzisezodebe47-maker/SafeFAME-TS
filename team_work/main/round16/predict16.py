"""明确指定预注册选型规则的州级数值推理，禁止根据测试分数静默换路。"""
import argparse
from common16 import *
from model16 import load_predictions

def validate(payload):
    require(isinstance(payload,dict) and set(payload)=={'state','horizon','rule','history'},'需要且仅接受state/horizon/rule/history')
    require(payload['state'] in [s['code'] for s in cfg()['states']],'州代码只能是CA/TX/IA');h=payload['horizon'];require(type(h) is int and h in cfg()['horizons'],'步长只能是4或12')
    require(payload['rule'] in ['CalChoice']+cfg()['fixed_rules'],'必须明确指定已经固定的选择规则');history=payload['history'];require(isinstance(history,list) and len(history)==52,'需要52个历史数值')
    require(all(type(v) in (int,float) for v in history),'历史必须是数值');x=np.asarray(history,float);require(np.isfinite(x).all() and ((x>=0)&(x<=500)).all(),'DSCI需位于0到500');return x

def forecast(payload):
    x=validate(payload);task=f'{payload["state"]}_h{payload["horizon"]}';rel=f'{payload["state"]}/scaler.json';manifest=read(LOCAL/'prepared/MANIFEST.json');require(sha(LOCAL/'prepared'/rel)==manifest['files'][rel],'scaler bytes changed');scale=read(LOCAL/'prepared'/rel);pred,_,seal=load_predictions(task,((x-scale['mean'])/scale['std'])[None,:]);name=seal['calibration_choice'] if payload['rule']=='CalChoice' else seal['fixed_choices'][payload['rule']];values=pred[name][0]*scale['std']+scale['mean'];require(np.isfinite(values).all(),'输出非有限值')
    return {'state':payload['state'],'horizon':payload['horizon'],'rule':payload['rule'],'model':name,'forecast_dsci':values.tolist(),'text_weight':0,'out_of_range':bool(((values<0)|(values>500)).any()),'role':cfg()['role']}

if __name__=='__main__':
    import torch
    torch.set_num_threads(2);ap=argparse.ArgumentParser();ap.add_argument('--state',required=True,choices=['CA','TX','IA']);ap.add_argument('--horizon',required=True,type=int,choices=[4,12]);ap.add_argument('--rule',required=True,choices=['CalChoice']+cfg()['fixed_rules']);group=ap.add_mutually_exclusive_group(required=True);group.add_argument('--example',action='store_true');group.add_argument('--history-json',type=Path);args=ap.parse_args()
    if args.example:
        z=np.load(LOCAL/f'prepared/{args.state}_h{args.horizon}/test_features.npz')['x'][-1];scale=read(LOCAL/f'prepared/{args.state}/scaler.json');history=(z*scale['std']+scale['mean']).tolist()
    else:history=read(args.history_json)
    print(json.dumps(forecast({'state':args.state,'horizon':args.horizon,'rule':args.rule,'history':history}),ensure_ascii=False,indent=2))
