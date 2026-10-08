"""独立复算已封存预测、检查输入/权重/因果反馈证据，生成紧凑验收结果。"""
from __future__ import annotations
import importlib.util
import platform
import subprocess
import joblib
from common import *
from adaptive import causal_replay
from threadpoolctl import threadpool_limits

def grid(path,meta,horizon):
    frame=pd.read_csv(path,float_precision='round_trip');out={}
    for name,g in frame.groupby('candidate'):
        g=g.sort_values(['origin_index','step'])
        require(not g.duplicated(['origin_index','step']).any(),'duplicate prediction')
        require(np.array_equal(g.origin_index,np.repeat(meta.origin_index.to_numpy(),horizon)),'origin alignment')
        require(np.array_equal(g.step,np.tile(np.arange(1,horizon+1),len(meta))),'step alignment')
        require(np.array_equal(g.origin_id.astype(str),np.repeat(meta.origin_id.astype(str).to_numpy(),horizon)),'origin ID alignment')
        values=g.y_pred.to_numpy().reshape(len(meta),horizon);require(np.isfinite(values).all(),'nonfinite prediction');out[name]=values
    return out

def files(folder,mapping):
    for rel,h in mapping.items():require(sha(folder/rel)==h,f'byte mismatch: {folder/rel}')

def verify():
    if not BUNDLE.exists() or not LOCAL.exists() or not OFFICIAL.exists():
        print(json.dumps({'status':'PARTIAL','reason':'Local evidence or official model source absent by design; full verification not executed','required_local_paths':[str(BUNDLE),str(LOCAL),str(OFFICIAL)]}));return
    import deep_baselines as deep
    import experiments as exp
    p=protocol();counts={'deep_runs':0,'comparison_scenarios':0,'rolling_cases':0};rows=[];training=[]
    bundle=verify_bundle(BUNDLE,p['bundle_signature'],SPEC)
    # Reuse the preceding independently implemented isolated-package verifier.
    module_spec=importlib.util.spec_from_file_location('round12_master',ROOT/'team_work/main/round12/run_master.py')
    master=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(master)
    for task in p['tasks']:
        package=LOCAL/'isolated'/task;master.verify_package(package,SPEC,BUNDLE)
        meta,f=features(task,'proxy');y=truth(task);h=y.shape[1];scale=read(BUNDLE/task/'numeric_fit.json')['std']
        folder=LOCAL/'deep'/task;seal=read(folder/'selection_seal.json')
        require(seal['protocol_sha256']==sha(HERE/'protocol.json'),'deep protocol')
        require(seal['code_sha256']==deep.code_anchors(),'deep source changed')
        select_meta,sf=selection(task,'proxy');cal=select_meta.segment.eq('calibration').to_numpy()
        for model in p['deep_models']:
            for seed in p['seeds']:
                sub=folder/model/str(seed);m=read(sub/'manifest.json');require(sha(sub/'manifest.json')==seal['model_manifests'][f'{model}/{seed}'],'deep manifest')
                files(sub,m['files']);require(m['code_sha256']==deep.code_anchors(),'run code')
                require(m['package_manifest_sha256']==sha(package/'MANIFEST.json'),'deep input')
                loss=mse(np.load(sub/'calibration_train_only.npy'),sf.targets_standardized[cal]);require(np.isclose(loss,m['calibration_train_only_mse'],rtol=1e-12),'calibration metric')
                training.append({k:m[k] for k in ('task','model','seed','best_epoch','seconds','parameters','calibration_train_only_mse')});counts['deep_runs']+=1
        commitment=read(folder/'test_prediction_commitment.json');require(commitment['prediction_sha256']==sha(folder/'test_predictions.csv'),'deep predictions changed');require(commitment['selection_seal_sha256']==sha(folder/'selection_seal.json'),'deep seal changed')
        dp=grid(folder/'test_predictions.csv',meta,h)
        for model in p['deep_models']:require(np.allclose(dp[model],np.mean([dp[f'{model}_s{s}'] for s in p['seeds']],axis=0),rtol=1e-6,atol=1e-7),'ensemble inconsistent')
        for name,pred in dp.items():rows.append({'task':task,'scenario':'proxy','candidate':name,'origins':len(y),'horizon':h,'mse':mse(pred,y),'mae':float(abs(pred-y).mean()),'raw_rmse':float(np.sqrt(mse(pred,y))*scale),'kind':'deep ensemble' if name in p['deep_models'] else 'deep individual seed'})
        for scenario in p['scenarios']:
            sub=LOCAL/'comparison'/task/scenario;s=read(sub/'selection_seal.json');files(sub,s['files']);require(s['code_sha256']==exp.code_anchors(),'comparison code changed');require(s['package_manifest_sha256']==sha(package/'MANIFEST.json'),'comparison input')
            c=read(sub/'static_prediction_commitment.json');require(c['prediction_sha256']==sha(sub/'static_test_predictions.csv') and c['selection_seal_sha256']==sha(sub/'selection_seal.json'),'static commitment')
            pred=grid(sub/'static_test_predictions.csv',meta,h);pred.update(grid(sub/'adaptive_test_predictions.csv',meta,h))
            a=read(HERE/'results'/task/scenario/'adaptive_summary.json');require(a['prediction_sha256']==sha(sub/'adaptive_test_predictions.csv'),'adaptive commitment')
            stored=np.load(sub/'selection_state.npz');di=stored['origins'];_,ff=features(task,scenario)
            origins=np.r_[di,ff.origin_index];base=np.r_[stored['base'],pred[s['fallback']]];text=np.r_[stored['text'],pred['N+S+Q+SF']];available=np.r_[stored['available'],ff.text_available & (not s['model_metadata']['N+S+Q+SF']['semantic_untrainable'])]
            all_y=np.r_[stored['truth'],y]
            replay,traces=causal_replay(origins,base,text,available,h,p['adaptive'],lambda i:all_y[i]);require(np.allclose(replay[len(di):],pred['AdaptiveRisk'],rtol=1e-12,atol=1e-12),'causal replay differs')
            trace=pd.read_csv(sub/'adaptive_trace.csv');expected=traces[len(di):]
            require(np.array_equal(trace.weight,[t['weight'] for t in expected]),'trace weights');require(np.array_equal(trace.origin_index,ff.origin_index),'trace origin')
            require(all(t['last_feedback_origin'] is None or t['last_feedback_origin']+h<=t['origin_index'] for t in expected),'future feedback')
            require(np.array_equal(pred['StrictSourceFallback'],pred[s['fallback']]),'strict fallback mismatch')
            reported=pd.read_csv(HERE/'results'/task/scenario/'comparison_metrics.csv',float_precision='round_trip')
            require(set(reported.candidate)==set(pred),'metric coverage')
            for r in reported.to_dict('records'):
                require(np.isclose(r['mse'],mse(pred[r['candidate']],y),rtol=1e-12,atol=1e-14),'reported MSE mismatch');rows.append({**{k:r[k] for k in ('task','scenario','candidate','origins','horizon','mse','mae','raw_rmse')},'kind':'fixed common refit / causal adaptation'})
            counts['comparison_scenarios']+=1
        # Dashboard curve is explicitly a historical test replay, one-step slice.
        frame=pd.read_csv(BUNDLE/task/'samples.csv');test_frame=frame[frame.segment.eq('test')]
        curves={'task':task,'origin_index':meta.origin_index.tolist(),'truth':y[:,0].tolist(),'role':'历史测试回放：仅绘制每个起点的第1步，表格指标涵盖全部预测步'}
        proxy=grid(LOCAL/'comparison'/task/'proxy/static_test_predictions.csv',meta,h);proxy.update(grid(LOCAL/'comparison'/task/'proxy/adaptive_test_predictions.csv',meta,h));proxy.update(dp)
        for name in ('Last','N','StaticGateReplay','AdaptiveRisk','DLinear','PatchTST'):curves[name]=proxy[name][:,0].tolist()
        dump(HERE/'results/demo'/f'{task}.json',curves)
    source=read(HERE/'results/source/source_acceptance.json');require(source['lineage_sha256']==sha(LINEAGE),'lineage changed');require(source['catalog_sha256']==sha(LOCAL/'source_catalog_audited.csv'),'source catalog changed')
    rolling=read(HERE/'results/robustness/rolling_numeric_manifest.json');rm=pd.read_csv(HERE/'results/robustness/rolling_numeric_metrics.csv',float_precision='round_trip')
    for case in rolling['cases']:
        domain,htext,ftext=case['case'].split('_');h=int(htext[1:]);fold=int(ftext[1:]);file=RAW/f'{domain}_numerical.csv';require(sha(file)==case['source_sha256'],'rolling source')
        output=LOCAL/'rolling_numeric'/case['case']/'predictions.npz';require(sha(output)==case['prediction_sha256'],'rolling predictions');z=np.load(output);raw=pd.read_csv(file).OT.to_numpy(float);values=(raw-case['train_mean'])/case['train_std'];y=np.stack([values[i:i+h] for i in z['origins']]);require(z['origins'].min()>=case['bounds'][2] and z['origins'].max()+h<=case['bounds'][3],'rolling test boundary')
        for r in rm[(rm.domain==domain)&(rm.horizon==h)&(rm.fold==fold)].to_dict('records'):require(np.isclose(mse(z[r['candidate']],y),r['mse'],rtol=1e-12),'rolling MSE')
        counts['rolling_cases']+=1
    require(counts=={'deep_runs':24,'comparison_scenarios':8,'rolling_cases':24},'incomplete experiments')
    # Original committed outcomes must retain their exact Git HEAD bytes.
    tracked=subprocess.check_output(['git','ls-files','team_work/main/round8','team_work/main/round9','team_work/main/round10','team_work/main/round11','team_work/main/round12'],cwd=ROOT).decode().splitlines()
    changed=subprocess.check_output(['git','diff','HEAD','--name-only','--','team_work/main/round8','team_work/main/round9','team_work/main/round10','team_work/main/round11','team_work/main/round12'],cwd=ROOT).decode().strip();require(not changed,'historical tracked evidence modified')
    csv(HERE/'results/final/all_metrics.csv',rows);csv(HERE/'results/final/deep_training.csv',training)
    compact=[r for r in rows if r['scenario']=='proxy' and r['candidate'] in ('Last','N','N+S+Q+SF','AR-Ridge','AR-Ridge-refit','StaticGateReplay','AdaptiveRisk','DLinear','PatchTST')]
    dump(HERE/'results/final/dashboard.json',{'metrics':compact,'tasks':p['tasks'],'counts':counts,'source':{'rows':source['audited_rows'],'strict_eligible':source['strict_source_eligible_rows']},'limits':['本轮为已知历史数据上的回顾性扩展，不是新的盲测。','深度与融合模型统一在训练段+校准段重拟合；旧交付部分模型另加入决策段，分数不可直接等同。','自适应门槛为启发式，尚未证明普遍收益或统计安全保证。','24个滚动案例仅为数值模型；深度模型在4个原任务上各运行3个种子。']})
    import torch, sklearn, scipy
    dump(HERE/'results/final/runtime.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'paths':{'runtime':str(DATA/'runtime_torch_cpu'),'evidence':str(LOCAL)},'PatchTST_commit':p['PatchTST']['commit']})
    dump(HERE/'results/final/verification.json',{'status':'PASS_WITH_RESEARCH_LIMITATIONS','counts':counts,'historical_tracked_files_unchanged':len(tracked),'checks':['bundle and isolated package verification','24 training manifests, input/code/weight hashes, calibration metrics','4 deep prediction commitments and seed ensembles','8 static/adaptive commitments, independently recomputed test MSE','8 causal feedback replays and strict fallback equality','24 rolling source hashes, boundary checks and recomputed MSE','original lineage/catalog byte integrity','historical tracked evidence unchanged'],'not_claimed':['new independent blind holdout','verified original publication timing for corpus','adaptive statistical safety guarantee','deep/text training over all 24 rolling cases'],'verifier_sha256':sha(Path(__file__))})
    print(json.dumps(counts));print(pd.DataFrame(compact).pivot(index='task',columns='candidate',values='mse').round(6).to_string())

if __name__=='__main__':
    with threadpool_limits(limits=1):verify()
