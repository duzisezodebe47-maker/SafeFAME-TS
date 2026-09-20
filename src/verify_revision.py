"""Recompute corrected gates, metrics and block intervals from saved origin evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from run_baselines import metrics
from run_safefame_v2 import moving_block_interval, VARIANTS
from validation_boundaries import validation_masks, decision_segments

ROOT = Path(__file__).resolve().parents[1]


def check(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-7, atol=1e-8, equal_nan=True):
        raise AssertionError(f'{label}: {actual} != {expected}')


def verify(replay_models=False):
    folder = ROOT/'outputs/safefame_v3'
    m = pd.read_csv(folder/'safefame_v3_metrics.csv').set_index(['domain','horizon','model'])
    audits = pd.read_csv(folder/'safefame_v3_selection_audit.csv')
    assert len(audits)==18 and len(m)==72
    count = 0
    for row in audits.itertuples(index=False):
        task = folder/'tasks'/f'{row.domain}_h{row.horizon}'
        detail = json.loads((task/'audit.json').read_text(encoding='utf-8'))
        frozen = json.loads((task/'frozen_selection.json').read_text(encoding='utf-8'))
        z = np.load(task/'predictions.npz')
        origins = np.arange(detail['train_end'], detail['validation_end']-row.horizon+1)
        cal, dec, cut = validation_masks(origins,row.horizon,detail['train_end'],detail['validation_end'])
        assert np.array_equal(origins[cal],z['calibration_origins'])
        assert np.array_equal(origins[dec],z['decision_origins'])
        masks = decision_segments(origins[dec],row.horizon,cut,detail['validation_end'])[:2]
        assert np.array_equal(masks[0],z['segment1_mask']) and np.array_equal(masks[1],z['segment2_mask'])
        scores = {name:metrics(z['calibration_actual'],z[f'{name}_calibration'])['mse']
                  for name in frozen['numeric_calibration_scores']}
        for name in scores: check(scores[name],frozen['numeric_calibration_scores'][name],'numeric calibration')
        assert min(scores,key=scores.get)==row.numeric_fallback==frozen['fallback']
        check(len(z['calibration_origins']),row.calibration_windows,'calibration count')
        check(len(z['decision_origins']),row.decision_windows,'decision count')
        fallback_dec = z[f'{row.numeric_fallback}_decision']
        eligible = []
        for v in VARIANTS:
            null = z[f'{v}_permutation_mse']
            assert null.shape==(99,) and np.isfinite(null).all()
            score = metrics(z['decision_actual'],z[f'{v}_decision'])['mse']
            p = (1+np.sum(null<=score))/100
            segment_wins = all(mask.any() and metrics(z['decision_actual'][mask],z[f'{v}_decision'][mask])['mse'] <
                               metrics(z['decision_actual'][mask],fallback_dec[mask])['mse'] for mask in masks)
            gate = score<metrics(z['decision_actual'],fallback_dec)['mse'] and segment_wins and p<=.025
            check(p,getattr(row,f'{v}_permutation_p_value'),'99 permutation p')
            check(score,getattr(row,f'{v}_decision_mse'),'decision MSE')
            assert bool(segment_wins)==getattr(row,f'{v}_segment_wins')
            assert bool(gate)==getattr(row,f'{v}_eligible')==frozen['candidates'][v]['eligible']
            if gate: eligible.append(v)
        selected = min(eligible,key=lambda v:metrics(z['decision_actual'],z[f'{v}_decision'])['mse']) if eligible else 'numeric_fallback'
        assert selected==row.selected_path==frozen['selected']
        check(z['selected_test'],z['fallback_test'] if selected=='numeric_fallback' else z[f'{selected}_test'],'route forecast')
        for name,key in [(row.numeric_fallback,'fallback_test'),('semantic_residual','semantic_residual_test'),
                         ('frequency_residual','frequency_residual_test'),('SafeFAME-TS-v3','selected_test')]:
            expect=m.loc[row.domain,row.horizon,name]
            for k,value in metrics(z['test_actual'],z[key]).items():check(value,expect[k],k)
            delta,low,high=moving_block_interval(z['test_actual'],z['fallback_test'],z[key],detail['block_length'])
            for value,k in zip((delta,low,high),('loss_difference_vs_fallback','block_ci_low','block_ci_high')):check(value,expect[k],k)
        if replay_models and row.numeric_fallback in ('PatchTST','DLinear-M'):
            import torch
            from train_dlinear import load_domain, make_windows, predict, DLinearTarget
            from train_patchtst import PatchTSTTarget
            p=ROOT/'references/external/Time-MMD/numerical'/row.domain/f'{row.domain}.csv'
            values,_,target,_=load_domain(p,detail['train_end'],'multivariate' if row.numeric_fallback=='DLinear-M' else 'univariate')
            forecasts=[]
            for seed in (2026,2027,2028):
                payload=torch.load(task/'models'/f'{row.numeric_fallback}_refit_s{seed}.pt',map_location='cpu',weights_only=True)
                x,y,_=make_windows(values,target,payload['model_args']['input_len'],row.horizon,z['test_origins'])
                cls=DLinearTarget if row.numeric_fallback=='DLinear-M' else PatchTSTTarget
                model=cls(**payload['model_args']);model.load_state_dict(payload['state_dict'])
                forecasts.append(predict(model,x,torch.device('cpu'),64))
            # GPU training / CPU inference allows float32 kernel roundoff.
            assert np.allclose(np.mean(forecasts,axis=0),z['fallback_test'],rtol=2e-5,atol=2e-5)
        count+=1
        print(f'verified v3 {row.domain} H{row.horizon}',flush=True)
    for name in ('reviewer_sensitivity_corrected','reviewer_sensitivity_v3'):
        base=ROOT/'outputs'/name
        s=pd.read_csv(base/'reviewer_sensitivity_results.csv')
        assert len(s)==36
        for row in s.itertuples(index=False):
            z=np.load(base/'evidence'/f'{row.domain}_h{row.horizon}_{row.variant}.npz')
            aligned=metrics(z['decision_actual'],z['aligned_decision'])['mse']
            p=(1+np.sum(z['shifted_mse']<=aligned))/1000
            check(p,row.circular_shift_p_value,'999 shift p')
            assert len(z['shifted_mse'])==999
            candidate=metrics(z['test_actual'],z['test_candidate'])['mse']
            control=metrics(z['test_actual'],z['test_control'])['mse']
            check(candidate,row.test_candidate_mse,'candidate MSE')
            check(control,row.test_matched_control_mse,'control MSE')
            delta,lo,hi=moving_block_interval(z['test_actual'],z['test_control'],z['test_candidate'],
                row.block_length,seed=int(z['bootstrap_seed']))
            check([100*lo/control,100*hi/control],[row.test_gain_ci_low_pct,row.test_gain_ci_high_pct],'matched CI')
        summary=json.loads((base/'reviewer_sensitivity_summary.json').read_text(encoding='utf-8'))
        assert summary['aligned_solver']==summary['shifted_solver']=='cholesky'
        assert int((s.circular_shift_p_value<=.025).sum())==summary['circular_shift_passes_at_0_025']
    manifest=json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
    causal={'run_safefame_v2.py','validation_boundaries.py','train_dlinear.py','train_patchtst.py','data_utils.py','run_baselines.py','run_famets_selective.py'}
    for name,digest in manifest['sha256'].items():
        p=Path(name)
        if p.suffix=='.npz' or p.suffix=='.csv' or p.name in causal:
            assert hashlib.sha256(p.read_bytes()).hexdigest()==digest, f'Experiment source changed: {p}'
    return {'status':'PASS','v3_tasks':count,'v3_metrics':72,'sensitivity_paths':72,
            'bootstrap_repeats':5000,'checkpoint_replay':replay_models,
            'scope':'saved origin predictions, target boundaries, all expert calibration scores, 99/999 null distributions, metrics, block intervals and selected deep checkpoints',
            'caveat':'does not certify unseen test data or true text publication timestamps'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--record',action='store_true');parser.add_argument('--replay-models',action='store_true')
    args=parser.parse_args()
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    with threadpool_limits(1): result=verify(args.replay_models)
    if args.record:
        paths=[]
        for folder in ('safefame_v3','reviewer_sensitivity_corrected','reviewer_sensitivity_v3'):
            paths += [p for p in (ROOT/'outputs'/folder).rglob('*') if p.is_file() and p.name!='verification.json']
        result['sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        (ROOT/'outputs/safefame_v3/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='sha256'},ensure_ascii=False,indent=2))
