"""独立核对校准分数究竟来自训练拟合还是训练+校准重拟合。仅访问非测试标签。"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'model_engine'))
from branches import FeatureBundle
from candidates import BranchResidualCandidate
from permutation import _fit
from numeric_fallbacks import numeric_baselines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    spec = json.loads(a.spec.read_text(encoding='utf-8'))
    rows = []
    with threadpool_limits(limits=1):
        for t in spec['tasks']:
            task = f"{t['domain']}_h{t['horizon']}_f{t['fold_id']}"
            td = a.bundle / task
            meta = pd.read_csv(td / 'samples.csv')
            parts = []
            for segment in ('train', 'calibration'):
                mask = meta.segment.eq(segment).to_numpy()
                def read(name):
                    return np.asarray(np.load(td / name, mmap_mode='r')[mask])
                parts.append(FeatureBundle(read('numeric_history.npy'), read('proxy/semantic.npy'),
                    read('proxy/quality.npy'), read('proxy/text_available.npy'), read('origin_index.npy'),
                    read('targets_raw.npy'), read('targets_standardized.npy')))
            train, cal = parts
            model = BranchResidualCandidate('N')
            model.fit_design(train)
            model.select_alphas(train, cal)
            model.refit(train)
            heldout = float(np.mean((model.predict(cal) - cal.targets_standardized)**2))
            legacy = _fit('N', train, cal)
            reused = float(np.mean((legacy.predict(cal) - cal.targets_standardized)**2))
            x = np.r_[train.numeric_history, cal.numeric_history]
            infer = FeatureBundle(x, np.zeros((len(x),1)), np.zeros((len(x),1)), np.zeros(len(x),bool), np.arange(len(x)))
            y = np.r_[train.targets_standardized, cal.targets_standardized]
            segs = np.r_[np.repeat('train',len(train.numeric_history)),np.repeat('calibration',len(cal.numeric_history))]
            bases = numeric_baselines(infer,segs,y,seasonal_period=spec['seasonal_periods'][t['domain']])
            scores = {c: float(np.mean((p[-len(cal.numeric_history):]-cal.targets_standardized)**2)) for c,p in bases['predictions'].items()}
            correct = dict(scores, N=heldout)
            old = dict(scores, N=reused)
            rows.append({'task':task,'N_cal_train_only':heldout,'N_cal_refit_including_cal':reused,
                'correct_fallback':min(correct,key=lambda c:(correct[c],c)),
                'legacy_fallback':min(old,key=lambda c:(old[c],c)),
                'calibration_scores_train_only':correct,'alpha':model.alpha_by_group,
                'issue':'Legacy calibration predictions include their own labels in refit. Use train-only predictions for fallback selection.'})
    a.out.write_text(json.dumps({'status':'ISSUE_CONFIRMED','test_truth_read':False,'rows':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False))

if __name__=='__main__': main()
