"""Round-two negative controls and four-task interface checks."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FIRST = ROOT / '辅助电脑02交付01次'
sys.path[:0] = [str(HERE), str(FIRST)]
from audit import require, write_json
from bundle import fit_numeric, validate_selected, validate_targets
from round2_pipeline import SCENARIOS, validate_contract_rows, verify_local


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    tests = []

    def reject(name, operation, text):
        try:
            operation()
        except ValueError as exc:
            require(text in str(exc), f'{name}: wrong rejection: {exc}')
            tests.append(dict(test=name, status='PASS', rejection=str(exc)))
        else:
            raise AssertionError(f'{name}: injected fault was accepted')

    verify_local(bundle, manifest['bundle_signature'])
    tests.append(dict(test='full_bundle_hash_inventory', status='PASS', files=len(manifest['files'])))
    contract = rows(bundle / 'samples.jsonl')
    validate_contract_rows(contract)
    tests.append(dict(test='contract_identity_and_targets', status='PASS', rows=len(contract)))
    bad = [dict(contract[0]), *contract[1:]]
    bad[0]['origin_id'] += '-wrong'
    reject('misaligned_origin_id', lambda: validate_contract_rows(bad), 'Misaligned origin ID')
    facts = pd.DataFrame(dict(text_id=['late'], source=['report'],
        start_date=pd.to_datetime(['2020-01-04']), end_date=pd.to_datetime(['2020-01-04']),
        source_url=[''], future_language_flag=[False]))
    reject('late_text', lambda: validate_selected(facts, pd.Timestamp('2020-01-03'),
        pd.Timestamp('2019-01-01'), 'proxy', 1), 'Future')
    reject('target_crossing', lambda: validate_targets(np.array([9]), np.ones((1, 2)),
        0, 10, 2, ['x']), 'crosses')
    reject('training_outside_fit', lambda: fit_numeric(np.arange(20.), 10, np.arange(11)), 'training segment')
    reject('unknown_target', lambda: validate_targets(np.array([3]), np.array([[np.nan]]),
        0, 10, 1, ['x']), 'Nonfinite')
    reject('stale_bundle_signature', lambda: verify_local(bundle, '0' * 64), 'signature mismatch')
    coverage = pd.read_csv(bundle / 'coverage.csv')
    require((coverage.coverage_loss_vs_proxy >= 0).all(), 'Stricter scenario increased coverage')
    tests.append(dict(test='conservative_coverage_nonincrease', status='PASS'))
    require(not list(bundle.glob('*/complete_source')), 'complete_source training arrays were emitted')
    require(coverage[coverage.scenario.eq('complete_source_audit_only')].text_available_origins.eq(0).all(),
            'complete_source audit should be empty without original URLs')
    tests.append(dict(test='complete_source_audit_only', status='PASS'))
    for task in sorted(p for p in bundle.iterdir() if p.is_dir()):
        n = np.load(task / 'numeric_history.npy', allow_pickle=False)
        y = np.load(task / 'targets_standardized.npy', allow_pickle=False)
        for scenario in SCENARIOS:
            require(len(np.load(task / scenario / 'semantic.npy', allow_pickle=False)) == len(n) == len(y),
                    f'{task.name}: scenario sample order/count differs')
        tests.append(dict(test=f'{task.name}_scenario_shared_numeric_targets', status='PASS', rows=len(n)))
    social = bundle / 'SocialGood_h3_f1'
    samples = pd.read_csv(social / 'samples.csv')
    train = samples.segment.eq('train').to_numpy()
    for scenario in SCENARIOS:
        semantic = np.load(social / scenario / 'semantic.npy', allow_pickle=False)
        available = np.load(social / scenario / 'text_available.npy', allow_pickle=False)
        require(not available[train].any(), f'SocialGood train unexpectedly has text in {scenario}')
        require(np.array_equal(semantic[train], np.zeros_like(semantic[train])),
                f'SocialGood no-text semantic is not exactly zero in {scenario}')
    tests.append(dict(test='socialgood_train_safe_no_text_signal', status='PASS', train_rows=int(train.sum())))
    report = dict(status='PASS', count=len(tests), tests=tests,
        bundle_signature=manifest['bundle_signature'], elapsed_seconds=time.perf_counter()-start,
        limitation='Engineering preview checks only; task hashes still await master freeze.')
    write_json(HERE / 'evidence/tests_round2.json', report)
    print(json.dumps(dict(status='PASS', tests=len(tests), contract_rows=len(contract),
        bundle_signature=manifest['bundle_signature']), indent=2))


if __name__ == '__main__':
    main()
