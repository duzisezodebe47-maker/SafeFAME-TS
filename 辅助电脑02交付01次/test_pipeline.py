"""No deep training: negative controls plus a reproducible real-data preview."""
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from audit import ROOT, digest, environment, peak_memory, publish_audit, require, run_audit, seal, signature, verify_artifact, write_json
from bundle import (build, fact_reasons, fit_numeric, propose_samples, read_bundle,
                    trace_origin, validate_selected, validate_targets)


def main():
    start = time.perf_counter()
    tests = []

    def rejected(name, fn, expected):
        try:
            fn()
        except ValueError as exc:
            require(expected in str(exc), f'{name}: unexpected rejection: {exc}')
            tests.append(dict(test=name, status='PASS', rejection=str(exc)))
        else:
            raise AssertionError(f'{name}: injected fault was not detected')

    facts = pd.DataFrame(dict(text_id=['past', 'future', 'missing'], source=['report']*3,
        start_date=pd.to_datetime(['2020-01-01', '2020-01-04', None]),
        end_date=pd.to_datetime(['2020-01-01', '2020-01-04', None]),
        source_url=['https://example.org/past']*3, future_language_flag=[False]*3))
    cutoff, history = pd.Timestamp('2020-01-03'), pd.Timestamp('2019-12-01')
    rejected('future_text', lambda: validate_selected(facts.iloc[[1]], cutoff, history, 'proxy', 1), 'Future')
    rejected('missing_text_date', lambda: validate_selected(facts.iloc[[2]], cutoff, history, 'proxy', 1), 'no valid date')
    rejected('target_crosses_boundary', lambda: validate_targets(np.array([9]), np.ones((1, 2)), 0, 10, 2, ['o9']), 'crosses')
    rejected('duplicate_origin_id', lambda: validate_targets(np.array([3, 4]), np.ones((2, 1)), 0, 10, 1, ['x', 'x']), 'Duplicate')
    rejected('nonfinite_target', lambda: validate_targets(np.array([3]), np.array([[np.inf]]), 0, 10, 1, ['x']), 'Nonfinite')
    values = np.arange(20, dtype=float)
    rejected('fit_outside_training', lambda: fit_numeric(values, 10, np.arange(11)), 'training segment')
    p1 = fit_numeric(values, 10, np.arange(10))
    values[10:] = 1e12
    require(p1 == fit_numeric(values, 10, np.arange(10)), 'Test perturbation changed training scale')
    tests.append(dict(test='test_values_cannot_change_training_scale', status='PASS'))
    tmp_root = ROOT / 'data_processed/team_data'
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fault-tests-', dir=tmp_root) as temp:
        folder = Path(temp)
        (folder / 'data.txt').write_text('known output', encoding='utf-8')
        old = seal(folder, dict(rule='end_date<cutoff', revision='pinned', split='A'))
        rejected('stale_rule_cache', lambda: verify_artifact(folder, signature(dict(rule='end_date+7<cutoff', revision='pinned', split='A'))), 'Stale cache')
        (folder / 'data.txt').write_text('corrupted output', encoding='utf-8')
        rejected('corrupt_cached_payload', lambda: verify_artifact(folder, old['signature']), 'content changed')
    reasons, _ = fact_reasons(facts, history, cutoff, 'proxy', 1, 32)
    require(reasons.tolist() == ['selected', 'not_strictly_before_cutoff', 'missing_or_invalid_date'], 'Wrong text exclusions')
    tests.append(dict(test='per_fact_exclusion_reasons', status='PASS'))
    delayed, _ = fact_reasons(facts.iloc[[0]], history, cutoff, 'conservative_lag', 2, 32)
    require(delayed.tolist() == ['lag_not_elapsed'], 'Equality at lag cutoff must be rejected')
    tests.append(dict(test='lag_equality_is_excluded', status='PASS'))
    positive = facts.iloc[[0]].copy()
    retained, _ = fact_reasons(positive, history, cutoff, 'complete_source', 1, 32)
    require(retained.tolist() == ['selected'], 'Complete-source implementation always rejects even valid metadata')
    tests.append(dict(test='complete_source_positive_control', status='PASS'))
    tiny = pd.DataFrame(dict(start_date=pd.date_range('2020-01-01', periods=20),
        end_date=pd.date_range('2020-01-01', periods=20), OT=np.arange(20, dtype=float)))
    tiny.loc[9, 'OT'] = np.nan
    tiny_task = dict(domain='Demo', fold_id='unit', input_len=2, horizon=2, bounds=[8, 12, 16, 20])
    tiny_rows = propose_samples(tiny, tiny_task)
    require(tiny_rows[tiny_rows.origin_index.isin([8, 9])].reason.eq('nonfinite_target').all(), 'Unknown target window survived')
    tests.append(dict(test='unknown_target_excluded_without_imputation', status='PASS'))

    audit_folder, audit_summary = run_audit()
    require(audit_folder is not None, f'Missing real smoke inputs: {audit_summary}')
    require(audit_summary['v4_target_compatible'], 'Raw target audit differs from v4; stop and investigate')
    # Explicitly a software fixture: these bounds/lag are not a master research protocol.
    task = dict(domain='Agriculture', fold_id='smoke', input_len=24, horizon=3, lag_days=31,
                bounds=[350, 370, 400, 420], numerical_sha256=digest(audit_folder / 'Agriculture_numerical.csv'))
    spec = dict(schema_version=1, status='engineering_preview', approved_by=None,
        target_boundary='disjoint_half_open', fit_policy='train_only', numeric_features=['OT'],
        complete_source_rule='original_url_and_valid_interval', max_per_source=32, tasks=[task],
        warning='Synthetic split settings on real data for software checks only; not frozen or selected using model scores')
    spec_path = tmp_root / 'engineering_smoke_spec.json'
    write_json(spec_path, spec)
    rejected('unfrozen_protocol', lambda: build(audit_folder, spec_path), 'master-frozen')
    output, first = build(audit_folder, spec_path, preview=True, threads=1)
    repeat, second = build(audit_folder, spec_path, preview=True, threads=2, replica='threads2')
    require(first['signature'] == second['signature'] and first['files'] == second['files'],
            'Independent 1/2-thread builds differ')
    tests.append(dict(test='real_data_repeat_thread_1_vs_2', status='PASS', signature=first['signature'], files=len(first['files'])))
    _, cached_manifest = build(audit_folder, spec_path, preview=True)
    require(cached_manifest == first, 'Identical-input cache reuse differs')
    tests.append(dict(test='identical_input_verified_cache_reuse', status='PASS'))
    rejected('preview_training_reader', lambda: read_bundle(output, 'Agriculture_h3_fsmoke', 'proxy', first['signature']), 'not a frozen')

    task_dir = output / 'Agriculture_h3_fsmoke'
    samples = pd.read_csv(task_dir / 'samples.csv')
    proposed = pd.read_csv(task_dir / 'sample_audit.csv')
    require(len(samples) == 388 and len(proposed) == 396, 'Real smoke sample count changed')
    require(proposed.reason.eq('target_crosses_boundary').sum() == 8, 'Boundary exclusion count changed')
    coverage = pd.read_csv(output / 'coverage.csv')
    require(coverage[coverage.scenario.eq('complete_source')].covered_origins.eq(0).all(), 'Original URLs unexpectedly available')
    require((coverage.coverage_loss_origins >= 0).all(), 'Stricter scenario increased coverage')
    # Check recomputed proxy vectors/quality against the existing frozen aggregation.
    old_cache = np.load(ROOT / 'data_processed/v4/semantic_features/Agriculture.npz', allow_pickle=False)
    old_positions = {int(o): i for i, o in enumerate(old_cache['origin_index'])}
    semantic = np.load(task_dir / 'proxy/semantic.npy')
    quality = np.load(task_dir / 'proxy/quality.npy')
    positions = [old_positions[int(o)] for o in samples.origin_index]
    old_semantic = np.c_[old_cache['report_embedding'][positions], old_cache['search_embedding'][positions]]
    require(np.allclose(semantic, old_semantic, atol=1e-6, rtol=1e-6), 'Proxy semantics differ from existing v4')
    require(np.allclose(quality, old_cache['quality'][positions], atol=1e-6, rtol=1e-6), 'Proxy quality differs from existing v4')
    tests.append(dict(test='real_proxy_matches_existing_v4', status='PASS', samples=len(samples),
                      max_abs_semantic_difference=float(np.max(np.abs(semantic-old_semantic)))))
    origin_id = samples.origin_id.iloc[-1]
    _, lineage = trace_origin(audit_folder, output, 'Agriculture_h3_fsmoke', origin_id, 'proxy')
    traces = [json.loads(s) for s in (task_dir / 'proxy/text_trace.jsonl').read_text(encoding='utf-8').splitlines()]
    selected = set(traces[-1]['report_ids'] + traces[-1]['search_ids'])
    require(set(lineage.loc[lineage.origin_reason.eq('selected'), 'text_id']) == selected, 'Raw lineage cannot reproduce selected IDs')
    tests.append(dict(test='real_origin_raw_row_trace', status='PASS', origin_id=origin_id,
                      traced_raw_rows=len(lineage), selected=len(selected)))
    report = dict(status='PASS', tests=tests, count=len(tests), audit_folder=str(audit_folder),
        preview_folder=str(output), independent_repeat_folder=str(repeat),
        sample_count=len(samples), cross_boundary_excluded=8, preview_signature=first['signature'],
        coverage=coverage.where(pd.notna(coverage), None).to_dict('records'), environment=environment(),
        elapsed_seconds=time.perf_counter()-start, memory=peak_memory(),
        limitation='Real smoke is Agriculture only; no master-frozen bundle or model performance evaluation')
    evidence = Path(__file__).parent / 'evidence'
    evidence.mkdir(exist_ok=True)
    write_json(evidence / 'tests.json', report)
    coverage.to_csv(evidence / 'smoke_coverage.csv', index=False)
    publish_audit(audit_folder, audit_summary, evidence)
    print(json.dumps(dict(status=report['status'], tests=len(tests), sample_count=len(samples),
                         signature=first['signature'], elapsed_seconds=report['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
