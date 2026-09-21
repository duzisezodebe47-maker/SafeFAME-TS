"""Build the four-task pre-freeze handoff without training or route selection."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FIRST = ROOT / '辅助电脑02交付01次'
sys.path.insert(0, str(FIRST))
from audit import digest, load_vectors, require, signature, verify_artifact, write_json
from bundle import fill_numeric, fit_numeric, load_canonical, propose_samples
from build_semantic_features import normalized_weighted_mean
from run_famets_selective import frequency_statistics

MASTER_BRANCH = 'origin/team/main-eval-20260921'
MASTER_SPLIT = 'team_work/main/protocol/split_spec.json'
MASTER_CONTRACT = 'team_work/main/protocol/prediction_contract.json'
BASELINE_COMMIT = 'c63793236d532d2ffdcccd4c7cb0c8e194f96eab'
TASKS = (
    ('Agriculture', 12, 1),
    ('Climate', 4, 2),
    ('SocialGood', 3, 1),
    ('Environment', 7, 2),
)
FREQUENCY_LAG = {'daily': 1, 'weekly': 7, 'monthly': 31}
SEGMENTS = ('train', 'calibration', 'decision', 'test')
SCENARIOS = ('proxy', 'conservative_lag')


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True,
                                   encoding='utf-8').strip()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n'
                            for r in rows), encoding='utf-8')


def latest_audit():
    pointer = FIRST / 'evidence/audit_summary.json'
    if pointer.is_file():
        candidate = Path(json.loads(pointer.read_text(encoding='utf-8'))['output'])
        if candidate.is_dir():
            verify_artifact(candidate)
            return candidate
    valid = []
    for candidate in (ROOT / 'data_processed/team_data').glob('audit-*'):
        try:
            verify_artifact(candidate)
            summary = json.loads((candidate / 'summary.json').read_text(encoding='utf-8'))
            if summary.get('v4_target_compatible'):
                valid.append(candidate)
        except (ValueError, OSError, json.JSONDecodeError):
            pass
    require(valid, 'No complete first-round audit is available')
    return max(valid, key=lambda p: p.stat().st_mtime_ns)


def master_protocol():
    split_text = git('show', f'{MASTER_BRANCH}:{MASTER_SPLIT}')
    contract_text = git('show', f'{MASTER_BRANCH}:{MASTER_CONTRACT}')
    split = json.loads(split_text)
    contract = json.loads(contract_text)
    require(split['basis'].endswith(BASELINE_COMMIT), 'Unexpected master baseline commit')
    require(split['status'].startswith('FROZEN_FOR_INTERFACE'), 'Master interface is not frozen')
    return split, contract, split_text + '\n', contract_text + '\n', git('rev-parse', MASTER_BRANCH)


def task_id(domain, horizon, fold):
    return f'{domain}_h{horizon}_f{fold}'


def prepare_request(audit_folder):
    config = json.loads((ROOT / 'configs/safefame_v4.json').read_text(encoding='utf-8'))
    split, contract, split_text, contract_text, master_commit = master_protocol()
    require(split['fold_boundaries'] == config['folds'], 'Master/config fold ratios differ')
    evidence = HERE / 'evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'master_split_spec_reference.json').write_text(split_text, encoding='utf-8')
    (evidence / 'master_prediction_contract_reference.json').write_text(contract_text, encoding='utf-8')
    rows, tasks = [], []
    for domain, horizon, fold in TASKS:
        settings = config['domains'][domain]
        path = audit_folder / f'{domain}_numerical.csv'
        frame = pd.read_csv(path)
        ratios = config['folds'][fold - 1]
        bounds = [math.floor(len(frame) * value) for value in ratios]
        require(bounds == sorted(set(bounds)) and len(bounds) == 4, 'Degenerate split bounds')
        unknown = int((~np.isfinite(pd.to_numeric(frame.OT, errors='coerce'))).sum())
        task = dict(domain=domain, fold_id=fold, input_len=settings['input_len'], horizon=horizon,
                    lag_days=FREQUENCY_LAG[settings['frequency']], bounds=bounds,
                    numerical_sha256=digest(path))
        audit = propose_samples(frame, task)
        rows.append(dict(task_id=task_id(domain, horizon, fold), domain=domain, fold_id=fold,
            n_rows=len(frame), input_len=settings['input_len'], horizon=horizon,
            train_end=bounds[0], calibration_end=bounds[1], decision_end=bounds[2], test_end=bounds[3],
            frequency=settings['frequency'], lag_days=task['lag_days'], unknown_target_rows=unknown,
            numerical_path=f'data_processed/team_data/{audit_folder.name}/{domain}_numerical.csv',
            numerical_sha256=task['numerical_sha256'], candidate_origins=len(audit),
            expected_retained_origins=int(audit.reason.eq('retained').sum()),
            status='AWAITING_MASTER_HASH_FREEZE'))
        tasks.append(task)
    registry = pd.DataFrame(rows)
    registry.to_csv(evidence / 'task_registry.csv', index=False)
    candidate = dict(schema_version=1, status='engineering_preview', approved_by=None,
        target_boundary='disjoint_half_open', fit_policy='train_only', numeric_features=['OT'],
        complete_source_rule='original_url_and_valid_interval', max_per_source=32, tasks=tasks,
        protocol_version=split['protocol_version'], master_protocol_commit=master_commit,
        master_split_spec_sha256=__import__('hashlib').sha256(split_text.encode()).hexdigest(),
        baseline_commit=BASELINE_COMMIT,
        warning='Freeze request only. Master must confirm all hashes and change status/approved_by before formal build.')
    write_json(evidence / 'split_spec_freeze_request.json', candidate)
    return candidate, registry, split, contract, master_commit


def source_views(facts):
    result = {}
    for source in ('report', 'search'):
        part = facts[facts.source.eq(source)].sort_values(
            ['end_date', 'text_id'], ascending=[True, False]).reset_index(drop=True)
        result[source] = (part, part.end_date.astype('int64').to_numpy())
    return result


def selected_at(views, vectors, vector_rows, history, cutoff, lag_days, scenario, maximum=32):
    embeddings, ids, ages_all, fractions, counts = [], {}, [], [], {}
    effective = cutoff - pd.Timedelta(days=lag_days) if scenario == 'conservative_lag' else cutoff
    for source in ('report', 'search'):
        part, ends = views[source]
        left = int(np.searchsorted(ends, history.value, side='left'))
        right = int(np.searchsorted(ends, effective.value, side='left'))
        available = part.iloc[left:right]
        selected = available.iloc[::-1].head(maximum)
        selected_ids = selected.text_id.tolist()
        ids[source] = selected_ids
        counts[source] = len(available)
        age = (cutoff - selected.end_date).dt.total_seconds().to_numpy() / 86400.
        ages_all.extend(age.tolist())
        half_life = max(float((cutoff - history).days) / 4., 1.)
        if selected_ids:
            rows = [vector_rows[i] for i in selected_ids]
            emb = normalized_weighted_mean(vectors[rows], np.exp(-np.log(2.) * age / half_life), 384)
            fraction = float(selected.future_language_flag.astype(bool).mean())
        else:
            emb, fraction = np.zeros(384, dtype=np.float32), 0.
        embeddings.append(emb)
        fractions.append(fraction)
    history_days = max(float((cutoff - history).days), 1.)
    quality = np.asarray([np.log1p(counts['report']), np.log1p(counts['search']),
        np.log1p(len(ids['report'])), np.log1p(len(ids['search'])),
        float(not ids['report']), float(not ids['search']),
        min(ages_all) / history_days if ages_all else 1.,
        np.mean(ages_all) / history_days if ages_all else 1., *fractions], dtype=np.float32)
    reason_counts = dict(total_facts=sum(len(v[0]) for v in views.values()),
        selected=sum(len(v) for v in ids.values()), available=sum(counts.values()),
        source_cap_exceeded=sum(max(0, v - maximum) for v in counts.values()))
    reason_counts['outside_lookback_or_cutoff_or_lag'] = reason_counts['total_facts'] - reason_counts['available']
    return np.concatenate(embeddings).astype(np.float32), quality, ids, reason_counts


def save_array(path, value):
    np.save(path, np.asarray(value), allow_pickle=False)


def validate_frozen_spec(spec, registry):
    require(spec.get('schema_version') == 1, 'Unsupported frozen spec schema')
    require(spec.get('status') == 'frozen' and bool(spec.get('approved_by')),
            'Formal build requires master status=frozen and approved_by')
    expected = registry.set_index('task_id')
    require(len(spec.get('tasks', [])) == len(expected), 'Frozen task count changed')
    for task in spec['tasks']:
        tid = task_id(task['domain'], int(task['horizon']), int(task['fold_id']))
        require(tid in expected.index, f'Unexpected frozen task: {tid}')
        row = expected.loc[tid]
        require(task['bounds'] == [int(row.train_end), int(row.calibration_end),
                                  int(row.decision_end), int(row.test_end)], f'{tid}: frozen bounds changed')
        require(task['numerical_sha256'] == row.numerical_sha256, f'{tid}: frozen snapshot hash changed')
        require(int(task['lag_days']) == int(row.lag_days), f'{tid}: frozen lag changed')


def build_bundle(audit_folder, candidate, registry, formal=False):
    started = time.perf_counter()
    corpus = load_canonical(audit_folder)
    vectors = load_vectors(corpus, ROOT / 'data_processed/v4/embeddings')
    vector_rows = dict(zip(corpus.text_id, range(len(corpus))))
    code = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(
        list(HERE.glob('*.py')) + list(FIRST.glob('*.py')))}
    mode = 'frozen' if formal else 'engineering_preview_awaiting_master_hash_freeze'
    inputs = dict(audit_manifest_sha256=digest(audit_folder / 'manifest.json'), split_spec=candidate,
                  code=code, mode=mode)
    bundle_sig = signature(inputs)
    output = ROOT / 'data_processed/team_data' / (('round2-bundle-' if formal else 'round2-preview-') + bundle_sig[:20])
    if output.exists():
        verify_artifact(output, bundle_sig)
        return output, json.loads((output / 'manifest.json').read_text(encoding='utf-8')), time.perf_counter()-started
    output.mkdir(parents=True)
    coverage_rows, contract_rows, audit_rows = [], [], []
    for task in candidate['tasks']:
        domain, fold, horizon = task['domain'], int(task['fold_id']), task['horizon']
        tid = task_id(domain, horizon, fold)
        task_dir = output / tid
        task_dir.mkdir()
        frame = pd.read_csv(audit_folder / f'{domain}_numerical.csv')
        raw = frame.OT.to_numpy(float)
        params = fit_numeric(raw, task['bounds'][0], np.arange(task['bounds'][0]))
        filled = fill_numeric(raw, params)
        audit = propose_samples(frame, task)
        audit.insert(0, 'task_id', tid)
        audit.insert(1, 'fold_id', fold)
        audit.insert(2, 'horizon', horizon)
        audit.to_csv(task_dir / 'sample_audit.csv', index=False)
        audit_rows.append(audit)
        valid = audit[audit.reason.eq('retained')].reset_index(drop=True)
        valid['origin_index'] = valid.origin_index.astype(int)
        valid['task_id'] = tid
        valid['fold_id'] = fold
        valid['horizon'] = horizon
        valid.to_csv(task_dir / 'samples.csv', index=False)
        origins = valid.origin_index.to_numpy(int)
        length = task['input_len']
        x_raw = np.stack([raw[o-length:o] for o in origins])
        x = (np.stack([filled[o-length:o] for o in origins]) - params['mean']) / params['std']
        y_raw = np.stack([raw[o:o+horizon] for o in origins])
        y = (y_raw - params['mean']) / params['std']
        require(np.isfinite(y).all(), 'Unknown target entered retained samples')
        common = dict(numeric_history=x.astype(np.float32), numeric_history_raw=x_raw.astype(np.float32),
            numeric_missing=(~np.isfinite(x_raw)), targets_raw=y_raw.astype(np.float32),
            targets_standardized=y.astype(np.float32), frequency=frequency_statistics(x).astype(np.float32),
            origin_index=origins.astype(np.int64),
            target_time=np.stack([frame.start_date.iloc[o:o+horizon].to_numpy(dtype='U32') for o in origins]))
        for name, value in common.items():
            save_array(task_dir / f'{name}.npy', value)
        write_json(task_dir / 'numeric_fit.json', params)
        facts = corpus[corpus.domain.eq(domain)].copy()
        views = source_views(facts)
        scenario_traces = {}
        for scenario in SCENARIOS:
            scenario_dir = task_dir / scenario
            scenario_dir.mkdir()
            semantic, quality, traces = [], [], []
            for row in valid.itertuples():
                emb, q, ids, reasons = selected_at(views, vectors, vector_rows,
                    pd.Timestamp(row.history_start), pd.Timestamp(row.cutoff_time),
                    task['lag_days'], scenario, candidate['max_per_source'])
                semantic.append(emb); quality.append(q)
                traces.append(dict(origin_id=row.origin_id, origin_index=int(row.origin_index),
                    report_ids=ids['report'], search_ids=ids['search'], reason_counts=reasons,
                    publication_time_status='end_date_proxy_unverified'))
            semantic, quality = np.asarray(semantic), np.asarray(quality)
            available = quality[:, 4:6].eq(0) if isinstance(quality, pd.DataFrame) else quality[:, 4:6] == 0
            save_array(scenario_dir / 'semantic.npy', semantic)
            save_array(scenario_dir / 'quality.npy', quality)
            save_array(scenario_dir / 'source_available.npy', available)
            save_array(scenario_dir / 'text_available.npy', available.any(axis=1))
            write_jsonl(scenario_dir / 'text_trace.jsonl', traces)
            scenario_traces[scenario] = traces
            for segment, group in valid.groupby('segment', sort=False):
                idx = group.index.to_numpy()
                selected_count = sum(traces[i]['reason_counts']['selected'] for i in idx)
                coverage_rows.append(dict(task_id=tid, fold_id=fold, scenario=scenario, segment=segment,
                    candidate_origins=int(audit.segment.eq(segment).sum()), retained_origins=len(idx),
                    text_available_origins=int(available[idx].any(axis=1).sum()),
                    selected_text_occurrences=int(selected_count),
                    unknown_target_excluded=int((audit.segment.eq(segment) & audit.reason.eq('nonfinite_target')).sum()),
                    cross_boundary_excluded=int((audit.segment.eq(segment) & audit.reason.str.contains('crosses_boundary')).sum())))
        # No per-origin arrays for complete_source: current raw files contain zero original URLs.
        for segment, group in valid.groupby('segment', sort=False):
            coverage_rows.append(dict(task_id=tid, fold_id=fold, scenario='complete_source_audit_only', segment=segment,
                candidate_origins=int(audit.segment.eq(segment).sum()), retained_origins=len(group),
                text_available_origins=0, selected_text_occurrences=0,
                unknown_target_excluded=int((audit.segment.eq(segment) & audit.reason.eq('nonfinite_target')).sum()),
                cross_boundary_excluded=int((audit.segment.eq(segment) & audit.reason.str.contains('crosses_boundary')).sum())))
        for i, row in valid.iterrows():
            contract_rows.append(dict(task_id=tid, fold_id=fold, origin_id=row.origin_id,
                origin_index=int(row.origin_index), segment=row.segment, horizon=horizon,
                cutoff_time=row.cutoff_time, target_time=common['target_time'][i].tolist(),
                target=y[i].astype(float).tolist(), target_raw=y_raw[i].astype(float).tolist(),
                cutoff_index=int(row.origin_index)-1, target_start_index=int(row.origin_index),
                snapshot_sha256=task['numerical_sha256'], bundle_signature=bundle_sig,
                preprocessing_fit_end=task['bounds'][0]))
        # Exact proxy compatibility against historical v4 feature cache.
        old = np.load(ROOT / f'data_processed/v4/semantic_features/{domain}.npz', allow_pickle=False)
        old_pos = {int(o): i for i, o in enumerate(old['origin_index'])}
        positions = [old_pos[int(o)] for o in origins]
        proxy_sem = np.load(task_dir / 'proxy/semantic.npy')
        expected = np.c_[old['report_embedding'][positions], old['search_embedding'][positions]]
        require(np.allclose(proxy_sem, expected, atol=1e-6, rtol=1e-6), f'{tid}: proxy semantics differ from v4')
        require(np.allclose(np.load(task_dir / 'proxy/quality.npy'), old['quality'][positions], atol=1e-6, rtol=1e-6),
                f'{tid}: proxy quality differs from v4')
    coverage = pd.DataFrame(coverage_rows)
    proxy = coverage[coverage.scenario.eq('proxy')].set_index(['task_id', 'segment']).text_available_origins
    coverage['proxy_text_available_origins'] = [int(proxy.loc[(r.task_id, r.segment)]) for r in coverage.itertuples()]
    coverage['coverage_loss_vs_proxy'] = coverage.proxy_text_available_origins - coverage.text_available_origins
    require((coverage.coverage_loss_vs_proxy >= 0).all(), 'Stricter scenario increased text coverage')
    coverage.to_csv(output / 'coverage.csv', index=False)
    pd.concat(audit_rows, ignore_index=True).to_csv(output / 'sample_audit.csv', index=False)
    write_jsonl(output / 'samples.jsonl', contract_rows)
    write_json(output / 'schema.json', dict(schema_version='team-data-2.0', scenarios=list(SCENARIOS),
        complete_source='audit only; no prediction arrays', semantic_dim=768, quality_dim=10,
        frequency_dim=10, numeric_interface='OT univariate', target_scale='train-only standardized plus raw truth',
        status='FROZEN' if formal else 'AWAITING_MASTER_HASH_FREEZE'))
    manifest = seal_local(output, inputs, bundle_sig)
    return output, manifest, time.perf_counter()-started


def seal_local(folder, inputs, bundle_sig):
    files = {}
    for path in sorted(p for p in folder.rglob('*') if p.is_file() and p.name != 'manifest.json'):
        files[path.relative_to(folder).as_posix()] = digest(path)
    manifest = dict(bundle_signature=bundle_sig, signature=bundle_sig, inputs=inputs, files=files)
    write_json(folder / 'manifest.json', manifest)
    return manifest


def verify_local(folder, expected):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    require(manifest['bundle_signature'] == expected, 'Bundle signature mismatch')
    actual = {p.relative_to(folder).as_posix(): digest(p) for p in sorted(
        p for p in folder.rglob('*') if p.is_file() and p.name != 'manifest.json')}
    require(actual == manifest['files'], 'Bundle file inventory/hash mismatch')
    return manifest


def validate_contract_rows(rows):
    required = {'task_id', 'fold_id', 'origin_id', 'origin_index', 'segment', 'horizon',
                'target', 'target_raw', 'cutoff_index', 'target_start_index', 'bundle_signature'}
    seen = set()
    for row in rows:
        require(required <= row.keys(), 'Contract row missing identity or target fields')
        key = (row['task_id'], row['origin_id'])
        require(key not in seen, 'Duplicate origin ID in contract')
        seen.add(key)
        expected = f"{row['task_id'].split('_h')[0]}:h{row['horizon']}:f{row['fold_id']}:o{row['origin_index']}"
        require(row['origin_id'] == expected, 'Misaligned origin ID')
        require(row['target_start_index'] == row['origin_index'] and row['cutoff_index'] < row['origin_index'],
                'Misaligned target/cutoff index')
        require(row['segment'] in SEGMENTS, 'Unknown segment')
        require(len(row['target']) == row['horizon'] and len(row['target_raw']) == row['horizon'],
                'Wrong target length')
        require(np.isfinite(row['target']).all() and np.isfinite(row['target_raw']).all(),
                'Nonfinite contract target')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'preview', 'build', 'verify'])
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--split-spec', type=Path)
    args = parser.parse_args()
    audit_folder = latest_audit()
    candidate, registry, split, contract, master_commit = prepare_request(audit_folder)
    if args.command == 'prepare':
        print(json.dumps(dict(status='AWAITING_MASTER_HASH_FREEZE', audit_folder=str(audit_folder),
            tasks=len(registry), master_protocol_commit=master_commit), ensure_ascii=False, indent=2))
        return
    if args.command == 'verify':
        require(args.bundle is not None, 'verify requires --bundle')
        manifest = json.loads((args.bundle / 'manifest.json').read_text(encoding='utf-8'))
        verify_local(args.bundle, manifest['bundle_signature'])
        print(json.dumps(dict(status='PASS', bundle=str(args.bundle), signature=manifest['bundle_signature']), indent=2))
        return
    formal = args.command == 'build'
    if formal:
        require(args.split_spec is not None, 'build requires master-frozen --split-spec')
        candidate = json.loads(args.split_spec.read_text(encoding='utf-8'))
        validate_frozen_spec(candidate, registry)
    output, manifest, elapsed = build_bundle(audit_folder, candidate, registry, formal=formal)
    evidence = HERE / 'evidence'
    pd.read_csv(output / 'coverage.csv').to_csv(evidence / 'coverage.csv', index=False)
    full_audit = pd.read_csv(output / 'sample_audit.csv')
    compact_audit = pd.concat([
        group.iloc[[0]] if len(group) == 1 else group.iloc[[0, -1]]
        for _, group in full_audit.groupby(['task_id', 'segment', 'reason'], sort=False)
    ], ignore_index=True)
    compact_audit.to_csv(evidence / 'sample_audit.csv', index=False)
    full_audit.groupby(['task_id', 'segment', 'reason']).size().rename('origins').reset_index().to_csv(
        evidence / 'sample_audit_summary.csv', index=False)
    registry = registry.copy()
    actual = full_audit.groupby('task_id').reason.apply(lambda x: int(x.eq('retained').sum()))
    registry['actual_retained_origins'] = registry.task_id.map(actual)
    registry.to_csv(evidence / 'task_registry.csv', index=False)
    result_status = 'FROZEN' if formal else 'ENGINEERING_PREVIEW_AWAITING_MASTER_HASH_FREEZE'
    write_json(evidence / ('build_result.json' if formal else 'preview_result.json'), dict(status=result_status,
        bundle_path=str(output), bundle_signature=manifest['bundle_signature'], files=len(manifest['files']),
        elapsed_seconds=elapsed, master_protocol_commit=master_commit,
        warning='Not a formal frozen Bundle; master has not approved task hashes.'))
    print(json.dumps(dict(status=result_status, output=str(output),
        signature=manifest['bundle_signature'], tasks=len(candidate['tasks']), elapsed_seconds=elapsed),
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
