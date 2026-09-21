"""Versioned feature interface. Formal export requires a master-frozen split spec."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit import (ROOT, code_hashes, digest, environment, load_vectors, require, seal,
                   signature, verify_artifact, write_json)
from build_semantic_features import QUALITY_NAMES, normalized_weighted_mean
from run_famets_selective import frequency_statistics

SEGMENTS = ('train', 'calibration', 'decision', 'test')
SCENARIOS = ('proxy', 'conservative_lag', 'complete_source')


def validate_spec(spec, preview=False):
    require(spec.get('schema_version') == 1, 'Unsupported split specification schema')
    require(spec.get('status') == 'frozen' or (preview and spec.get('status') == 'engineering_preview'),
            'Formal FeatureBundle requires master-frozen split_spec.json')
    require(bool(spec.get('approved_by')) or preview, 'Missing master approval identifier')
    require(spec.get('target_boundary') == 'disjoint_half_open', 'Unsupported target boundary policy')
    require(spec.get('fit_policy') == 'train_only', 'Preprocessing must fit training rows only')
    require(spec.get('numeric_features') == ['OT'], 'This interface exports OT histories only; agree multivariate schema separately')
    require(spec.get('complete_source_rule') == 'original_url_and_valid_interval', 'Unsupported source completeness rule')
    require(isinstance(spec.get('max_per_source'), int) and spec['max_per_source'] > 0,
            'max_per_source must be positive')
    require(bool(spec.get('tasks')), 'No tasks in split specification')
    ids = []
    for task in spec['tasks']:
        require(re.fullmatch(r'[A-Za-z0-9_-]+', task['domain']) is not None, 'Invalid domain identity')
        require(re.fullmatch(r'[A-Za-z0-9_-]+', str(task['fold_id'])) is not None, 'Invalid fold identity')
        for name in ('input_len', 'horizon', 'lag_days'):
            require(type(task[name]) is int and task[name] > 0, f'{name} must be a positive integer')
        require(task['input_len'] >= 2, 'Frequency features require at least two history observations')
        bounds = task['bounds']
        require(len(bounds) == 4 and all(type(x) is int for x in bounds), 'Bounds must be four integer row offsets')
        require(task['input_len'] < bounds[0] < bounds[1] < bounds[2] < bounds[3], 'Invalid chronological bounds')
        require(re.fullmatch(r'[0-9a-f]{64}', task['numerical_sha256']) is not None, 'Missing frozen numerical snapshot hash')
        ids.append((task['domain'], task['horizon'], str(task['fold_id'])))
    require(len(ids) == len(set(ids)), 'Duplicate task/fold identity')


def fact_reasons(facts, history_start, cutoff, scenario, lag_days, maximum):
    """One mutually exclusive reason per canonical fact at this origin."""
    require(scenario in SCENARIOS, 'Unknown availability scenario')
    start, end = facts.start_date, facts.end_date
    bad_date = start.isna() | end.isna()
    reason = pd.Series('available', index=facts.index, dtype='object')
    reason.loc[bad_date] = 'missing_or_invalid_date'
    reason.loc[~bad_date & (end < start)] = 'reversed_interval'
    reason.loc[reason.eq('available') & (end < history_start)] = 'outside_lookback'
    reason.loc[reason.eq('available') & (end >= cutoff)] = 'not_strictly_before_cutoff'
    if scenario == 'conservative_lag':
        reason.loc[reason.eq('available') & (end + pd.Timedelta(days=lag_days) >= cutoff)] = 'lag_not_elapsed'
    if scenario == 'complete_source':
        has_url = facts.source_url.fillna('').str.match(r'^https?://[^/\s]+')
        reason.loc[reason.eq('available') & ~has_url] = 'missing_original_url'
    reason.loc[reason.eq('available') & ~facts.source.isin(['report', 'search'])] = 'unsupported_source'
    available_counts = {}
    for source in ('report', 'search'):
        candidates = facts[reason.eq('available') & facts.source.eq(source)].sort_values(
            ['end_date', 'text_id'], ascending=[False, True])
        available_counts[source] = len(candidates)
        reason.loc[candidates.index[:maximum]] = 'selected'
        reason.loc[candidates.index[maximum:]] = 'source_cap_exceeded'
    return reason, available_counts


def validate_selected(selected, cutoff, history_start, scenario, lag_days):
    require(not selected.text_id.duplicated().any(), 'Duplicate selected text identity')
    require(selected.start_date.notna().all() and selected.end_date.notna().all(), 'Text has no valid date')
    require((selected.start_date <= selected.end_date).all(), 'Reversed text interval')
    require((selected.end_date < cutoff).all(), 'Future or same-origin text included')
    require((selected.end_date >= history_start).all(), 'Text outside history interval')
    if scenario == 'conservative_lag':
        require((selected.end_date + pd.Timedelta(days=lag_days) < cutoff).all(), 'Lag scenario violated')
    if scenario == 'complete_source':
        require(selected.source_url.fillna('').str.match(r'^https?://[^/\s]+').all(), 'Missing original source URL')


def fit_numeric(values, train_end, fit_rows):
    require(np.array_equal(fit_rows, np.arange(train_end)), 'Fit rows extend beyond or omit training segment')
    train = np.asarray(values, dtype=float)[fit_rows]
    finite = train[np.isfinite(train)]
    require(len(finite) > 1, 'Insufficient finite training targets')
    params = dict(mean=float(finite.mean()), std=float(finite.std()), median=float(np.median(finite)),
                  fit_rows_sha256=signature(fit_rows.tolist()), train_end=int(train_end), ddof=0)
    require(params['std'] > 1e-8, 'Degenerate training scale')
    return params


def fill_numeric(values, params):
    clean = pd.Series(np.asarray(values, dtype=float)).replace([np.inf, -np.inf], np.nan)
    return clean.ffill().fillna(params['median']).to_numpy(float)


def validate_targets(origins, y, lo, hi, horizon, origin_ids):
    require(len(origin_ids) == len(set(origin_ids)), 'Duplicate origin_id')
    require(y.shape == (len(origins), horizon), 'Wrong target array shape')
    require(np.isfinite(y).all(), 'Nonfinite target entered bundle')
    require(((origins >= lo) & (origins + horizon <= hi)).all(), 'Target window crosses split boundary')


def propose_samples(frame, task):
    starts, ends = pd.to_datetime(frame.start_date), pd.to_datetime(frame.end_date)
    require(starts.is_monotonic_increasing and not starts.duplicated().any(), 'Unordered or duplicate numerical dates')
    require(starts.notna().all() and ends.notna().all() and (ends >= starts).all(), 'Invalid numerical time')
    length, horizon = task['input_len'], task['horizon']
    bounds = [0] + task['bounds']
    require(bounds[-1] <= len(frame), 'Split extends beyond observed data')
    rows = []
    for segment, lo, hi in zip(SEGMENTS, bounds[:-1], bounds[1:]):
        for origin in range(max(length, lo), hi):
            reason = 'retained'
            if origin + horizon > hi:
                reason = 'target_crosses_boundary'
            elif not np.isfinite(frame.OT.iloc[origin:origin + horizon]).all():
                reason = 'nonfinite_target'
            elif (ends.iloc[origin - length:origin] >= starts.iloc[origin]).any():
                reason = 'history_not_available_at_cutoff'
            elif hi < len(frame) and (ends.iloc[origin:origin + horizon] >= starts.iloc[hi]).any():
                reason = 'target_interval_crosses_boundary'
            rows.append(dict(origin_index=origin, segment=segment, reason=reason,
                origin_id=f"{task['domain']}:h{horizon}:f{task['fold_id']}:o{origin}",
                cutoff_time=starts.iloc[origin].isoformat(), history_start=starts.iloc[origin-length].isoformat(),
                target_end_index=origin + horizon, segment_lo=lo, segment_hi=hi))
    return pd.DataFrame(rows)


def features_at_origin(facts, vectors, vector_rows, history_start, cutoff, scenario, task, maximum):
    reasons, counts = fact_reasons(facts, history_start, cutoff, scenario, task['lag_days'], maximum)
    selected = facts[reasons.eq('selected')]
    validate_selected(selected, cutoff, history_start, scenario, task['lag_days'])
    history_days = max(float((cutoff - history_start).days), 1.)
    half_life = max(history_days / 4., 1.)
    embeddings, fractions, ids, ages = [], [], {}, []
    for source in ('report', 'search'):
        group = selected[selected.source.eq(source)].sort_values(['end_date', 'text_id'], ascending=[False, True])
        source_ids = group.text_id.tolist()
        ids[source] = source_ids
        age = (cutoff - group.end_date).dt.total_seconds().to_numpy() / 86400.
        ages.extend(age.tolist())
        embeddings.append(normalized_weighted_mean(vectors[[vector_rows[i] for i in source_ids]],
            np.exp(-np.log(2.) * age / half_life), 384))
        fractions.append(float(group.future_language_flag.astype(float).mean()) if len(group) else 0.)
    quality = np.asarray([np.log1p(counts['report']), np.log1p(counts['search']),
        np.log1p(len(ids['report'])), np.log1p(len(ids['search'])),
        float(not ids['report']), float(not ids['search']),
        min(ages) / history_days if ages else 1., np.mean(ages) / history_days if ages else 1.,
        *fractions], dtype=np.float32)
    return np.concatenate(embeddings), quality, ids, reasons.value_counts().to_dict()


def load_canonical(audit_folder):
    corpus = pd.read_csv(audit_folder / 'fact_corpus.csv', parse_dates=['start_date', 'end_date'])
    lineage = pd.read_csv(audit_folder / 'text_lineage.csv', keep_default_na=False)
    selected = lineage[lineage.reason.eq('retained')].set_index('text_id')
    require(not selected.index.duplicated().any(), 'Ambiguous raw-row to canonical mapping')
    require(set(corpus.text_id) == set(selected.index), 'Incomplete raw text lineage')
    corpus['source_url'] = corpus.text_id.map(selected.source_url).fillna('')
    return corpus


def build(audit_folder, spec_path, preview=False, threads=1, expected_signature=None, replica=None):
    require(type(threads) is int and threads > 0, 'Threads must be a positive integer')
    audit_manifest = verify_artifact(audit_folder)
    spec = json.loads(Path(spec_path).read_text(encoding='utf-8'))
    validate_spec(spec, preview)
    # Recheck the current source snapshot before trusting an audited cache.
    for relative, sha in audit_manifest['inputs']['raw_and_cache'].items():
        require(digest(ROOT / relative) == sha, f'Raw/cache changed since audit: {relative}')
    inputs = dict(audit_signature=audit_manifest['signature'], audit_manifest_sha256=digest(audit_folder / 'manifest.json'),
                  split_spec=spec, split_spec_sha256=digest(spec_path), code=code_hashes(),
                  environment=environment(), mode='engineering_preview' if preview else 'frozen')
    sig = signature(inputs)
    if expected_signature is not None:
        require(sig == expected_signature, 'Stale cache: new inputs do not match requested frozen signature')
    require(replica is None or (preview and re.fullmatch(r'[A-Za-z0-9_-]+', replica)),
            'Replicas are restricted to named engineering checks')
    suffix = '-' + replica if replica else ''
    output = ROOT / 'data_processed/team_data' / (('preview-' if preview else 'bundle-') + sig[:20] + suffix)
    if output.exists():
        return output, verify_artifact(output, sig)
    corpus = load_canonical(audit_folder)
    vectors = load_vectors(corpus, ROOT / 'data_processed/v4/embeddings')
    vector_rows = dict(zip(corpus.text_id, range(len(corpus))))
    output.mkdir(parents=True)
    coverage = []
    with threadpool_limits(limits=threads):
        for task in spec['tasks']:
            name = f"{task['domain']}_h{task['horizon']}_f{task['fold_id']}"
            path = audit_folder / f"{task['domain']}_numerical.csv"
            require(digest(path) == task['numerical_sha256'], 'Split numerical snapshot does not match audit')
            frame = pd.read_csv(path)
            raw = frame.OT.to_numpy(float)
            params = fit_numeric(raw, task['bounds'][0], np.arange(task['bounds'][0]))
            filled = fill_numeric(raw, params)
            rows = propose_samples(frame, task)
            require(not rows.origin_id.duplicated().any(), 'Duplicate origin_id')
            valid = rows[rows.reason.eq('retained')].reset_index(drop=True)
            require(len(valid) > 0, 'No valid samples')
            task_dir = output / name
            task_dir.mkdir()
            rows.to_csv(task_dir / 'sample_audit.csv', index=False)
            write_json(task_dir / 'numeric_fit.json', params)
            origins = valid.origin_index.to_numpy(int)
            length, horizon = task['input_len'], task['horizon']
            xraw = np.stack([raw[o-length:o] for o in origins])
            x = (np.stack([filled[o-length:o] for o in origins]) - params['mean']) / params['std']
            y = np.stack([raw[o:o+horizon] for o in origins])
            for segment, group in valid.groupby('segment', sort=False):
                idx = group.index.to_numpy()
                validate_targets(origins[idx], y[idx], int(group.segment_lo.iloc[0]), int(group.segment_hi.iloc[0]),
                                 horizon, group.origin_id.tolist())
            shared = dict(numeric_history=x, numeric_history_raw=xraw, numeric_missing=~np.isfinite(xraw),
                targets=y, targets_standardized=(y-params['mean']) / params['std'],
                frequency=frequency_statistics(x), origin_index=origins,
                target_time=np.stack([frame.start_date.iloc[o:o+horizon].to_numpy(dtype='U32') for o in origins]),
                target_end_time=np.stack([frame.end_date.iloc[o:o+horizon].to_numpy(dtype='U32') for o in origins]))
            for key, value in shared.items():
                np.save(task_dir / f'{key}.npy', value, allow_pickle=False)
            valid.to_csv(task_dir / 'samples.csv', index=False)
            facts = corpus[corpus.domain.eq(task['domain'])].reset_index(drop=True)
            for scenario in SCENARIOS:
                scenario_dir = task_dir / scenario
                scenario_dir.mkdir()
                semantic, quality, traces = [], [], []
                for row in valid.itertuples():
                    e, q, ids, reasons = features_at_origin(facts, vectors, vector_rows,
                        pd.Timestamp(row.history_start), pd.Timestamp(row.cutoff_time), scenario, task, spec['max_per_source'])
                    semantic.append(e)
                    quality.append(q)
                    traces.append(dict(origin_id=row.origin_id, report_ids=ids['report'], search_ids=ids['search'],
                                       reason_counts=reasons, publication_time_status='end_date_proxy_unverified'))
                semantic, quality = np.asarray(semantic), np.asarray(quality)
                require(np.isfinite(semantic).all() and np.isfinite(quality).all(), 'Nonfinite feature')
                mask = quality[:, 4:6] == 0
                for key, value in dict(semantic=semantic, quality=quality, source_available=mask,
                                       text_available=mask.any(axis=1)).items():
                    np.save(scenario_dir / f'{key}.npy', value, allow_pickle=False)
                (scenario_dir / 'text_trace.jsonl').write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n'
                    for t in traces), encoding='utf-8')
                for segment, group in valid.groupby('segment', sort=False):
                    idx = group.index.to_numpy()
                    unique_ids = {i for j in idx for s in ('report_ids', 'search_ids') for i in traces[j][s]}
                    coverage.append(dict(task_id=name, domain=task['domain'], fold_id=task['fold_id'], horizon=horizon,
                        scenario=scenario, segment=segment, valid_origins=len(idx),
                        covered_origins=int(mask[idx].any(axis=1).sum()), unique_selected_facts=len(unique_ids),
                        excluded_origins=int((rows.segment.eq(segment) & rows.reason.ne('retained')).sum())))
    coverage = pd.DataFrame(coverage)
    keys = ['task_id', 'segment']
    baseline = coverage[coverage.scenario.eq('proxy')].set_index(keys).covered_origins
    coverage['baseline_covered_origins'] = [int(baseline.loc[(r.task_id, r.segment)]) for r in coverage.itertuples()]
    coverage['coverage_loss_origins'] = coverage.baseline_covered_origins - coverage.covered_origins
    coverage['coverage_loss_pct_of_baseline'] = np.where(coverage.baseline_covered_origins > 0,
        coverage.coverage_loss_origins * 100 / coverage.baseline_covered_origins, np.nan)
    coverage.to_csv(output / 'coverage.csv', index=False)
    write_json(output / 'schema.json', dict(schema_version=1, semantic_dim=768, quality_names=QUALITY_NAMES,
        sources=['report', 'search'], frequency_dim=10, numeric_features=['OT'],
        numeric_unit='train-only standardized OT; raw values and targets retain source units',
        frequency_order=['band_0', 'band_1', 'band_2', 'band_3', 'centroid', 'entropy', 'dominant', 'concentration', 'slope', 'volatility'],
        pca='not fitted: raw frozen 384+384 source embeddings; downstream must fit only allowed training rows',
        missing_mask='True = imputed numeric history; source_available/text_available True = selected text exists',
        audit_folder_relative=audit_folder.relative_to(ROOT).as_posix(),
        status=inputs['mode'], scenarios=list(SCENARIOS)))
    return output, seal(output, inputs)


def read_bundle(folder, task_id, scenario, expected_signature):
    manifest = verify_artifact(Path(folder), expected_signature)
    schema = json.loads((Path(folder) / 'schema.json').read_text(encoding='utf-8'))
    require(schema['status'] == 'frozen', 'Engineering preview is not a frozen training interface')
    require(scenario in SCENARIOS, 'Unknown scenario')
    require(re.fullmatch(r'[A-Za-z0-9_-]+', task_id) is not None, 'Invalid task_id')
    task = Path(folder) / task_id
    arrays = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in task.glob('*.npy')}
    arrays.update({p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in (task / scenario).glob('*.npy')})
    samples = pd.read_csv(task / 'samples.csv')
    require(len(samples) == len(arrays['targets']) == len(arrays['semantic']), 'Inconsistent sample count')
    return samples, arrays, manifest


def trace_origin(audit_folder, bundle_folder, task_id, origin_id, scenario):
    verify_artifact(audit_folder)
    manifest = verify_artifact(bundle_folder)
    require(re.fullmatch(r'[A-Za-z0-9_-]+', task_id) is not None, 'Invalid task_id')
    require(manifest['inputs']['audit_manifest_sha256'] == digest(audit_folder / 'manifest.json'), 'Wrong audit for bundle')
    spec = manifest['inputs']['split_spec']
    task = next(t for t in spec['tasks'] if f"{t['domain']}_h{t['horizon']}_f{t['fold_id']}" == task_id)
    rows = pd.read_csv(bundle_folder / task_id / 'sample_audit.csv')
    found = rows[rows.origin_id.eq(origin_id)]
    require(len(found) == 1, 'Unknown or ambiguous origin')
    row = found.iloc[0]
    corpus = load_canonical(audit_folder)
    facts = corpus[corpus.domain.eq(task['domain'])].reset_index(drop=True)
    reasons, _ = fact_reasons(facts, pd.Timestamp(row.history_start), pd.Timestamp(row.cutoff_time),
                              scenario, task['lag_days'], spec['max_per_source'])
    mapping = dict(zip(facts.text_id, reasons))
    lineage = pd.read_csv(audit_folder / 'text_lineage.csv', keep_default_na=False)
    lineage = lineage[lineage.domain.eq(task['domain'])].copy()
    lineage['origin_reason'] = [mapping.get(r.text_id, r.reason) if r.reason == 'retained' else r.reason
                                for r in lineage.itertuples()]
    if row.reason != 'retained':
        lineage['origin_reason'] = 'sample_excluded:' + str(row.reason)
    return row.to_dict(), lineage
