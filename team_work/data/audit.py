"""Read-only source audit built on the existing Time-MMD loaders."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from audit_timemmd import _valid_text
from build_text_index import load_corpus, normalize_fact, stable_id
from data_utils import numerical_intervals

REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DATA_REVISION = '00281e2d86058286d5548b15a7670e8eda57ef62'
TARGET_MEANINGS = {
    'Agriculture': 'Retail Broiler Composite', 'Climate': 'Drought Level',
    'Economy': 'International Trade Balance', 'Energy': 'Gasoline Prices',
    'Environment': 'Air Quality Index', 'Health_AFR': 'Influenza Patients Proportion',
    'Health_US': 'Influenza Patients Proportion', 'Security': 'Disaster and Emergency Grants',
    'SocialGood': 'Unemployment Rate', 'Traffic': 'Travel Volumn (upstream spelling)',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def environment():
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages={name: importlib.metadata.version(name)
                          for name in ('numpy', 'pandas', 'scikit-learn', 'threadpoolctl')},
                seed=2026, encoding='reuse existing pinned vectors; no model download or training')


def peak_memory():
    if sys.platform != 'win32':
        return dict(measured=False, reason='Windows process counter is unavailable')
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess
    process.restype = ctypes.c_void_p
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    require(get_info(process(), ctypes.byref(counters), counters.cb) != 0, 'Cannot read process memory counter')
    return dict(measured=True, peak_working_set_bytes=counters.PeakWorkingSetSize,
                scope='whole Python process, including imports and native arrays; Windows GetProcessMemoryInfo')


def code_hashes():
    paths = list(Path(__file__).parent.glob('*.py')) + [ROOT / 'src' / name for name in
        ('audit_timemmd.py', 'build_text_index.py', 'data_utils.py', 'build_semantic_features.py',
         'run_baselines.py', 'run_famets_selective.py', 'encode_text.py')]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def seal(folder, inputs):
    records = {p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob('*'))
               if p.is_file() and p.name != 'manifest.json'}
    manifest = dict(signature=signature(inputs), inputs=inputs, files=records)
    write_json(folder / 'manifest.json', manifest)
    return manifest


def verify_artifact(folder, expected=None):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    require(signature(manifest['inputs']) == manifest['signature'], 'Manifest input signature changed')
    if expected is not None:
        require(manifest['signature'] == expected, 'Stale cache: input/rule signature mismatch')
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()
              and p != folder / 'manifest.json'}
    require(actual == set(manifest['files']), 'Incomplete or unexpected artifact files')
    for relative, sha in manifest['files'].items():
        require(digest(folder / relative) == sha, f'Artifact content changed: {relative}')
    return manifest


def audit_numerical(path, frequency):
    raw = pd.read_csv(path)
    start, end = numerical_intervals(raw, path)
    target = pd.to_numeric(raw['OT'], errors='coerce')
    frame = pd.DataFrame(dict(raw_row=np.arange(len(raw)), start_date=start, end_date=end,
                              raw_target=raw['OT'], OT=target))
    invalid = start.isna() | end.isna() | (end < start)
    require(not invalid.any(), f'Invalid numerical intervals require adjudication: {path}')
    frame = frame.sort_values(['start_date', 'end_date', 'raw_row'], kind='stable')
    duplicates = frame.start_date.duplicated(keep=False)
    frame['duplicate_date'] = duplicates
    frame['nonfinite_target'] = ~np.isfinite(frame.OT)
    frame['status'] = np.where(duplicates, 'duplicate_date_unknown',
                             np.where(frame.nonfinite_target, 'unknown_target', 'retained'))
    frame.loc[duplicates | frame.nonfinite_target, 'OT'] = np.nan
    unique = frame.drop_duplicates('start_date', keep='first').reset_index(drop=True)
    valid_positions = np.flatnonzero(unique.OT.notna())
    require(len(valid_positions) > 0, f'No finite target: {path}')
    first, last = valid_positions[[0, -1]]
    clean = unique.iloc[first:last + 1][['start_date', 'end_date', 'OT']].reset_index(drop=True)
    date_to_row = dict(zip(clean.start_date, range(len(clean))))
    frame['clean_index'] = frame.start_date.map(date_to_row).astype('Int64')
    frame.loc[frame.clean_index.isna(), 'status'] = 'trimmed_unknown_edge'
    frame['raw_start'] = raw.loc[frame.raw_row, next(c for c in
        ('start_date', 'date', 'Date', 'MapDate', 'Month') if c in raw)].to_numpy()
    frame['raw_end'] = raw.loc[frame.raw_row, 'end_date'].to_numpy() if 'end_date' in raw else frame.raw_start
    ordered_dates = clean.start_date
    steps = ordered_dates.diff().dt.total_seconds().div(86400).dropna()
    if frequency == 'monthly':
        gaps = ordered_dates.dt.to_period('M').astype('int64').diff().dropna().ne(1)
    else:
        gaps = steps.ne({'weekly': 7, 'daily': 1, 'hourly': 1 / 24}.get(frequency))
    summary = dict(raw_rows=len(raw), clean_rows=len(clean),
                   duplicate_rows=int(duplicates.sum()), duplicate_dates=int(frame.loc[duplicates, 'start_date'].nunique()),
                   collapsed_rows=len(frame) - len(unique), trimmed_edge_rows=int(first + len(unique) - last - 1),
                   unknown_target_rows=int(clean.OT.isna().sum()), raw_nonfinite=int((~np.isfinite(target)).sum()),
                   raw_infinite=int(np.isinf(target).sum()), irregular_steps=int(gaps.sum()), frequency=frequency,
                   start=clean.start_date.min().isoformat(), end=clean.end_date.max().isoformat(),
                   fields=list(raw.columns), target='OT', unit='unverified: see upstream DescriptionOfOT.png / paper Appendix C',
                   modal_step_days=float(steps.mode().iloc[0]) if len(steps) else None)
    return clean, frame, summary


def text_lineage(path, corpus):
    raw = pd.read_csv(path)
    domain, source = path.parent.name, path.stem.rsplit('_', 1)[-1]
    start = pd.to_datetime(raw.start_date, errors='coerce')
    end = pd.to_datetime(raw.end_date, errors='coerce')
    valid = _valid_text(raw.fact)
    normalized = raw.fact.map(normalize_fact)
    selected = {(r.source, r.fact): r.text_id for r in corpus.itertuples() if r.domain == domain}
    ids = [stable_id(domain, source, a, b, t) if pd.notna(a) and pd.notna(b) else ''
           for a, b, t in zip(start, end, normalized)]
    canonical_ids = [selected.get((source, t), '') for t in normalized]
    unambiguous = raw.start_date.astype(str).str.match(r'^\d{4}-\d{2}-\d{2}$') & raw.end_date.astype(str).str.match(r'^\d{4}-\d{2}-\d{2}$')
    reason = np.select([~valid, start.isna() | end.isna(), ~unambiguous, end < start,
                        np.asarray(ids) != np.asarray(canonical_ids)],
                       ['missing_fact', 'missing_or_invalid_date', 'ambiguous_date_format', 'reversed_interval', 'duplicate_fact_later'],
                       default='retained')
    result = pd.DataFrame(dict(raw_row=np.arange(len(raw)), domain=domain, source=source,
        raw_start=raw.start_date, raw_end=raw.end_date, fact=raw.fact, normalized_fact=normalized,
        start_date=start, end_date=end, text_id=ids, canonical_text_id=canonical_ids, reason=reason))
    # Multiple exact copies can have the same stable ID; only one raw row is canonical.
    same = result.reason.eq('retained') & result.text_id.duplicated(keep='first')
    result.loc[same, 'reason'] = 'duplicate_fact_exact'
    result['source_file'] = path.relative_to(ROOT).as_posix()
    # CSV provenance is not the original publication URL or a verified publication time.
    url_columns = [c for c in raw if c.lower() in ('url', 'source_url', 'article_url')]
    result['source_url'] = raw[url_columns[0]].fillna('').astype(str) if url_columns else ''
    result['publication_verified'] = False
    result['pred_fields_excluded'] = ','.join(c for c in ('pred', 'preds') if c in raw)
    return result


def publish_audit(folder, summary, evidence):
    evidence.mkdir(exist_ok=True, parents=True)
    write_json(evidence / 'audit_summary.json', dict(output=str(folder) if folder else None, **summary))
    if folder is not None:
        for name in ('numerical_sources.csv', 'textual_sources.csv', 'numerical_field_audit.csv',
                     'text_exclusion_counts.csv', 'ett_sources.csv', 'v4_compatibility.csv', 'legacy_proxy_coverage.csv'):
            shutil.copy2(folder / name, evidence / name)


def audit_legacy_origins(project, folder, corpus, config):
    """Verify all historical index rows without deciding any new train/test split."""
    index = pd.read_csv(project / 'data_processed/v4/text/sample_text_index.csv')
    require(not index.duplicated(['domain', 'origin_index']).any(), 'Duplicate legacy origin identity')
    records = []
    for domain, settings in config['domains'].items():
        frame = pd.read_csv(folder / f'{domain}_numerical.csv', parse_dates=['start_date', 'end_date'])
        rows = index[index.domain.eq(domain)]
        require(sorted(rows.origin_index) == list(range(settings['input_len'], len(frame))), 'Incomplete legacy origin index')
        sources = {}
        for source in ('report', 'search'):
            group = corpus[corpus.domain.eq(domain) & corpus.source.eq(source)].sort_values(
                ['end_date', 'text_id'], ascending=[False, True])
            sources[source] = (group.end_date.astype('int64').to_numpy(), group.text_id.to_numpy())
        for row in rows.itertuples():
            cutoff = frame.start_date.iloc[row.origin_index]
            history = frame.start_date.iloc[row.origin_index - settings['input_len']]
            require(cutoff == pd.Timestamp(row.forecast_start) and history == pd.Timestamp(row.history_start),
                    'Legacy text index uses a different numerical timeline')
            selected_count, available_count = 0, 0
            for source, (ends, ids) in sources.items():
                usable = ids[(ends >= history.value) & (ends < cutoff.value)]
                chosen = json.loads(getattr(row, source + '_text_ids'))
                require(chosen == usable[:32].tolist(), f'Legacy text IDs violate strict proxy rule: {domain} {row.origin_index}')
                require(len(usable) == getattr(row, source + '_available_count'), 'Legacy available count changed')
                require(len(chosen) == getattr(row, source + '_selected_count'), 'Legacy selected count changed')
                selected_count += len(chosen)
                available_count += len(usable)
            require(selected_count == row.total_selected_count, 'Legacy total text count changed')
            records.append(dict(domain=domain, origin_index=row.origin_index, cutoff_time=cutoff.isoformat(),
                selected_facts=selected_count, available_facts=available_count,
                excluded_by_lookback_or_cutoff=int((corpus.domain == domain).sum())-available_count,
                publication_time_status='end_date_proxy_unverified'))
    details = pd.DataFrame(records)
    details.to_csv(folder / 'legacy_origin_audit.csv', index=False)
    summary = details.groupby('domain').agg(origins=('origin_index', 'size'),
        covered_origins=('selected_facts', lambda v: int(v.gt(0).sum()))).reset_index()
    summary['coverage_pct'] = summary.covered_origins * 100 / summary.origins
    summary.to_csv(folder / 'legacy_proxy_coverage.csv', index=False)
    return len(details)


def load_vectors(corpus, cache_root):
    meta = json.loads((cache_root / 'fact_embedding_metadata.json').read_text(encoding='utf-8'))
    require(meta['model'] == MODEL and meta['revision'] == REVISION, 'Wrong encoder model/revision')
    index = pd.read_csv(cache_root / 'fact_embedding_index.csv')
    require(not index.text_id.duplicated().any(), 'Duplicate embedding identity')
    require(index.row.tolist() == list(range(len(index))), 'Embedding index row order changed')
    require(index.text_id.tolist() == corpus.text_id.tolist(), 'Raw corpus differs from pinned embedding index')
    h = hashlib.sha256()
    for row in corpus.itertuples():
        h.update(f'{row.text_id}\0{row.fact}\n'.encode('utf-8'))
    require(h.hexdigest() == meta['corpus_sha256'], 'Raw corpus differs from encoded corpus')
    vectors = np.load(cache_root / 'fact_embeddings.npy', mmap_mode='r', allow_pickle=False)
    require(list(vectors.shape) == meta['shape'] == [len(corpus), 384], 'Embedding shape mismatch')
    require(np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4),
            'Invalid/non-normalized embedding cache')
    return vectors


def run_audit(project=ROOT):
    raw_root = project / 'references/external/Time-MMD'
    cache_root = project / 'data_processed/v4/embeddings'
    config_path = project / 'configs/safefame_v4.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    expected = [raw_root / 'numerical' / d / f'{d}.csv' for d in config['domains']]
    expected += [raw_root / 'textual' / d / f'{d}_{s}.csv' for d in config['domains'] for s in ('report', 'search')]
    expected += [cache_root / n for n in ('fact_embedding_metadata.json', 'fact_embedding_index.csv', 'fact_embeddings.npy')]
    expected += [project / 'data_processed/v4/numerical' / d / f'{d}.csv' for d in config['domains']]
    expected += [project / 'data_processed/v4/text/sample_text_index.csv']
    expected += [project / 'references/external/ETDataset/ETT-small' / n for n in ('ETTh1.csv', 'ETTh2.csv')]
    missing = [str(p) for p in expected if not p.is_file()]
    if missing:
        return None, dict(status='MISSING_INPUTS', missing=missing)
    sources = set(expected) | set((raw_root / 'textual').glob('*/*.csv')) | set((raw_root / 'numerical').glob('*/*.csv'))
    sources |= {raw_root / 'readme.MD', raw_root / 'DescriptionOfOT.png'}
    files = {p.relative_to(project).as_posix(): digest(p) for p in sorted(sources)}
    ett = sorted((project / 'references/external/ETDataset/ETT-small').glob('ETTh*.csv'))
    files.update({p.relative_to(project).as_posix(): digest(p) for p in ett})
    inputs = dict(raw_and_cache=files, config_sha256=digest(config_path), code=code_hashes(),
                  environment=environment(), rules='v4-compatible-target-cleaning; canonical-load_corpus; row-lineage-v1')
    folder = project / 'data_processed/team_data' / ('audit-' + signature(inputs)[:20])
    if folder.exists():
        verify_artifact(folder, signature(inputs))
        return folder, json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    corpus, _ = load_corpus(raw_root, list(config['domains']))
    load_vectors(corpus, cache_root)
    folder.mkdir(parents=True)
    corpus.to_csv(folder / 'fact_corpus.csv', index=False)
    numerical, textual, lineage_frames, comparisons, field_audit = [], [], [], [], []
    for domain, settings in config['domains'].items():
        path = raw_root / 'numerical' / domain / f'{domain}.csv'
        clean, lineage, stats = audit_numerical(path, settings['frequency'])
        clean.to_csv(folder / f'{domain}_numerical.csv', index=False)
        lineage.to_csv(folder / f'{domain}_numeric_lineage.csv', index=False)
        old = pd.read_csv(project / 'data_processed/v4/numerical' / domain / f'{domain}.csv')
        compatible = (len(clean) == len(old) and np.array_equal(clean.OT, old.OT, equal_nan=True)
                      and clean.start_date.equals(pd.to_datetime(old.start_date))
                      and clean.end_date.equals(pd.to_datetime(old.end_date)))
        comparisons.append(dict(domain=domain, v4_target_and_dates_equal=compatible))
        numerical.append(dict(domain=domain, source_file=path.relative_to(project).as_posix(),
            sha256=digest(path), target_meaning=TARGET_MEANINGS[domain],
            meaning_source='Time-MMD/DescriptionOfOT.png; frequency/counts follow actual CSV, not image', **stats))
        raw_frame = pd.read_csv(path)
        for col in raw_frame:
            coerced = pd.to_numeric(raw_frame[col], errors='coerce')
            field_audit.append(dict(domain=domain, column=col, raw_dtype=str(raw_frame[col].dtype),
                raw_missing=int(raw_frame[col].isna().sum()), nonnumeric_nonmissing=int((raw_frame[col].notna() & coerced.isna()).sum()),
                infinite_numeric=int(np.isinf(coerced).sum()), used_in_bundle=(col == 'OT'),
                policy='OT only; other raw columns retained in original snapshot, audited but not converted into features'))
        for path in sorted((raw_root / 'textual' / domain).glob('*.csv')):
            lineage = text_lineage(path, corpus)
            lineage_frames.append(lineage)
            counts = lineage.reason.value_counts().to_dict()
            textual.append(dict(domain=domain, source=path.stem.rsplit('_', 1)[-1],
                source_file=path.relative_to(project).as_posix(), sha256=digest(path), raw_rows=len(lineage),
                retained=int(lineage.reason.eq('retained').sum()),
                original_url_rows=int(lineage.source_url.str.match(r'^https?://').sum()),
                reasons=counts, fields=list(pd.read_csv(path, nrows=0).columns)))
    all_lineage = pd.concat(lineage_frames, ignore_index=True)
    all_lineage.to_csv(folder / 'text_lineage.csv', index=False)
    require(not all_lineage.reason.eq('ambiguous_date_format').any(),
            f'Ambiguous dates require source adjudication; see {folder / "text_lineage.csv"}')
    require(set(all_lineage.loc[all_lineage.reason.eq('retained'), 'text_id']) == set(corpus.text_id),
            'Raw lineage and canonical corpus differ')
    pd.DataFrame(field_audit).to_csv(folder / 'numerical_field_audit.csv', index=False)
    pd.concat(lineage_frames, ignore_index=True).groupby(['domain', 'source', 'reason']).size().rename('rows').reset_index().to_csv(
        folder / 'text_exclusion_counts.csv', index=False)
    for name, values in [('numerical_sources', numerical), ('textual_sources', textual), ('v4_compatibility', comparisons)]:
        pd.DataFrame(values).to_csv(folder / f'{name}.csv', index=False)
    ett_stats = []
    for path in ett:
        _, _, stats = audit_numerical(path, 'hourly')
        stats['unit'] = 'oil temperature; numerical unit not reverified from source'
        ett_stats.append(dict(dataset=path.stem, sha256=digest(path), **stats))
    pd.DataFrame(ett_stats).to_csv(folder / 'ett_sources.csv', index=False)
    legacy_origins = audit_legacy_origins(project, folder, corpus, config)
    git_revision = subprocess.check_output(['git', '-C', str(raw_root), 'rev-parse', 'HEAD'], text=True).strip()
    require(git_revision == DATA_REVISION, 'Time-MMD revision differs from documented input snapshot')
    summary = dict(status='AUDIT_COMPLETE_PROTOCOL_PENDING', series=len(numerical), domains=9,
        raw_numerical_rows=sum(r['raw_rows'] for r in numerical),
        clean_numerical_rows=sum(r['clean_rows'] for r in numerical),
        canonical_facts=len(corpus), raw_text_rows=sum(r['raw_rows'] for r in textual),
        legacy_proxy_origins_checked=legacy_origins,
        original_url_rows=sum(r['original_url_rows'] for r in textual),
        unknown_target_rows=sum(r['unknown_target_rows'] for r in numerical),
        v4_target_compatible=all(r['v4_target_and_dates_equal'] for r in comparisons),
        timemmd_revision=git_revision, expected_revision=DATA_REVISION,
        encoder_revision=REVISION, embedding_cache_valid=True, ett_files=len(ett),
        sources=dict(TimeMMD='https://github.com/AdityaLab/Time-MMD', ETT='https://github.com/zhouhaoyi/ETDataset'),
        license_status='Time-MMD: ODC-By stated in prior project literature audit; no local LICENSE. ETT: upstream terms; no new license verification.',
        missing=['master-frozen split_spec.json', 'master-approved lag schedule and completeness rule',
                 'verified per-record publication times and original URLs', 'joint FeatureBundle schema approval'],
        caveat='end_date is a proxy; source labels do not establish publication reliability')
    write_json(folder / 'summary.json', summary)
    seal(folder, inputs)
    return folder, summary
