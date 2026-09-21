"""Create compact trace examples and local full raw-row evidence for four tasks."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FIRST = ROOT / '辅助电脑02交付01次'
sys.path.insert(0, str(FIRST))
from audit import digest, require, write_json
from bundle import trace_origin


def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('audit', type=Path)
    args = parser.parse_args()
    bundle, audit = args.bundle.resolve(), args.audit.resolve()
    local_root = ROOT / 'data_processed/team_data/round2_traces' / bundle.name
    local_root.mkdir(parents=True, exist_ok=True)
    compact = []
    for task_dir in sorted(p for p in bundle.iterdir() if p.is_dir()):
        samples = pd.read_csv(task_dir / 'samples.csv')
        sample_audit = pd.read_csv(task_dir / 'sample_audit.csv')
        available = np.load(task_dir / 'proxy/text_available.npy', allow_pickle=False)
        traces = json_lines(task_dir / 'proxy/text_trace.jsonl')
        candidates = {}
        with_positions = np.flatnonzero(available)
        without_positions = np.flatnonzero(~available)
        if len(with_positions):
            candidates['with_text'] = samples.iloc[int(with_positions[0])]
        if len(without_positions):
            candidates['without_text'] = samples.iloc[int(without_positions[0])]
        boundary = sample_audit[sample_audit.reason.str.contains('crosses_boundary')]
        if len(boundary):
            candidates['boundary_excluded'] = boundary.iloc[0]
        task_out = local_root / task_dir.name
        task_out.mkdir(exist_ok=True)
        domain = task_dir.name.split('_h')[0]
        numeric_lineage = pd.read_csv(audit / f'{domain}_numeric_lineage.csv')
        for kind in ('with_text', 'without_text', 'boundary_excluded'):
            if kind not in candidates:
                compact.append(dict(task_id=task_dir.name, example_type=kind, available=False,
                    reason='No origin of this class exists under the preregistered split'))
                continue
            row = candidates[kind]
            row_dict, text_lineage = trace_origin(audit, bundle, task_dir.name, row.origin_id, 'proxy')
            target_end = int(row.target_end_index)
            history_start = int(row.origin_index) - int(task_dir.name.startswith('Environment') and 56 or
                task_dir.name.startswith('Climate') and 52 or 24)
            numeric = numeric_lineage[pd.to_numeric(numeric_lineage.clean_index, errors='coerce').between(
                history_start, target_end - 1)].copy()
            text_path = task_out / f'{kind}_text_lineage.csv'
            numeric_path = task_out / f'{kind}_numeric_lineage.csv'
            text_lineage.to_csv(text_path, index=False)
            numeric.to_csv(numeric_path, index=False)
            trace = next((t for t in traces if t['origin_id'] == row.origin_id), None)
            compact.append(dict(task_id=task_dir.name, example_type=kind, available=True,
                origin_id=row.origin_id, origin_index=int(row.origin_index), segment=row.segment,
                sample_reason=row.reason, cutoff_time=row.cutoff_time,
                numeric_raw_rows=int(len(numeric)), numeric_raw_source_rows=[int(x) for x in numeric.raw_row.tolist()],
                selected_report_ids=[] if trace is None else trace['report_ids'],
                selected_search_ids=[] if trace is None else trace['search_ids'],
                exclusion_counts={} if trace is None else trace['reason_counts'],
                full_text_trace=f'data_processed/team_data/round2_traces/{bundle.name}/{task_dir.name}/{text_path.name}',
                full_text_trace_sha256=digest(text_path),
                full_numeric_trace=f'data_processed/team_data/round2_traces/{bundle.name}/{task_dir.name}/{numeric_path.name}',
                full_numeric_trace_sha256=digest(numeric_path)))
    out = HERE / 'evidence/trace_examples.jsonl'
    out.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in compact), encoding='utf-8')
    write_json(HERE / 'evidence/trace_result.json', dict(status='PASS', examples=len(compact),
        available_examples=sum(bool(r['available']) for r in compact), local_full_trace_root=str(local_root),
        limitation='Git stores compact examples and hashes; full raw-row traces stay in ignored D-drive evidence.'))
    print(json.dumps(dict(status='PASS', examples=len(compact), output=str(out)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
