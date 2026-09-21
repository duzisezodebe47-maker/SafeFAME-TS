"""Read-only SHY interface for a transferred round-two Bundle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fail_unless(condition, message):
    if not condition:
        raise ValueError(message)


def verify(folder):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    actual = {p.relative_to(folder).as_posix(): digest(p) for p in sorted(
        p for p in folder.rglob('*') if p.is_file() and p.name != 'manifest.json')}
    fail_unless(actual == manifest['files'], 'Bundle file list or SHA256 differs from manifest')
    return manifest


def read_task(folder, task_id, scenario, allow_preview=False):
    manifest = verify(folder)
    mode = manifest['inputs']['mode']
    fail_unless(mode == 'frozen' or allow_preview, 'Refusing non-frozen engineering preview')
    fail_unless(scenario in ('proxy', 'conservative_lag'), 'complete_source has no training arrays')
    task = folder / task_id
    samples = pd.read_csv(task / 'samples.csv')
    arrays = {}
    for path in task.glob('*.npy'):
        arrays[path.stem] = np.load(path, mmap_mode='r', allow_pickle=False)
    for path in (task / scenario).glob('*.npy'):
        arrays[path.stem] = np.load(path, mmap_mode='r', allow_pickle=False)
    count = len(samples)
    for key in ('numeric_history', 'targets_standardized', 'targets_raw', 'semantic',
                'quality', 'frequency', 'text_available'):
        fail_unless(len(arrays[key]) == count, f'{key}: sample count mismatch')
    segment_masks = {name: samples.segment.eq(name).to_numpy()
                     for name in ('train', 'calibration', 'decision', 'test')}
    return samples, arrays, segment_masks, manifest['bundle_signature']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('task_id')
    parser.add_argument('--scenario', choices=['proxy', 'conservative_lag'], default='proxy')
    parser.add_argument('--allow-preview', action='store_true', help='Engineering verification only; never train formal models')
    args = parser.parse_args()
    samples, arrays, masks, bundle_signature = read_task(
        args.bundle.resolve(), args.task_id, args.scenario, args.allow_preview)
    result = {
        'bundle_signature': bundle_signature,
        'task_id': args.task_id,
        'scenario': args.scenario,
        'rows_by_segment': {name: int(mask.sum()) for name, mask in masks.items()},
        'shapes': {name: list(arrays[name].shape) for name in (
            'numeric_history', 'targets_standardized', 'targets_raw', 'semantic',
            'quality', 'frequency', 'text_available')},
        'first_origin_id': samples.origin_id.iloc[0],
        'first_raw_truth': arrays['targets_raw'][0].astype(float).tolist(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
