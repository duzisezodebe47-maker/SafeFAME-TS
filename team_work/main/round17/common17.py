"""统一入口只读历史证据，新增输出仅写入第17轮。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROJECT = ROOT.parent
DATA = PROJECT / '主控接收数据'
RESULTS = HERE / 'results'
ROUNDS = (13, 14, 15, 16)
RESOURCE_PATHS = {
    'r13': '主控加强_20261007',
    'bundle': 'SafeFAME_TS_第7轮数据_02_a06dfe1/release_unpacked/inputs/formal_bundle',
    'lineage': 'SafeFAME_TS_第7轮数据_02_a06dfe1/release_unpacked/inputs/text_lineage.csv',
    'raw_old': 'SafeFAME_TS_第4轮_2d530ec7',
    'patchtst': 'PatchTST_official_204c21e/PatchTST_supervised',
    'r14': '主控第14轮_USDM_20261008',
    'r15': '主控第15轮_选择稳定性与归因_20261008',
    'r16': '主控第16轮_三州验证_20261008',
}
DEPENDENCIES = {
    13: ['r13', 'bundle', 'lineage', 'raw_old', 'patchtst'],
    14: ['r14', 'patchtst'],
    15: ['r14', 'r15', 'patchtst'],
    16: ['r16'],
}
MODULES = {
    13: ('verify_strengthening', 'predictor', 'results/final/verification.json'),
    14: ('verify_round14', 'predictor14', 'results/final/verification.json'),
    15: ('verify15', 'predict15', 'results/verification.json'),
    16: ('verify16', 'predict16', 'results/verification.json'),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def beneath(base, rel):
    """清单只能引用给定根目录内的文件。"""
    target = (Path(base) / rel).resolve()
    if not target.is_relative_to(Path(base).resolve()):
        raise ValueError('清单路径越界')
    return target


def env():
    e = os.environ.copy()
    e.update(PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8',
             OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
             TEMP=str(DATA / 'runtime_tmp'), TMP=str(DATA / 'runtime_tmp'),
             MPLCONFIGDIR=str(DATA / 'mpl_cache'))
    e['PYTHONPATH'] = os.pathsep.join(str(DATA / p) for p in ('runtime_torch_cpu', 'runtime_py312'))
    return e


def worker(round_id, action, payload=None):
    proc = subprocess.run([sys.executable, str(HERE / 'worker17.py'), str(round_id), action],
                          input=json.dumps(payload, ensure_ascii=False), text=True, encoding='utf-8',
                          capture_output=True, cwd=ROOT, env=env(), timeout=900)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-6000:] or proc.stdout[-6000:] or '子进程失败')
    return json.loads(proc.stdout)


def check_map(base, mapping, alternatives=None):
    issues = []
    for rel, digest in mapping.items():
        p = beneath(base, rel)
        if not p.is_file():
            issues.append({'path': str(p), 'reason': 'required_file_missing'})
        elif sha(p) not in {digest, (alternatives or {}).get(rel, digest)}:
            issues.append({'path': str(p), 'reason': 'hash_mismatch'})
    return issues


def check_tracked(reg, root=ROOT):
    compatibility = read(HERE / 'checkout_bytes.json')
    if compatibility['registry_sha256'] != sha(HERE / 'registry.json'):
        return [{'reason': 'checkout_registry_mismatch'}]
    alternatives = compatibility['git_blob_sha256']
    if set(alternatives) != set(reg['checkout_normalized_files']):
        return [{'reason': 'checkout_normalization_list_mismatch'}]
    return check_map(root, reg['tracked_files'], alternatives)


def check_resources(index, keys, data=DATA):
    """整个非Git证据根缺失可PARTIAL；已有根内部缺文件或损坏必须FAIL。"""
    missing, issues, checked = [], [], 0
    for key in keys:
        item = index[key]
        base = beneath(data, item['path'])
        if not base.exists():
            missing.append({'resource': key, 'path': str(base)})
            continue
        if item['kind'] == 'file':
            if not base.is_file() or sha(base) != item['sha256']:
                issues.append({'path': str(base), 'reason': 'hash_mismatch_or_wrong_type'})
            checked += 1
        elif not base.is_dir():
            issues.append({'path': str(base), 'reason': 'expected_directory'})
        else:
            issues.extend(check_map(base, item['files']))
            checked += len(item['files'])
    return {'status': 'FAIL' if issues else ('PARTIAL' if missing else 'PASS'),
            'missing_resources': missing, 'issues': issues, 'checked_files': checked}
