"""登记当前已提交的历史源码/结果和本机证据；不是重新认证历史实验。"""
import subprocess
from common17 import *


def main():
    dest = HERE / 'registry.json'
    if dest.exists():
        raise ValueError('登记已存在，拒绝覆盖；新基准需要新版本')
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    selected = [p for p in files if p and
                (p.startswith('team_work/main/') or p.startswith('src/') or p.startswith('configs/'))
                and not p.startswith('team_work/main/round17/')]
    frozen = {}
    normalized = []
    for rel in selected:
        actual = sha(ROOT / rel)
        original = hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT)).hexdigest()
        if actual != original:
            clean = subprocess.check_output(['git', 'hash-object', '--path=' + rel, rel], cwd=ROOT, text=True).strip()
            obj = subprocess.check_output(['git', 'rev-parse', 'HEAD:' + rel], cwd=ROOT, text=True).strip()
            if clean != obj:
                raise ValueError('历史文件存在未提交修改，不能登记: ' + rel)
            normalized.append(rel)
        frozen[rel] = actual
    resources = {}
    for key, rel in RESOURCE_PATHS.items():
        p = DATA / rel
        if not p.exists():
            raise ValueError('登记所需本地证据缺失: ' + str(p))
        if p.is_file():
            resources[key] = {'path': rel, 'kind': 'file', 'sha256': sha(p), 'bytes': p.stat().st_size}
        else:
            members = [f for f in sorted(p.rglob('*')) if f.is_file() and not
                       any(x in f.parts for x in ('__pycache__', '.git'))]
            resources[key] = {'path': rel, 'kind': 'directory',
                              'files': {f.relative_to(p).as_posix(): sha(f) for f in members},
                              'bytes': sum(f.stat().st_size for f in members)}
    catalog = []
    for n in range(1, 17):
        prefix = f'team_work/main/round{n}/' if n > 1 else 'team_work/main/主控电脑交付01次'
        members = [f for f in frozen if f.startswith(prefix)]
        catalog.append({'round': n, 'files': members,
                        'protocols': [f for f in members if 'protocol' in f or 'split_spec' in f],
                        'tables': [f for f in members if f.endswith('.csv')],
                        'code': [f for f in members if f.endswith(('.py', '.ps1'))],
                        'replay_supported': n in ROUNDS,
                        'local_resources': DEPENDENCIES.get(n, []),
                        'coverage': '可选本轮重新复算' if n in ROUNDS else '历史索引与字节检查，未接入本轮复算'})
    dump(dest, {'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'registration_role': '当前文件身份登记；历史结果有效性另由原复算器检验',
                'checkout_normalized_files': normalized,
                'tracked_files': frozen, 'resources': resources, 'catalog': catalog})
    print(json.dumps({'tracked': len(frozen), 'resources': len(resources),
                      'bytes': sum(v['bytes'] for v in resources.values())}))


if __name__ == '__main__':
    main()
