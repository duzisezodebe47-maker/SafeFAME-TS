"""已提交版本的D盘技术交付包；不复制运行环境和大型历史数据。"""
import argparse
import shutil
from common17 import *


def main(receipt_path):
    dest = PROJECT / 'Codex交付文件/主控技术整合_第17轮_20261009'
    if dest.exists():
        raise ValueError('交付已存在，拒绝覆盖')
    record = read(RESULTS / 'verification_replay.json')
    if record['status'] != 'PASS_REPLAYED' or not read(RESULTS / 'tests.json')['passed']:
        raise ValueError('尚未通过验收')
    if check_map(HERE, read(RESULTS / 'source_manifest.json')['files']):
        raise ValueError('源码已变，需重新核验和生成报告')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if branch != 'team/main-eval-20260921':
        raise ValueError('错误分支')
    if subprocess.check_output(['git', 'status', '--porcelain', '--', 'team_work/main/round17', 'README.md'], cwd=ROOT).strip():
        raise ValueError('第17轮或README尚未提交')
    receipt = read(receipt_path)
    pushed = receipt.get('pushed') is True and receipt.get('commit') == commit == receipt.get('remote_commit')
    dest.mkdir(parents=True)
    copies = []
    def copy(src, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        if sha(src) != sha(target):
            raise ValueError('复制字节不一致')
        copies.append({'file': target.relative_to(dest).as_posix(), 'sha256': sha(target)})
    for src in HERE.rglob('*'):
        if src.is_file() and '__pycache__' not in src.parts:
            copy(src, dest / '第17轮代码与验收' / src.relative_to(HERE))
    copy(ROOT / 'README.md', dest / '仓库首页快照.md')
    for name in ('运行与验收说明.txt', '技术贡献与证据边界.txt'):
        copy(HERE / name, dest / name)
    copy(RESULTS / '00_第17轮验收与技术结论.txt', dest / '00_第17轮验收与技术结论.txt')
    copy(RESULTS / '历史轮次索引.txt', dest / '历史轮次索引.txt')
    copy(receipt_path, dest / 'push_receipt.json')
    for label, args in [('运行统一验收', "@('verify','--replay')"),
                        ('运行三州历史示例', "@('example','--round','16','--state','CA','--horizon','4','--rule','three_block_minimax_relative_regret')")]:
        ps = "$ErrorActionPreference='Stop'\n$project=(Get-Item $PSScriptRoot).Parent.Parent.FullName\n"
        ps += "$runner=Join-Path $project 'SafeFAME-TS_主控技术工作区/team_work/main/round17/Run-Round17.ps1'\n"
        ps += "if(-not (Test-Path -LiteralPath $runner)){ throw 'Original repository not found; see run instructions' }\n"
        ps += f'& $runner -Arguments {args}\nexit $LASTEXITCODE\n'
        (dest / (label + '.ps1')).write_text(ps, encoding='utf-8-sig')
    dump(dest / 'VERSION.json', {'branch': branch, 'commit': commit, 'pushed': pushed,
        'live_receipt': receipt, 'merged_main': False, 'runtime_included': False,
        'evidence_location': str(DATA), 'replay_status': record['status'],
        'historical_rounds_replayed': list(ROUNDS), 'inference_cases': 14})
    dump(dest / 'copy_verification.json', {'status': 'PASS', 'files': copies})
    manifest = {p.relative_to(dest).as_posix(): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
    dump(dest / 'MANIFEST.json', {'files': manifest})
    if check_map(dest, manifest):
        raise ValueError('交付清单验证失败')
    print(json.dumps({'delivery': str(dest), 'files': len(manifest) + 1, 'pushed': pushed,
                      'copied_verified': len(copies)}, ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--receipt', type=Path, required=True)
    main(ap.parse_args().receipt)
