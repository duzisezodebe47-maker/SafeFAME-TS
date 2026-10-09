"""由Git对象构造无大型证据的临时副本，实际检查PARTIAL和缺文件FAIL。"""
import shutil
import tempfile
from common17 import *


if __name__ == '__main__':
    reg = read(HERE / 'registry.json')
    with tempfile.TemporaryDirectory(prefix='round17_checkout_', dir=DATA / 'runtime_tmp') as tmp:
        clone = Path(tmp) / 'repo'; clone.mkdir()
        for rel in reg['tracked_files']:
            p = beneath(clone, rel); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(subprocess.check_output(['git', 'show', reg['baseline_commit'] + ':' + rel], cwd=ROOT))
        entry = clone / 'team_work/main/round17'
        entry.mkdir(parents=True)
        for p in HERE.iterdir():
            if p.is_file():
                shutil.copy2(p, entry / p.name)
        checks = []
        for scenario in ('absent_evidence', 'missing_tracked_file'):
            if scenario == 'missing_tracked_file':
                (clone / 'team_work/main/round16/protocol.json').unlink()
            p = subprocess.run([sys.executable, str(entry / 'portal17.py'), 'verify', '--replay'],
                               cwd=clone, env=env(), capture_output=True, text=True, encoding='utf8', timeout=120)
            result = json.loads(p.stdout)
            expected, code = ('PARTIAL', 2) if scenario == 'absent_evidence' else ('FAIL', 1)
            if result['status'] != expected or p.returncode != code or any(r['replay_executed'] for r in result['rounds']):
                raise AssertionError(result)
            checks.append({'scenario': scenario, 'status': result['status'], 'exit_code': p.returncode,
                           'replay_executed': False})
        dump(RESULTS / 'portable_check.json', {'status': 'PASS', 'checks': checks,
              'scope': '由基准Git对象构造代码副本，不复制大型证据；没有安装新环境'})
        print(json.dumps(checks))
