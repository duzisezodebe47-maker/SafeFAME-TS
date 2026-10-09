"""针对错误通过、证据缺失/损坏混淆及跨轮导入的集成测试。"""
import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch
from common17 import *
import portal17 as portal
import worker17 as child


class Acceptance(unittest.TestCase):
    def test_missing_root_is_partial(self):
        with tempfile.TemporaryDirectory(dir=DATA / 'runtime_tmp') as d:
            index = {'x': {'kind': 'directory', 'path': 'absent', 'files': {'model': '0'}}}
            self.assertEqual(check_resources(index, ['x'], Path(d))['status'], 'PARTIAL')

    def test_missing_member_is_failure(self):
        with tempfile.TemporaryDirectory(dir=DATA / 'runtime_tmp') as d:
            (Path(d) / 'existing').mkdir()
            index = {'x': {'kind': 'directory', 'path': 'existing', 'files': {'model': '0'}}}
            self.assertEqual(check_resources(index, ['x'], Path(d))['status'], 'FAIL')

    def test_corruption_overrides_absence(self):
        with tempfile.TemporaryDirectory(dir=DATA / 'runtime_tmp') as d:
            (Path(d) / 'bad').write_text('tampered')
            index = {'x': {'kind': 'file', 'path': 'bad', 'sha256': '0'},
                     'y': {'kind': 'directory', 'path': 'missing', 'files': {'a': '0'}}}
            result = check_resources(index, ['y', 'x'], Path(d))
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(len(result['missing_resources']), 1)

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            beneath(HERE, '../round16/protocol.json')

    def test_input_contract(self):
        valid = {'round': 16, 'payload': {'state': 'CA', 'horizon': 4,
                 'rule': 'CalChoice', 'history': [1.0] * 52}}
        portal.validate_request(valid)
        for field, value in [('state', '../CA'), ('horizon', True), ('rule', 'test_best'),
                             ('history', [float('nan')] * 52), ('history', [True] * 52),
                             ('history', [501] * 52), ('history', [1] * 51)]:
            with self.subTest(field=field, value=str(value)[:20]), self.assertRaises(ValueError):
                portal.validate_request({'round': 16, 'payload': {**valid['payload'], field: value}})
        with self.assertRaises(ValueError):
            portal.validate_request({**valid, 'future_truth': [0]})
        with self.assertRaises(ValueError):
            portal.validate_request({'round': 16, 'payload': {**valid['payload'], 'model_path': 'bad'}})

    def test_incomplete_evidence_never_predicts(self):
        payload = {'round': 14, 'payload': {'horizon': 4, 'history': [1.0] * 52}}
        with patch.object(portal, 'preflight', return_value={'status': 'PARTIAL'}), patch.object(portal, 'worker') as call:
            self.assertFalse(portal.predict(payload)['prediction_executed'])
            call.assert_not_called()

    def test_replay_exception_not_pass(self):
        with patch.object(portal, 'registry', return_value={'tracked_files': {}, 'resources': {}}), \
             patch.object(portal, 'check_tracked', return_value=[]), \
             patch.object(portal, 'check_resources', return_value={'status': 'PASS'}), \
             patch.object(portal, 'worker', side_effect=RuntimeError('failed')), patch.object(portal, 'dump'):
            self.assertEqual(portal.verify([16], True)['status'], 'FAIL')

    def test_replay_requires_fresh_record(self):
        from types import SimpleNamespace
        # Even an older PASS on disk cannot count if this invocation wrote nothing.
        with patch.object(child.importlib, 'import_module', return_value=SimpleNamespace(verify=lambda: None)):
            with self.assertRaisesRegex(ValueError, '未生成验收记录'):
                child.execute(16, 'replay', None)

    def test_frozen_rounds_unchanged(self):
        self.assertEqual(check_map(ROOT, portal.registry()['tracked_files']), [])

    def test_only_registered_byte_variants_allowed(self):
        import hashlib
        a, b = b'a\r\n', b'a\n'
        with tempfile.TemporaryDirectory(dir=DATA / 'runtime_tmp') as d:
            p = Path(d) / 'data'; p.write_bytes(b)
            mapping = {'data': hashlib.sha256(a).hexdigest()}
            variants = {'data': hashlib.sha256(b).hexdigest()}
            self.assertEqual(check_map(Path(d), mapping, variants), [])
            p.write_bytes(b'changed\n')
            self.assertTrue(check_map(Path(d), mapping, variants))


def integration():
    """14个历史起点，覆盖所有接入任务；与原入口直接调用逐项相等。"""
    cases = [(13, {'task': t}) for t in read(ROOT / 'team_work/main/round13/protocol.json')['tasks']]
    cases += [(n, {'horizon': h}) for n in (14, 15) for h in (4, 12)]
    cases += [(16, {'state': s, 'horizon': h, 'rule': 'three_block_minimax_relative_regret'})
              for s in ('CA', 'TX', 'IA') for h in (4, 12)]
    rows = []
    for n, args in cases:
        print(f'真实推理: round{n} {args}', file=sys.stderr, flush=True)
        p = worker(n, 'example', args)['value']
        actual = portal.predict({'round': n, 'payload': p})
        expected = worker(n, 'predict', p)['value']
        if actual['status'] != 'PASS' or actual['result'] != expected:
            raise AssertionError('统一入口与原入口不一致')
        rows.append({'round': n, 'case': args, 'status': 'PASS', 'result': actual['result']})
        tag = args.get('task', f"{args.get('state', 'CONUS')}_h{args.get('horizon')}")
        dump(RESULTS / 'examples' / f'round{n}_{tag}.json',
             {'status': 'PASS', 'request': {'round': n, 'payload': p}, 'prediction': actual,
              'role': '最后一个历史测试起点的历史输入，不是当前实时数据'})
    dump(RESULTS / 'inference_parity.json', {'status': 'PASS', 'cases': rows, 'count': len(rows)})


if __name__ == '__main__':
    buffer = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Acceptance)
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
    print(buffer.getvalue())
    dump(RESULTS / 'tests.json', {'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
                                 'errors': len(result.errors), 'failures': len(result.failures),
                                 'source_sha256': sha(Path(__file__))})
    (RESULTS / 'tests.log').write_text(buffer.getvalue(), encoding='utf8')
    if not result.wasSuccessful():
        sys.exit(1)
    integration()
