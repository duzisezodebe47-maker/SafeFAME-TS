"""每轮独立进程导入，避免common等同名模块污染；历史复算只重定向输出。"""
import contextlib
import importlib
import io
from common17 import *


def redirect_outputs(module, round_id):
    old = ROOT / f'team_work/main/round{round_id}/results'
    new = RESULTS / 'replays' / f'round{round_id}'
    written = []
    for name in ('dump', 'csv'):
        if not hasattr(module, name):
            continue
        original = getattr(module, name)
        def redirected(path, value, fn=original):
            path = Path(path).resolve()
            if not path.is_relative_to(old.resolve()):
                raise ValueError('复算器试图写入未授权历史目录: ' + str(path))
            target = new / path.relative_to(old)
            result = fn(target, value)
            written.append(target.resolve())
            return result
        setattr(module, name, redirected)
    return written


def execute(round_id, action, payload):
    if round_id not in ROUNDS:
        raise ValueError('未接入该轮')
    sys.path.insert(0, str(ROOT / f'team_work/main/round{round_id}'))
    if action == 'replay':
        module = importlib.import_module(MODULES[round_id][0])
        written = redirect_outputs(module, round_id)
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(2)
        with threadpool_limits(limits=1):
            module.verify()
        rel = Path(MODULES[round_id][2]).relative_to('results')
        path = RESULTS / 'replays' / f'round{round_id}' / rel
        if path.resolve() not in written:
            raise ValueError('本次复算未生成验收记录，不得复用旧PASS')
        return {'record': read(path), 'output': path.relative_to(HERE).as_posix()}
    module = importlib.import_module(MODULES[round_id][1])
    if action == 'predict':
        import torch
        torch.set_num_threads(2)
        return module.forecast(payload)
    if action != 'example':
        raise ValueError('未知动作')
    if round_id == 13:
        example = module.example(payload['task'])
        return {'task': payload['task'], 'model': 'StrictSourceFallback', 'history': example['history']}
    if round_id in (14, 15):
        return {'horizon': payload['horizon'], 'history': module.example(payload['horizon'])['history']}
    import numpy as np
    state, h = payload['state'], payload['horizon']
    f = module.LOCAL / f'prepared/{state}_h{h}/test_features.npz'
    scale = read(module.LOCAL / f'prepared/{state}/scaler.json')
    x = np.load(f)['x'][-1]
    return {**payload, 'history': (x * scale['std'] + scale['mean']).tolist()}


if __name__ == '__main__':
    round_id, action = int(sys.argv[1]), sys.argv[2]
    payload = json.load(sys.stdin)
    logs = io.StringIO()
    with contextlib.redirect_stdout(logs):
        value = execute(round_id, action, payload)
    print(json.dumps({'value': value, 'log': logs.getvalue()}, ensure_ascii=False, allow_nan=False))
