"""统一目录、分级验收与历史数值推理；不训练、不调参、不启用未经核验文本。"""
import argparse
import math
from common17 import *


def registry():
    return read(HERE / 'registry.json')


def validate_request(value):
    if not isinstance(value, dict) or set(value) != {'round', 'payload'}:
        raise ValueError('仅接受round和payload，禁止未来标签、外部路径等附加字段')
    n, p = value['round'], value['payload']
    if type(n) is not int or n not in ROUNDS or not isinstance(p, dict):
        raise ValueError('round必须为13至16的整数，payload必须为对象')
    required = {13: {'task', 'model', 'history'}, 14: {'horizon', 'history'},
                15: {'horizon', 'history'}, 16: {'state', 'horizon', 'rule', 'history'}}[n]
    if set(p) != required:
        raise ValueError('该轮字段必须为: ' + ','.join(sorted(required)))
    if n == 13:
        protocol = read(ROOT / 'team_work/main/round13/protocol.json')
        if p['task'] not in protocol['tasks'] or p['model'] != 'StrictSourceFallback':
            raise ValueError('统一入口第13轮只使用已固定的严格回退模型和四个登记任务')
    else:
        if type(p['horizon']) is not int or p['horizon'] not in (4, 12):
            raise ValueError('horizon只能为4或12')
        if n == 16:
            rules = read(ROOT / 'team_work/main/round16/protocol.json')['fixed_rules']
            if p['state'] not in ('CA', 'TX', 'IA') or p['rule'] not in ['CalChoice'] + rules:
                raise ValueError('州别或固定规则无效')
    history = p['history']
    if not isinstance(history, list) or not history or any(type(x) not in (float, int) or not math.isfinite(x) for x in history):
        raise ValueError('history必须为有限数值列表，不接受布尔值')
    if n != 13 and (len(history) != 52 or any(x < 0 or x > 500 for x in history)):
        raise ValueError('DSCI需要52个0至500的历史数值')
    return n, p


def preflight(n, reg=None):
    reg = reg or registry()
    issues = check_tracked(reg)
    result = check_resources(reg['resources'], DEPENDENCIES[n])
    result['issues'] += issues
    if issues:
        result['status'] = 'FAIL'
    return result


def predict(request):
    n, payload = validate_request(request)
    check = preflight(n)
    if check['status'] != 'PASS':
        return {'status': check['status'], 'prediction_executed': False, 'evidence': check}
    result = worker(n, 'predict', payload)['value']
    if result.get('text_weight') != 0:
        raise ValueError('严格入口不允许未经核实文本进入预测')
    return {'status': 'PASS', 'prediction_executed': True, 'round': n, 'result': result,
            'boundary': '保存的历史模型推理；未校验用户序列的实际时间与来源，不代表实时或前瞻效果'}


def verify(rounds=ROUNDS, replay=False):
    reg = registry()
    static_issues = check_tracked(reg)
    rows = []
    for n in rounds:
        check = check_resources(reg['resources'], DEPENDENCIES[n])
        row = {'round': n, 'evidence': check, 'replay_executed': False}
        if static_issues or check['status'] == 'FAIL':
            row['status'] = 'FAIL'
        elif check['status'] == 'PARTIAL':
            row['status'] = 'PARTIAL'
        elif not replay:
            row['status'] = 'PASS_INTEGRITY_ONLY'
        else:
            try:
                result = worker(n, 'replay')
                dump(RESULTS / 'logs' / f'round{n}.json', result)
                row.update(replay_executed=True, replay_record=result['value'])
                row['status'] = 'PASS' if result['value']['record']['status'].startswith('PASS') else 'FAIL'
            except Exception as e:
                row.update(status='FAIL', error=str(e))
        rows.append(row)
        print(f"第{n}轮: {row['status']}", file=sys.stderr, flush=True)
    # 输出重定向不应导致任何历史文件变动。
    after = check_tracked(reg)
    status = ('FAIL' if static_issues or after or any(r['status'] == 'FAIL' for r in rows)
              else 'PARTIAL' if any(r['status'] == 'PARTIAL' for r in rows)
              else 'PASS_REPLAYED' if replay else 'PASS_INTEGRITY_ONLY')
    result = {'status': status, 'rounds': rows, 'historical_files_checked': len(reg['tracked_files']),
              'static_issues': static_issues, 'after_issues': after,
              'scope': '仅所选第13至16轮可重放；第1至12轮只检查已登记文件。未重训，不是新实验。',
              'registry_sha256': sha(HERE / 'registry.json'),
              'entry_sha256': sha(Path(__file__))}
    dump(RESULTS / ('verification_replay.json' if replay else 'verification_integrity.json'), result)
    return result


def example(n, task=None, state=None, horizon=None, rule=None):
    # 示例同样先检查证据，禁止缺数据时生成猜测值。
    p = {'task': task} if n == 13 else {'horizon': horizon}
    if n == 16:
        p.update(state=state, rule=rule)
    if n == 13 and task not in read(ROOT / 'team_work/main/round13/protocol.json')['tasks']:
        raise ValueError('第13轮必须明确指定登记任务')
    if n != 13 and horizon not in (4, 12):
        raise ValueError('必须明确指定horizon=4或12')
    if n == 16:
        validate_request({'round': n, 'payload': {**p, 'history': [0.0] * 52}})
    check = preflight(n)
    if check['status'] != 'PASS':
        return {'status': check['status'], 'example_executed': False, 'evidence': check}
    request = {'round': n, 'payload': worker(n, 'example', p)['value']}
    prediction = predict(request)
    result = {'status': prediction['status'], 'request': request, 'prediction': prediction,
              'role': '该轮最后一个历史测试起点的输入示例，不是今日实时预测'}
    tag = task if n == 13 else f'{state or "CONUS"}_h{horizon}_{rule or "fixed"}'
    dump(RESULTS / 'examples' / f'round{n}_{tag}.json', result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest='command', required=True)
    subs.add_parser('catalog')
    v = subs.add_parser('verify'); v.add_argument('--replay', action='store_true'); v.add_argument('--round', type=int, choices=ROUNDS, nargs='+', default=list(ROUNDS))
    e = subs.add_parser('example'); e.add_argument('--round', type=int, choices=ROUNDS, required=True)
    e.add_argument('--task'); e.add_argument('--state', choices=['CA', 'TX', 'IA']); e.add_argument('--horizon', type=int, choices=[4, 12]); e.add_argument('--rule')
    p = subs.add_parser('predict'); p.add_argument('--input', type=Path, required=True)
    args = ap.parse_args()
    try:
        if args.command == 'catalog':
            result = {'catalog': registry()['catalog'], 'scope': '目录不等于验收通过，需运行verify'}
        elif args.command == 'verify':
            result = verify(args.round, args.replay)
        elif args.command == 'example':
            result = example(args.round, args.task, args.state, args.horizon, args.rule)
        else:
            result = predict(read(args.input))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if result.get('status') == 'FAIL' else 2 if result.get('status') == 'PARTIAL' else 0
    except Exception as e:
        print(json.dumps({'status': 'FAIL', 'error': str(e)}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    sys.exit(main())
