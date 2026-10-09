"""从实际验收记录生成交付结论；没有结果时拒绝输出完成声明。"""
from common17 import *


def main():
    verification = read(RESULTS / 'verification_replay.json')
    tests = read(RESULTS / 'tests.json')
    parity = read(RESULTS / 'inference_parity.json')
    portable = read(RESULTS / 'portable_check.json')
    reg = read(HERE / 'registry.json')
    assert verification['status'] == 'PASS_REPLAYED'
    assert [r['round'] for r in verification['rounds']] == list(ROUNDS)
    assert all(r['replay_executed'] and r['status'] == 'PASS' for r in verification['rounds'])
    assert verification['entry_sha256'] == sha(HERE / 'portal17.py')
    assert verification['registry_sha256'] == sha(HERE / 'registry.json')
    assert tests['passed'] and tests['source_sha256'] == sha(HERE / 'test17.py')
    assert parity['status'] == 'PASS' and parity['count'] == 14
    assert portable['status'] == 'PASS'
    assert not check_map(ROOT, reg['tracked_files'])
    lines = [
        'SafeFAME-TS 第17轮技术整合与验收', '完成日期：2026-10-09', '',
        '一、结论',
        '支持性分析：统一目录、分级验收和推理入口已实际运行。第13至16轮重新复算全部通过，14个历史起点的统一预测与原入口逐项一致。新增工程整合没有改变旧模型或选型规则。',
        '对立性视角：本轮提高的是可使用性和可复核性，没有产生新的性能提升证据。首次公开文本证据缺口仍在，严格推理text_weight=0。不能把历史回放、来源合规或回退机制说成多模态优越性和理论安全保证。', '',
        '二、已交付内容',
        '1. 第1至16轮目录、代码、协议和结果索引，登记483个历史源码与结果文件及8类本地证据资源。',
        '2. catalog / verify / example / predict统一命令；各轮在独立进程中调用原实现。',
        '3. PASS_INTEGRITY_ONLY、PASS_REPLAYED、PARTIAL、FAIL四类结果；后两者分别退出码2和1。',
        '4. 第13至16轮的14个历史输入示例；明确标注历史模型、固定策略和文本禁用限制。',
        '5. README已恢复三人组队、第4周方案提交、第10周PPT分享等课件要求。',
        '6. 技术贡献与证据边界、运行说明和本次验收记录。', '',
        '三、实际验收',
        f"异常与完整性测试：{tests['tests_run']}项通过，错误{tests['errors']}、失败{tests['failures']}。",
        f"原入口与统一入口预测对照：{parity['count']}个案例全部相等。",
        f"历史文件前后字节检查：{verification['historical_files_checked']}个，未改写。",
        '34个旧文件存在Git正常换行转换；兼容清单只接受登记时本地字节或已核对的基准Git对象字节，不接受任意格式变化。',
        '从Git对象构造无大型证据的副本：实际返回PARTIAL和退出码2；删除一个应存在的协议文件后返回FAIL和退出码1，均未伪称执行复算。', '',
        '四、各轮复算范围（不能机械相加为独立实验数）',
    ]
    for row in verification['rounds']:
        record = row['replay_record']['record']
        lines.append(f"第{row['round']}轮：{record['status']}；新记录：{row['replay_record']['output']}")
    lines += ['', '第13轮：24条深度训练记录、8个对照情景、24个数值滚动案例；复算已保存证据，并未重新训练全部模型。',
              '第14轮：354份原始快照、2630个窗口起点、12条主深度路线、2次因果反馈回放。',
              '第15轮：58条指标、464次在线选择、6个固定选择、2个残差分解；仍是已看历史结果上的探索。',
              '第16轮：54条主指标、18条种子指标、1392次在线选择，三州六任务。', '',
              '五、交付与版本',
              '推送状态以交付目录VERSION.json及远端回执为准。本报告不提前声称上传成功。',
              '分支：team/main-eval-20260921；未合并main。',
              '交付目录：D:\\aaa2026人工智能课程报告\\Codex交付文件\\主控技术整合_第17轮_20261009',
              '交付包含代码、索引、运行说明、示例和验收结果；原始数据与模型继续保留在主控接收数据，不重复打包或上传Git。',
              '交付包不是独立便携运行环境；根目录启动脚本指向本机原仓库，源码副本用于归档。', '',
              '六、尚未完成与下一步',
              '第1至12轮本轮只做索引和文件核查，未重新执行全部历史实验；旧DOCX和正式PPT未在本轮改写。',
              '建议转入课程成果整合：明确研究问题和可证实贡献，引用成功与失败案例，用统一入口做真实演示。',
              '若再开展研究，应优先解决首次公开版本或真正未来数据验证，不能靠继续添加模型替代独立证据。']
    (RESULTS / '00_第17轮验收与技术结论.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')
    catalog = ['第1至16轮历史成果索引', '', '以下路径相对于SafeFAME-TS_主控技术工作区。',
               '各轮文件数量不代表实验数量；未重新复算的旧轮次不标记为本轮PASS。', '']
    for item in reg['catalog']:
        catalog += [f"第{item['round']}轮：{item['coverage']}",
                    f"登记文件{len(item['files'])}；代码/脚本{len(item['code'])}；CSV结果表{len(item['tables'])}。"]
        catalog += ['协议：' + p for p in item['protocols']]
        notes = [p for p in item['files'] if p.endswith(('.md', '.txt')) and not p.endswith('requirements.txt')]
        catalog += ['说明：' + p for p in notes[:8]]
        if item['round'] in ROUNDS:
            catalog += [f"重放：Run-Round17.ps1 -Arguments @('verify','--replay','--round','{item['round']}')",
                        '本地证据：' + ', '.join(reg['resources'][k]['path'] for k in item['local_resources'])]
        catalog.append('')
    (RESULTS / '历史轮次索引.txt').write_text('\n'.join(catalog).rstrip() + '\n', encoding='utf-8-sig')
    source = {p.name: sha(p) for p in HERE.iterdir() if p.is_file()}
    dump(RESULTS / 'source_manifest.json', {'files': source, 'role': '第17轮最终源码/说明/登记清单字节'})
    print('第17轮技术报告已按实际验收记录生成')


if __name__ == '__main__':
    main()
