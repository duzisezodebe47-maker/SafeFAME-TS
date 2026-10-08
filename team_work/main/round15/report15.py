"""根据已复核数值生成技术结论、图表及下一步边界。"""
from common15 import *

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':10})
    frames=[pd.read_csv(HERE/f'results/h{h}/metrics.csv') for h in (4,12)];metrics=pd.concat(frames,ignore_index=True);csv(HERE/'results/all_metrics.csv',metrics.to_dict('records'))
    captions={'OriginalCalibrationChoice':'原校准期选择','decision_mean':'决策段均值选择','three_block_minimax_relative_regret':'三段最坏相对遗憾','OnlineMatured26':'26起点在线选择','OnlineMatured52':'52起点在线选择'}
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for h,ax in zip((4,12),axes):
        df=metrics[(metrics.horizon==h)&(metrics.group=='numeric_selection')&metrics.candidate.isin(captions)]
        ax.bar([captions[n] for n in df.candidate],df.mse,color=['#687c92','#337da8','#46a8a6','#e28755','#337da8']);ax.tick_params(axis='x',rotation=22);ax.set_ylabel('MSE（训练期标准化尺度）');ax.set_title(f'{h}周预测');ax.grid(axis='y',alpha=.2)
    fig.suptitle('模型选择规则比较：短窗口在线切换也会明显退化\n沿用已检查时段的探索性分析，不是新的盲测',fontsize=13);fig.tight_layout();fig.savefig(HERE/'results/模型选择稳定性.png',dpi=180);plt.close(fig)
    labels={'SelectedNumeric':'原选定数值','OriginalText':'完整文本修正','ConstantResidual':'仅整体偏差修正','NumericStateResidual':'仅数值状态修正','CenteredText':'去均值文本修正'}
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for h,ax in zip((4,12),axes):
        df=metrics[(metrics.horizon==h)&(metrics.group=='attribution')&metrics.candidate.isin(labels)]
        ax.bar([labels[n] for n in df.candidate],df.mse,color=['#687c92','#e28755','#337da8','#46a8a6','#ad6473']);ax.tick_params(axis='x',rotation=22);ax.set_ylabel('MSE（训练期标准化尺度）');ax.set_title(f'{h}周预测 · 保持原数值基线不变');ax.grid(axis='y',alpha=.2)
    fig.suptitle('同一组训练残差的配对对照：文本贡献尚未独立确立',fontsize=13);fig.tight_layout();fig.savefig(HERE/'results/文本残差归因.png',dpi=180);plt.close(fig)
    lines=['SafeFAME-TS 第15轮：模型选择稳定性与文本残差归因','日期：2026-10-08',
    '\n一、结论',
    '本轮完成了可运行的多时段数值选择、成熟反馈在线切换、整体偏差/数值状态/去均值文本对照、文本陈旧度对照及严格数值推理入口。',
    '支持性分析：固定数值模型权重后，决策段均值选择和三段最坏相对遗憾选择均在4周任务选Ridge、12周任务选Last，改善上一轮选型结果。这说明应优先修复选择规则，而不是单纯扩大网络。',
    '对立性视角：两种固定规则本轮选到相同模型，不能证明三段规则本身优于均值规则。12周短窗口在线选择明显失败；决策段通过的去均值文本门槛在后续时段也失败。启发式安全门槛不是普遍安全保证。',
    '\n二、实验身份与时间边界',
    '本轮明确属于已经查看过的第14轮数据上的后续探索分析，不是全新留出测试，不是前瞻性预测。新增协议在本轮计算前封存，但无法消除研究者已知道上一轮结果的影响。',
    '沿用官方CONUS DSCI任务及4/12周目标。数值模型权重固定为第14轮训练加校准重拟合结果，不使用决策段或测试段重新训练。决策段截至2024-05-14；后续评价目标范围2024-05-21至2026-09-22。',
    '比较Last、SeasonalNaive、Ridge、DLinear、PatchTST五个固定数值候选。三段规则先把决策起点按顺序切为三个连续块，再对每个模型取“块MSE/该块最佳MSE-1”的最大值，选择最小者；同分按名称排序。',
    '这三块来自同一条序列，预测窗口会重叠，不是三个独立数据集，也不是三次重新训练的滚动交叉验证。我们比较的是同一批固定模型的选型规则。',
    '在线选择分别保留26或52个最近完整成熟起点。对过去起点i，仅在i+H不晚于当前起点时允许读取其完整目标；起点有空缺时仍按真实周索引判定。',
    '\n三、量化结果']
    for h in (4,12):
        lines.append(f'\n预测{h}周：')
        for group,names in [('numeric_selection',list(captions)),('attribution',list(labels))]:
            frame=metrics[(metrics.horizon==h)&(metrics.group==group)].set_index('candidate')
            for name in names:
                row=frame.loc[name];lines.append(f'  {captions.get(name,labels.get(name,name))}：MSE={row.mse:.6f}，相对第14轮原选定数值模型改善={row.gain_vs_round14_selected_pct:.3f}%。')
        summary=read(HERE/f'results/h{h}/summary.json')
        for name,info in summary['online'].items():lines.append(f'  {name}：切换{info["switches"]}次；模型使用次数={info["choices"]}。')
    lines += ['\n四、文本归因怎么做',
    '为避免将数值模型变化混进文本贡献分析，全部文本对照始终固定第14轮原选定数值模型：4周Last、12周PatchTST。使用完全相同的99/91个训练段OOF残差起点，以及相同的文本可用掩码。',
    '仅整体偏差修正：训练OOF残差按预测步分别取均值，只在相同文本可用起点上加回该常数。它不使用正文，但保留相同可用性信息；并不是对全部起点无条件加偏差。',
    '仅数值状态修正：以最后观测值、4周变化、13周变化、13周标准差、年周期正弦和余弦为输入。特征尺度只用同一批OOF训练起点估计。Ridge候选0.1/1/10/100在原校准段上选择；4周alpha=100、12周alpha=10。校准对照使用原训练期模型的校准预测，而非已在校准段重拟合的预测。',
    '去均值文本修正：从原文本修正中减去训练OOF残差均值。完整文本修正=整体偏差项+去均值项，逐数组验证恒等式。它是分解诊断，不是重新训练的新文本模型，也不是“纯语义效应”的因果估计。',
    '4周完整文本MSE=0.094477，整体偏差修正=0.091398；简单修正更好。12周完整文本=0.476433、整体偏差修正=0.508873，但两者配对块Bootstrap区间包含0。不能据此宣布文本具有稳定独立收益。',
    '\n五、陈旧度与门槛失败',
    '分别限制报告距离最后已观测地图周0/1/4/8周；超限时退回相同数值基线。两步长均完整报告全部设置，不根据本轮后续误差选择最好阈值。',
    '4周测试期可用文本起点数依次62/89/114/119，12周依次55/81/106/111。更短陈旧度条件使部分结果改善，但未证明语义时效性，可能仅来自更少启用文本或样本构成改变。',
    '每个对照另在决策段选择0/0.25/0.5/0.75/1权重，非零权重要求前后两半平均收益均为正。去均值文本在两步长都选择权重1，却在后续时段分别退化到MSE 0.105506和0.731332。该失败证据必须保留，不能将规则称作保证不退化。',
    '多数其他对照权重为0。直接报告的探索对照分数与经过门槛的可用路线分数分表保存，不能拿前者替换后者。',
    '\n六、统计与来源边界',
    '使用2000次循环块Bootstrap，块长H和2H，种子2026。报告逐项探索区间，不作多重比较后的显著性结论。12周大块长仅约4个块，支持有限。',
    '逐篇首次公开时间的严格合格文本仍为0；本轮没有新增或伪造时间证明，严格预测入口文本权重保持0。CONUS DSCI仍是相对旧Climate目标单独定义的任务。',
    '\n七、工程验证与运行',
    'verify15.py核对74个登记输入/代码文件，独立复算58条MSE指标、464次在线选择、6次固定选择、2套修正分解。详见verification.json。',
    '11项回归测试覆盖最坏块规则与均值规则的区别、同分排序、异常损失、起点缺口、未来标签扰动、时间顺序、双半段门槛、报告年龄边界、历史特征、Bootstrap及两步长真实推理与保存结果一致。',
    'predict15.py提供严格数值预测：默认使用事先定义的三段最坏相对遗憾规则，固定4周Ridge、12周Last。显式标明探索性策略，未替换第14轮历史记录或演示。',
    '\n八、下一步优先事项',
    '1. 若继续提出新的选型或回退规则，应在另立评估设计中比较，最终等待真正未使用的后续数据验证。不能持续在当前时段试验，再把最好的设置包装为泛化提升。',
    '2. 优先解决首次公开版本证据和多序列验证；当前不建议继续以增加模型数量为目标。',
    '3. 对短窗口切换与门槛失败，应研究状态变化和模型误差持久性，采用预先约定的比较，而非事后调整阈值消除失败记录。',
    '4. 现有工程已能支持完整的课程技术讲解：数据来源、时点约束、强基线、反事实、失败与可复算代码。科研上稳定文本增益和前瞻有效性仍未完成。']
    (HERE/'results/01_第15轮技术结论.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    instructions=rf'''第15轮运行和文件说明
源码目录：{HERE}
大型证据目录：{LOCAL}
依赖第14轮原始准备结果和模型：{PREVIOUS}

进入源码目录后，使用现有Python3.12环境：
powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-Round15.ps1 -Entry verify15.py
powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-Round15.ps1 -Entry test15.py

严格预测示例（在PowerShell中）：
.\Run-Round15.ps1 -Entry predict15.py -Arguments @('--horizon','4','--example')
.\Run-Round15.ps1 -Entry predict15.py -Arguments @('--horizon','12','--example')
--example取第14轮最后一个评价历史窗口，不是今天的实时数据。输出是原始DSCI点数。
自定义预测：将52个0至500之间的历史数值保存为JSON数组文件，然后用--history-json 'D:\路径\history.json'替换--example。

从头复算顺序：run15.py → verify15.py → test15.py → report15.py。
run15.py拒绝覆盖已存在第15轮大型证据。需要重跑时，在另一个完整项目副本中恢复相同相对目录的输入后运行，保留当前封存记录。不要手工改哈希。
results/all_metrics.csv是58条完整指标；numeric_selection、attribution、decision_gated_controls是三个不同问题，不能混成同一场候选模型竞赛。
results/h*/selection_scores.csv记录固定选择依据；decision_block_scores.csv给出三个连续块的实际得分。
results/h*/gate_selection.csv记录每种修正的决策期权重选择；text_control_intervals.csv给出配对区间。
大型证据h*/OnlineMatured*_trace.csv记录每个预测时点选择及最后反馈起点。
predict15.py每次推理核对已登记输入和选择策略；不接受正文输入或任意模型文件。

本轮是后续探索：已知道第14轮结果，不能表述为新的盲测或前瞻验证。
所有本轮文件在D盘，调用已有解释器，不新安装组件。交付包不含完整Python便携环境。
git仅推送主控分支；原始数据、权重和大型逐样本证据保存在D盘专门目录，不上传main。
'''
    (HERE/'results/02_运行说明.txt').write_text(instructions,encoding='utf-8-sig')
    print('Reports and figures saved')

if __name__=='__main__':main()
