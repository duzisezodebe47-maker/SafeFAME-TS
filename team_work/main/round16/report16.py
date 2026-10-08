"""从已经复算的六任务结果生成技术说明、完整失败记录和图表。"""
from common16 import *

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':10})
    metrics=pd.read_csv(HERE/'results/all_metrics.csv');aggregate=pd.read_csv(HERE/'results/aggregate.csv');source=read(HERE/'results/source/summary.json');verified=read(HERE/'results/verification.json');tests=read(HERE/'results/tests.json')
    labels={'Last':'Last','Ridge':'Ridge','DLinear':'DLinear','CalChoice':'校准期选型','decision_mean':'决策段均值','three_block_minimax_relative_regret':'三段最坏相对遗憾','recent_26_origins':'近26起点选型','Online26':'在线26','Online52':'在线52'}
    tasks=[f'{s["code"]}_h{h}' for s in cfg()['states'] for h in cfg()['horizons']]
    methods=['Ridge','DLinear','CalChoice','decision_mean','three_block_minimax_relative_regret','Online26','Online52']
    mat=np.array([[metrics[(metrics.state==task.split('_h')[0])&(metrics.horizon==int(task.split('_h')[1]))&(metrics.candidate==name)].iloc[0].gain_vs_last_pct for name in methods] for task in tasks])
    fig,ax=plt.subplots(figsize=(11,5.2));im=ax.imshow(mat,cmap='RdBu',vmin=-35,vmax=35,aspect='auto');ax.set_xticks(range(len(methods)),[labels[n] for n in methods],rotation=20);ax.set_yticks(range(len(tasks)),[t.replace('_h',' · ')+'周' for t in tasks])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):ax.text(j,i,f'{mat[i,j]:+.1f}%',ha='center',va='center',color='white' if abs(mat[i,j])>23 else '#182633')
    fig.colorbar(im,ax=ax,label='相对Last的MSE改善率（%）；负值为退化');ax.set_title('三个州 × 两个步长：所有预先固定任务均保留\n同领域历史验证，不能解释为六次独立实验');fig.tight_layout();fig.savefig(HERE/'results/三州模型对比.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.7));g=aggregate[aggregate.candidate!='Last'];x=np.arange(len(g));ax.bar(x,g.wins_vs_last,color='#327da7',label='优于Last');ax.bar(x,g.ties_vs_last,bottom=g.wins_vs_last,color='#a8b3bf',label='相同');ax.bar(x,g.losses_vs_last,bottom=g.wins_vs_last+g.ties_vs_last,color='#d17857',label='差于Last');ax.set_xticks(x,[labels[n] for n in g.candidate],rotation=20);ax.set_ylabel('任务数（总计6，任务之间相关）');ax.set_ylim(0,7.2);ax.legend(ncol=3,loc='upper right');ax.set_title('胜负汇总：简单基线值得保留，复杂规则存在失败');fig.tight_layout();fig.savefig(HERE/'results/规则胜负汇总.png',dpi=180);plt.close(fig)
    seeds=pd.read_csv(HERE/'results/dlinear_seed_metrics.csv');csv(HERE/'results/dlinear_seed_summary.csv',seeds.groupby(['state','horizon']).mse.agg(['mean','std','min','max']).reset_index().to_dict('records'))
    lines=['SafeFAME-TS 第16轮技术交付：三州验证与发布时间证据核查','2026-10-08',
    '\n一、结论先行',
    '支持性分析：完成了加利福尼亚CA、得克萨斯TX、艾奥瓦IA三个预先固定州的独立训练与统一评价。每州1395个连续周、两个步长，共6个任务；18条DLinear种子训练路线。正式增加了数据身份检查、完整目标期隔离、选型封存、因果反馈复放和可验证推理入口。',
    '对立性视角：该结果不支持“模型越复杂越好”。Ridge六个任务都优于Last，DLinear有两个退化任务；三段最坏相对遗憾规则只在两个任务改善、四个持平，未胜过固定Ridge。不能把保守选择不退化与最优准确率混为一谈。',
    '本轮没有把全国周报直接套到州级目标做文本融合。首次公开内容仍缺少逐篇证明，严格文本准入仍关闭。因此本轮加强的是跨序列数值验证和来源审计，不是多模态增益证明。',
    '\n二、预先固定范围与时间划分',
    '三个州、候选模型、训练参数和选型规则在首次下载州级数据前写入protocol.json，并在protocol_registration.json登记时间和哈希。没有根据后续评价结果更换州、步长、算法或参数。',
    '原始数据来自USDM官方StateStatistics/GetDSCI，FIPS分别06/48/19，响应中的州名与预期逐项核对。数据范围2000-01-04至2026-09-22，每州1395周，总4185个州周观测。',
    '训练截止2021-12-28，校准截止2022-12-27，决策截止2024-05-14，后续评价目标期2024-05-21至2026-09-22。输入52周，输出4/12周；目标跨分段边界的窗口剔除。各州均有120/112个评价起点。',
    '这是前面CONUS探索之后、方案固定的同领域多序列历史评价。三个州共同受到大范围气候因素影响，两个步长目标重叠；6个任务不是6次独立复制。它不是跨领域泛化、真正前瞻预测或历史版本已认证的回放。',
    '\n三、模型与迁移规则',
    '候选集合固定为Last、Ridge、DLinear。迁移第15轮的选型算法，不迁移CONUS模型权重；每州重新训练。这一轮三候选集合与第15轮五候选集合不同，不能把分数差异全部归因于地区变化。',
    '数值尺度只由训练段估计。Ridge候选alpha为0.01/0.1/1/10/100/1000，仅按训练模型在校准段误差选择。',
    'DLinear复用项目中的分解结构：25点移动平均分出趋势与剩余项，两条线性映射输出H步。种子2026/2027/2028，AdamW学习率0.001、权重衰减0.0001、batch64、最多60轮、patience8、min_delta=1e-6。',
    '各种子先以训练段拟合并选定校准最佳轮数，再以训练加校准段重拟合相同轮数。18条主训练路线各含这两个阶段，不把36次拟合当成36个独立样本。三个种子的预测先平均得到主DLinear分数，单种子均值/标准差另存。',
    '固定规则包括校准期赢家、决策段均值最小、三段最坏相对遗憾最小、最近26个决策起点均值最小。三段规则把决策段等分三个连续块，对每个模型计算max(块MSE/块最佳MSE-1)，选择最小值；同分按名称排序。',
    '在线规则分别保留26/52个最近已完整成熟起点，只有过去起点i+H不晚于当前起点时才读取完整标签。六个任务选择封印全部保存后，先保存固定预测与哈希，再统一评分。',
    '\n四、完整量化结果（相对Last的MSE改善，负数为退化）']
    for row in aggregate.itertuples():lines.append(f'{labels[row.candidate]}：胜{row.wins_vs_last}/平{row.ties_vs_last}/负{row.losses_vs_last}，六任务等权平均改善{row.equal_task_mean_gain_pct:.3f}%，最差任务{row.worst_task_gain_pct:.3f}%。')
    lines.append('\n各州各步长的固定模型MSE（按各州训练段标准化，跨州原始尺度比较请看raw_rmse_dsci）：')
    for task in tasks:
        code,hs=task.split('_h');f=metrics[(metrics.state==code)&(metrics.horizon==int(hs))].set_index('candidate');lines.append(task+'：'+', '.join(f'{name}={f.loc[name,"mse"]:.6f}' for name in ['Last','Ridge','DLinear','decision_mean','three_block_minimax_relative_regret','Online26','Online52']))
    lines += ['六任务等权平均改善是六个相对比例的算术平均，不是把所有观测混在一起计算的总体改善率，也未进行独立样本的显著性推断。',
    'IA的12周任务中，Ridge和DLinear分别有约32.56%和33.21%的描述性改善，但固定决策规则仍选Last。这是选型失败证据，不能在看到后续结果后替换当时选择。',
    'CA的12周任务中，DLinear约退化12.39%，在线52起点选择约退化16.45%。这两个失败均保留在表格和图中。',
    '\n五、逐篇公开时间及历史版本核查',
    f'核对既存{source["archived_xml_checked"]}份原始XML及其哈希：{source["parsed"]}份可解析、{source["parse_failures"]}份格式错误。可解析文档仅观察到item/entry/title/forecastPeriod/assessment字段，未发现本轮所列首次发布日期候选字段。',
    f'{source["modification_before_schedule"]}份文件的Last-Modified早于按官方流程推算的公开时间。修改时间、地图对应日期、首次公开时间是不同概念；不能将前两者当成第三者。',
    f'按档案顺序等距选择6份文档重新抓取，{source["repeat_identical"]}/6与同日上一轮保存字节相同。这只能说明这两次抓取内容一致，不能证明历史原始版本从未被修改。',
    '官方生产流程页面说明通常周四08:30美国东部时间公开地图和叙述。官方Metadata页主要说明GIS、表格等数据的结构与处理，不提供本项目所需的逐篇叙述首次公开版本证明。检查范围限于本轮列出的字段、文件头、重复样本和官方页面，不宣称所有外部档案都不存在证据。',
    '因此source/summary.json记录first_publication_verified=0，predict16.py固定text_weight=0。当前数值也是获取日的历史档案快照，历史修订的不确定性仍然存在。',
    '\n六、验收与可运行入口',
    f'独立复核：{verified["status"]}。核查{verified["raw_snapshots_checked"]}份新下载快照、{verified["input_origins"]}个窗口起点、{verified["metrics"]}条主指标、{verified["seed_metrics"]}条种子指标、{verified["online_predictions"]}次在线选择。',
    f'{tests["tests_run"]}项测试通过，覆盖州别错误、周缺口/重复、范围与非有限数值、跨界目标剔除、未来标签扰动、延迟反馈、标准化文件篡改，以及六个任务真实推理与保存预测一致。',
    '提供命令行州级推理，必须明确指定预先固定的选型规则，不根据测试最优分数静默换模型。输出原始DSCI点数；超范围不事后裁剪，而是明确返回out_of_range。',
    '\n七、当前技术成熟度与下一步',
    '工程方面：已具备官方数据接入、按时间隔离、强基线、深度训练、多序列检验、预注册规则、延迟反馈、来源审计、真实推理、独立验收和Git版本交付。可以据此组织完整的技术答辩。',
    '研究方面：稳定且独立的文本增益、逐篇首次公开版本、真正未使用的未来数据验证仍未完成。当前不应宣称多模态优越性、理论安全保证或跨领域泛化。',
    '建议下一阶段先整合已有技术成果和课程要求，把可证明的贡献、失败案例和演示入口讲清；若继续研究，则优先补真实首次发布版本或开展真正前瞻记录，避免无止境地增加候选模型。',
    '\n八、可核查资料（访问日期2026-10-08）',
    '[1] U.S. Drought Monitor. Web Service Information. https://droughtmonitor.unl.edu/DmData/DataDownload/WebServiceInfo.aspx',
    '[2] National Drought Mitigation Center. What is the Timeline for Production? https://drought.unl.edu/Education/Tutorials/Usdm/Section-Six.aspx',
    '[3] U.S. Drought Monitor. Narrative archive. https://droughtmonitor.unl.edu/Data/narrative/',
    '[4] U.S. Drought Monitor. Metadata. https://droughtmonitor.unl.edu/DmData/Metadata.aspx',
    '[5] U.S. Drought Monitor. Drought Severity and Coverage Index. https://droughtmonitor.unl.edu/About/AbouttheData/DSCI.aspx',
    '[6] Zeng, A.; Chen, M.; Zhang, L.; Xu, Q. (2022). Are Transformers Effective for Time Series Forecasting? arXiv:2205.13504. https://arxiv.org/abs/2205.13504 ; PDF https://arxiv.org/pdf/2205.13504',
    '引用对应：[1][5]为第二节数据接口和指标口径；[6]为第三节DLinear模型出处；[2][3][4]为第五节来源核查。模型为当前项目实现和参数设置，不宣称完整复现原论文全部实验。']
    (HERE/'results/01_第16轮技术结论与验收.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    instructions=rf'''第16轮运行说明
源码：{HERE}
原始数据、处理数据、权重与逐样本输出：{LOCAL}

先进入源码目录，再在PowerShell执行：
.\Run-Round16.ps1 -Entry verify16.py
.\Run-Round16.ps1 -Entry test16.py
.\Run-Round16.ps1 -Entry predict16.py -Arguments @('--state','CA','--horizon','4','--rule','CalChoice','--example')
.\Run-Round16.ps1 -Entry predict16.py -Arguments @('--state','TX','--horizon','12','--rule','three_block_minimax_relative_regret','--example')
--example使用最后一个评价历史窗口，并非当前实时观测。自定义输入请用--history-json替换--example，文件内容为52个0至500的数值组成的JSON数组。
规则可选CalChoice、decision_mean、three_block_minimax_relative_regret、recent_26_origins；必须明确指定。

从头复算阶段：collect16.py → prepare16.py → train16.py → evaluate16.py → source_audit16.py → verify16.py → test16.py → report16.py。
既有prepare/evaluate拒绝覆盖；train只在来源和权重哈希一致时跳过已封存任务。重跑请使用独立完整项目副本，保留当前实验，勿手工改哈希。
需要Python3.12、现有torch2.8.0+cpu及D盘运行依赖。运行脚本复用本机既有Python解释器；不新安装到C盘。
依赖src/train_dlinear.py及其导入，以及round15/analysis15.py。source_audit16还依赖第14轮保留的351份原始XML，关联原文在交付包中保留。

结果：all_metrics.csv共54条；dlinear_seed_metrics.csv共18条；aggregate.csv是按任务等权的描述性汇总。
各任务selection.json/choice_scores.csv记录评分前选择。prepared包含选择区与单独测试特征/目标，models保存训练日志、原始及重拟合权重、选择封印。
evaluation/fixed_commitment.json证明固定预测文件在评分流程中的保存顺序；它不是第三方时间戳认证。
source目录给出逐篇审计和6份重复抓取比较；未找到证据不能写成已完成首次公开证明。
交付目录包含证据、源码和关联输入，不包含完整便携Python环境。Git只推送主控分支的代码和紧凑结果。
推送状态以交付包内实时远端核验回执为准，不使用可能过期的本地远端跟踪引用作为成功证明。
'''
    (HERE/'results/02_运行与复现说明.txt').write_text(instructions,encoding='utf-8-sig');print('Reports and figures ready')

if __name__=='__main__':main()
