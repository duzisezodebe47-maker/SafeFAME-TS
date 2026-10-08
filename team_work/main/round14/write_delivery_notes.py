"""将冻结指标和检查记录转成可阅读的技术交付说明。"""
from r14_common import *

def main():
    final=HERE/'results/final';metrics=pd.read_csv(final/'all_metrics.csv');checks=read(final/'verification.json');tests=read(final/'tests.json')
    lines=['SafeFAME-TS 第14轮技术交付与验收结论','日期：2026-10-08','一、结论先行',
    '已完成官方数据接入、独立任务定义、后续时间段评价、数值候选选择、文本反事实诊断及严格模式演示。工程证据比上轮完整；本轮没有证实稳定、独立的文本语义增益。',
    '不宣称“多模态普遍优于单模态”，不把安全回退等同于预测精度提升。回退只是在本轮保持了选定数值模型的结果，不保证优于所有数值模型。',
    '\n二、数据身份与来源',
    '官方美国本土 CONUS DSCI：1395个连续周，2000-01-04至2026-09-22。下载351份官方周报XML；339份可解析、235份符合正文规则，12份XML格式失败、104份正文不足等原因排除。',
    '与旧 Climate 目标重叠1272周，在0.01容限下没有一周相同。因此这是气候领域内新定义的DSCI任务，不能拼接旧目标，也不构成跨领域验证。',
    '原始响应、请求URL、抓取时间、Last-Modified、内容哈希、解析异常及逐预测来源均保留。354份原始快照包括351份XML和数值、目录、发布时间说明3份文件。',
    '逐篇首次公开时间的严格合格文本仍为0。常规实验依据官方周四08:30美国东部时间发布流程推算；这不是逐篇公开时间证明。Last-Modified保守情景也不证明历史首次版本。',
    '数值和文本均为本次获取的历史档案，可能包含历史修订；本轮是历史后续时段评价，不是前瞻性盲测。',
    '\n三、实验设计与工程实现',
    '协议在下载新增数值前登记，输入52周，预测4/12周。训练截至2021-12-28，校准截至2022-12-27，决策截至2024-05-14，测试目标期2024-05-21至2026-09-22。完整预测目标必须位于同一分段，跨界窗口剔除。',
    '比较Last、SeasonalNaive、Ridge、DLinear、PatchTST；数值模型按校准误差选择并冻结。两步长×两深度模型×三个随机种子，共12条主训练路线；每条含校准选择和训练加校准重拟合。H12文本残差还执行3折×3种子的9次PatchTST内层训练。',
    '固定随机种子2026/2027/2028。数值标准化仅拟合训练段，OOF内层各自仅拟合历史数据。保存逐轮损失、最佳轮次、模型权重、参数量和选择理由。复用已固定版本的官方PatchTST实现，不宣称完整复现原论文全部设置。',
    '文本移除展望段和数字，训练段TF-IDF加16维SVD。修正器使用3折按时间排序、清除目标重叠后的数值样本外误差。H4/H12实际可用文本残差样本分别99/91，不能把一千多个数值窗口当成一千多个独立文本样本。',
    'OOF数值权重按折限制历史，但模型家族及训练轮数由外层校准期选择；这是条件于外层选择的残差训练，不是完全嵌套的泛化评估。训练文本表示在训练段整体拟合，最终测试完全位于其后。',
    '两种步长静态权重均为0；自适应只读完整到期的历史H步误差，本轮测试所有起点也均为0。风险门槛是启发式，不是有限样本安全定理。',
    '无当周合格报告时允许复用最近8周内报告；文本覆盖率包括复用，不能解释为每个起点都有新周报。测试H4有119/120个起点存在可用文本，H12为111/112。',
    '\n四、真实测试结果（MSE为训练段标准化尺度，越小越好）']
    for h in (4,12):
        frame=metrics[metrics.horizon==h].set_index('candidate');s=read(HERE/f'results/h{h}/test_summary.json')
        lines.append(f'\n预测{h}周，测试起点{s["test_origins"]}，校准期选定{s["selected_numeric"]}：')
        for name in ['Last','Ridge','DLinear','PatchTST','SelectedNumeric','DirectFusion','StaticFallback','AdaptiveFallback','StrictSourceFallback','Delay_1w','Delay_4w','Missing_50pct','PastMismatch']:
            row=frame.loc[name];lines.append(f'  {name}: MSE={row.mse:.6f}，相对选定数值模型改善={row.gain_vs_selected_numeric_pct:.3f}%')
        ci=pd.read_csv(HERE/f'results/h{h}/block_diagnostics.csv');row=ci[(ci.candidate=='DirectFusion')&(ci.block==h)].iloc[0]
        lines.append(f'  直接融合平均MSE收益的块Bootstrap诊断区间（块长{h}）=[{row.ci_low:.6f}, {row.ci_high:.6f}]，包含0。')
    lines += ['\n五、从老师和审稿人视角看',
    '支持性分析：数据来源可追溯；能够区分地图日期和文本发布时间；先冻结路线再评分；有强数值基线、延迟/缺失/错配对照、独立复算和可操作演示。这些是可以实际展示的技术贡献。',
    '对立性视角：校准期选中的模型在后续时段不是最好，H12 PatchTST尤其落后于DLinear/Ridge；直接文本改善只相对这个选定模型成立。延迟和缺失文本反而更好，提示文本修正可能含偏置、旧状态信息或偶然效应，不能归因于及时语义。',
    '三种块长的直接融合区间均包含0；H12大块长仅约4或2块，支持有限。这些是逐项诊断区间，未进行多重比较校正，不构成统计显著性证据。',
    '三个种子的单模型MSE均值和标准差另存deep_seed_summary.csv；它与三种子预测平均后的ensemble MSE不同，不能混用。H4 PatchTST种子波动较大，应报告而非只取最好种子。',
    '\n六、已完成验收',
    f'核心复核状态：{checks["status"]}；验证{checks["raw_snapshot_files_verified"]}份原始快照、{checks["prediction_origins_checked"]}个输入起点，两步长因果回放与全部30条指标复算。',
    f'回归测试：{tests["tests_run"]}项，成功={tests["passed"]}。包含清洗、夏令时、来源篡改、输入边界、成熟反馈、未来扰动不影响过去、保存模型复推和HTTP接口。',
    '首次接口测试受本机网络沙箱限制，获准本地连接后重测；另修正测试中精确比较浮点数的断言为1e-12容限。未为此修改任何训练权重或评分结果。',
    '浏览器实际检查记录见browser_qa.json及演示截图；仅覆盖本机桌面，不宣称手机、其他浏览器或学校投影验证。',
    '\n七、后续技术优先级（由主控负责）',
    '1. 优先研究数值模型选择稳定性：另立协议，按多个连续历史校准块选择模型，和Last/Ridge/DLinear比较；本轮测试已经公开，不能再将同一段称为未见测试集。',
    '2. 拆解文本残差：加入仅截距修正、常数/时间/滞后状态等对照，确认收益是否只是偏置修正。对正文复用设置0/1/4/8周边界，并在新的验证流程内选择。',
    '3. 如要声称“发布当时可用”，需要逐篇首次发布或不可修改版本证据；没有证据继续保留严格回退，不能用当前网址或Last-Modified冒充证明。',
    '4. 若继续增添模型，必须先说明它解决哪个已发现问题；当前没有必要单纯扩充模型数量、参数规模或代码行数。',
    '本轮任务已完成；以上是下一阶段研究方向，尚未执行，不记为本轮成果。',
    '\n八、原始资料（2026-10-08核查）',
    'USDM官方数据接口说明：https://droughtmonitor.unl.edu/DmData/DataDownload/WebServiceInfo.aspx',
    '官方周报原文档案：https://droughtmonitor.unl.edu/Data/narrative/',
    '官方发布时间说明：https://drought.unl.edu/Education/Tutorials/Usdm/Section-Six.aspx',
    'DSCI定义及Akyuz (2017)建议引用：https://droughtmonitor.unl.edu/About/AbouttheData/DSCI.aspx']
    (final/'01_技术结果与验收结论.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    instructions=rf'''SafeFAME-TS 第14轮运行说明
项目：{PROJECT}
源码：{HERE}
独立原始数据、处理数据、模型及逐样本预测：{LOCAL}

一、直接查看
先读01_技术结果与验收结论.txt。图表对应all_metrics.csv；完整逐步结果位于原始证据/evaluation。
演示为本机服务，双击交付目录的Start-Demo.cmd，保持窗口开启，再用浏览器打开 http://127.0.0.1:8874/ 。用完关闭窗口或Ctrl+C。
如果端口被占用，不终止其他程序，可手动指定另一端口：Run-Round14.ps1 -Entry serve14.py -Arguments @('--port','8875')。

二、验证现有结果（在源码目录运行PowerShell）
powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-Round14.ps1 -Entry verify_round14.py
powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-Round14.ps1 -Entry test_round14.py
verify读取原始证据、模型及预测逐项复算；test另需允许本机网络。缺少整套本地证据时verify报告PARTIAL，不能当成完整验收。已有证据的字节不符会失败。

三、从头复现的阶段顺序
collect_official.py --phase all → prepare_data.py → train_models.py → evaluate_test.py → verify_round14.py → finalize_results.py → test_round14.py → write_delivery_notes.py
这些步骤通过Run-Round14.ps1的Entry/Arguments传入。训练、数据准备、测试评分对既有封存结果拒绝覆盖。这是防止误用的保护。
应在另一个完整项目副本中恢复原始raw快照及protocol_registration.json，保留同样相对目录和协议，再重新生成prepared/models/evaluation。不要删除或改写当前已交付实验，不手动调整哈希。重新从网络下载可能遇到历史修订，不保证得到相同字节。
官方PatchTST依赖位于主控接收数据/PatchTST_official_204c21e；固定commit 204c21efe0b39603ad6e2ca640ef5896646ab1a9。第14轮复用round13/deep_baselines.py及adaptive.py，源码锚定记录在selection_seal.json。
训练流程：先校准选择、保存模型及选择封印，固定预测保存之后才读取标签评分。种子补充分析在评分后执行，仅作诊断，不用于选择。

四、环境与可迁移边界
Python3.12，当前复用已有Python解释器；torch2.8.0+cpu、numpy2.5.3、scipy1.18.1、scikit-learn1.9.1、pandas3.0.1。
依赖增量和临时目录均在D盘主控接收数据。运行器引用现有C盘解释器，仅调用它，不在C盘新安装或下载。
独立交付目录是代码与实验档案，不是含所有Python依赖的便携环境；Start-Demo.cmd依赖当前D盘工作区和已存在环境。迁移到其他电脑需按上述目录准备Python3.12和依赖。
模型joblib仅可加载本项目经哈希核对的可信文件，不加载他人替换的序列化文件。

五、文件解释
protocol.json：本轮预先固定配置。
raw/*.meta.json：原始URL、抓取时间和字节哈希。
prepared/source_catalog.csv、parse_errors.json：正文规则与排除证据。
models/h*/selection_details.json：候选选择、OOF折、深度训练、融合权重。
evaluation/h*/fixed_commitment.json：评分前固定预测记录。
evaluation/h*/adaptive_trace.csv：已到期反馈与逐起点权重。
results/final/all_metrics.csv：全部30条测试指标。
results/h*/block_diagnostics.csv：块Bootstrap诊断，不是显著性保证。
MANIFEST.json：交付文件字节清单；VERSION.json：本地Git版本和推送状态。
'''
    (final/'02_运行与复现说明.txt').write_text(instructions,encoding='utf-8-sig')
    print('Delivery notes saved')

if __name__=='__main__':main()
