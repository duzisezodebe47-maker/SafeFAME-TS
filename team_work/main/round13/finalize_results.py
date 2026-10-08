"""由已验收CSV生成科研图、种子统计、技术结论与交付说明。"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(__file__).resolve().parents[4]/'主控接收数据/mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from common import *
from robustness import interval_sensitivity

def main():
    interval_sensitivity()
    df=pd.read_csv(HERE/'results/final/all_metrics.csv',float_precision='round_trip');proxy=df[df.scenario.eq('proxy')];out=HERE/'results/final';figs=out/'figures';figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    names={'Agriculture':'农业','Climate':'气候','Environment':'环境','SocialGood':'社会公益'};candidates=['Last','N','AR-Ridge-refit','N+S+Q+SF','AdaptiveRisk','DLinear','PatchTST'];labels=['Last','数值N','AR重拟合','完整融合','自适应','DLinear','PatchTST'];colors=['#8195a6','#217c99','#559eaa','#5375b8','#df9047','#8f98c6','#1b455e']
    fig,axes=plt.subplots(2,2,figsize=(13,8.3))
    for ax,task in zip(axes.flat,protocol()['tasks']):
        g=proxy[proxy.task.eq(task)].set_index('candidate');values=[g.loc[n,'mse'] for n in candidates];bars=ax.bar(np.arange(len(values)),values,color=colors);ax.set_xticks(range(len(values)),labels,rotation=18,ha='right');ax.set_ylabel('测试MSE（训练段标准化尺度）');ax.set_title(f"{names[task.split('_')[0]]} / {task} / {int(g.iloc[0].origins)}个起点");ax.set_ylim(0,max(values)*1.2)
        for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+max(values)*.015,f'{value:.4f}',ha='center',fontsize=8)
    fig.suptitle('图1 统一训练范围下的四领域历史测试比较',fontsize=16);fig.text(.5,.012,'来源：all_metrics.csv。深度模型为3种子预测均值；不同领域不可直接比较MSE。已知历史数据的回顾性扩展。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(figs/'01_四领域模型比较.png',dpi=180);fig.savefig(figs/'01_四领域模型比较.svg');plt.close(fig)
    rows=[]
    for task in protocol()['tasks']:
        for model in protocol()['deep_models']:
            g=proxy[proxy.task.eq(task)&proxy.candidate.str.startswith(model+'_s')];ensemble=proxy[proxy.task.eq(task)&proxy.candidate.eq(model)].iloc[0]
            rows.append({'task':task,'model':model,'seeds':len(g),'seed_mse_mean':float(g.mse.mean()),'seed_mse_std_ddof1':float(g.mse.std()),'seed_mse_min':float(g.mse.min()),'seed_mse_max':float(g.mse.max()),'ensemble_mse':float(ensemble.mse),'note':'Seed MSE mean is not ensemble MSE; 3 runs do not establish broad stability.'})
    csv(out/'deep_seed_summary.csv',rows)
    fig,axes=plt.subplots(2,2,figsize=(13,7))
    for ax,task in zip(axes.flat,protocol()['tasks']):
        trace=pd.read_csv(LOCAL/'comparison'/task/'proxy/adaptive_trace.csv');ax.step(np.arange(len(trace)),trace.weight,where='post',color='#d38b42');ax.set_ylim(-.05,1.05);ax.set_xlabel('测试预测起点序号（从0开始）');ax.set_ylabel('文本融合权重');ax.set_title(names[task.split('_')[0]]+f' / 文本启用{int((trace.weight>0).sum())}/{len(trace)}')
    fig.suptitle('图2 自适应融合的因果反馈回放（proxy场景）',fontsize=16);fig.text(.5,.015,'来源：各任务adaptive_trace.csv。权重只使用完整到期的历史标签；启用文本不等于实现误差改善。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(figs/'02_自适应权重轨迹.png',dpi=180);plt.close(fig)
    rolling=pd.read_csv(HERE/'results/robustness/rolling_numeric_metrics.csv');wins=[]
    for candidate,g in rolling.groupby('candidate'):wins.append({'candidate':candidate,'cases':len(g),'better_than_Last_cases':int((g.gain_vs_Last_pct>1e-9).sum()),'median_gain_vs_Last_pct':float(g.gain_vs_Last_pct.median()),'minimum_gain_vs_Last_pct':float(g.gain_vs_Last_pct.min()),'note':'Overlapping temporal folds are descriptive diagnostics, not independent replications.'})
    csv(out/'rolling_summary.csv',wins)
    source=read(HERE/'results/source/source_acceptance.json');tests=read(out/'tests.json');require(tests['status']=='PASS','tests not passed');verification=read(out/'verification.json');require(verification['status']=='PASS_WITH_RESEARCH_LIMITATIONS','verification incomplete')
    summary=read(out/'dashboard.json');table=proxy[proxy.candidate.isin(candidates)].pivot(index='task',columns='candidate',values='mse').to_string(float_format=lambda x:f'{x:.6f}')
    report=f'''SafeFAME-TS 主控第13轮技术加强验收说明
完成日期：2026-10-08；开始运行日期：2026-10-07（因此内部证据目录保留20261007）。
分支：team/main-eval-20260921。此轮为本地交付；是否上传以交付目录的版本记录为准。

一、已经落地的五项工作
1. 统一深度基线：4任务×DLinear/PatchTST×3随机种子=24次训练，保存早停、校准预测、重拟合权重与逐步测试预测。它们在旧项目已有实现基础上接入主控协议，并非新发明这两种模型。
2. 来源审计：逐行检查{source['audited_rows']}条四领域原始谱系记录（text_id去重后{source['unique_text_ids']}个）；127条空信息标记、192条文本年份晚于区间的待复核标记。标记只提示人工核验，不能直接证明泄漏。原始网址及真实公开时间均未建立完整链路，严格来源合格数为0。四个案例查证状态另见source目录，不把相关网页冒充原始来源。
3. 稳健性：24个数值滚动案例（4领域×2预测长度×3时间折）、32个文本扰动场景、36行块长敏感性。深度模型/文本模型没有在24个滚动案例全部重训。时间折存在重叠，不是24个独立外部样本。
4. 自适应融合：已实现完整到期标签释放、历史收益门槛、有限权重网格、无可用文本回退、严格来源回退，并逐起点重演核验。公式和局限见运行说明。本轮结果未证明它普遍胜过固定路线。
5. 本地演示：四任务对照、历史曲线、六种数值推理选择、输入校验、模型权重校验和本机HTTP接口。严格来源模式只做数值回退；任意新文本的抽取/来源认证不在演示能力内。

二、统一比较结果（MSE越小越好，不同领域尺度不宜横比）
{table}

农业Last=0.123532，低于完整融合0.214386、自适应0.283698和PatchTST 0.178700。因此“相对AR-Ridge改善”不足以支持“优于简单方法”。
气候完整融合历史MSE=0.112671，低于N=0.133936，但未据此改选旧正式路线；不可以用已知测试结果倒过来调门槛。
环境PatchTST=0.245259，低于N=0.256917，约改善4.54%；这是当前配置、当前历史任务的描述结果，不是统计显著或外部有效性的证明。
社会公益AR重拟合=0.026729，低于N=0.027641。文本分支不可训练时回退与N相同，不能算文本贡献。
自适应在农业优于冻结AR，但差于固定选择；环境略差于N；气候和社会公益没有启用文本。不得包装成“安全地普遍提精度”。

三、新旧结果为何不能混为一谈
旧交付predict_test.py在冻结超参数后对train+calibration+decision重新拟合。第13轮深度与融合均只在train+calibration重拟合，以统一训练信息。StaticGateReplay表示沿用历史候选名称、按新范围重拟合，绝非旧预测文件原样复刻。旧评分和路线未修改。
旧AR只训练段拟合，这是既有协议，故补充AR-Ridge-refit作为公平对照。首次保存AR参数时发现float32均值与既有float64实现不同；已保存诊断版本到本地diagnostics/pre_fix_ar_precision，并修复、重新选择封存与预测。未手工改哈希。
农业测试仅42个重叠起点；块长48已超样本量，拒绝给出退化的伪窄区间，输出NOT_ESTIMABLE。其余低块数结果仅作诊断。

四、验收证据
自动回归测试：{tests['tests']}项全部通过，含24个任务/推理模式组合、16个预测文件逐值对照及输入、HTTP、未来标签、重复起点等负向测试。
实验独立验收：24次训练、8场景预测、24滚动案例的哈希与分数复算通过；8条场景的自适应轨迹由历史输入重新计算，与记录一致。源语料哈希一致。浏览器视觉与交互验收见browser_qa.json。
本轮仍是回顾性实验：我们之前已经知道这些测试数据的表现。文件封存能检查一致性，不能把已知测试集变成新的盲测。

五、老师视角的技术判断
支持性判断：项目现已具备真实模型训练、强弱基线比较、多随机种子、滚动诊断、来源审计、延迟反馈实现、自动测试和可交互演示，工程链条足以形成有内容的课程展示。
质疑性判断：目前最薄弱的是文本公开时间证据与独立泛化，不是代码量。来源全部未核实且部分简单基线更优，不能宣称实现“可信文本普遍提高预测精度”。合理主线是“文本收益的边界与可审计回退”。
技术进度：这一轮列定的实现与实验已完成；研究假设尚未获得全面支持。后续若继续投入，应先建立有原始URL/首次发布时间的文本样本，再在预先冻结的新时间段验证。若无新数据，应保留负结果，停止反复对当前测试集调参。
'''
    (out/'主控第13轮_技术验收与结论.txt').write_text(report,encoding='utf-8')
    print('REPORT_AND_FIGURES_DONE',figs)

if __name__=='__main__':main()
