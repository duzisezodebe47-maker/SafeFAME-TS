"""Summarize every completed rolling task, without tuning the frozen protocol."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from run_safefame_v4 import ROOT,DATA,write_json,interval,VARIANTS
from run_baselines import metrics
from plot_style import configure_fonts,finish_fonts

OUT=ROOT/'outputs/safefame_v4'

def main():
    manifest=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    cfg=manifest['config']
    completed=list((OUT/'tasks').glob('*/completed.json'))
    assert len(completed)==120,'Final summaries require every configured task'
    frames=[];tasks=[]
    index=pd.read_csv(DATA/'text/sample_text_index.csv').set_index(['domain','origin_index'])
    for p in sorted(completed):
        task=p.parent;a=json.loads((task/'audit.json').read_text(encoding='utf-8'))
        frozen=json.loads((task/'frozen_selection.json').read_text(encoding='utf-8'))
        z=np.load(task/'predictions.npz');frames.append(pd.read_csv(task/'metrics.csv'))
        coverage=index.loc[[(a['domain'],int(o)) for o in z['test_origins']],'total_selected_count']
        train_coverage=index.loc[[(a['domain'],int(o)) for o in z['train_origins']],'total_selected_count']
        block=a['block_length'];sel=a['selected_metrics']['mse'];base=a['fallback_metrics']['mse']
        tasks.append(dict(domain=a['domain'],horizon=a['horizon'],fold=a['fold'],fallback=frozen['fallback'],
            selected=frozen['selected'],fallback_mse=base,selected_mse=sel,selected_gain_pct=100*(1-sel/base),
            selected_delta=a['selected_delta'][0],selected_ci_low=a['selected_delta'][1],selected_ci_high=a['selected_delta'][2],
            cal_windows=a['windows']['cal'],decision_windows=a['windows']['dec'],test_windows=a['windows']['test'],
            block_length=block,decision_blocks=a['windows']['dec']//block,
            segment1_windows=a['segment_windows'][0],segment2_windows=a['segment_windows'][1],
            text_coverage_pct=float((coverage>0).mean()*100),train_text_coverage_pct=float((train_coverage>0).mean()*100),seconds=a['seconds']))
    frame=pd.concat(frames,ignore_index=True);taskframe=pd.DataFrame(tasks)
    frame.to_csv(OUT/'candidate_results.csv',index=False);taskframe.to_csv(OUT/'task_results.csv',index=False)
    bydomain=[]
    for d in cfg['domains']:
        t=taskframe[taskframe.domain==d];c=frame[frame.domain==d]
        bydomain.append(dict(domain=d,task_folds=len(t),selected=int((t.selected!='numeric_fallback').sum()),
            selected_positive=int((t.selected_gain_pct>0).sum()),selected_negative=int((t.selected_gain_pct<0).sum()),
            semantic_median_gain=float(c.loc[c.variant==VARIANTS[0],'improvement_vs_fallback_pct'].median()),
            frequency_median_gain=float(c.loc[c.variant==VARIANTS[1],'improvement_vs_fallback_pct'].median()),
            selected_mean_gain=float(t.selected_gain_pct.mean()),selected_median_gain=float(t.selected_gain_pct.median()),
            median_decision_blocks=float(t.decision_blocks.median()),min_decision_blocks=int(t.decision_blocks.min()),
            mean_text_coverage_pct=float(t.text_coverage_pct.mean())))
    domains=pd.DataFrame(bydomain);domains.to_csv(OUT/'domain_results.csv',index=False)
    family=frame.groupby(['domain','horizon','variant'],as_index=False).agg(
        mean_fold_gain_pct=('improvement_vs_fallback_pct','mean'),median_fold_gain_pct=('improvement_vs_fallback_pct','median'),
        gate_passes=('eligible','sum'))
    family.to_csv(OUT/'task_family_results.csv',index=False)
    summary=dict(version=cfg['version'],role=cfg['role'],series=10,domains=9,task_families=40,task_folds=120,
        candidate_paths=240,seeds=cfg['seeds'],row_permutations=cfg['row_permutations'],circular_shifts=cfg['circular_shifts'],
        bootstrap_repeats=cfg['bootstrap_repeats'],text_selected=int((taskframe.selected!='numeric_fallback').sum()),
        numeric_fallback=int((taskframe.selected=='numeric_fallback').sum()),selected_point_positive=int((taskframe.selected_gain_pct>0).sum()),
        selected_point_negative=int((taskframe.selected_gain_pct<0).sum()),selected_ci_positive=int((taskframe.selected_ci_low>0).sum()),
        selected_ci_negative=int((taskframe.selected_ci_high<0).sum()),
        selected_mean_gain_pct=float(taskframe.selected_gain_pct.mean()),selected_median_gain_pct=float(taskframe.selected_gain_pct.median()),
        calibration_windows=int(taskframe.cal_windows.sum()),decision_windows=int(taskframe.decision_windows.sum()),
        test_windows=int(taskframe.test_windows.sum()),decision_blocks_lt3=int((taskframe.decision_blocks<3).sum()),
        decision_blocks_ge8=int((taskframe.decision_blocks>=8).sum()),minimum_decision_blocks=int(taskframe.decision_blocks.min()),
        median_decision_blocks=float(taskframe.decision_blocks.median()),total_task_seconds=float(taskframe.seconds.sum()),
        zero_train_text_tasks=[f'{r.domain}_h{r.horizon}_f{r.fold}' for r in taskframe[taskframe.train_text_coverage_pct==0].itertuples()],
        calibration_checkpoints=len(list((OUT/'tasks').glob('*/models/*_cal_s*.pt'))),
        refit_checkpoints=len(list((OUT/'tasks').glob('*/models/*_refit_s*.pt'))),
        caveats=['task-folds and horizons are not independent samples','previously available historical data, not prospective confirmation',
                 'Bonferroni threshold within each task-fold only','text availability based on end_date proxy',
                 'changes from v3 include split widths, seeds, tasks and permutation count; no single-factor attribution'])
    for v in VARIANTS:
        part=frame[frame.variant==v];families=family[family.variant==v]
        summary[v]=dict(point_positive=int((part.improvement_vs_fallback_pct>0).sum()),ci_positive=int((part.ci_low>0).sum()),
            ci_negative=int((part.ci_high<0).sum()),row_pass=int((part.p_value<=cfg['p_threshold']).sum()),
            circular_pass=int((part.circular_p_value<=cfg['p_threshold']).sum()),eligible=int(part.eligible.sum()),
            matched_ci_positive=int((part.matched_ci_low>0).sum()),matched_ci_negative=int((part.matched_ci_high<0).sum()),
            task_family_mean_positive=int((families.mean_fold_gain_pct>0).sum()),
            median_gain_pct=float(part.improvement_vs_fallback_pct.median()))
    write_json(OUT/'summary.json',summary)
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    configure_fonts()
    fig,ax=plt.subplots(figsize=(9.8,3.1))
    colors=['#9eb7ce','#74a9a0','#e5bd72','#6386a4']
    labels=['训练','校准','资格决策','测试']
    for f,bounds in enumerate(cfg['folds']):
        left=0
        for j,right in enumerate(bounds):
            ax.barh(3-f,right-left,left=left,color=colors[j],edgecolor='white',height=.62)
            ax.text((left+right)/2,3-f,f'{labels[j]} {(right-left)*100:.0f}%',ha='center',va='center',fontsize=10)
            left=right
    ax.set_yticks([1,2,3],['滚动折3','滚动折2','滚动折1']);ax.set_xlim(0,1)
    ax.set_xlabel('清洗后序列的时间位置比例');ax.set_title('每折目标跨界窗口剔除 三折测试区间互不重叠')
    finish_fonts(fig);fig.tight_layout();fig.savefig(figures/'rolling_design.png',dpi=240);fig.savefig(figures/'rolling_design.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.8,5.8))
    y=np.arange(len(domains))
    for offset,key,label,color in [(-.12,'semantic_median_gain','语义候选','#326b9b'),(.12,'frequency_median_gain','频率候选','#d88b27')]:
        axes[0].scatter(domains[key],y+offset,label=label,color=color,s=28)
    axes[0].axvline(0,color='gray',lw=1);axes[0].set_yticks(y,domains.domain);axes[0].invert_yaxis()
    axes[0].set_xlabel('12个任务—折的改善中位数（%）');axes[0].set_title('文本候选相对数值回退');axes[0].legend(frameon=False)
    axes[0].grid(axis='x',alpha=.2)
    axes[1].barh(y,domains.selected_positive,label='启用后点改善',color='#3c8d7d')
    axes[1].barh(y,domains.selected_negative,left=domains.selected_positive,label='启用后点变差',color='#c97468')
    axes[1].set_yticks(y,domains.domain);axes[1].invert_yaxis();axes[1].set_xlim(0,12)
    axes[1].set_xlabel('实际启用文本的任务—折数（每组共12）');axes[1].set_title('门控启用后的测试表现');axes[1].legend(frameon=False)
    finish_fonts(fig);fig.tight_layout();fig.savefig(figures/'domain_results.png',dpi=240);fig.savefig(figures/'domain_results.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.7,4))
    values=[taskframe.loc[taskframe.domain==d,'decision_blocks'].to_numpy() for d in domains.domain]
    ax.boxplot(values,tick_labels=domains.domain,showfliers=False)
    ax.axhline(8,color='#b85043',ls='--',label='8块参考线（不是功效保证）')
    ax.set_yscale('log');ax.set_ylabel('决策期完整非重叠块数（对数轴）')
    ax.tick_params(axis='x',rotation=30);ax.legend(frameon=False);ax.set_title('扩大决策时段后的样本量诊断')
    finish_fonts(fig);fig.tight_layout();fig.savefig(figures/'decision_blocks.png',dpi=240);fig.savefig(figures/'decision_blocks.pdf');plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
