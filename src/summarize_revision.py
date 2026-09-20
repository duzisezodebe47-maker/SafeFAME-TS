"""Summarize historical v2 and target-disjoint v3 without replacing either experiment."""
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import configure_fonts, finish_fonts

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'outputs/revision_comparison';output.mkdir(exist_ok=True)
    summaries={};tasks={}
    for version in ('v2','v3'):
        base=ROOT/'outputs'/f'safefame_{version}'
        m=pd.read_csv(base/f'safefame_{version}_metrics.csv')
        a=pd.read_csv(base/f'safefame_{version}_selection_audit.csv')
        tasks[version]=a
        s={'tasks':len(a),'text_selected':int(a.selected_path.ne('numeric_fallback').sum()),
           'calibration_windows':int(a.calibration_windows.sum()),'decision_windows':int(a.decision_windows.sum()),
           'test_windows':int(a.test_windows.sum())}
        for variant in ('semantic_residual','frequency_residual'):
            r=m[m.model.eq(variant)]
            s[variant]={'point_positive':int((r.improvement_vs_fallback_pct>0).sum()),
                        'ci_positive':int((r.block_ci_low>0).sum()),'ci_negative':int((r.block_ci_high<0).sum()),
                        'row_permutation_pass':int((a[variant+'_permutation_p_value']<=.025).sum())}
        summaries[version]=s
    summaries['v3']['removed_crossing_windows']=int(tasks['v3'].validation_crossing_windows_removed.sum())
    summaries['v3']['insufficient_segment_tasks']=int(tasks['v3'].segment_status.ne('sufficient').sum())
    for name in ('reviewer_sensitivity','reviewer_sensitivity_corrected','reviewer_sensitivity_v3'):
        summaries[name]=json.loads((ROOT/'outputs'/name/'reviewer_sensitivity_summary.json').read_text(encoding='utf-8'))
    combined=tasks['v2'][['domain','horizon','numeric_fallback']].merge(tasks['v3'],on=['domain','horizon'],suffixes=('_v2','_v3'))
    summaries['fallback_changes']=combined.loc[combined.numeric_fallback_v2.ne(combined.numeric_fallback_v3),
        ['domain','horizon','numeric_fallback_v2','numeric_fallback_v3']].to_dict('records')
    combined.to_csv(output/'task_comparison.csv',index=False)
    (output/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    configure_fonts()
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    keys=['校准起点','决策起点','测试起点'];x=list(range(3));w=.34
    axes[0].bar([v-w/2 for v in x],[summaries['v2'][k] for k in ('calibration_windows','decision_windows','test_windows')],w,label='历史 v2',color='#8997A5')
    axes[0].bar([v+w/2 for v in x],[summaries['v3'][k] for k in ('calibration_windows','decision_windows','test_windows')],w,label='修订 v3',color='#2F6690')
    axes[0].set_xticks(x,keys);axes[0].set_ylabel('18项任务起点数之和');axes[0].legend(frameon=False)
    for container in axes[0].containers:axes[0].bar_label(container,padding=3,fontsize=10)
    labels=['99次置换达阈值','完整门控启用','999次敏感性达阈值']
    axes[1].bar(range(3),[sum(summaries['v3'][v]['row_permutation_pass'] for v in ('semantic_residual','frequency_residual')),summaries['v3']['text_selected'],summaries['reviewer_sensitivity_v3']['circular_shift_passes_at_0_025']],color=['#2F6690','#78828A','#D97706'])
    axes[1].set_xticks(range(3),labels,rotation=12);axes[1].set_ylabel('36条候选路径数');axes[1].set_ylim(0,3)
    for container in axes[1].containers:axes[1].bar_label(container,padding=4)
    axes[0].set_title('移除目标跨界窗口后的样本量');axes[1].set_title('单项检验通过不等于获得部署资格')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    finish_fonts(fig);fig.tight_layout()
    fig.savefig(output/'boundary_revision.png',dpi=300,bbox_inches='tight');fig.savefig(output/'boundary_revision.pdf',bbox_inches='tight');plt.close(fig)
    print(json.dumps(summaries['v3'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
