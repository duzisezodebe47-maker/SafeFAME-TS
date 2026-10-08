"""汇总已冻结结果；种子分析仅作事后诊断，不参与重新选择。"""
from r14_common import *
from evaluate_test import load_trusted
from train_models import deep

def main():
    import torch,matplotlib
    torch.set_num_threads(2);matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':10})
    final=HERE/'results/final';final.mkdir(exist_ok=True)
    frames=[pd.read_csv(HERE/f'results/h{h}/test_metrics.csv') for h in cfg()['horizons']];metrics=pd.concat(frames,ignore_index=True);csv(final/'all_metrics.csv',metrics.to_dict('records'))
    tasks={str(h):read(HERE/f'results/h{h}/test_summary.json') for h in cfg()['horizons']}
    dump(final/'dashboard.json',{'tasks':tasks,'metrics':metrics.to_dict('records'),'data':read(HERE/'results/data/data_acceptance.json')})
    seed_rows=[]
    for h in cfg()['horizons']:
        obj,_=load_trusted(h);x=np.load(LOCAL/f'prepared/h{h}/test_features/x.npy');y=np.load(LOCAL/f'prepared/h{h}/test_truth.npy')
        for name in ('DLinear','PatchTST'):
            for seed,model in zip(cfg()['seeds'],obj['numeric'][name]):seed_rows.append({'horizon':h,'model':name,'seed':seed,'test_mse':mse(deep.predict(model,x),y)})
    df=pd.DataFrame(seed_rows);csv(final/'deep_seed_metrics.csv',seed_rows)
    aggregate=df.groupby(['horizon','model']).test_mse.agg(['mean','std','min','max']).reset_index();csv(final/'deep_seed_summary.csv',aggregate.to_dict('records'))
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for h,ax in zip(cfg()['horizons'],axes):
        subset=metrics[(metrics.horizon==h)&metrics.candidate.isin(['Last','Ridge','DLinear','PatchTST','DirectFusion','AdaptiveFallback'])];colors=['#ee9457' if n=='DirectFusion' else '#327eab' for n in subset.candidate]
        ax.bar(subset.candidate,subset.mse,color=colors);ax.tick_params(axis='x',rotation=30);ax.set_title(f'{h}周预测：校准期选定 {tasks[str(h)]["selected_numeric"]}');ax.set_ylabel('测试 MSE（训练期标准化尺度）');ax.grid(axis='y',alpha=.2)
    fig.suptitle('固定路线的新时间段检验：文本融合未超过全部数值基线');fig.tight_layout();fig.savefig(final/'模型对比.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    labels={'DirectFusion':'原对齐文本','Delay_1w':'延迟1周','Delay_4w':'延迟4周','Missing_50pct':'缺失50%','PastMismatch':'历史错配'}
    for h,ax in zip(cfg()['horizons'],axes):
        subset=metrics[(metrics.horizon==h)&metrics.candidate.isin(labels)];ax.bar([labels[v] for v in subset.candidate],subset.gain_vs_selected_numeric_pct,color='#e69854');ax.set_title(f'{h}周预测');ax.set_ylabel('相对选定数值模型的 MSE 改善率（%）');ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.2)
    fig.suptitle('反事实诊断：延迟与缺失文本更好，尚不能归因为及时语义');fig.tight_layout();fig.savefig(final/'文本反事实诊断.png',dpi=180);plt.close(fig)
    print('Final tables, seed diagnostics and figures saved:',final)

if __name__=='__main__':main()
