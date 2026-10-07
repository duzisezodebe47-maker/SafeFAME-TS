"""主控技术闭环验收：只复核冻结预测，禁止重新预测或用测试结果选模型。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROJECT = ROOT.parent
sys.path[:0] = [str(HERE), str(ROOT / 'team_work/main/src')]
import numpy as np
import pandas as pd
import run_master as master
from threadpoolctl import threadpool_limits

TASKS = ('SocialGood_h3_f1', 'Environment_h7_f2')
SCENARIOS = ('proxy', 'conservative_lag')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sealed(directory, code):
    route = read(directory / 'sealed_route.json')
    require(master.sha(directory/'sealed_route.json') == (directory/'sealed_route.json.sha256').read_text().strip(), 'route hash')
    for rel, expected in route['code_sha256'].items():
        require(master.sha(code/rel) == expected, f'code changed: {rel}')
    for rel, expected in route['evidence_sha256'].items():
        require(master.sha(directory/rel) == expected, f'evidence changed: {rel}')
    return route

def audit_test_csv(path, bundle, task, score):
    """复算已存预测的指标；不调用模型、不训练、不改变任何预测。"""
    t = master.task_spec(read(ROOT/'team_work/main/round2/split_spec_v2.json'), task)
    source = bundle/task
    meta = pd.read_csv(source/'samples.csv')
    mask = meta.segment.eq('test').to_numpy()
    meta = meta[mask]
    truth = np.asarray(np.load(source/'targets_standardized.npy', mmap_mode='r')[mask], float)
    df = pd.read_csv(path, float_precision='round_trip')
    require(not df.duplicated(['candidate','origin_index','step']).any(), 'duplicate test prediction')
    require(np.isfinite(df.y_pred).all(), 'nonfinite test prediction')
    observed = {}
    for candidate, metrics in score['metrics'].items():
        sub = df[df.candidate.eq(candidate)].sort_values(['origin_index','step'])
        require(len(sub) == len(meta)*t['horizon'], 'test row count')
        require(np.array_equal(sub.origin_index, np.repeat(meta.origin_index.to_numpy(), t['horizon'])), 'test origins')
        require(np.array_equal(sub.origin_id, np.repeat(meta.origin_id.to_numpy(), t['horizon'])), 'test origin identity')
        require(np.array_equal(sub.step, np.tile(np.arange(1,t['horizon']+1),len(meta))), 'test steps')
        e = sub.y_pred.to_numpy().reshape(truth.shape)-truth
        mse, mae = float(np.mean(e*e)), float(np.mean(abs(e)))
        require(np.isclose(mse,metrics['mse'],rtol=1e-13,atol=1e-15), 'MSE differs')
        require(np.isclose(mae,metrics['mae'],rtol=1e-13,atol=1e-15), 'MAE differs')
        observed[candidate] = {'mse':mse,'mae':mae}
    require(set(df.candidate)==set(score['metrics']), 'unexpected test candidate')
    return {'status':'PASS','rows':len(df),'origins':len(meta),'metrics_recomputed':observed}

def historical_bound(directory, domain, task, bundle, score):
    manifest = read(directory/'scoring_manifest.json')
    files = manifest.get('files_sha256', manifest.get('files',{}))
    byte_checks=[]
    for rel, h in files.items():
        current=master.sha(directory/rel)
        status='EXACT'
        if current!=h:
            b=(directory/rel).read_bytes()
            variants=[b.replace(b'\r\n',b'\n'),b.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')]
            if any(hashlib.sha256(v).hexdigest()==h for v in variants):
                status='LINE_ENDING_ONLY_DIFFERENCE'
            elif rel.endswith('.log'):
                status='SKIP_STALE_NON_SCORING_LOG'
            else:
                raise ValueError(f'historical scoring content changed: {rel}')
        byte_checks.append({'file':rel,'historical_sha256':h,'current_sha256':current,'status':status})
    df = pd.read_csv(directory/f'{domain}_test_predictions_bound.csv',float_precision='round_trip')
    meta = pd.read_csv(bundle/task/'samples.csv'); mask=meta.segment.eq('test').to_numpy(); meta=meta[mask]
    truth = np.asarray(np.load(bundle/task/'targets_standardized.npy',mmap_mode='r')[mask],float)
    require(not df.duplicated(['origin_index','step']).any(), 'historical duplicate')
    require(np.array_equal(df.origin_index,np.repeat(meta.origin_index,truth.shape[1])), 'historical origins')
    require(np.array_equal(df.origin_id,np.repeat(meta.origin_id,truth.shape[1])), 'historical identities')
    require(np.array_equal(df.step,np.tile(np.arange(1,truth.shape[1]+1),len(meta))), 'historical steps')
    require(np.array_equal(df.y_true_standardized.to_numpy().reshape(truth.shape),truth), 'historical truth mapping')
    if domain=='Agriculture':
        pairs = [('selected',score['selected_mse']),('fallback',score['fallback_mse'])]
        for name, value in pairs:
            p=df[f'{name}_pred_standardized'].to_numpy().reshape(truth.shape)
            require(np.isclose(np.mean((p-truth)**2),value,rtol=1e-13), 'historical MSE')
    else:
        for name, metric in score['metrics_standardized_and_raw'].items():
            p=df[f'{name}_pred_standardized'].to_numpy().reshape(truth.shape)
            require(np.isclose(np.mean((p-truth)**2),metric['mse'],rtol=1e-13), 'historical MSE')
    return {'status':'PASS_CURRENT_SCORE_WITH_HISTORICAL_BYTE_RESERVATIONS' if any(x['status']!='EXACT' for x in byte_checks) else 'PASS',
            'origins':len(meta),'bound_truth_source_exact':True,'historical_byte_checks':byte_checks,
            'scope':'Stored predictions and scores audited only. Historical model training and permutation runs were not rerun.'}

def finalize(bundle, packages, out):
    out.mkdir(parents=True,exist_ok=True)
    spec=ROOT/'team_work/main/round2/split_spec_v2.json'
    require(master.sha(spec)==master.SPEC_SHA,'split spec')
    master.verify_bundle(bundle,master.SIGNATURE,spec)
    audits=[]; rows=[]
    cal=read(HERE.parent/'round11/results/calibration_integrity_audit.json')
    for task in TASKS:
        intake=master.verify_package(packages/task,spec,bundle)
        for scenario in SCENARIOS:
            old=HERE.parent/'round11/results'/task/scenario; new=HERE/'results'/task/scenario
            r0=sealed(old,HERE.parent/'round11'); r1=sealed(new,HERE)
            require(r0['selected']==r1['selected']==r0['fallback']==r1['fallback']=='N','route changed')
            require(read(old/'models.json')==read(new/'models.json'),'selected model weights/alpha changed')
            p0=pd.read_csv(old/'selection_predictions.csv',float_precision='round_trip')
            p1=pd.read_csv(new/'selection_predictions.csv',float_precision='round_trip')
            d0=p0[p0.segment.eq('decision')].reset_index(drop=True); d1=p1[p1.segment.eq('decision')].reset_index(drop=True)
            require(d0.equals(d1),'decision prediction changed')
            audit_row=next(x for x in cal['rows'] if x['task']==task)
            require(r1['calibration_mse']==audit_row['calibration_scores_train_only'],'independent calibration mismatch')
            score=read(old/'test_score.json')
            require(score['prediction_sha256']==master.sha(old/'test_predictions.csv'),'test bytes changed')
            require(score['route_sha256']==master.sha(old/'sealed_route.json'),'historical route binding')
            test_audit=audit_test_csv(old/'test_predictions.csv',bundle,task,score)
            record={'status':'PASS_CORRECTION_WITH_UNCHANGED_TEST','task':task,'scenario':scenario,'package_audit':intake,
                    'train_only_calibration_crosscheck':True,'all_model_weights_and_alphas_unchanged':True,'decision_predictions_exact_equal':True,
                    'original_route_sha256':master.sha(old/'sealed_route.json'),'corrected_route_sha256':master.sha(new/'sealed_route.json'),
                    'original_test_prediction_sha256':master.sha(old/'test_predictions.csv'),'original_test_score_sha256':master.sha(old/'test_score.json'),
                    'test_audit':test_audit,'new_test_predictions_created':False,
                    'scope':'Retrospective calibration correction after original test. Original route was frozen before original test. No new blind test is claimed.'}
            master.write_json(new/'correction_acceptance.json',record); audits.append(record)
        require(master.sha(HERE.parent/'round11/results'/task/'proxy/test_predictions.csv')==master.sha(HERE.parent/'round11/results'/task/'conservative_lag/test_predictions.csv'),'numeric scenario predictions differ')
        score=read(HERE.parent/'round11/results'/task/'proxy/test_score.json'); metrics=score['metrics']; comp=score['comparisons']['AR-Ridge']
        rows.append({'domain':task.split('_')[0],'task':task,'scenario':'proxy;conservative_lag','selected':'N','fallback':'N','test_origins':score['n_origins'],'horizon':score['horizon'],
                     'selected_mse':metrics['N']['mse'],'comparison':'AR-Ridge','comparison_mse':metrics['AR-Ridge']['mse'],'gain_pct':comp['gain_pct'],
                     'delta_ci95_low':comp['ci95_origin_delta'][0],'delta_ci95_high':comp['ci95_origin_delta'][1],
                     'conclusion':'CI crosses zero; no clear gain vs AR-Ridge','source_score':f'team_work/main/round11/results/{task}/proxy/test_score.json'})
    ag=read(HERE.parent/'round8/results/Agriculture_test_score.json'); cl=read(HERE.parent/'round10/results/Climate_test_score.json')
    hist=[]
    for domain,task,round_name,score in [('Agriculture','Agriculture_h12_f1','round8',ag),('Climate','Climate_h4_f2','round10',cl)]:
        hist.append({'domain':domain,**historical_bound(HERE.parent/round_name/'results',domain,task,bundle,score)})
    require(master.sha(HERE.parent/'round8/results/sealed_route.json')==ag['route_sha256'],'Agriculture route')
    require(master.sha(HERE.parent/'round9/results/sealed_route.json')==cl['route_sha256'],'Climate route')
    climate_original=PROJECT/'主控接收数据/SafeFAME_TS_第9轮模型_SHY_008a722/audit_detached_worktree/03 辅助电脑二 交付九次/test_prediction/N_test_predictions.csv'
    require(master.sha(climate_original)==cl['selected_prediction_sha256'],'Climate original prediction bytes')
    # Preserve the old committed INCOMPLETE record. Current bytes receive a separate master audit.
    acceptance9=read(HERE.parent/'round10/results/round9_model_acceptance.json')
    master.write_json(out/'Climate_evidence_reservation.json',{'status':'MASTER_SCORE_BYTES_PASS_WITH_HISTORICAL_RESERVATION',
        'current_original_prediction_sha256':master.sha(climate_original),'current_route_sha256':cl['route_sha256'],
        'historical_committed_gate':acceptance9['committed_gate'],'historical_independent_gate':acceptance9['independent_gate'],
        'reservation':acceptance9['reservation'],'old_gate_record_overwritten':False,
        'scope':'Current scoring files, origin mapping, truth mapping and scores verified. Historical scoring manifest has line-ending differences and stale log; all retained in separate audit. No claim that the old manifests or auxiliary committed gate record were repaired.'})
    rows.insert(0,{'domain':'Climate','task':cl['task'],'scenario':'proxy','selected':'N','fallback':'N','test_origins':cl['n_origins'],'horizon':cl['horizon'],
       'selected_mse':cl['metrics_standardized_and_raw']['N']['mse'],'comparison':'Last','comparison_mse':cl['metrics_standardized_and_raw']['Last']['mse'],
       'gain_pct':cl['comparisons_vs_N']['Last']['gain_pct'],'delta_ci95_low':cl['comparisons_vs_N']['Last']['block_bootstrap_ci95'][0],
       'delta_ci95_high':cl['comparisons_vs_N']['Last']['block_bootstrap_ci95'][1],'conclusion':'CI crosses zero vs Last; improves vs AR-Ridge in this fold','source_score':'team_work/main/round10/results/Climate_test_score.json'})
    rows.insert(0,{'domain':'Agriculture','task':ag['task_id'],'scenario':'proxy','selected':ag['selected'],'fallback':ag['fallback'],'test_origins':ag['test_windows'],'horizon':12,
       'selected_mse':ag['selected_mse'],'comparison':'AR-Ridge','comparison_mse':ag['fallback_mse'],'gain_pct':ag['gain_pct'],'delta_ci95_low':ag['delta_ci95'][0],
       'delta_ci95_high':ag['delta_ci95'][1],'conclusion':'Positive bootstrap delta in this fold; source-time limitation remains','source_score':'team_work/main/round8/results/Agriculture_test_score.json'})
    master.write_csv(out/'four_domain_test_summary.csv',rows)
    import scipy, sklearn, matplotlib, threadpoolctl
    runtime={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'matplotlib':matplotlib.__version__,'threadpoolctl':threadpoolctl.__version__}
    master.write_json(out/'runtime_versions.json',runtime)
    (HERE/'requirements-master.txt').write_text('\n'.join(f'{k}=={v}' for k,v in runtime.items() if k!='python')+'\n',encoding='utf-8',newline='\n')
    source=PROJECT/'主控接收数据/SafeFAME_TS_第9轮模型_SHY_008a722/audit_detached_worktree/03 辅助电脑二 交付九次/model'
    provenance=[]
    for p in sorted((HERE/'model_engine').glob('*.py')):
        original=source/p.name
        provenance.append({'file':p.name,'master_sha256':master.sha(p),'upstream_sha256':master.sha(original) if original.exists() else None,
                           'same_bytes':p.read_bytes()==original.read_bytes() if original.exists() else False})
    master.write_json(out/'source_provenance.json',{'upstream_branch':'SHY','upstream_commit':'008a7223c8570cc8fe3fd5d7202b287cfa8f2f78','files':provenance,
       'changes':['Use isolated spectral function instead of importing broad project runner.','Constant-quality columns skip Q and record effective branches.','No training text skips semantic branches.','Master selection records train-only calibration predictions before train+cal refit.'],
       'credit':'Model engine extends teammate delivery; not claimed as wholly new master code.'})
    final={'status':'PASS_FOR_DECLARED_SCOPE','bundle_signature':master.SIGNATURE,'split_spec_sha256':master.SPEC_SHA,
           'new_task_scenario_audits':audits,'historical_score_audits':hist,'calibration_audit_sha256':master.sha(HERE.parent/'round11/results/calibration_integrity_audit.json'),
           'summary':rows,'limits':['Only four named task/fold/horizon experiments closed, not all possible folds or datasets.','Original URLs and true publication times remain unverified; lag is a sensitivity scenario.','Correction replay is retrospective and is not a new unseen holdout.','Agriculture and Climate original training and null refits were not rerun this round.','Climate auxiliary committed gate retains INCOMPLETE reservation.','Rejected SocialGood/Environment gates did not run 999 permutations; their necessary conditions already failed.','Bootstrap block equals horizon; no full block-length sensitivity study claimed.']}
    master.write_json(out/'final_acceptance.json',final)
    plots(rows,out)
    report(rows,out,runtime)
    print(json.dumps({'status':final['status'],'new_task_scenarios':len(audits),'historical_domains':len(hist),'output':str(out)},ensure_ascii=False))

def plots(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, ticker
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.exists(): font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font_path)).get_name() if font_path.exists() else 'DejaVu Sans',
                         'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    domains={'Agriculture':'农业','Climate':'气候','SocialGood':'社会公益','Environment':'环境'}
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,r in zip(axs.flat,rows):
        ax.bar(['冻结路线',r['comparison']],[r['selected_mse'],r['comparison_mse']],color=['#165f9e','#aebbc9'],width=.55)
        ax.set_title(f"{domains[r['domain']]} | {r['test_origins']}个测试起点，H={r['horizon']}")
        ax.set_ylabel('标准化测试 MSE')
        ax.text(.02,.94,f"路线：{r['selected']}\n相对图中对照改善：{r['gain_pct']:.2f}%",transform=ax.transAxes,va='top',fontsize=9)
        ax.set_ylim(0,max(r['selected_mse'],r['comparison_mse'])*1.4)
    fig.suptitle('四领域冻结测试结果（标准化与目标不同，仅在各子图内比较）')
    fig.savefig(out/'four_domain_MSE.png'); plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
    for ax,r in zip(axs.flat,rows):
        lo,hi=r['delta_ci95_low'],r['delta_ci95_high']; center=r['comparison_mse']-r['selected_mse']
        ax.errorbar([center],[0],xerr=[[center-lo],[hi-center]],fmt='o',capsize=5,color='#165f9e'); ax.axvline(0,color='#9f2929',ls='--',lw=1)
        ax.set_title(domains[r['domain']]); ax.set_yticks([]); ax.set_xlabel('对照 MSE − 冻结路线 MSE')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.text(.04,.9,r['comparison'],transform=ax.transAxes,fontsize=9)
    fig.suptitle('移动块 Bootstrap 95%差值区间（正值支持冻结路线）')
    fig.savefig(out/'four_domain_delta_CI.png'); plt.close(fig)
    cm=pd.read_csv(HERE/'results/Environment_h7_f2/proxy/candidate_metrics.csv').set_index('candidate')
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    x=np.arange(2)
    for offset,(c,col) in enumerate(zip(('N','N+S+Q','N+S+Q+SF'),('#165f9e','#c28126','#408777'))):
        ax.bar(x+(offset-1)*.23,cm.loc[c,['first_half_mse','second_half_mse']].to_numpy(float),width=.23,label=c,color=col)
    ax.set_xticks(x,['选择段前半期','选择段后半期']); ax.set_ylabel('标准化 MSE'); ax.legend()
    ax.set_title('环境领域：文本候选前半期更差，安全门控拒绝启用')
    fig.savefig(out/'Environment_gate_halves.png'); plt.close(fig)

def report(rows,out,runtime):
    text=['SafeFAME-TS 主控接管技术闭环交付（2026-10-07）','一、已完成的技术范围',
      '主控独立完成 SocialGood_h3_f1、Environment_h7_f2 的数据物理隔离、两种时间情景、候选训练、门控、路线冻结、测试评分与字节验收。连同已完成的 Agriculture_h12_f1、Climate_h4_f2，共闭合四个指定领域任务。',
      '候选为 N、N+Q、N+S+Q、N+S+Q+SF；基线为 Last、SeasonalNaive、AR-Ridge。N 是数值 Ridge 分支，不是深度神经网络。S 是冻结语义输入，Q 是质量输入，SF 是语义与数值谱交互。模型实现基于队友已有引擎，主控完成隔离、退化修正、校准修正和完整验收。',
      '二、关键结果（不同领域标准化不同，不应跨领域排名）']
    for r in rows:
        text.append(f"{r['domain']}：{r['selected']}；测试起点 {r['test_origins']}，预测步长 {r['horizon']}；MSE={r['selected_mse']:.9f}，对照 {r['comparison']} MSE={r['comparison_mse']:.9f}，改善 {r['gain_pct']:.4f}%；差值95%区间 [{r['delta_ci95_low']:.9f}, {r['delta_ci95_high']:.9f}]。")
    text+=['Agriculture 当前折支持文本交互路线，但不能推断所有领域或所有时间范围均有效。Climate 对 AR-Ridge 有正差值区间，但对 Last 区间跨零；SocialGood 和 Environment 对 AR-Ridge 的区间均跨零。',
      '三、真实修正与保留记录',
      '发现旧接口把 train+cal 重拟合后的 calibration 预测作为校准评分，造成乐观偏差。已独立复核四领域 train-only 校准评分，回退模型选择均不变。round12 修正 SocialGood/Environment：先 train-only 预测 calibration，再 train+cal 重拟合用于 decision；独立校准审计一致，全部候选权重与参数、decision 预测和既有测试预测均未改变。',
      '校准用于 alpha 选择，仍是选模评分，不能描述为无偏泛化性能。修正发生在原测试之后，属于回溯纠错，不把它包装为新的盲测；原 round11 路线在原测试之前已冻结。round11 保存旧流程证据，round12 为后续执行入口，勿混用。',
      'Climate 周期以冻结 spec 的52为准，旧任务文档的12是错误描述；本次验收使用52。Climate 辅助交付原已提交门禁仍有末尾换行哈希保留意见；主控旧评分清单也出现Git换行转换及旧日志差异。逐文件确认评分内容仅换行改变，并从正式数据复算分数、核对原预测字节；新验收单独保存，未修改旧清单哈希，也未声称旧清单全部通过。',
      '四、门控与反证',
      'SocialGood 训练段340个起点均无可用文本且质量列恒定，语义不能训练；候选真实退化为N，不能声称完成该领域有效多模态学习。两种情景仅校准起点385的覆盖不同，最终数值预测字节一致。',
      'Environment 全段有文本，但两种文本候选均在decision前半段差于N，未满足双半段改善。已经拒绝的候选不执行999次置换；记录p=null、null_refits=0，不能声称该部分完成置换显著性检验。必要条件短路不增加候选可通过集合。',
      '五、证据边界与技术成熟度',
      '已具备四指定任务的代码实验闭环、冻结预测、基线比较、安全回退及异常输入核验。适合作为课程项目技术主体；尚不是完成全部滚动折、全部预测步长、部署产品或来源发布时间确认的系统。',
      '原文URL和真实发布时间仍未核实；proxy/lag只是时点敏感性情景。bootstrap按预测步长设块长，未声称做完整块长敏感性。最后阶段文书、PPT、答辩叙事和学校信息仍需处理。',
      '六、复核与复现',
      'Python版本：'+runtime['python']+'；精确依赖见requirements-master.txt和runtime_versions.json。随机种子2026，alpha网格与划分均来自冻结spec。依赖、缓存、临时文件均位于D盘。',
      'run_master.py分prepare/select/test/verify四阶段。test拒绝覆盖已有预测。对于本轮已揭示测试真值的数据，仅运行验收或复算分数，不重做选模再声称盲测。新的可比较研究应预先确定新的未见时间段或外部数据，并单独冻结协议。',
      '校正验收见各task/scenario/correction_acceptance.json；整体验收final_acceptance.json；数据包位置见输入位置与复核说明.txt；回归测试日志在results/final。',
      '七、下一阶段顺序',
      '1. 以本交付汇总四领域结果并写课程最终报告，突出“识别文本无益并安全回退”，不追求四领域都获胜。',
      '2. 按老师课件要求完成三人贡献说明和第10周进展PPT；工程由主控接管不等于虚构三人实际贡献。',
      '3. 如需扩展技术证据，先确定新的未见评估数据和协议，再增加滚动折或深度基线；禁止继续调优已揭示测试段。',
      '4. 统一合入main必须单独执行；当前成果保留在主控分支。']
    (out/'主控技术进展与验收说明.txt').write_text('\n\n'.join(text)+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--bundle',type=Path,required=True); ap.add_argument('--packages',type=Path,required=True); ap.add_argument('--out',type=Path,default=HERE/'results/final'); ap.add_argument('--require-full',action='store_true'); a=ap.parse_args()
    # Large selection grids and formal arrays are local-only by repository policy.
    # A clean clone must report the missing scope; missing tracked compact evidence
    # still fails within finalize() and is never silently skipped.
    local_required=[a.bundle/'MANIFEST.json'] if not a.bundle.is_dir() else []
    for task in TASKS:
        local_required.append(a.packages/task/'MANIFEST.json')
        for scenario in SCENARIOS:
            local_required.extend([HERE.parent/'round11/results'/task/scenario/'selection_predictions.csv',HERE/'results'/task/scenario/'selection_predictions.csv'])
    local_required.append(PROJECT/'主控接收数据/SafeFAME_TS_第9轮模型_SHY_008a722/audit_detached_worktree/03 辅助电脑二 交付九次/test_prediction/N_test_predictions.csv')
    missing=[str(p) for p in local_required if not p.exists()]
    if missing:
        if a.require_full: raise FileNotFoundError('Full local evidence required: '+repr(missing))
        result={'status':'PARTIAL_VERIFICATION','missing_local_only_evidence':missing,'verified':[],
                'skipped':['source slices','selection replay equality','historical original Climate prediction bytes','test score recomputation'],
                'interpretation':'No experiment acceptance claimed. Restore local input bundles/grids and run --require-full for full declared-scope audit.'}
        master.write_json(a.out/'partial_verification.json',result); print(json.dumps(result,ensure_ascii=False))
    else:
        with threadpool_limits(limits=1): finalize(a.bundle,a.packages,a.out)
