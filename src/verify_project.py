"""Verify the final SafeFAME-TS v2 evidence chain without retraining."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sys
import statistics
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs" / "tables" / "v2"
FINAL = ROOT / "outputs" / "tables" / "final"
SENSITIVITY = ROOT / "outputs" / "reviewer_sensitivity"
ATTRIBUTION = ROOT / "outputs" / "attribution_audit_v4"
SKIPPED: list[str] = []


def skip(label: str) -> None:
    if label not in SKIPPED:
        SKIPPED.append(label)
        print(f"SKIP: {label}", file=sys.stderr)


def is_lfs_pointer(path: Path) -> bool:
    return path.is_file() and path.read_bytes().startswith(
        b"version https://git-lfs.github.com/spec/"
    )


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f"{label}: {actual} != {expected}")


def check_docx(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        xml = archive.read("word/document.xml").decode("utf-8")
        for marker in ("TODO", "待补充", "XXXXX"):
            assert marker not in xml, f"placeholder remains in {path.name}: {marker}"
        assert "Test-Time Training for Multimodal Time Series Forecasting" not in xml


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submission() -> dict:
    config = json.loads((ROOT / 'submission.json').read_text(encoding='utf-8'))
    assert [r['role'] for r in config['reports']] == ['week4', 'week10', 'final']
    for index, report in enumerate(config['reports'], 1):
        found = list(ROOT.glob(f'SafeFAME-TS_0{index}_*.docx'))
        assert len(found) == 1 and found[0].name == report['docx'], 'Ambiguous root report version'
        keys = ['docx', 'mirror']
        if config.get('pdf_required', False):
            keys.extend(('pdf', 'mirror_pdf'))
        for key in keys:
            path = (ROOT / report[key]).resolve()
            assert path.is_relative_to(ROOT) and path.is_file(), f'Missing/unsafe report: {report[key]}'
        assert digest(ROOT/report['docx']) == digest(ROOT/report['mirror']), 'DOCX mirrors differ'
        if config.get('pdf_required', False):
            assert digest(ROOT/report['pdf']) == digest(ROOT/report['mirror_pdf']), 'PDF mirrors differ'
    return config


def document_text(path: Path) -> tuple[str, ET.Element]:
    with zipfile.ZipFile(path) as z:
        tree=ET.fromstring(z.read('word/document.xml'))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    return '\n'.join(''.join(p.itertext()) for p in tree.findall('.//w:p',ns)), tree


def check_report_tables(tree: ET.Element, metrics, audit, sensitivity, ett) -> None:
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    tables=[[[ ''.join(c.itertext()) for c in r.findall('w:tc',ns)]
             for r in t.findall('w:tr',ns)] for t in tree.findall('.//w:tbl',ns)]
    def table(header):
        found=[t for t in tables if t[0]==header]
        assert len(found)==1, f'Missing/ambiguous report table: {header}'
        return found[0][1:]
    semantic={(r['domain'],r['horizon']):r for r in metrics if r['model']=='semantic_residual'}
    data=table(['领域','H','MSE','MAE','RMSE','相对改善','块区间'])
    assert len(data)==18
    for r in data:
        v=semantic[r[0],r[1]]
        expected=[v['domain'],v['horizon'],*[f'{float(v[k]):.4f}' for k in ('mse','mae','rmse')],
                  f"{float(v['improvement_vs_fallback_pct']):.2f}%",
                  f"[{float(v['block_ci_low']):.4f}, {float(v['block_ci_high']):.4f}]"]
        assert r==expected, f'DOCX result differs from CSV: {r}'
    data=table(['领域','H','数值回退','置换p','两段均胜','最终路径'])
    lookup={(r['domain'],r['horizon']):r for r in audit}
    assert len(data)==18
    for r in data:
        v=lookup[r[0],r[1]]
        assert r[2:]==[v['numeric_fallback'],f"{float(v['semantic_residual_permutation_p_value']):.2f}",
                      '是' if v['semantic_residual_segment_wins'].lower()=='true' else '否','回退']
    data=table(['数据','H','最佳模型','MSE均值±标准差','最佳非Patch','改善'])
    lookup={(r['dataset'],r['horizon']):r for r in ett}
    assert len(data)==6
    for r in data:
        v=lookup[r[0],r[1]]
        assert r[2:]==[v['best_model'],f"{float(v['best_mse']):.4f}±{float(v['best_mse_std']):.4f}",
                      v['best_non_patch_model'],f"{float(v['patchtst_gain_vs_best_non_patch_pct']):.2f}%"]
    data=table(['领域','H','决策起点','块长','非重叠块','语义近似MDE','频率近似MDE'])
    lookup={(r['domain'],r['horizon'],r['variant']):r for r in sensitivity}
    assert len(data)==18
    for r in data:
        v=lookup[r[0],r[1],'semantic_residual']
        assert r[2:5]==[v[k] for k in ('decision_windows','block_length','nonoverlapping_decision_blocks')]
        for i,variant in enumerate(('semantic_residual','frequency_residual'),5):
            value=lookup[r[0],r[1],variant]['approx_mde_vs_control_pct']
            assert r[i]==(f'{float(value):.1f}%' if value else '不足3块')
    data=table(['指标（18任务、36路径）','历史v2','审查后v3'])
    assert data==[
        ['校准 / 决策 / 测试起点','648 / 655 / 2764','571 / 586 / 2764'],
        ['完整门槛放行 / 最终回退任务','0 / 18','0 / 18'],
        ['99次检验p≤0.025的路径','0','2'],
        ['语义点改善 / 区间正 / 区间负','13 / 3 / 4','12 / 3 / 2'],
        ['频率点改善 / 区间正 / 区间负','8 / 2 / 3','8 / 3 / 3'],
        ['999次诊断p≤0.025的路径','0（历史混合求解器）','1（统一cholesky）'],
        ['纯数值残差消融区间正 / 负','2 / 9','2 / 8'],
        ['功效诊断不足3块的路径','32 / 36','34 / 36']]


def check_crossrefs(tree: ET.Element) -> None:
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    paragraphs=[''.join(p.itertext()) for p in tree.findall('.//w:body/w:p',ns)]
    captions=[]
    body=[]
    for p in paragraphs:
        match=re.match(r'^([图表])\s*([AB]?\d+)\s',p)
        if match: captions.append(match.groups())
        else: body.append(p)
    text='\n'.join(body)
    for kind,number in captions:
        assert re.search(kind+r'\s*'+number+r'(?!\d)',text),f'Unreferenced caption: {kind}{number}'


def check_v4() -> dict:
    folder=ROOT/'outputs/safefame_v4'
    data_root=ROOT/'data_processed'
    task_root=folder/'tasks'
    manifest=json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
    cfg=manifest['config']
    assert hashlib.sha256(json.dumps(dict(config=cfg,hashes=manifest['hashes']),sort_keys=True).encode()).hexdigest()==manifest['signature']
    assert cfg==json.loads((ROOT/'configs/safefame_v4.json').read_text(encoding='utf-8'))
    assert (cfg['row_permutations'],cfg['circular_shifts'],cfg['bootstrap_repeats'])==(999,999,5000)
    assert cfg['seeds']==[2026,2027,2028,2029,2030] and cfg['p_threshold']==.025
    for name,sha in manifest['hashes'].items():
        path=ROOT/name
        if path.is_file():
            assert digest(path)==sha,('v4 source changed',name)
        elif name.startswith('data_processed/'):
            assert not data_root.is_dir(),('missing v4 input in existing data tree',name)
            skip('v4 full input hashes (data_processed is not tracked)')
        else:
            raise AssertionError(('missing tracked v4 source',name))
    record=json.loads((folder/'verification.json').read_text(encoding='utf-8'))
    assert record['status']=='PASS' and record['replay_models'] is True
    assert (record['task_folds'],record['candidate_paths'])==(120,240)
    for name,sha in record['sha256'].items():
        path=ROOT/name
        if path.is_file():
            assert digest(path)==sha,('v4 evidence changed',name)
        elif name.startswith('outputs/safefame_v4/tasks/'):
            assert not task_root.is_dir(),('missing v4 evidence in existing task tree',name)
            skip('v4 task-level hashes and model replay (tasks is not tracked)')
        else:
            raise AssertionError(('missing tracked v4 evidence',name))
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    candidates=rows(folder/'candidate_results.csv');tasks=rows(folder/'task_results.csv')
    assert len(candidates)==240 and len(tasks)==120
    expected={(d,str(h),str(f)) for d,s in cfg['domains'].items() for h in s['horizons'] for f in (1,2,3)}
    assert {(t['domain'],t['horizon'],t['fold']) for t in tasks}==expected
    if task_root.is_dir():
        expected_dirs={f'{d}_h{h}_f{f}' for d,h,f in expected}
        actual_dirs={p.name for p in task_root.iterdir() if p.is_dir()}
        assert actual_dirs==expected_dirs,('incomplete v4 task tree',sorted(expected_dirs-actual_dirs))
        for t in tasks:
            task=task_root/f"{t['domain']}_h{t['horizon']}_f{t['fold']}"
            audit=json.loads((task/'audit.json').read_text(encoding='utf-8'))
            frozen=json.loads((task/'frozen_selection.json').read_text(encoding='utf-8'))
            for key,segment in [('train_text_coverage_pct','train'),('text_coverage_pct','test')]:close(float(t[key]),record['text_coverage_pct'][task.name][segment],'v4 text coverage')
            assert (t['fallback'],t['selected'])==(frozen['fallback'],frozen['selected'])
            close(float(t['selected_mse']),audit['selected_metrics']['mse'],'v4 selected mse')
            close(float(t['fallback_mse']),audit['fallback_metrics']['mse'],'v4 fallback mse')
            close(float(t['selected_gain_pct']),100*(1-float(t['selected_mse'])/float(t['fallback_mse'])),'v4 gain')
            for key,value in zip(('selected_delta','selected_ci_low','selected_ci_high'),audit['selected_delta']):close(float(t[key]),value,key)
            for key,segment in [('cal_windows','cal'),('decision_windows','dec'),('test_windows','test')]:assert int(t[key])==audit['windows'][segment]
            originals=rows(task/'metrics.csv')
            matches=[r for r in candidates if (r['domain'],r['horizon'],r['fold'])==(t['domain'],t['horizon'],t['fold'])]
            assert len(matches)==2
            for original in originals:
                r=next(x for x in matches if x['variant']==original['variant'])
                for key,value in original.items():
                    if key in ('domain','variant','fallback','selected_path','segment_wins','eligible') or value=='':assert r[key]==value
                    else:close(float(r[key]),float(value),'v4 aggregate '+key)
    else:
        skip('v4 task CSV/audit/frozen-route cross-checks (tasks is not tracked)')
    assert (summary['series'],summary['domains'],summary['task_families'],summary['task_folds'],summary['candidate_paths'])==(10,9,40,120,240)
    for v in ('semantic_residual','frequency_residual'):
        group=[r for r in candidates if r['variant']==v]
        checks={'point_positive':('improvement_vs_fallback_pct',lambda x:x>0),'ci_positive':('ci_low',lambda x:x>0),
                'ci_negative':('ci_high',lambda x:x<0),'row_pass':('p_value',lambda x:x<=.025),
                'circular_pass':('circular_p_value',lambda x:x<=.025),'matched_ci_positive':('matched_ci_low',lambda x:x>0),
                'matched_ci_negative':('matched_ci_high',lambda x:x<0)}
        for key,(column,predicate) in checks.items():assert summary[v][key]==sum(predicate(float(r[column])) for r in group)
        assert summary[v]['eligible']==sum(r['eligible'].lower()=='true' for r in group)
    assert summary['text_selected']==sum(t['selected']!='numeric_fallback' for t in tasks)
    assert summary['numeric_fallback']==120-summary['text_selected']
    for key,column,predicate in [('selected_point_positive','selected_gain_pct',lambda x:x>0),('selected_point_negative','selected_gain_pct',lambda x:x<0),
                                 ('selected_ci_positive','selected_ci_low',lambda x:x>0),('selected_ci_negative','selected_ci_high',lambda x:x<0),
                                 ('decision_blocks_ge8','decision_blocks',lambda x:x>=8),('decision_blocks_lt3','decision_blocks',lambda x:x<3)]:
        assert summary[key]==sum(predicate(float(t[column])) for t in tasks)
    close(summary['selected_mean_gain_pct'],statistics.mean(float(t['selected_gain_pct']) for t in tasks),'v4 mean gain')
    close(summary['selected_median_gain_pct'],statistics.median(float(t['selected_gain_pct']) for t in tasks),'v4 median gain')
    for key,col in [('calibration_windows','cal_windows'),('decision_windows','decision_windows'),('test_windows','test_windows')]:assert summary[key]==sum(int(t[col]) for t in tasks)
    if task_root.is_dir():
        assert summary['calibration_checkpoints']==len(list(task_root.glob('*/models/*_cal_s*.pt')))==1200
    else:
        skip('v4 calibration checkpoint count (tasks is not tracked)')
    assert summary['refit_checkpoints']==record['replayed_refit_checkpoints']
    assert sorted(summary['zero_train_text_tasks'])==sorted(name for name,coverage in record['text_coverage_pct'].items() if coverage['train']==0)
    return summary


def check_attribution_audit() -> dict:
    summary=json.loads((ATTRIBUTION/'summary.json').read_text(encoding='utf-8'))
    protocol=json.loads((ATTRIBUTION/'protocol.json').read_text(encoding='utf-8'))
    assert summary['analysis_role'].startswith('post-review attribution diagnostics')
    assert (summary['selected_routes'],summary['selected_semantic_routes'],summary['selected_frequency_routes'])==(10,9,1)
    assert (summary['original_selected_point_positive'],summary['original_selected_point_negative'])==(8,2)
    assert (summary['original_selected_ci_positive'],summary['original_selected_ci_negative'])==(5,0)
    assert (summary['legacy_numeric_ablation_selected_ci_positive'],summary['legacy_numeric_ablation_selected_ci_negative'])==(3,0)
    assert summary['joint_posthoc_supported_routes']==1
    assert summary['joint_posthoc_supported_list']==['Agriculture H6 F1 semantic_residual']
    close(summary['all_120_mean_gain_pct'],0.7675926100111714,'attribution all-route mean')
    close(summary['all_120_mean_without_largest_gain_pct'],0.32911975258455495,'attribution leave-largest-out mean')
    assert (summary['fallback_anchored_full_point_positive'],summary['fallback_anchored_full_ci_positive'],summary['fallback_anchored_full_ci_negative'])==(6,6,1)
    assert (summary['full_vs_dimension_matched_point_positive'],summary['full_vs_dimension_matched_ci_positive'],summary['full_vs_dimension_matched_ci_negative'])==(9,6,0)
    assert summary['multiplicity']['candidate_paths']==240
    assert (summary['multiplicity']['raw_p_le_0_025'],summary['multiplicity']['holm_p_le_0_05'],summary['multiplicity']['bh_q_le_0_05'])==(38,0,22)
    correction=summary['corrected_frequency_permutation']
    assert len(correction)==1 and correction[0]['domain']=='Agriculture' and correction[0]['horizon']==12 and correction[0]['fold']==1
    assert (correction[0]['permutations'],correction[0]['legacy_row_p'],correction[0]['corrected_row_p'])==(999,.021,.085)
    assert (correction[0]['legacy_circular_p'],correction[0]['corrected_circular_p'])==(.06,.068)
    assert summary['bootstrap_repeats']==5000
    assert protocol['routing_effect']=='none' and protocol['scope']=='ten routes enabled by the frozen v4 gate'
    for name,sha in summary['sha256'].items():
        path={'script':ROOT/'src/run_attribution_audit_v4.py','v4_summary':ROOT/'outputs/safefame_v4/summary.json',
              'v4_tasks':ROOT/'outputs/safefame_v4/task_results.csv','v4_candidates':ROOT/'outputs/safefame_v4/candidate_results.csv'}[name]
        assert digest(path)==sha,('attribution source changed',name)
    expected={'selected_route_attribution.csv','selected_route_comparisons.csv','block_length_sensitivity.csv',
              'corrected_frequency_permutation.csv','cross_fold_replication.csv','fig_selected_route_attribution.png',
              'protocol.json','summary.json'}
    assert expected.issubset({p.name for p in ATTRIBUTION.iterdir() if p.is_file()})
    evidence=ATTRIBUTION/'evidence'
    if evidence.is_dir():
        assert len(list(evidence.glob('*.npz')))==11
    else:
        skip('attribution NPZ evidence (evidence is not tracked)')
    return summary


def check_archive(config: dict) -> None:
    path=ROOT/config['archive']
    sha_path=ROOT/config['sha256']
    inventory=ROOT/config['inventory']
    inventory_mirror=ROOT/'paper/final'/config['inventory']
    assert inventory.is_file() and inventory_mirror.is_file(), 'Missing support inventory or mirror'
    assert digest(inventory)==digest(inventory_mirror), 'inventory mirrors differ'
    assert path.is_file()==sha_path.is_file(), 'Support ZIP and SHA256 side file must coexist'
    if not path.is_file():
        skip('local support ZIP, object SHA256, and manifest (generated on demand, not tracked by Git)')
        return
    if is_lfs_pointer(path):
        skip('support ZIP contents, object SHA256, and manifest (legacy Git LFS object not downloaded)')
        return
    sha=sha_path.read_text(encoding='utf-8').strip().split(maxsplit=1)
    assert sha == [digest(path), path.name], 'Archive SHA256 mismatch'
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        names=z.namelist()
        assert len(names)==len(set(names)), 'Duplicate ZIP entries'
        for name in names:
            assert not any(x in name.lower() for x in ['.venv','__pycache__','reference_2026','local_private','paper/archive'])
            assert not name.endswith(('.zip','.pyc')), name
            if not config.get('pdf_required', False):
                assert not name.lower().endswith('.pdf'), f'PDF present in DOCX-only release: {name}'
            assert not name.startswith('/') and '..' not in Path(name).parts
        manifest=list(csv.DictReader(io.StringIO(z.read('manifest.tsv').decode('utf-8')),delimiter='\t'))
        assert set(names)=={r['path'] for r in manifest}|{'manifest.tsv','SHA256SUMS.txt'}
        sums=dict(line.split('  ',1)[::-1] for line in z.read('SHA256SUMS.txt').decode('utf-8').splitlines())
        for r in manifest:
            data=z.read(r['path'])
            assert len(data)==int(r['bytes'])
            assert hashlib.sha256(data).hexdigest()==r['sha256']==sums[r['path']]
            assert (ROOT/r['path']).is_file() and digest(ROOT/r['path'])==r['sha256'], f'Stale archived file: {r["path"]}'
        required={'README.md','submission.json','src/verify_project.py',config['inventory'],
                  'paper/final/'+config['inventory'],'outputs/safefame_v3/verification.json',
                  'outputs/revision_comparison/summary.json'}
        required |= {'configs/safefame_v4.json','outputs/safefame_v4/verification.json','outputs/safefame_v4/summary.json'}
        required |= {r['mirror'] for r in config['reports']}
        if config.get('pdf_required', False):
            required |= {r['mirror_pdf'] for r in config['reports']}
        required |= {'outputs/attribution_audit_v4/summary.json','outputs/attribution_audit_v4/protocol.json',
                     'src/verify_attribution_audit_v4.py'}
        required |= {p.relative_to(ROOT).as_posix() for p in (ATTRIBUTION/'evidence').glob('*.npz')}
        assert required.issubset(names), required-set(names)


def main() -> int:
    SKIPPED.clear()
    v4=check_v4()
    attribution=check_attribution_audit()
    v3=ROOT/'outputs/safefame_v3'
    if v3.is_dir():
        assert (v3/'verification.json').is_file(), 'Incomplete v3 evidence tree'
        record=json.loads((v3/'verification.json').read_text(encoding='utf-8'))
        assert record['status']=='PASS' and record['checkpoint_replay'] is True
        assert (record['v3_tasks'],record['v3_metrics'],record['sensitivity_paths'])==(18,72,72)
        for name, expected_hash in record['sha256'].items():
            assert digest(ROOT/name)==expected_hash, f'Revision evidence changed since replay: {name}'
        new_audit=rows(v3/'safefame_v3_selection_audit.csv')
        assert len(new_audit)==18 and all(r['selected_path']=='numeric_fallback' for r in new_audit)
        assert sum(int(r['calibration_windows']) for r in new_audit)==571
        assert sum(int(r['decision_windows']) for r in new_audit)==586
        assert sum(float(r[f'{variant}_permutation_p_value'])<=.025 for r in new_audit for variant in ('semantic_residual','frequency_residual'))==2
        new_protocol=json.loads((v3/'safefame_v3_protocol.json').read_text(encoding='utf-8'))
        assert new_protocol['permutations']==99 and new_protocol['solver']=='cholesky'
        assert 'target-disjoint' in new_protocol['split']['validation_use']
    else:
        skip('v3 task evidence, checkpoint replay, and hashes (safefame_v3 is not tracked)')
    for folder,expected_pass in [('reviewer_sensitivity_corrected',0),('reviewer_sensitivity_v3',1)]:
        evidence_folder=ROOT/'outputs'/folder
        path=evidence_folder/'reviewer_sensitivity_summary.json'
        if evidence_folder.is_dir():
            assert path.is_file(), f'Incomplete sensitivity evidence: {folder}'
            summary=json.loads(path.read_text(encoding='utf-8'))
            assert summary['circular_shift_permutations']==999 and summary['bootstrap_repeats']==5000
            assert summary['aligned_solver']==summary['shifted_solver']=='cholesky'
            assert summary['circular_shift_passes_at_0_025']==expected_pass
        else:
            skip(f'{folder} full sensitivity evidence (directory is not tracked)')
    claims = json.loads((V2 / "verified_claims.json").read_text(encoding="utf-8"))
    metrics = rows(ROOT / "outputs" / "safefame_v2" / "safefame_v2_metrics.csv")
    audit = rows(ROOT / "outputs" / "safefame_v2" / "safefame_v2_selection_audit.csv")
    candidates = [row for row in metrics if row["model"] in {"semantic_residual", "frequency_residual"}]
    semantic = [row for row in candidates if row["model"] == "semantic_residual"]
    frequency = [row for row in candidates if row["model"] == "frequency_residual"]

    assert len(audit) == claims["tasks"] == 18
    assert len(candidates) == claims["total_candidate_pathways"] == 36
    assert sum(row["selected_path"] != "numeric_fallback" for row in audit) == claims["selected_text_tasks"]
    assert sum(float(row["improvement_vs_fallback_pct"]) > 0 for row in semantic) == claims["semantic_point_improvement_tasks"]
    assert sum(float(row["block_ci_low"]) > 0 for row in semantic) == claims["semantic_significant_improvement_tasks"]
    assert sum(float(row["block_ci_high"]) < 0 for row in semantic) == claims["semantic_significant_harm_tasks"]
    assert sum(float(row["improvement_vs_fallback_pct"]) > 0 for row in frequency) == claims["frequency_point_improvement_tasks"]
    assert sum(float(row["block_ci_low"]) > 0 for row in frequency) == claims["frequency_significant_improvement_tasks"]
    assert sum(float(row["block_ci_high"]) < 0 for row in frequency) == claims["frequency_significant_harm_tasks"]
    assert sum(int(row["test_windows"]) for row in audit) == claims["test_windows"]
    assert len({r['domain'] for r in audit})==6
    assert sum(int(r['calibration_windows']) for r in audit)==claims['calibration_windows']==648
    assert sum(int(r['decision_windows']) for r in audit)==claims['decision_windows']==655
    assert claims['test_windows']==2764 and claims['selected_text_tasks']==0
    assert (claims['semantic_point_improvement_tasks'],claims['semantic_significant_improvement_tasks'],claims['semantic_significant_harm_tasks'],claims['frequency_point_improvement_tasks'])==(13,3,4,8)
    by_task={(r['domain'],r['horizon'],r['model']):r for r in metrics}
    for row in audit:
        eligible=[]
        for variant in ('semantic_residual','frequency_residual'):
            p=float(row[f'{variant}_permutation_p_value'])
            assert abs(p*100-round(p*100))<1e-8 and 0.01<=p<=1
            wins=row[f'{variant}_segment_wins'].lower()=='true'
            gate=float(row[f'{variant}_decision_mse'])<float(row['numeric_decision_mse']) and wins and p<=0.025
            assert gate==(row[f'{variant}_eligible'].lower()=='true')
            if gate: eligible.append(variant)
            result=by_task[row['domain'],row['horizon'],variant]
            fallback=by_task[row['domain'],row['horizon'],row['numeric_fallback']]
            close(float(result['improvement_vs_fallback_pct']),100*(1-float(result['mse'])/float(fallback['mse'])),'candidate gain')
        selected=min(eligible,key=lambda v:float(row[f'{v}_decision_mse'])) if eligible else 'numeric_fallback'
        assert selected==row['selected_path']
        final=by_task[row['domain'],row['horizon'],'SafeFAME-TS-v2']
        fallback=by_task[row['domain'],row['horizon'],row['numeric_fallback']]
        for key in ('mse','mae','rmse'): close(float(final[key]),float(fallback[key]),'all fallback output')
    assert sum(float(r[f'{v}_permutation_p_value'])<=0.025 for r in audit for v in ('semantic_residual','frequency_residual'))==claims['permutation_passes_at_0_025']==0
    assert sum(r[f'{v}_segment_wins'].lower()=='true' for r in audit for v in ('semantic_residual','frequency_residual'))==claims['segment_stability_passes']==6

    sensitivity_summary = json.loads((SENSITIVITY / "reviewer_sensitivity_summary.json").read_text(encoding="utf-8"))
    sensitivity = rows(SENSITIVITY / "reviewer_sensitivity_results.csv")
    assert len(sensitivity) == sensitivity_summary["pathways"] == 36
    assert sum(float(row["circular_shift_p_value"]) <= 0.025 for row in sensitivity) == sensitivity_summary["circular_shift_passes_at_0_025"] == 0
    assert sum(float(row["test_gain_ci_low_pct"]) > 0 for row in sensitivity) == sensitivity_summary["matched_control_test_ci_positive"] == 2
    assert sum(float(row["test_gain_ci_high_pct"]) < 0 for row in sensitivity) == sensitivity_summary["matched_control_test_ci_negative"] == 9
    assert sum(row["approx_mde_loss"] == "" for row in sensitivity) == sensitivity_summary["power_audit_unavailable_lt3_blocks"] == 32
    assert sensitivity_summary['circular_shift_permutations']==999
    assert all(int(r['circular_shift_permutations'])==999 for r in sensitivity)
    assert sensitivity_summary['bootstrap_repeats']==5000
    assert 'post-review' in sensitivity_summary['analysis_role']
    assert 'no model selection' in sensitivity_summary['bootstrap_role']

    old_claims = json.loads((FINAL / "verified_claims.json").read_text(encoding="utf-8"))
    ett = rows(FINAL / "table_ett_external.csv")
    assert len(ett) == old_claims["ett_tasks"]
    close(sum(float(row["patchtst_gain_vs_best_non_patch_pct"]) for row in ett) / len(ett), old_claims["ett_mean_patchtst_gain_pct"], "ETT mean gain")
    assert len(ett)==6 and all(r['best_model']=='PatchTST-M' for r in ett)
    assert round(old_claims['ett_mean_patchtst_gain_pct'],2)==12.69

    config=submission()
    docx_paths = [ROOT/r['docx'] for r in config['reports']]
    pdf_paths = [ROOT/r['pdf'] for r in config['reports']] if config.get('pdf_required', False) else []
    expected = [*docx_paths, *pdf_paths]
    expected += list((ROOT / "outputs" / "figures" / "v2").glob("*.png"))
    expected += list(SENSITIVITY.glob("*.png"))
    expected += [
        ROOT / "outputs" / "safefame_v2" / "safefame_v2_protocol.json",
        ROOT / "outputs" / "safefame_v2" / "safefame_v2_metrics.csv",
        ROOT / "outputs" / "safefame_v2" / "safefame_v2_selection_audit.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected if not path.is_file() or path.stat().st_size == 0]
    assert not missing, f"missing artifacts: {missing}"
    for path in docx_paths:
        check_docx(path)
        text,tree=document_text(path)
        for phrase in ('99次逐行文本/质量特征错位','999次循环移位','5000次移动块Bootstrap','不参与模型选择','个人课程设计','无需课堂汇报'):
            assert phrase in text, f'Missing phase/scope statement: {path.name}: {phrase}'
        for phrase in ('事后','目标','重叠','已查看','v3'):
            assert phrase in text, f'Missing revision caveat: {path.name}: {phrase}'
        for phrase in ('v4','120次滚动评估','999次逐行联合错位','不是新增未来时间段的独立确认',
                       f'完整门控实际启用文本{v4["text_selected"]}次，数值回退{v4["numeric_fallback"]}次'):
            assert phrase in text, f'Missing v4 scope/results: {path.name}: {phrase}'
        assert '后半保持未见' not in text and '验证期后半独立用于资格判定' not in text
        assert '式（5）用文本位置置换后的 999' not in text
        assert '语义主成分前8维' not in text
        ns={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
        assert len(tree.findall('.//m:oMath',ns)) in (1,4,7), 'Native equations missing'
        check_crossrefs(tree)
        if path==docx_paths[-1]:
            check_report_tables(tree,metrics,audit,sensitivity,ett)
            ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            tables=[[[ ''.join(c.itertext()) for c in r.findall('w:tc',ns)] for r in t.findall('w:tr',ns)] for t in tree.findall('.//w:tbl',ns)]
            result=next(t for t in tables if t[0]==['指标（每类120条）','语义候选','频率候选'])
            keys=['point_positive','ci_positive','ci_negative','row_pass','eligible','circular_pass','matched_ci_positive','matched_ci_negative']
            assert len(result)==9
            for r,key in zip(result[1:],keys):assert r[1:]==[str(v4[v][key]) for v in ('semantic_residual','frequency_residual')]
            attribution_table=next(t for t in tables if t[0]==['对照问题','点改善','区间正','区间负','证据含义'])
            assert attribution_table[1:]==[
                ['回退锚定完整特征 对 实际回退','6/10','6/10','1/10','去除Last锚点不对称'],
                ['完整特征 对 等维纯数值','9/10','6/10','0/10','匹配输入宽度'],
                ['语义特征 对 纯数值','7/10','4/10','0/10','不含质量特征'],
                ['质量特征 对 纯数值','9/10','4/10','0/10','文本元数据增量'],
                ['完整特征 对 质量特征','7/10','4/10','0/10','控制质量后的语义相关增量'],
                ['完整特征 对 语义特征','9/10','3/10','0/10','控制语义后的质量相关增量']]
            assert str(attribution['joint_posthoc_supported_routes'])+'条同时满足' in text
    for path in pdf_paths:
        assert path.read_bytes()[:5] == b"%PDF-", f"invalid PDF: {path.name}"
        pages=len(re.findall(rb'/Type\s*/Page\b',path.read_bytes()))
        assert pages>0, f'Cannot verify page count: {path}'
        info=next(r for r in config['reports'] if r['pdf']==path.name)
        assert pages==info['rendered_docx_pages'], f'PDF pages changed: {path}'
    qa=json.loads((ROOT/'docs/submission_qa.json').read_text(encoding='utf-8'))
    assert qa['all_pages_visually_reviewed'] is True
    for r in config['reports']:
        assert qa['reports'][r['role']]['docx_sha256']==digest(ROOT/r['docx'])
        assert qa['reports'][r['role']]['rendered_pages']==r['rendered_docx_pages']
        if config.get('pdf_required', False):
            assert qa['reports'][r['role']]['pdf_sha256']==digest(ROOT/r['pdf'])

    protocol = json.loads((ROOT / "outputs" / "safefame_v2" / "safefame_v2_protocol.json").read_text(encoding="utf-8"))
    assert protocol["permutations"] == 99
    close(float(protocol["permutation_p_threshold"]), 0.025, "permutation threshold")
    assert protocol["seeds"] == [2026, 2027, 2028]
    assert 'frozen eligibility gate' in protocol['permutation_role']
    assert 'diagnostics only' in protocol['sensitivity_role']
    assert 'never model or route selection' in protocol['bootstrap_role']
    assert protocol['test_interval']=='5000-repeat moving-block bootstrap of paired origin losses'
    check_archive(config)

    artifact_mode='DOCX/PDF' if config.get('pdf_required', False) else 'DOCX-only'
    if SKIPPED:
        print(f"PARTIAL PASS: repository-resident compact evidence, protocols, report tables, and {artifact_mode} mirrors are consistent.")
        print('SKIPPED: '+'; '.join(SKIPPED))
        print('This is not a full evidence replay and does not validate the skipped items.')
    else:
        print(f"PASS: historical v2/v3, expanded v4 (120 task-folds), attribution audit, sensitivity, ETT, protocols, report tables, {artifact_mode} mirrors and ZIP hashes are consistent. This check is not retraining or proof of unseen test data.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, ValueError, OSError, ET.ParseError, zipfile.BadZipFile) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
