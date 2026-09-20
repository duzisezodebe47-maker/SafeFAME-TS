"""Apply the 2026-09-19 evidence-based edits to the user's root DOCX versions.

Run with the document runtime. Preserve native equations, drawings and styles.
Original documents are retained under paper/archive/20260919_before_revision.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / 'paper/archive/20260919_before_revision'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
POLICY = ('协议口径：99次逐行文本/质量特征错位置换用于主冻结资格门控；'
          '999次循环移位为外部审查后敏感性分析，不改变冻结路由；'
          '5000次移动块Bootstrap只用于测试期损失差区间诊断，不参与模型选择。'
          '本项目为个人课程设计，无需PPT，也无需课堂汇报。')


def replace(p, old, new):
    nodes = p.findall('.//w:t', NS)
    text = ''.join(n.text or '' for n in nodes)
    if old not in text:
        return False
    start = text.index(old)
    end = start + len(old)
    pos = 0
    inserted = False
    for n in nodes:
        s = n.text or ''
        a, b = pos, pos + len(s)
        pos = b
        if b <= start or a >= end:
            continue
        prefix = s[:max(0, start-a)]
        suffix = s[max(0, end-a):]
        n.text = prefix + (new if not inserted else '') + suffix
        n.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        inserted = True
    return True


def main():
    BACKUP.mkdir(parents=True, exist_ok=True)
    files = [p for p in ROOT.rglob('*') if p.is_file() and '.venv' not in p.parts
             and '.git' not in p.parts and '__pycache__' not in p.parts]
    audit = ROOT / 'tmp/revision_20260919'
    audit.mkdir(parents=True, exist_ok=True)
    # Inventory includes unrelated local assets but never enters the support ZIP.
    (audit / 'inventory_before.txt').write_text('\n'.join(str(p.relative_to(ROOT)) for p in files), encoding='utf-8')
    protected = [p for p in files if (p.suffix in {'.csv', '.npy', '.npz', '.pt'} and
                 p.parts[len(ROOT.parts)] in {'outputs','data_processed','references'})]
    if not (audit / 'protected_hashes.json').exists():
        (audit / 'protected_hashes.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}, indent=2), encoding='utf-8')
    roles = []
    log = []
    common = [
        ('Spectral Text Fusion for Multimodal Time Series Forecasting', 'Spectral Text Fusion: A Frequency-Aware Approach to Multimodal Time-Series Forecasting'),
        ('Lin J, Wang Y, Luo H, Pei Z, Wang J', 'Lin J, Wang Y, Luo H, Wang J, Pei Z'),
        ('6.1 数据划分与独立样本单位', '6.1 数据划分与损失汇总单位'),
        ('重叠起点并非独立样本，而非每个 H 步误差', '每个起点先汇总H步误差，重叠起点并非独立样本'),
        ('说明零放行结论不依赖逐行随机置换这一种反事实构造', '提供方向一致的补充证据，但不证明两种检验等价，也不改变冻结路由'),
        ('说明此前差异主要来自残差结构而非可识别的文本增量', '提示残差结构是部分改善的可能解释，不能直接归因于文本'),
        ('每一步输出到新目录，不覆盖原始数据', '输出按阶段目录保存；默认重跑会覆盖该阶段输出，宜在副本中执行，原始数据不覆盖'),
        ('12:     ŷ_t ← r^c_t  若 gate_c 成立；否则 ŷ_t ← b_t', '12:     合格路径按决策段MSE选最优；无合格路径则 ŷ_t ← b_t'),
        ('14: 5000 次移动块 Bootstrap 报告配对损失差的 95% 区间（仅用于诊断）', '14: 5000次移动块Bootstrap诊断测试损失差；999次循环移位仅作审查后敏感性，不回写路由'),
        ('99次文本置换检验', '99次逐行文本/质量特征错位置换检验'),
        ('99次文本错位置换', '99次逐行文本/质量特征错位置换'),
        ('语义主成分前8维的乘积交互', '全部保留的语义主成分与前8个标准化频率特征的乘积交互'),
        ('频率统计及其与全部保留的语义主成分与前8个标准化频率特征的乘积交互', '频率统计，以及全部保留的语义主成分与前8个标准化频率特征的乘积交互'),
        ('独立统计单位是预测起点', '损失汇总单位是预测起点，重叠起点并非独立样本'),
        ('块长至少为H并最多取24或季节周期', '块长取max(H,min(季节周期,24))并限制在当前起点数内'),
        ('768维拼接输入 最多保留24维 仅训练期拟合', '768维拼接；最多24维；门控前仅训练段拟合，冻结后训练加验证段重拟合'),
        ('再仅用训练期数据拟合PCA', '在资格判定阶段仅用训练期数据拟合PCA；冻结后在训练加验证段重拟合'),
        ('冻结 m*、PCA、α 与 gate；在 train+validation 上按固定轮数重训', '冻结 m*、α 与 gate；在 train+validation 上重拟合残差PCA和模型，深度模型按固定轮数重训'),
        ('12: ŷ_t ← r^c_t 若 gate_c 成立；否则 ŷ_t ← b_t', '12: 多条合格路径按决策段MSE选最优；若无合格路径，则 ŷ_t ← b_t'),
        ('99 次文本错位置换', '99 次逐行文本/质量特征错位置换'),
        ('每个模型保存逐预测起点结果，而不只保存平均指标；', '开发基线和ETT保存预测或逐起点误差；v2保存逐任务指标与门控审计，未持久化逐起点预测；'),
        ('核心脚本固定随机种子并保存逐预测起点结果', '核心脚本固定随机种子；v2保存逐任务指标与门控审计，逐起点预测未持久化'),
        ('容量对齐', '相同Ridge家族与调参范围，输入维数不同'),
        ('相同的数值窗口、Ridge 容量与频率统计', '相同的数值窗口、Ridge模型家族和调参范围，频率对照保留频率统计'),
        ('候选差异主要来自残差结构，而非可识别的文本增量', '部分候选优势可由数值残差结构解释；仅两条区间为正，不能统一归因为文本'),
        ('否定结论不依赖神经句向量的选择', '开发阶段两种表征均未呈现稳定对齐优势；不能排除其他编码器有效'),
        ('文本单独几乎不具备预测力', '该开发设置中纯文本误差高于早融合，不代表所有纯文本方法'),
        ('无需PPT展示', '无需PPT或课堂汇报'),
    ]
    for index in range(1,4):
        matches = list(ROOT.glob(f'SafeFAME-TS_0{index}_*.docx'))
        assert len(matches)==1, matches
        path = matches[0]
        source = BACKUP / path.name
        if not source.exists(): shutil.copy2(path, source)
        with ZipFile(source) as z:
            contents = {i.filename: z.read(i.filename) for i in z.infolist()}
        tree = E.fromstring(contents['word/document.xml'])
        for p in tree.findall('.//w:p', NS):
            for old,new in common:
                if replace(p,old,new): log.append([path.name,old,new])
            text = ''.join(p.xpath('.//w:t/text()',namespaces=NS))
            revisions = {}
            if text.startswith('式（5）用文本位置置换后的'):
                revisions[text] = '式（5）使用主冻结协议的99次逐行文本/质量特征错位对照，加一经验p值的分母为100，用于资格门控；999次循环移位属于审查后敏感性分析，不回写路由。小p值只是在该检验构造下反对错位对照的证据，不等同于因果效应。'
            if text.startswith('表征探针进一步表明'):
                revisions[text] = '表征探针属于开发阶段四领域12项任务，未采用v2的独立校准与路径决策双分段门控。MiniLM和TF-IDF各有3项对齐优于错位，纯文本误差也较高，提示应保留数值对照；这些结果不能证明零放行与编码能力无关，不能替代v2结果或排除其他表征有效。（见表10）'
            if text.startswith('PatchTST 被选中8次'):
                revisions[text] = 'PatchTST 被选中8次，SeasonalNaive与AR-Ridge各3次，Last与DLinear-M各2次。经济任务全部选择AR-Ridge，交通任务全部选择季节朴素。开发阶段4领域×3跨度×2模型共24个模型—任务组均有三种子结果；以样本标准差除以均值计算测试MSE变异系数，中位为3.05%、均值为3.82%、最大为20.46%（Energy H24的DLinear-M）。这是开发性训练随机性诊断，不是v2全部18项任务的统计量，也不加入冻结资格门槛。（见图3）'
            if text.startswith('更新的多模态工作进一步引入'):
                revisions[text] = '相关多模态工作包括混合专家融合TiMi[16]、大模型在环解释TimeXL[17]和生成式跨域预测Aurora[18]。本文保留这些方法作为背景，不复现其大型结构；第8.4节的表征探针仅讨论本项目开发设置，不能排除其他编码器有效。'
            if text.startswith('Security 测试段相对训练段'):
                revisions[text] = 'Security 测试段相对训练段发生明显尺度漂移，标准化测试误差远高于其他领域；当前文本候选不能稳定修正这种漂移。该现象揭示当前预测设置的限制，不能据此判断因果适配能力，也不能把相关文本当作导致目标变化的证据。（见图9）'
            if text.startswith('以 Climate 的 H=4 任务为例'):
                revisions[text] = '以Climate H=4为例，目标为美国大陆降水量。语料中包含对已发生降水的回顾性描述，其与历史数值的信息可能重叠；本次未对具体文本逐条实施归因实验，不能断言信息已被数值窗口完全包含。语义候选有22.04%的点估计改善且移动块区间完全为正（表A1），但验证期置换门槛未通过，不能据此部署。'
            if text.startswith('Economy 三项任务则呈现相反情形'):
                revisions[text] = 'Economy三项任务的语义候选相对数值回退变化分别为−51.98%、−59.01%、−108.10%，测试损失差区间均完全小于零（表A1）。宏观摘要的滞后性和主题漂移是可能解释，但未进行逐条事实归因；不能把这些机制写成已经证实的原因。'
            if text.startswith('式（4）以正则化最小二乘'):
                revisions[text] = '式（4）学习相对Last锚点的多步残差，输入包含数值、文本与质量特征，不能将全部拟合贡献归为文本。δ为回归系数，α为正则强度；实现另拟合不受惩罚的截距。'
            if index==2 and text.startswith('本报告把开发过程中'):
                revisions[text] = '本报告保留开发失败与阶段风险，按第10周书面检查节点组织；其中测试期区间汇总为后续复核补记，不声称所有补记均在第10周前完成，也不用于修改冻结门槛。个人课程设计无需PPT、无需课堂汇报。'
            for old,new in revisions.items():
                if replace(p,old,new): log.append([path.name,old,new])
        # Add explicit phase clarification next to method text, using existing body style.
        body=tree.find('w:body',NS)
        # Restore missing cross-references without changing the research.
        for p in body.findall('w:p',NS):
            t=''.join(p.xpath('.//w:t/text()',namespaces=NS))
            if index==1 and t.startswith('数值侧从 Last'):
                replace(p,t,t+'Crossformer[5]与iTransformer[7]提供跨变量建模背景，TimeXer[8]提供外生变量建模背景，均不新增为本项目模型。')
            if index==2 and t.startswith('研究目标保持不变'):
                replace(p,t,t+'数据依据Time-MMD[1]；数值基线参考PatchTST与DLinear[2-3]；周期与变量建模背景见[4-8]；语言表征与融合参考[9-14]，块重采样依据[15]。')
            if index==3 and t.startswith('PatchTST 的分块与通道独立结构'):
                replace(p,t,t+'Crossformer[5]作为跨变量依赖背景，不列入实验模型。')
            if index==3 and t.startswith('课程设计要求形成'):
                replace(p,t,t+'课程要求落实位置见附录表B1。')
        # Repeated header applies to data tables, not the one-cell pseudocode box.
        for table in tree.findall('.//w:tbl',NS):
            rows=table.findall('w:tr',NS)
            if len(rows)>1:
                props=rows[0].find('w:trPr',NS)
                if props is None: props=E.SubElement(rows[0],'{'+NS['w']+'}trPr')
                if props.find('w:tblHeader',NS) is None: E.SubElement(props,'{'+NS['w']+'}tblHeader')
        anchor=next(p for p in body.findall('w:p',NS) if ''.join(p.xpath('.//w:t/text()',namespaces=NS)).startswith('式（1）'))
        newp=E.Element('{'+NS['w']+'}p')
        props=anchor.find('w:pPr',NS)
        if props is not None:
            import copy
            newp.append(copy.deepcopy(props))
        run=E.SubElement(newp,'{'+NS['w']+'}r')
        rp=E.SubElement(run,'{'+NS['w']+'}rPr')
        fonts=E.SubElement(rp,'{'+NS['w']+'}rFonts')
        for k,v in [('ascii','Times New Roman'),('hAnsi','Times New Roman'),('eastAsia','宋体')]: fonts.set('{'+NS['w']+'}'+k,v)
        E.SubElement(run,'{'+NS['w']+'}t').text=POLICY if index!=1 else '最终修订口径说明：'+POLICY+'本段为方案复核补记，不把审查后分析追溯为第4周已预注册内容。'
        anchor.addnext(newp)
        if index==1:
            for p in tree.findall('.//w:p',NS):
                replace(p,'99次冻结置换 加循环移位敏感性','99次主置换；999次仅作审查后敏感性')
        if index==3:
            for row in tree.findall('.//w:tr',NS):
                cells=row.findall('w:tc',NS)
                values=[''.join(c.xpath('.//w:t/text()',namespaces=NS)) for c in cells]
                if len(values)==7 and values[:2]==['Energy','12']:
                    for p in cells[-1].findall('w:p',NS): replace(p,'—','558.9%')
            for p in tree.findall('.//w:p',NS):
                replace(p,'冻结 m*、PCA、α 与 gate','冻结 m*、α 与 gate')
            # The PCA refit stage is part of the actual implementation, not a new model.
            for p in tree.findall('.//w:p',NS):
                if replace(p,'专家选择只根据校准段MSE。','专家选择只根据校准段MSE。资格判定前PCA及残差特征缩放仅拟合训练段；门控与超参数冻结后，残差特征变换在训练加验证段重拟合，测试段始终不参与拟合。'):
                    break
        # Align image accessibility text with the actual caption, without touching pixels.
        paras=body.findall('w:p',NS)
        for i,p in enumerate(paras):
            if i+1<len(paras):
                caption=''.join(paras[i+1].xpath('.//w:t/text()',namespaces=NS))
                if caption.startswith('图 '):
                    for node in p.findall('.//wp:docPr',NS): node.set('descr',caption)
        contents['word/document.xml']=E.tostring(tree,xml_declaration=True,encoding='UTF-8',standalone=True)
        with ZipFile(path,'w',ZIP_DEFLATED) as z:
            for name,data in contents.items(): z.writestr(name,data)
        mirror=ROOT/'paper/final'/path.name
        shutil.copy2(path,mirror)
        roles.append({'role':['week4','week10','final'][index-1], 'docx':path.name,
                      'pdf':path.with_suffix('.pdf').name,'mirror':'paper/final/'+path.name,
                      'mirror_pdf':'paper/final/'+path.with_suffix('.pdf').name})
    config={'revision':'2026-09-19','authority':'root SafeFAME-TS_01/02/03 DOCX; paper/final mirrors',
            'reports':roles,'archive':'SafeFAME-TS_v2_支撑材料.zip','sha256':'SafeFAME-TS_v2_支撑材料.sha256',
            'inventory':'支撑材料清单_v2.txt'}
    (ROOT/'submission.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (audit/'document_edits.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
    # Preserve old deliverables privately, removing ambiguity from paper/final.
    allowed={r[k].split('/')[-1] for r in roles for k in ('docx','pdf')}
    for p in (ROOT/'paper/final').iterdir():
        if (p.suffix in {'.docx','.pdf','.zip'} and p.name not in allowed
                and p.name!='SafeFAME-TS_v2_支撑材料.zip'):
            dest=BACKUP/'paper_final'/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists(): shutil.move(str(p),dest)
    private=ROOT/'references/local_private'
    if (ROOT/'reference_2026.md').exists():
        private.mkdir(parents=True,exist_ok=True)
        shutil.move(str(ROOT/'reference_2026.md'),private/'reference_2026.md')
    print(f'Edited {len(roles)} documents; {len(log)} targeted replacements. Originals archived.')


if __name__=='__main__': main()
