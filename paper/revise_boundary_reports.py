"""Apply audited boundary corrections, preserving the user's report styles and v2 tables."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
import json
from docx import Document
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree as E
from revise_submission import replace, NS

ROOT=Path(__file__).resolve().parents[1]
BACKUP=ROOT/'paper/archive/20260919_before_boundary_fix'
NOTICE=('版本说明：原始v2按预测起点分半，校准与决策目标存在重叠；本报告保留其历史结果。'
        '审查后v3剔除跨界窗口并统一Ridge求解器，18项任务仍全部数值回退。'
        '修订使用已查看过的测试集，是事后方法纠错，不是新增未见测试或事前预注册。')
COMMON=[
 ('严格排除未来信息后','按可得时间代理限制信息后'),
 ('独立路径决策段','按用途划分的决策段'),
 ('该结论由验证段产生，而不是看到测试结果后作出','该次路由由验证段决定，不保证迭代开发中测试始终未见'),
 ('消融、跨领域确认、外部数据','消融、多领域回测、外部数据'),
 ('验证期后半独立用于资格判定','历史v2验证期后半按起点用于资格判定（目标与校准段有重叠）'),
 ('独立决策段','按起点划分的历史决策段'),
 ('v2的独立校准与路径决策双分段门控','v2的校准与路径决策分段门控'),
 ('和独立选择协议下','和明确标注边界的选择协议下'),
 ('校准 决策 测试三重隔离','用途分离；历史校准与决策目标有重叠'),
 ('测试集在选择完成后才计算','单次运行内先选择后评估；不代表跨开发阶段测试始终未见'),
 ('测试期只做一次性评估，不再调整任何规则','单次运行内固定路由再评估；后续纠错沿用已查看测试集'),
 ('冻结后一次评估与区间估计','本次路由确定后的评估与区间诊断'),
 ('避免与预注册式主结果混写','避免与历史v2主结果混写；迭代开发不构成事前预注册'),
 ('严格门槛降低了假阳性，也可能提高假阴性','严格门槛旨在限制错误启用，但当前置换构造不保证假阳性控制，也可能提高假阴性'),
 ('少样本任务另报告决策起点数','少样本任务另报告决策起点数'),
 ('按当前冻结实现','按历史v2冻结实现'),
 ('截至第10周计划节点，课程设计已完成','本报告按第10周计划节点组织，后续复核材料显示已完成'),
 ('截至该节点，六领域数值审计与时点语料已冻结','历史v2记录中，六领域数值审计与时点语料已冻结'),
 ('这不是系统失败，而是选择器按预先标准拒绝缺乏稳健证据的文本路径','这是历史运行按设定条件回退的结果；不构成全局未见测试下的安全保证'),
 ('后半保持未见，用于文本路径资格','后半起点不重合，但多步目标与前半有重叠，不能称为完全未见；修订见第9.4节'),
 ('循环移位保留各数据段内部文本序列的局部结构，只破坏文本与数值起点的同步关系','循环移位保留段内顺序（首尾接缝除外）；历史实现另有求解器差异，修订版才统一求解器'),
 ('保证公式与代码一致','使该定义与实现对应'),
 ('损失汇总单位是预测起点，重叠起点并非独立样本，而非每个H步误差','损失汇总单位是预测起点，重叠起点并非独立样本'),
]

def text(p): return ''.join(p.xpath('.//w:t/text()',namespaces=NS))

def font(p, heading=False):
    for r in p.runs:
        r.font.name='Times New Roman'
        rf=r._element.get_or_add_rPr().rFonts
        rf.set(qn('w:eastAsia'),'黑体' if heading else '宋体')

def insert(anchor,doc,content,style=None):
    p=anchor.insert_paragraph_before(content,style=style)
    if not style:
        p.paragraph_format.first_line_indent=Pt(22)
        p.paragraph_format.space_after=Pt(5)
    font(p, bool(style and style.startswith('Heading')))
    if style=='Caption':
        p.alignment=1
        p.paragraph_format.first_line_indent=Pt(0)
    return p

def table_before(anchor,doc,headers,rows):
    table=doc.add_table(rows=1,cols=len(headers))
    table.style=doc.tables[0].style
    for cell,value in zip(table.rows[0].cells,headers): cell.text=value
    for row in rows:
        for cell,value in zip(table.add_row().cells,row): cell.text=str(value)
    header=OxmlElement('w:tblHeader');table.rows[0]._tr.get_or_add_trPr().append(header)
    for row in table.rows:
        row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        for cell in row.cells:
            for p in cell.paragraphs:
                font(p)
                p.paragraph_format.space_after=Pt(3)
                for r in p.runs:r.font.size=Pt(9)
    # Same three-line table convention as the retained report.
    borders=OxmlElement('w:tblBorders')
    for edge in ('top','bottom','left','right','insideH','insideV'):
        e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single' if edge in ('top','bottom') else 'nil');e.set(qn('w:sz'),'8');borders.append(e)
    table._tbl.tblPr.append(borders)
    for cell in table.rows[0].cells:
        for p in cell.paragraphs:
            for run in p.runs:run.bold=True
        b=OxmlElement('w:tcBorders');e=OxmlElement('w:bottom');e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');b.append(e);cell._tc.get_or_add_tcPr().append(b)
    anchor._p.addprevious(table._tbl)
    return table

def main():
    log=[]
    config=json.loads((ROOT/'submission.json').read_text(encoding='utf-8'))
    for index,record in enumerate(config['reports'],1):
        path=ROOT/record['docx']
        with ZipFile(BACKUP/path.name) as z: contents={n:z.read(n) for n in z.namelist()}
        tree=E.fromstring(contents['word/document.xml'])
        for p in tree.findall('.//w:p',NS):
            for old,new in COMMON:
                if old!=new and replace(p,old,new): log.append([path.name,old,new])
            t=text(p)
            if index==3:
                replacements={
                    '相对于匹配容量纯数值残差对照，只有': '历史v2相对匹配数值对照的区间为2正9负；目标不重叠v3为2正8负。均提示部分候选改善不能直接归因于文本，也不能证明文本普遍有效或无效。',
                    '路径决策段在32/36条候选上不足': '历史v2有32/36条候选不足3个完整非重叠块，v3为34/36；零放行同时受门槛与低功效限制，不能写成文本普遍无效。',
                    '999次循环移位将原校准得到的': '历史999次分析固定校准α，但对齐路径用lsqr、移位对照用cholesky，因此并非只改变文本位置。原结果为0/36达到p≤0.025（见图11）；保留旧划分、统一cholesky的复核仍为0/36。目标不重叠的v3有1/36达到阈值，详见第9.4节，不能把旧版零通过结论泛化到所有修订设置。',
                    '999次循环移位置换仍无候选达到': '历史999次循环移位为0/36；统一求解器、保留旧划分仍为0/36；目标不重叠的v3为1/36。三者均是敏感性诊断，不回写99次门控或证明文本普遍无效。',
                    '第一，Time-MMD 的 end_date': '第一，end_date只是可得时间代理；第二，历史v2校准与决策目标重叠，修订v3已剔除跨界窗口，但同段窗口仍依赖，v3有34/36条候选不足3个完整块；第三，Last锚点与回退专家不同，容量对照也非精确等参数；第四，99次逐行错位不保留时序依赖、p值离散，循环移位也依赖平稳性且有首尾接缝；第五，v3沿用已查看测试集，且边界、求解器和线程一起规范化，变化不能单独归因于边界；第六，轻量编码、领域漂移和硬件误差限制外推，所有结果均非因果证据。',
                    '系统按数据审计、时点语料': '审查发现v2内部验证目标重叠，另行运行目标不重叠的v3：18项仍全回退，语义12/18项点改善、3项区间为正、2项为负；99次检验有2条达阈值但完整门槛未通过，999次敏感性有1条达阈值。v3保存逐起点预测、零分布和模型权重。旧结果与修订结果分列；修订沿用已查看测试集，不是新的独立确认，不能解释为因果关系。',
                }
                for prefix,new in replacements.items():
                    if t.startswith(prefix): replace(p,t,new);log.append([path.name,t,new]);break
                if t.startswith('冻结规则没有在18项任务'):replace(p,'冻结规则没有','历史v2冻结规则没有')
                if t.startswith('结果说明，文本价值'):replace(p,'999次循环移位检验仍无候选','历史v2的999次循环移位检验无候选')
                if t.startswith('在当前六领域18项任务和冻结门槛下'):replace(p,t,'在六领域18项任务中，历史v2与审查后v3均未放行文本路径。该一致性只说明两套已执行规则的路由结果相同，不能替代独立样本上的外部确认。')
        contents['word/document.xml']=E.tostring(tree,xml_declaration=True,encoding='UTF-8',standalone=True)
        images={2:['fig_architecture.png','fig02_text_coverage.png','v2/fig_v2_candidate_gains.png','v2/fig_v2_fallback_experts.png','v2/fig_v2_permutation_gate.png'],
                3:['fig02_text_coverage.png','fig_architecture.png','v2/fig_v2_fallback_experts.png','v2/fig_v2_permutation_gate.png','v2/fig_v2_candidate_gains.png','refig6.png','fig06_ett_external_results.png','fig09_input_robustness.png','fig08_security_distribution_shift.png','../reviewer_sensitivity/fig_v3_matched_control_ci.png','refig11.png']}
        for n,name in enumerate(images.get(index,[]),1):contents[f'word/media/image{n}.png']=(ROOT/'outputs/figures'/name).read_bytes()
        with ZipFile(path,'w',ZIP_DEFLATED) as z:
            for n,b in contents.items():z.writestr(n,b)
        doc=Document(path)
        # Notice on the existing summary page, before keywords or its page break.
        if index==3:
            anchor=next(p for p in doc.paragraphs if p.text=='5 SafeFAME-TS v2 模型')
            insert(anchor,doc,NOTICE)
            ett=next(p for p in doc.paragraphs if p.text=='8 稳健性与误差分析')
            insert(ett,doc,'ETT口径补充：PatchTST-M使用通道独立骨干，当前损失与指标只评价OT通道；固定权重时改变非OT输入不会改变OT输出。因此六项最优与12.69%改善是该OT预测设置的结果，不能解释为验证了跨变量融合增益。')
            anchor=next(p for p in doc.paragraphs if p.text=='10 模型评价与结论')
            insert(anchor,doc,'9.4 审查后目标边界修订与证据复核','Heading 2')
            insert(anchor,doc,'本节为事后纠错，保留第5至8节和附录的历史v2数据。v2按验证起点序列对半切分，18项任务的两段目标共享H−1个时间点；Security H12原决策段全部10个起点均涉及这种重叠。因此“后半独立未见”的原表述不成立。修订保持六领域、18任务、两类候选、五类专家与外层70/10/20划分，不引入新模型。')
            insert(anchor,doc,'以原始时间索引的验证中点c划分：校准起点t必须满足t+H≤c，决策起点必须满足t≥c；丢弃跨界目标窗口。决策段前后两半也按原始时间中点剔除跨界窗口；任一半无样本则拒绝启用。表12与图12对照两版本。v3共减少146个校准或决策起点，保留571/586/2764个校准/决策/测试起点；Economy H12和Security H12的双分段样本不足。')
            insert(anchor,doc,'表 12 历史v2与审查后v3的口径对照','Caption')
            table_before(anchor,doc,['指标（18任务、36路径）','历史v2','审查后v3'],[
                ['校准 / 决策 / 测试起点','648 / 655 / 2764','571 / 586 / 2764'],
                ['完整门槛放行 / 最终回退任务','0 / 18','0 / 18'],
                ['99次检验p≤0.025的路径','0','2'],
                ['语义点改善 / 区间正 / 区间负','13 / 3 / 4','12 / 3 / 2'],
                ['频率点改善 / 区间正 / 区间负','8 / 2 / 3','8 / 3 / 3'],
                ['999次诊断p≤0.025的路径','0（历史混合求解器）','1（统一cholesky）'],
                ['匹配对照区间正 / 负','2 / 9','2 / 8'],
                ['功效诊断不足3块的路径','32 / 36','34 / 36']])
            insert(anchor,doc,'注：区间均为5000次移动块Bootstrap的测试期诊断，不参与选择；999次是敏感性分析，不是99次主门控。保留旧划分而仅统一求解器的中间复核仍为0/36，匹配区间仍为2正9负。数据来源为outputs/revision_comparison/summary.json及两版逐任务CSV。')
            p=insert(anchor,doc,'');p.paragraph_format.first_line_indent=Pt(0)
            picture=p.add_run().add_picture(str(ROOT/'outputs/revision_comparison/boundary_revision.png'),width=Inches(6.1))
            picture._inline.docPr.set('descr','历史v2与修订v3的校准、决策和测试起点数，以及99次、完整门槛和999次敏感性通过路径数对照。')
            p.paragraph_format.keep_with_next=True
            insert(anchor,doc,'图 12 边界修订前后的样本量与资格结果','Caption')
            insert(anchor,doc,'v3的Climate H24与Economy H12语义路径均在99次检验达到p=0.01，但前者未在两半均胜出，后者一半无有效目标窗口，均未满足完整门槛。999次敏感性中仅Climate H24语义达到p=0.009，不回写路由。回退专家改变3项：Climate H24由DLinear-M变为AR-Ridge，Economy H6由AR-Ridge变为PatchTST，Security H3由PatchTST变为AR-Ridge。完整门槛仍为0/36，不能将其解释成“所有单项检验均不通过”。')
            insert(anchor,doc,'复现修订统一对齐和错位Ridge为cholesky，将新运行的BLAS与PyTorch CPU线程设为1，MiniLM固定到历史缓存版本1110a243fdf4706b3f48f1d95db1a4f5529b4d41。历史v2的lsqr在不同线程下存在数值差异，不能只固定随机种子就保证逐位一致。v3逐任务保存先于测试评估的选择记录、所有专家校准分数、逐起点预测、99次零分布、早停日志和模型权重；两套修订敏感性保存999次零分布与预测。')
            insert(anchor,doc,'已从保存预测复算72行v3指标、72条修订敏感性路径、全部相应5000次区间，并回放选中深度专家权重，核验通过。该过程不是全部ETT重训。新结果沿用旧测试数据；边界、求解器与线程同时规范化，不将变化单独归因于其中一项。逐行置换的可交换性和循环移位的平稳性未获证明，经验p值不能当成有保证的因果或错误率证据。')
        else:
            anchor=next(p for p in doc.paragraphs if p.text=='参考文献')
            insert(anchor,doc,'审查后修订补记','Heading 2')
            insert(anchor,doc,NOTICE+' 本补记不虚构第4周或第10周已经完成这些重跑。')
            insert(anchor,doc,'修订规则以目标时间而非仅预测起点隔离校准和决策，任一决策半段无样本即拒绝启用；99次逐行错位属于主门控，999次循环移位和5000次移动块Bootstrap只作诊断。新运行共571个校准、586个决策和2764个测试起点；99次有2条达阈值但完整门槛均未通过，999次有1条达阈值。语义点改善12/18、区间3正2负，频率点改善8/18。此前13/18、3正4负及999次0通过均仅指历史v2。')
            insert(anchor,doc,'已统一修订分析求解器，固定编码器版本和线程，保存预测、零分布、选择记录、日志与权重。原始数据和历史结果不覆盖，最终报告第9.4节及docs/20记录差异。低统计功效、漂移、Last锚点不对称与发布时间代理仍是限制；匹配对照仅约束模型家族和调参范围，不构成因果识别或严格等参数实验。')
        doc.core_properties.comments='2026-09-19 boundary audit revision; original v2 results retained'
        doc.save(path)
    (ROOT/'tmp/revision_20260919/boundary_document_edits.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Updated three root reports; Word export and visual review still required')

if __name__=='__main__':main()
