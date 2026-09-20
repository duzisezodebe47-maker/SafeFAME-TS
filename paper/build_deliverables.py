"""Build the Week 4, Week 10 and final SafeFAME-TS course-design documents."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "final"
FIG = ROOT / "outputs" / "figures"
V2_FIG = FIG / "v2"
NAVY = "234E70"
PALE = "EFF5F8"
GRID = "D9D9D9"


REFERENCES = [
    "[1] Liu H, Xu S, Zhao Z, et al. Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis. Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2024. DOI: 10.52202/079017-2476.",
    "[2] Nie Y, Nguyen N H, Sinthong P, Kalagnanam J. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. International Conference on Learning Representations, 2023.",
    "[3] Zeng A, Chen M, Zhang L, Xu Q. Are Transformers Effective for Time Series Forecasting? Proceedings of the AAAI Conference on Artificial Intelligence, 2023, 37(9):11121-11128. DOI: 10.1609/aaai.v37i9.26317.",
    "[4] Wu H, Hu T, Liu Y, et al. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. International Conference on Learning Representations, 2023.",
    "[5] Zhang Y, Yan J. Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting. International Conference on Learning Representations, 2023.",
    "[6] Zhou T, Ma Z, Wen Q, et al. FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting. Proceedings of the 39th International Conference on Machine Learning, PMLR 162:27268-27286, 2022.",
    "[7] Liu Y, Hu T, Zhang H, et al. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. International Conference on Learning Representations, 2024.",
    "[8] Wang Y, Wu H, Dong J, et al. TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables. Advances in Neural Information Processing Systems, 2024. DOI: 10.52202/079017-0015.",
    "[9] Jin M, Wang S, Ma L, et al. Time-LLM: Time Series Forecasting by Reprogramming Large Language Models. International Conference on Learning Representations, 2024.",
    "[10] Reimers N, Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP, 2019:3982-3992. DOI: 10.18653/v1/D19-1410.",
    "[11] Wang W, Wei F, Dong L, et al. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. Advances in Neural Information Processing Systems, 2020.",
    "[12] Liu C, Xu Q, Miao H, et al. TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment. Proceedings of the AAAI Conference on Artificial Intelligence, 2025, 39(18):18780-18788. DOI: 10.1609/aaai.v39i18.34067.",
    "[13] Chowdhury A M, Akter R, Arib S H. T3Time: Tri-Modal Time Series Forecasting via Adaptive Multi-Head Alignment and Residual Fusion. Proceedings of the AAAI Conference on Artificial Intelligence, 2026, 40(25):20597-20605. DOI: 10.1609/aaai.v40i25.39196.",
    "[14] Spectral Text Fusion for Multimodal Time Series Forecasting. arXiv:2602.01588, 2026. 预印本.",
    "[15] Künsch H R. The Jackknife and the Bootstrap for General Stationary Observations. The Annals of Statistics, 1989, 17(3):1217-1241. DOI: 10.1214/aos/1176347265.",
]


def set_repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    node = OxmlElement("w:tblHeader")
    node.set(qn("w:val"), "true")
    tr_pr.append(node)


def set_cell(cell, fill: str | None = None, margin: int = 100) -> None:
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    tc_pr = cell._tc.get_or_add_tcPr()
    if fill:
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), fill)
        tc_pr.append(shd)
    tc_mar = OxmlElement("w:tcMar")
    for name in ("top", "start", "bottom", "end"):
        n = OxmlElement(f"w:{name}")
        n.set(qn("w:w"), str(margin))
        n.set(qn("w:type"), "dxa")
        tc_mar.append(n)
    tc_pr.append(tc_mar)


def set_table_borders(table) -> None:
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "4")
        node.set(qn("w:color"), GRID)
        borders.append(node)
    table._tbl.tblPr.append(borders)


def configure(doc: Document, short_title: str) -> None:
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin, section.bottom_margin = Cm(2.25), Cm(1.8)
    section.left_margin, section.right_margin = Cm(2.45), Cm(2.25)
    section.different_first_page_header_footer = True
    normal = doc.styles["Normal"]
    normal.font.name = "宋体"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    for name, size, font in (("Title", 22, "黑体"), ("Heading 1", 16, "黑体"), ("Heading 2", 13, "黑体"), ("Heading 3", 11, "黑体")):
        style = doc.styles[name]
        style.font.name = font
        style._element.rPr.rFonts.set(qn("w:eastAsia"), font)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
    title_ppr = doc.styles["Title"]._element.get_or_add_pPr()
    title_border = title_ppr.find(qn("w:pBdr"))
    if title_border is not None:
        title_ppr.remove(title_border)
    doc.styles["Caption"].font.name = "宋体"
    doc.styles["Caption"]._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    doc.styles["Caption"].font.size = Pt(9)
    doc.styles["Caption"].font.color.rgb = RGBColor(0, 0, 0)
    header = section.header.paragraphs[0]
    header.text = short_title
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.runs[0].font.size = Pt(8)
    header.runs[0].font.color.rgb = RGBColor(96, 96, 96)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("第 ")
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    footer._p.append(fld)
    footer.add_run(" 页")


def apply_font_policy(doc: Document) -> None:
    """Use 宋体/黑体 for Chinese and Times New Roman for Latin glyphs."""
    heading_names = {"Title", "Heading 1", "Heading 2", "Heading 3"}

    def set_rfonts(run, east: str) -> None:
        run.font.name = "Times New Roman"
        rpr = run._element.get_or_add_rPr()
        rfonts = rpr.get_or_add_rFonts()
        for key in ("ascii", "hAnsi", "cs"):
            rfonts.set(qn(f"w:{key}"), "Times New Roman")
        rfonts.set(qn("w:eastAsia"), east)

    for style in doc.styles:
        if not hasattr(style, "font"):
            continue
        east = "黑体" if style.name in heading_names else "宋体"
        style.font.name = "Times New Roman"
        rpr = style._element.get_or_add_rPr()
        rfonts = rpr.get_or_add_rFonts()
        for key in ("ascii", "hAnsi", "cs"):
            rfonts.set(qn(f"w:{key}"), "Times New Roman")
        rfonts.set(qn("w:eastAsia"), east)

    def apply_paragraph(paragraph) -> None:
        style_name = paragraph.style.name if paragraph.style else "Normal"
        east = "黑体" if style_name in heading_names else "宋体"
        for run in paragraph.runs:
            set_rfonts(run, east)

    for paragraph in doc.paragraphs:
        apply_paragraph(paragraph)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    apply_paragraph(paragraph)
    for section in doc.sections:
        for paragraph in section.header.paragraphs + section.footer.paragraphs:
            apply_paragraph(paragraph)


def cover(doc: Document, title: str, subtitle: str, stage: str) -> None:
    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(78)
    p.paragraph_format.space_after = Pt(22)
    p.add_run(title)
    direct_border = p._p.get_or_add_pPr().find(qn("w:pBdr"))
    if direct_border is not None:
        p._p.get_or_add_pPr().remove(direct_border)
    p_borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "nil")
    p_borders.append(bottom)
    p._p.get_or_add_pPr().append(p_borders)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(subtitle)
    r.bold = True
    r.font.name = "黑体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    r.font.size = Pt(15)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    p.add_run(stage).font.size = Pt(12)
    doc.add_paragraph().paragraph_format.space_after = Pt(92)
    for label in ("课程名称  人工智能", "作业性质  个人课程设计", "姓名学号  ____________________", "提交日期  ____________________"):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(8)
        p.add_run(label).font.size = Pt(11)
    doc.add_page_break()


def heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_heading(text, level=level)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(10 if level == 1 else 7)
    p.paragraph_format.space_after = Pt(5)


def body(doc: Document, text: str, indent: bool = True) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(21) if indent else Pt(0)
    p.paragraph_format.line_spacing = 1.45
    p.paragraph_format.space_after = Pt(5)
    p.add_run(text)


def bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.left_indent = Cm(0.75)
        p.paragraph_format.line_spacing = 1.35
        p.paragraph_format.space_after = Pt(3)
        p.add_run(item)


def equation(doc: Document, text: str, number: int) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(5)
    # These strings are the readable Word rendering of the LaTeX definitions
    # used in the source specification; source syntax is intentionally hidden.
    display_map = {
        1: "x̃ₜ = (xₜ − μ_train) / σ_train",
        2: "m* = arg minₘ MSE_calibration(m)",
        3: "zₜ = Normalize(Σᵢwᵢeᵢ / Σᵢwᵢ)，wᵢ = exp(−ln2·ageᵢ / τ)",
        4: "δ̂ = arg minδ {Σₜ‖Yₜ − Last(Xₜ) − Φ(Xₜ,Zₜ,qₜ)δ‖₂² + α‖δ‖₂²}",
        5: "p_perm = [1 + #(MSE_perm ≤ MSE_aligned)] / (99 + 1)",
        6: "ŷ = r_best（总体胜出且两段均胜出且 p_perm ≤ 0.025），否则 ŷ = b",
        7: "MSE = (1 / NH) ΣₙΣₕ (yₙ,ₕ − ŷₙ,ₕ)²",
    }
    display = display_map.get(number, text)
    r = p.add_run(f"{display}    （{number}）")
    r.font.name = "Cambria Math"
    r._element.rPr.rFonts.set(qn("w:ascii"), "Cambria Math")
    r._element.rPr.rFonts.set(qn("w:hAnsi"), "Cambria Math")
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    r.font.size = Pt(11)


def caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph(style="Caption")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(5)
    p.add_run(text)


def figure(doc: Document, path: Path, text: str, width_cm: float = 15.5) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    shape = p.add_run().add_picture(str(path), width=Cm(width_cm))
    shape._inline.docPr.set("descr", text)
    caption(doc, text)


def table(doc: Document, title: str, headers: list[str], rows: list[list[object]], widths: list[float] | None = None, size: float = 8.5) -> None:
    caption(doc, title)
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    set_table_borders(t)
    set_repeat_header(t.rows[0])
    for i, value in enumerate(headers):
        c = t.rows[0].cells[i]
        set_cell(c, NAVY)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(value)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(size)
    for ridx, values in enumerate(rows):
        row = t.add_row()
        cant_split = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(cant_split)
        for i, value in enumerate(values):
            c = row.cells[i]
            set_cell(c, PALE if ridx % 2 else None)
            p = c.paragraphs[0]
            is_text = isinstance(value, str) and len(value) > 12
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if is_text else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.15
            r = p.add_run(str(value))
            r.font.size = Pt(size)
    if widths:
        for row in t.rows:
            for i, value in enumerate(widths):
                row.cells[i].width = Cm(value)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def toc(doc: Document, items: list[str]) -> None:
    heading(doc, "目录", 1)
    for item in items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(3)
    doc.add_page_break()


def remove_last_paragraph(doc: Document) -> None:
    paragraph = doc.paragraphs[-1]
    paragraph._element.getparent().remove(paragraph._element)


def add_references(doc: Document, full: bool = True) -> None:
    heading(doc, "参考文献", 1)
    refs = REFERENCES if full else REFERENCES[:12]
    for ref in refs:
        p = doc.add_paragraph(ref)
        p.paragraph_format.hanging_indent = Cm(0.65)
        p.paragraph_format.line_spacing = 1.2
        p.paragraph_format.space_after = Pt(4)


def add_literature_table(doc: Document, caption: str = "表 1 相关研究 证据等级与本文取舍") -> None:
    rows = [
        ["Time-MMD", "NeurIPS 2024", "多领域数值与文本对齐数据", "提供数据基础；不照抄其结论"],
        ["DLinear", "AAAI 2023", "分解线性预测与强简单基线", "作为可解释深度基线"],
        ["PatchTST", "ICLR 2023", "分块与通道独立", "作为主要数值专家"],
        ["TimesNet / FEDformer", "ICLR 2023 / ICML 2022", "多周期与频域建模", "启发频率统计分支"],
        ["TimeXer", "NeurIPS 2024", "外生变量建模", "支持将文本视为外生信息"],
        ["Time-LLM / TimeCMA", "ICLR 2024 / AAAI 2025", "语言模型重编程与跨模态对齐", "说明融合方法空间；本文采用轻量实现"],
        ["T3Time", "AAAI 2026", "时域 频域 提示三分支与跨度条件门控", "借鉴多模态和跨度条件思想"],
        ["Spectral Text Fusion", "2026 预印本", "频谱与文本交互", "启发候选频率交互；须独立验证"],
    ]
    table(doc, caption, ["研究", "身份", "主要启发", "本文使用边界"], rows, [3.1, 3.2, 4.3, 6.0], 8.0)


def data_rows() -> list[list[object]]:
    audit = pd.read_csv(ROOT / "outputs" / "audit" / "timemmd_numerical_audit.csv")
    keep = audit[audit.domain.isin(["Climate", "Energy", "Economy", "Traffic", "Agriculture", "Security"])]
    return [[r.domain, int(r.rows), int(r.numeric_columns), str(r.start)[:7], str(r.end)[:7], int(r.rows_moved_by_stable_sort), int(r.ot_missing)] for r in keep.itertuples()]


def build_week4() -> Path:
    doc = Document()
    configure(doc, "人工智能课程设计  第4周选题与初步技术方案")
    cover(doc, "文本增强多领域时间序列预测", "严格时点对齐 反事实检验与安全回退", "第4周课程设计选题与初步技术方案")
    heading(doc, "摘要", 1)
    body(doc, "本课程设计拟研究外部文本在多领域时间序列预测中的真实增益。核心问题不是简单增加文本输入，而是在严格排除未来信息后，判断文本语义何时能稳定改善预测，以及证据不足时如何退回可靠的纯数值模型。项目计划以 Time-MMD 为主数据，在气候、能源、经济、交通、农业和安全六个领域构建18项领域与预测跨度组合，并以 ETTh1、ETTh2 作为纯数值外部基准。候选系统由数值专家池、冻结句向量、语义残差、频率交互、文本置换检验和验证期安全选择组成。最终交付包括源代码、逐步数据审计、实验表图、书面报告和交互式原型。")
    body(doc, "本方案的可检验目标是：若文本候选不能在预先保留的验证期同时超过数值回退、通过分段稳定性和文本置换门槛，则系统不启用文本。该设计允许最终结论为否定结果，从而避免只报告成功案例。")
    p = doc.add_paragraph()
    p.add_run("关键词  ").bold = True
    p.add_run("多模态时间序列  时点对齐  反事实检验  安全融合  负迁移")
    doc.add_page_break()
    toc(doc, ["1 选题背景与问题定义", "2 文献调研与方案取舍", "3 数据与任务设计", "4 初步算法方案", "5 实验与验证计划", "6 进度安排 风险与交付", "参考文献"])

    heading(doc, "1 选题背景与问题定义", 1)
    heading(doc, "1.1 研究对象", 2)
    body(doc, "现实预测通常同时面对历史数值序列与新闻、报告、政策或事件文本。文本可能解释数值突变，但也可能晚于预测时点、偏离主题或重复既有趋势。Time-MMD 提供多领域成对数据，为这一问题提供实验基础[1]。本项目将文本视为受信息可得性约束的外生变量，而不是默认有效的额外模态。")
    heading(doc, "1.2 拟解决的具体问题", 2)
    bullets(doc, [
        "在文本结束时间早于预测起点的约束下，语义特征能否超过纯数值基线。",
        "正确对齐文本能否优于随机置换文本，以排除模板结构、文本数量和时间趋势造成的伪增益。",
        "频域统计与文本语义交互是否在较长预测跨度更有效。",
        "当文本质量或稳定性不足时，能否用冻结规则自动回退到数值专家。",
        "数值骨干在独立 ETT 数据与输入扰动下是否保持可复现性能。",
    ])
    heading(doc, "1.3 输入 输出与边界", 2)
    body(doc, "输入为长度 L 的历史数值窗口 X 和预测时点前可获得的事实集合 Z；输出为未来 H 步目标序列。数据中的 end_date 仅作为可得时间代理，不能等同于真实发布时间；包含未来预测内容的 preds 字段不进入模型。")
    equation(doc, "ŷₜ₊₁:ₜ₊ᴴ = fθ(Xₜ₋ᴸ₊₁:ₜ , Z<t)", 1)

    doc.add_page_break()
    heading(doc, "2 文献调研与方案取舍", 1)
    add_literature_table(doc)
    body(doc, "数值侧从 Last、季节朴素和 AR-Ridge 起步，再加入 DLinear 与 PatchTST[2-3]。频率分支受 TimesNet 与 FEDformer 的多周期和频域思想启发[4,6]，但只构造低维频率统计和交互，避免复制大型结构。文本侧使用冻结 MiniLM 句向量[10-11]，以受正则化残差模型检验是否存在增量信息。Time-LLM、AAAI 2025 的 TimeCMA 与 AAAI 2026 的 T3Time 提供语言重编程、跨模态对齐和三分支融合依据[9,12-13]；Spectral Text Fusion 仍为预印本[14]，仅作为频谱交互启发。")

    heading(doc, "3 数据与任务设计", 1)
    table(doc, "表 2 Time-MMD 六领域数据审计结果", ["领域", "行数", "数值列", "起始", "结束", "重排数", "OT缺失"], data_rows(), [2.6, 1.6, 1.8, 2.4, 2.4, 1.8, 1.8], 8.2)
    body(doc, "气候文件原为倒序，经济文件顺序混乱，建模前必须按解析后的时间稳定升序排序。六领域目标 OT 均无缺失；Security 仅有一个数值变量且样本最少，适合作为分布漂移和少样本压力测试。文本仅使用 fact，按同来源和规范化文本去重，最多保留每来源32条最近事实。")
    table(doc, "表 3 任务与评价口径", ["数据", "任务数", "划分", "主要指标", "用途"], [
        ["Time-MMD 六领域", 18, "70%训练 10%验证 20%测试", "标准化 MSE MAE RMSE", "主实验"],
        ["ETTh1 ETTh2", 6, "固定时间划分", "标准化 MSE 与三种子均值", "外部数值验证"],
        ["输入扰动", 24, "高斯 缺失 尖峰 尾段缺失", "相对退化率", "稳健性"],
    ], [3.4, 1.8, 4.8, 3.5, 3.8], 8.2)

    heading(doc, "4 初步算法方案", 1)
    heading(doc, "4.1 数值专家池", 2)
    body(doc, "候选专家包括 Last、SeasonalNaive、AR-Ridge、DLinear-M 和 PatchTST。简单模型提供可解释下限，深度模型固定2026、2027、2028三个随机种子。数值专家仅在验证期前半段比较，测试集不参与选择。")
    heading(doc, "4.2 严格时点语义表征", 2)
    body(doc, "对每个预测起点，仅索引历史窗口内且 end_date 早于预测起点的 fact。每条文本经冻结 all-MiniLM-L6-v2 编码；同来源向量按时间衰减加权平均并做 L2 归一化。历史窗口长度的四分之一作为半衰期。")
    equation(doc, "zₜ = Normalize(Σᵢ exp[-ln2·ageᵢ/τ] eᵢ / Σᵢ exp[-ln2·ageᵢ/τ])", 2)
    heading(doc, "4.3 残差候选与安全选择", 2)
    body(doc, "语义候选以 Last 预测为锚点，使用 Ridge 学习由数值窗口、文本语义和质量特征共同解释的残差；频率候选再加入历史窗口频谱统计与语义交互。由于候选锚点与最终数值回退可能不同，候选相对回退的改善不能单独归因于文本，必须结合匹配容量纯数值残差对照和文本错位置换解释。验证期再对半：前半选择正则系数和数值专家，后半只作路径资格判定。候选必须同时满足后半总体优于回退、前后两段均获胜、99次文本置换检验经 Bonferroni 校正后 p≤0.025。")
    equation(doc, "δ̂ = arg minδ ||y - Last(x) - Φ(x,z)δ||²₂ + α||δ||²₂", 3)
    equation(doc, "route = text  iff  MSEtext<MSEfallback ∧ two-segment wins ∧ pperm≤0.025", 4)

    heading(doc, "5 实验与验证计划", 1)
    table(doc, "表 4 实验矩阵与成功标准", ["实验", "对照", "验证方式", "预先判据"], [
        ["数值基线", "5类专家", "时间顺序留出 三随机种子", "报告全部模型 不只报告赢家"],
        ["语义增量", "语义候选 对 数值回退", "独立路径决策段", "总体和两分段均更优"],
        ["文本反事实", "对齐文本 对 错位文本", "99次冻结置换 加循环移位敏感性", "p≤0.025"],
        ["测试不确定性", "逐预测起点配对损失", "5000次移动块Bootstrap", "报告95%区间"],
        ["消融", "无频率 对 频率交互", "同一划分同一回退", "不以单任务胜利概括总体"],
        ["稳健性", "干净输入 对 四类扰动", "三次重复", "报告均值和标准差"],
    ], [3.1, 4.1, 4.4, 5.5], 8.0)
    body(doc, "由于滑动预测窗口相互重叠，普通独立同分布Bootstrap会低估不确定性，因此使用移动块Bootstrap保留局部依赖[15]。评价值均在训练期均值与标准差定义的标准化空间计算；不同领域的绝对 MSE 不直接横向比较。")

    heading(doc, "6 进度安排 风险与交付", 1)
    table(doc, "表 5 第4周以后实施计划", ["周次", "主要工作", "验收证据"], [
        ["第4至6周", "数据审计 时点索引 简单基线", "审计表 自检日志 基线结果"],
        ["第7至9周", "DLinear PatchTST 文本编码与开发实验", "三种子模型 训练日志 开发图表"],
        ["第10周", "书面中期检查 冻结最终协议", "中期报告 协议JSON 风险清单"],
        ["第11至14周", "六领域确认实验 ETT外部验证 稳健性", "测试结果 区间估计 消融表"],
        ["第15至17周", "报告 原型 支撑材料 审稿检查", "DOCX PDF 代码 数据清单 校验报告"],
    ], [2.5, 8.0, 6.6], 8.3)
    body(doc, "主要风险包括文本真实发布时间缺失、搜索文本主题漂移、验证样本较少、不同领域尺度不一致和深度模型随机性。对应措施是把 end_date 明确表述为代理变量、保留文本置换与缺失指示、按领域标准化、固定三随机种子并报告限制。少样本任务另报告决策起点数、非重叠块数和近似最小可检测效应。若文本未通过门槛，否定结果仍作为完整结论，不临时放宽标准。")
    heading(doc, "最终交付文件", 2)
    bullets(doc, ["第4周选题与初步技术方案；", "第10周书面中期进展报告；", "最终课程设计报告 DOCX 与 PDF；", "完整 Python 源代码、依赖与统一运行说明；", "原始数据来源和固定提交信息、处理后审计表、实验明细与图表；", "交互式结果原型、文件哈希与最终检查报告。"])
    add_references(doc, full=True)
    path = OUT / "第4周_课程设计选题与初步技术方案.docx"
    doc.core_properties.title = "文本增强多领域时间序列预测 第4周课程设计选题与初步技术方案"
    doc.core_properties.author = ""
    apply_font_policy(doc)
    doc.save(path)
    return path


def build_week10() -> Path:
    claims = json.loads((ROOT / "outputs" / "tables" / "v2" / "verified_claims.json").read_text(encoding="utf-8"))
    doc = Document()
    configure(doc, "人工智能课程设计  第10周书面中期进展")
    cover(doc, "文本增强多领域时间序列预测", "数据审计 基线复现与严格融合协议", "第10周课程设计书面中期进展报告")
    heading(doc, "中期结论摘要", 1)
    body(doc, "截至第10周计划节点，课程设计已完成数据审计、时间顺序修复、严格时点文本索引、五类数值专家、冻结文本表征、语义与频率残差候选、反事实置换检验和外部数值基准。主实验覆盖六个 Time-MMD 领域与三个预测跨度，共18项任务；测试期共有2764个预测起点。中期检查发现，早期直接拼接或宽松门控容易产生负迁移，因此已将最终协议修订为验证期双分段选择与证据不足回退。")
    body(doc, "本报告把开发过程中出现的失败结果纳入方法修订依据，不使用测试集调整最终资格门槛。第10周以后将按冻结协议完成一次性汇总、移动块Bootstrap区间、输入扰动、论文与支撑材料。教师已说明个人完成且不再进行PPT展示，因此本节点以书面进展报告替代演示文稿。")
    doc.add_page_break()
    toc(doc, ["1 课题与技术路线", "2 已完成的数据工作", "3 已完成的模型与实现", "4 阶段实验发现", "5 协议修订与冻结", "6 后续计划与风险", "参考文献"])

    heading(doc, "1 课题与技术路线", 1)
    body(doc, "研究目标保持不变：检验外部事实文本在严格时点约束下是否为多领域时间序列提供可泛化的增量信息，并在证据不足时自动退回验证期选定的数值预测器。项目规模包括六领域18项多模态任务、ETTh1与ETTh2六项长跨度数值任务、四类输入扰动以及完整原型。")
    figure(doc, FIG / "fig01_technical_route.png", "图 1 项目总体技术路线", 15.4)
    heading(doc, "阶段完成度", 2)
    table(doc, "表 1 第10周任务完成情况", ["模块", "状态", "形成文件", "主要检查"], [
        ["数据审计", "完成", "审计CSV JSON", "日期顺序 缺失 重复 单位"],
        ["时点文本索引", "完成", "语料与样本索引", "fact可得时间严格早于起点"],
        ["数值基线", "完成", "预测明细与训练日志", "简单基线 深度模型 三种子"],
        ["文本候选", "完成", "语义缓存与置换结果", "对齐文本不使用preds"],
        ["最终选择协议", "已冻结", "protocol.json", "校准 决策 测试三重隔离"],
        ["报告与原型", "进行中", "图表与网页原型", "一致性 可复现 无PPT"],
    ], [3.0, 2.3, 5.2, 6.5], 8.2)

    doc.add_page_break()
    heading(doc, "2 已完成的数据工作", 1)
    table(doc, "表 2 六领域数值数据审计", ["领域", "行数", "数值列", "起始", "结束", "重排数", "OT缺失"], data_rows(), [2.6, 1.6, 1.8, 2.4, 2.4, 1.8, 1.8], 8.2)
    body(doc, "原始文件未被覆盖。Climate 的1272行全部因倒序而重排，Economy 的447行全部改变原行位置；如果直接按CSV顺序划分，会把未来样本置于训练段。所有模型统一调用时间排序函数。Health_AFR、Health_US、Environment 和 SocialGood 在数据审计中保留，但因目标缺失、重复日期、日频算力或文本质量问题未进入主实验。")
    figure(doc, FIG / "fig02_text_coverage.png", "图 2 严格时点约束下的文本覆盖", 15.4)
    body(doc, "去重后语料共14281条事实。每个样本最多选择报告和搜索各32条，使用预测起点前的事实结束时间构造索引。end_date 只是 Time-MMD 提供的区间终点，因此本文只把它称为可得时间代理，不推断其等于真实发布日期。")

    heading(doc, "3 已完成的模型与实现", 1)
    heading(doc, "3.1 数值专家", 2)
    body(doc, "数值专家池包括 Last、SeasonalNaive、AR-Ridge、DLinear-M 和 PatchTST。DLinear 与 PatchTST 采用2026、2027、2028三个随机种子；早停轮数由校准段确定，冻结后在训练加验证数据上按同轮数重训。")
    heading(doc, "3.2 文本候选", 2)
    body(doc, "文本由冻结 all-MiniLM-L6-v2 编码，报告与搜索分别以指数时间衰减聚合。两路各384维向量拼接为768维原始语义输入，再仅用训练期数据拟合PCA。语义残差候选使用数值窗口、降维语义向量与覆盖质量特征；频率残差候选额外加入频率统计及其与语义主成分的乘积。两个候选均以 Last 为残差锚点，并用 Ridge 控制容量。由于最终比较对象可能是其他数值专家，后续增加匹配容量纯数值残差对照，避免把结构差异误写成文本贡献。")
    equation(doc, "L(δ)=Σ||y−Last(x)−Φ(x,z)δ||²₂+α||δ||²₂", 1)
    heading(doc, "3.3 工程复现", 2)
    bullets(doc, ["数据、代码、输出、图表、报告分目录保存；", "所有随机过程固定种子；", "每个模型保存逐预测起点结果，而不只保存平均指标；", "核心脚本包含最小自检；", "测试集在选择完成后才计算；", "所有正式图表由结果CSV自动生成。"])

    heading(doc, "4 阶段实验发现", 1)
    body(doc, "开发性实验表明，文本增益对领域和跨度高度敏感。早期直接融合路径在若干任务上优于数值基线，却在经济和安全任务产生明显负迁移；频率全分支也没有稳定胜过无频率版本。这一结果否定了“加入文本或频率模块必然更好”的初始直觉。")
    figure(doc, V2_FIG / "fig_v2_candidate_gains.png", "图 3 冻结候选的跨任务点估计收益与负迁移", 16.0)
    body(doc, f"按当前冻结实现，语义候选在{claims['semantic_point_improvement_tasks']}/18项任务上取得更低测试MSE，但相对于数值回退的移动块Bootstrap只有{claims['semantic_significant_improvement_tasks']}项95%区间完全大于零，同时有{claims['semantic_significant_harm_tasks']}项区间完全小于零。这里比较的是完整候选路径与回退专家，不等同于纯文本效应；该测试期汇总只用于说明风险，不用于返回修改门槛。")
    figure(doc, V2_FIG / "fig_v2_fallback_experts.png", "图 4 各类数值专家在验证期校准中的入选次数", 13.7)
    body(doc, "PatchTST 在8项任务成为回退专家，但 Last、季节朴素、AR-Ridge 和 DLinear-M 仍分别在不同领域胜出，说明不能按模型知名度预设统一骨干。")

    heading(doc, "5 协议修订与冻结", 1)
    body(doc, "针对开发阶段暴露的数据泄漏和选择偏差风险，最终协议做出四项修订：第一，验证期一分为二，前半校准、后半资格判定；第二，数值回退只能由校准段选择；第三，文本资格同时要求总体胜出、决策段前后两半均胜出和置换检验 p≤0.025；第四，测试不确定性使用5000次移动块Bootstrap，而非把重叠窗口视作独立样本。")
    table(doc, "表 3 冻结后的数据隔离规则", ["数据段", "比例", "允许用途", "禁止用途"], [
        ["训练", "70%", "拟合模型与PCA", "查看未来验证和测试目标"],
        ["校准", "约5%", "选数值专家 正则系数 早停轮数", "判定文本资格"],
        ["路径决策", "约5%", "资格门槛与置换检验", "再次调参"],
        ["测试", "20%", "冻结后一次评估与区间估计", "模型或门槛选择"],
    ], [3.1, 2.0, 6.2, 5.7], 8.2)
    figure(doc, V2_FIG / "fig_v2_permutation_gate.png", "图 5 文本路径启用前的置换检验门槛", 15.6)
    body(doc, f"36条候选路径中，{claims['segment_stability_passes']}条通过分段稳定性，但没有一条通过 Bonferroni 后的 p≤0.025 门槛。因此若按冻结协议执行，当前18项任务均应回退；这不是系统失败，而是选择器按预先标准拒绝缺乏稳健证据的文本路径。")

    heading(doc, "6 后续计划与风险", 1)
    table(doc, "表 4 第10周以后工作安排", ["工作", "输出", "完成判据"], [
        ["冻结协议一次性汇总", "18任务结果与资格审计", "结果与protocol.json一致"],
        ["移动块不确定性", "配对损失95%区间", "逐任务5000次重采样"],
        ["ETT外部与扰动", "6项数值任务 4类扰动", "三随机种子和重复实验"],
        ["报告与支撑材料", "DOCX PDF 代码 数据清单", "数字 图表 代码一致"],
        ["原型与最终审稿", "交互网站 检查报告 哈希", "无夸大 无缺页 可复算"],
    ], [4.0, 6.0, 7.0], 8.2)
    body(doc, "剩余最大风险是验证期样本较少导致路径选择功效不足，以及 end_date 不能证明真实可得时间。最终报告将补充决策起点数、非重叠块数、近似最小可检测效应、循环移位置换和匹配容量纯数值残差对照，并把“未放行文本”解释为当前数据与门槛下的结论，而非证明文本永远无效。")
    add_references(doc, full=True)
    path = OUT / "第10周_课程设计书面中期进展报告.docx"
    doc.core_properties.title = "文本增强多领域时间序列预测 第10周书面中期进展报告"
    doc.core_properties.author = ""
    apply_font_policy(doc)
    doc.save(path)
    return path


def build_final() -> Path:
    claims = json.loads((ROOT / "outputs" / "tables" / "v2" / "verified_claims.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(ROOT / "outputs" / "safefame_v2" / "safefame_v2_metrics.csv")
    audit = pd.read_csv(ROOT / "outputs" / "safefame_v2" / "safefame_v2_selection_audit.csv")
    ett = pd.read_csv(ROOT / "outputs" / "tables" / "final" / "table_ett_external.csv")
    old_claims = json.loads((ROOT / "outputs" / "tables" / "final" / "verified_claims.json").read_text(encoding="utf-8"))
    sensitivity = pd.read_csv(ROOT / "outputs" / "reviewer_sensitivity" / "reviewer_sensitivity_results.csv")
    sensitivity_summary = json.loads((ROOT / "outputs" / "reviewer_sensitivity" / "reviewer_sensitivity_summary.json").read_text(encoding="utf-8"))
    doc = Document()
    configure(doc, "人工智能课程设计  SafeFAME TS")
    cover(doc, "文本何时有助于时间序列预测", "多领域严格时点反事实检验与安全回退", "SafeFAME TS 人工智能课程设计报告")
    heading(doc, "摘要", 1)
    body(doc, "外部文本可能解释时间序列变化，也可能因发布时间不明、主题漂移或样本量有限而造成负迁移。本文以 Time-MMD 为主数据，研究在严格时点约束下文本是否能为多领域预测提供可泛化的增量信息。研究覆盖气候、能源、经济、交通、农业和安全六个领域，构造18项领域与预测跨度任务；另在 ETTh1、ETTh2 上进行六项长跨度数值外部验证。文本侧仅使用 fact，排除数据中生成式预测字段 preds；共保留14281条去重事实，并要求事实 end_date 早于预测起点。需强调，end_date 只是数据提供的可得时间代理，不等同于已核验的真实发布时间。")
    body(doc, "本文提出 SafeFAME-TS v2。系统先从 Last、季节朴素、AR-Ridge、DLinear-M 与 PatchTST 中按验证校准段选择数值回退，再构造语义残差和频率语义残差候选。验证期后半独立用于资格判定：候选必须总体超过回退、在决策段前后两半均胜出，并在99次文本错位置换中达到每任务两条候选 Bonferroni 校正后的 p≤0.025。测试期采用5000次移动块Bootstrap评估重叠预测窗口下的损失差区间。外部审查后另做999次循环移位置换和匹配容量纯数值残差对照，作为不改变冻结路由的敏感性检查。")
    body(doc, f"冻结规则没有在18项任务中放行任何文本路径。测试期事后分析显示，语义候选虽在{claims['semantic_point_improvement_tasks']}/18项任务取得更低点估计MSE，但只有{claims['semantic_significant_improvement_tasks']}项95%移动块区间完全支持改善，同时有{claims['semantic_significant_harm_tasks']}项区间完全支持变差；频率候选仅在{claims['frequency_point_improvement_tasks']}/18项任务取得点估计改善。数值外部验证中，PatchTST-M 在六项 ETT 任务均优于所比较的非PatchTST基线，平均MSE改善{old_claims['ett_mean_patchtst_gain_pct']:.2f}%。扰动实验表明其对高斯噪声与10%随机缺失的平均退化分别为{old_claims['robustness_patchtst_gaussian_mean_degradation_pct']:.2f}%和{old_claims['robustness_patchtst_random_missing_mean_degradation_pct']:.2f}%，但对尖峰和末段连续缺失平均退化达{old_claims['robustness_patchtst_spikes_mean_degradation_pct']:.2f}%和{old_claims['robustness_patchtst_tail_missing_mean_degradation_pct']:.2f}%。")
    body(doc, f"结果说明，文本价值不能由少数测试任务或模型复杂度推断，必须在预测时点约束、反事实对照和独立选择协议下逐任务验证。999次循环移位检验仍无候选达到p≤0.025；相对于匹配容量纯数值残差对照，只有{sensitivity_summary['matched_control_test_ci_positive']}条候选的测试期区间完全大于零，{sensitivity_summary['matched_control_test_ci_negative']}条完全小于零。SafeFAME-TS v2 的主要贡献是将是否启用文本转化为可证伪的统计决策，并在当前证据不足时保留数值回退；结论不应外推为文本在所有时间序列任务中无效。")
    p = doc.add_paragraph()
    p.add_run("关键词  ").bold = True
    p.add_run("多模态时间序列预测  时点对齐  文本置换  移动块Bootstrap  安全回退  负迁移")
    doc.add_page_break()
    toc(doc, ["1 问题重述与研究目标", "2 相关研究与方案依据", "3 数据审计与预处理", "4 模型假设与符号", "5 SafeFAME TS v2模型", "6 实验设计与求解", "7 实验结果", "8 稳健性与误差分析", "9 系统实现与可复现性", "10 模型评价与结论", "参考文献", "附录"])

    heading(doc, "1 问题重述与研究目标", 1)
    body(doc, "课程设计要求形成完整的问题描述、系统或实验设计、算法描述、实验结果与分析图表，并提交源代码和详细实验过程资料。结合时间序列研究基础，本文把任务定义为：在只能使用预测起点前信息的条件下，判断外部事实文本是否改善未来多步预测，并设计一个在文本证据不足时不会被迫采用文本的系统。")
    heading(doc, "1.1 研究问题", 2)
    bullets(doc, ["RQ1 严格时点文本语义是否稳定超过纯数值预测。", "RQ2 正确对齐文本是否超过随机置换文本。", "RQ3 频率与语义交互是否提高跨跨度稳定性。", "RQ4 验证期选择能否避免已观察到的文本负迁移。", "RQ5 数值骨干能否在外部数据和输入扰动下保持可复现性能。"])
    heading(doc, "1.2 闭环对应关系", 2)
    table(doc, "表 1 原题要求 数据 方法 输出与验证对应", ["要求", "数据证据", "模型方法", "输出", "验证"], [
        ["问题与数据", "六领域数值和fact", "时点索引", "18项任务", "审计日志"],
        ["算法设计", "历史窗口和文本", "专家池与残差候选", "多步预测", "基线比较"],
        ["改进路线", "覆盖和频率特征", "文本置换与安全选择", "路径决策", "分段门槛"],
        ["实验结果", "2764个测试起点", "冻结后评估", "MSE MAE RMSE", "块Bootstrap"],
        ["系统实现", "结果CSV", "交互原型", "任务审计视图", "复现脚本"],
    ], [3.0, 3.6, 4.4, 3.2, 3.2], 8.0)

    doc.add_page_break()
    heading(doc, "2 相关研究与方案依据", 1)
    add_literature_table(doc, "表 2 相关研究 证据等级与本文取舍")
    body(doc, "PatchTST 的分块与通道独立结构[2]、DLinear 对简单基线的强调[3]、TimesNet 与 FEDformer 的多周期或频域建模[4,6]共同构成数值侧依据。iTransformer 与 TimeXer 分别讨论变量中心表示和外生变量[7-8]。文本侧以 Sentence-BERT 和 MiniLM 提供轻量句向量[10-11]。Time-LLM 与 TimeCMA 展示语言表征与时间序列对齐的不同路径[9,12]。本文没有复现所有大型模型，因为研究变量是“文本是否提供增量信息”，不是模型排行榜；使用冻结编码与线性残差头可以降低容量混杂。")
    body(doc, "TimeCMA 已正式发表于 AAAI 2025[12]。T3Time 的正式题名为 Tri-Modal Time Series Forecasting via Adaptive Multi-Head Alignment and Residual Fusion，已发表于 AAAI 2026[13]；本文借鉴其时域、频域和提示三分支以及跨度条件门控，不再把它误写为测试时训练。Spectral Text Fusion 截至核验时仍为 arXiv:2602.01588 预印本[14]，只作为频谱与文本交互启发，不直接采用其报告数字作为本文证据。")

    heading(doc, "3 数据审计与预处理", 1)
    heading(doc, "3.1 数据身份与范围", 2)
    body(doc, "主数据来自 Time-MMD 官方论文和官方仓库[1]。本文选取六个领域，覆盖周频与月频、单变量与多变量、长序列与少样本。原始文件保留不变，所有清洗结果写入 data_processed 和 outputs。")
    table(doc, "表 3 六领域数值数据与排序审计", ["领域", "行数", "数值列", "起始", "结束", "重排数", "OT缺失"], data_rows(), [2.6, 1.6, 1.8, 2.4, 2.4, 1.8, 1.8], 8.2)
    heading(doc, "3.2 数值预处理", 2)
    body(doc, "先解析日期区间并按 start_date、end_date 稳定升序排序。目标 OT 以训练期均值和总体标准差标准化，验证与测试只使用训练期参数。多变量模型的各数值列同样以训练期统计量标准化。样本按滑动窗口构造，但训练、验证和测试边界处不允许目标跨段。")
    equation(doc, "x̃ₜ=(xₜ−μtrain)/σtrain", 1)
    heading(doc, "3.3 文本预处理与泄漏控制", 2)
    body(doc, "只保留非空 fact，排除 preds 与 pred；同领域同来源的规范化完全重复事实只保留最早结束区间。对预测起点 t，文本必须位于同一历史窗口内且满足 end_date<t。该规则比允许同周期文本更保守，但 end_date 并非经外部核验的发布日期，因此结果仍属于基于可得时间代理的回测。")
    figure(doc, FIG / "fig02_text_coverage.png", "图 1 严格时点约束下各领域文本覆盖", 15.5)
    body(doc, "去重语料共14281条；每个样本每来源最多选32条。文本中含 forecast、will、expect 等未来措辞仅作为审计特征而不自动删除，因为这些措辞可能出现在预测起点前已发布的历史报告中。模型同时接收来源缺失、可用和选中数量、文本年龄与未来措辞比例等10维质量特征。")

    heading(doc, "4 模型假设与符号", 1)
    heading(doc, "4.1 模型假设", 2)
    bullets(doc, ["在每个数据文件内部，稳定升序后的日期能够表示观测先后顺序。", "训练期标准化参数可用于后续时间段；明显分布漂移将作为误差来源而非被隐去。", "end_date 早于预测起点是文本可得性的必要代理条件，但不是充分的真实发布日期证明。", "同一任务内逐预测起点损失存在局部相关，因此区间估计按连续块重采样。", "验证段能够用于有限的模型与路径选择；小样本任务的统计功效可能不足。"])
    heading(doc, "4.2 主要符号", 2)
    table(doc, "表 4 主要符号与含义", ["符号", "含义", "类型或单位"], [
        ["Xₜ", "截至预测起点的长度L数值窗口", "标准化数值"], ["Yₜ,H", "未来H步OT目标", "标准化数值"], ["Zₜ", "report与search事实的聚合向量", "384+384=768维后降维"], ["qₜ", "文本覆盖 年龄 来源与措辞质量", "10维特征"], ["bₜ", "验证期选定的数值回退预测", "H维"], ["rₜ", "文本残差候选预测", "H维"], ["α", "Ridge正则系数", "超参数"], ["pperm", "文本置换检验经验p值", "0至1"],
    ], [2.8, 9.0, 5.0], 8.4)

    heading(doc, "5 SafeFAME TS v2模型", 1)
    heading(doc, "5.1 数值专家池", 2)
    body(doc, "专家包括 Last、SeasonalNaive、AR-Ridge、DLinear-M 与 PatchTST。Last 复制窗口最后值；季节朴素使用对应周期滞后；AR-Ridge 把长度L窗口映射至H步；DLinear-M 对趋势与季节分量建立线性映射；PatchTST 将单目标窗口切为重叠patch并用Transformer编码。专家选择只根据校准段MSE。")
    equation(doc, "m* = arg minₘ MSEcalibration(m)", 2)
    heading(doc, "5.2 文本聚合", 2)
    body(doc, "事实向量由冻结 all-MiniLM-L6-v2 生成。报告与搜索文本分开聚合，权重仅由事实年龄决定；质量特征不进入聚合权重，而是在残差模型中作为独立输入。这一区分保证公式与代码一致。")
    equation(doc, "zₜ = Normalize(Σᵢwᵢeᵢ/Σᵢwᵢ),  wᵢ=exp(−ln2·ageᵢ/τ)", 3)
    heading(doc, "5.3 语义与频率残差候选", 2)
    body(doc, "语义候选的设计矩阵由标准化历史窗口、训练期拟合的PCA语义主成分和标准化质量特征组成。频率候选额外加入窗口频率统计及其与语义主成分前8维的乘积交互。两者预测 Last 锚点的H步残差，而不是直接与任意数值专家输出拼接。由于路径资格比较对象是校准选出的数值回退，候选相对回退的差异同时包含残差结构和文本输入，不能直接称为纯文本效应。本文因此增加结构匹配的纯数值残差对照：语义路径对照只保留数值窗口，频率路径对照保留数值窗口和频率统计。")
    equation(doc, "δ̂ = arg minδ Σ||Yₜ−Last(Xₜ)−Φ(Xₜ,Zₜ,qₜ)δ||²₂+α||δ||²₂", 4)
    heading(doc, "5.4 资格判定与输出", 2)
    body(doc, "验证期按时间对半。前半选择数值专家、Ridge正则和深度模型早停轮数；后半保持未见，用于文本路径资格。冻结协议对每个候选执行99次文本与质量行置换，经验p值采用加一校正。Bonferroni校正族定义为同一领域与跨度任务内的语义和频率两条候选，因此阈值取0.05/2=0.025；该校正不覆盖18项任务之间的探索性测试期区间。99次置换的p值分辨率仅为0.01，所以另以固定原校准正则系数的999次循环移位置换做敏感性检查。循环移位保留各数据段内部文本序列的局部结构，只破坏文本与数值起点的同步关系。")
    equation(doc, "pperm=(1+#{MSEperm≤MSEaligned})/(99+1)", 5)
    equation(doc, "ŷ = rbest  若总体胜出∧两分段胜出∧pperm≤0.025；否则 ŷ=b", 6)
    heading(doc, "5.5 算法过程", 2)
    table(doc, "表 5 SafeFAME TS v2算法步骤", ["步骤", "输入", "处理", "输出"], [
        [1, "原始CSV", "日期稳定排序 训练期标准化", "时间窗口"], [2, "fact与日期", "去重 严格时点索引 冻结编码", "Z与q"], [3, "训练和校准", "训练五类专家并按校准MSE选回退", "b"], [4, "训练和校准", "PCA与Ridge选择语义 频率候选", "候选参数"], [5, "独立决策段", "总体 分段 99次置换", "资格标记"], [6, "训练加验证", "按冻结参数重训", "最终模型"], [7, "测试", "MSE MAE RMSE和移动块区间", "结果与审计"],
    ], [2.0, 4.1, 7.5, 3.4], 8.2)

    heading(doc, "6 实验设计与求解", 1)
    heading(doc, "6.1 数据划分与独立样本单位", 2)
    body(doc, f"Time-MMD 每个领域按70%训练、10%验证、20%测试顺序划分，验证再分为校准和路径决策。18项任务共{claims['calibration_windows']}个校准起点、{claims['decision_windows']}个决策起点和{claims['test_windows']}个测试起点。独立统计单位是预测起点，而非每个H步误差；每个起点先对H步和变量取平均损失。")
    heading(doc, "6.2 超参数与训练", 2)
    table(doc, "表 6 核心超参数", ["模块", "设置"], [
        ["随机种子", "2026 2027 2028"], ["深度训练", "AdamW 学习率3e-4 权重衰减1e-4 最大60轮 patience 8"], ["PatchTST", "d_model 32 四头 两层 d_ff 64 dropout 0.1"], ["语义PCA", "768维拼接输入 最多保留24维 仅训练期拟合"], ["Ridge α", "1e-4至1e4候选"], ["冻结置换", "99次逐行错位 加一经验p值"], ["敏感性置换", "999次循环移位 固定原校准α"], ["移动块Bootstrap", "5000次 块长=max(H,min(季节周期,24))"],
    ], [4.0, 13.0], 8.5)
    heading(doc, "6.3 评价指标", 2)
    equation(doc, "MSE=(1/NH)ΣₙΣₕ(yₙ,ₕ−ŷₙ,ₕ)²", 7)
    body(doc, "同时报告 MAE 与 RMSE。MSE、MAE、RMSE 均在各领域训练期标准化空间计算，因此同一任务内可比较模型，不把 Security 的绝对值与 Climate 直接比较。候选相对改善定义为100×(1−MSEcandidate/MSEfallback)。")
    heading(doc, "6.4 不确定性", 2)
    body(doc, "滑动窗口高度重叠，逐起点损失不是独立样本。本文对回退损失减候选损失做移动块Bootstrap，块长至少为H并最多取24或季节周期，重复5000次，报告2.5%与97.5%分位数[15]。区间完全大于零表示当前重采样设定支持候选改善，完全小于零表示支持候选变差，跨零表示当前样本不足以下定论；这些测试期区间用于诊断，不用于路径选择，也未按18项任务再次进行多重比较校正。")
    power_rows = []
    for (domain, horizon), group in sensitivity.groupby(["domain", "horizon"], sort=False):
        semantic = group[group.variant.eq("semantic_residual")].iloc[0]
        frequency = group[group.variant.eq("frequency_residual")].iloc[0]
        semantic_mde = "不足3块" if pd.isna(semantic.approx_mde_vs_control_pct) else f"{semantic.approx_mde_vs_control_pct:.1f}%"
        frequency_mde = "不足3块" if pd.isna(frequency.approx_mde_vs_control_pct) else f"{frequency.approx_mde_vs_control_pct:.1f}%"
        power_rows.append([domain, int(horizon), int(semantic.decision_windows), int(semantic.block_length), int(semantic.nonoverlapping_decision_blocks), semantic_mde, frequency_mde])
    table(doc, "表 7 路径决策段样本量与近似功效审计", ["领域", "H", "决策起点", "块长", "非重叠块", "语义近似MDE", "频率近似MDE"], power_rows, [2.7, 0.9, 2.0, 1.3, 2.0, 3.5, 3.5], 7.3)
    body(doc, f"在36条候选路径中，有{sensitivity_summary['power_audit_unavailable_lt3_blocks']}条因决策段不足3个完整非重叠块而不报告近似最小可检测效应。能够估计的4条路径也需要约26.6%至558.9%的相对损失差才达到近似80%功效。该计算基于非重叠块均值和正态近似，只是功效警示，不是正式功效证明；它说明严格门槛下的零放行结果同时受到低假阳性目标和小样本低功效影响。")

    heading(doc, "7 实验结果", 1)
    heading(doc, "7.1 数值回退选择", 2)
    figure(doc, V2_FIG / "fig_v2_fallback_experts.png", "图 2 验证期选定的数值回退专家", 13.8)
    body(doc, "PatchTST 被选中8次，SeasonalNaive与AR-Ridge各3次，Last与DLinear-M各2次。经济任务全部选择AR-Ridge，交通任务全部选择季节朴素；这说明专家选择与领域结构一致，也说明单一深度骨干不能替代简单基线。")
    heading(doc, "7.2 路径资格结果", 2)
    figure(doc, V2_FIG / "fig_v2_permutation_gate.png", "图 3 两条文本候选的验证期置换检验p值", 15.6)
    body(doc, f"共36条文本候选路径，其中{claims['segment_stability_passes']}条在决策段前后两半均胜过回退，但没有一条达到p≤0.025。最接近门槛的候选p=0.03，仍按预先规则拒绝。最终18项任务全部输出数值回退；该结论由验证段产生，而不是看到测试结果后作出。")
    heading(doc, "7.3 测试期事后诊断", 2)
    figure(doc, V2_FIG / "fig_v2_candidate_gains.png", "图 4 未被选中文本候选的测试期相对MSE变化", 16.0)
    body(doc, f"语义候选在{claims['semantic_point_improvement_tasks']}项任务有正点估计，频率候选仅在{claims['frequency_point_improvement_tasks']}项为正；经济三项和若干安全任务损失明显增加。点估计不能绕过验证期资格，因为这相当于用测试集选择路径。")
    figure(doc, V2_FIG / "fig_v2_semantic_block_ci.png", "图 5 语义候选逐任务移动块Bootstrap区间", 15.8)
    body(doc, f"相对于校准选出的数值回退，语义候选有{claims['semantic_significant_improvement_tasks']}项区间完全大于零，分别为 Climate H4、Climate H12 和 Agriculture H12；有{claims['semantic_significant_harm_tasks']}项完全小于零，包括 Economy H3、H6、H12 与 Security H12。由于候选还包含数值残差结构，这里只能称为完整候选路径差异，不能直接归因为文本。Traffic 三项虽有23.35%至37.31%的点估计改善，但区间均跨零，且验证期置换门槛未通过，因此不能声称已证明交通文本有效。")
    rows = []
    for r in audit.itertuples():
        rows.append([r.domain, int(r.horizon), r.numeric_fallback, f"{r.semantic_residual_permutation_p_value:.2f}", "是" if r.semantic_residual_segment_wins else "否", "回退"])
    table(doc, "表 8 语义候选资格审计", ["领域", "H", "数值回退", "置换p", "两段均胜", "最终路径"], rows, [3.0, 1.2, 3.6, 2.2, 3.0, 2.5], 7.8)

    heading(doc, "7.4 ETT外部数值验证", 2)
    rows = [[r.dataset, int(r.horizon), r.best_model, f"{r.best_mse:.4f}±{r.best_mse_std:.4f}", r.best_non_patch_model, f"{r.patchtst_gain_vs_best_non_patch_pct:.2f}%"] for r in ett.itertuples()]
    table(doc, "表 9 ETTh1与ETTh2外部验证", ["数据", "H", "最佳模型", "MSE均值±标准差", "最佳非Patch", "改善"], rows, [2.5, 1.4, 3.2, 4.6, 3.4, 2.5], 8.0)
    figure(doc, FIG / "fig06_ett_external_results.png", "图 6 ETT外部基准结果", 15.6)
    body(doc, f"PatchTST-M 在六项ETT任务全部取得最低MSE，相对最佳非PatchTST基线平均改善{old_claims['ett_mean_patchtst_gain_pct']:.2f}%。这是单目标OT协议，不能与使用全通道平均指标的论文表格直接拼接；其作用是验证本项目的数值训练与评估管线。")

    heading(doc, "8 稳健性与误差分析", 1)
    heading(doc, "8.1 输入扰动", 2)
    figure(doc, FIG / "fig09_input_robustness.png", "图 7 数值模型在四类输入扰动下的相对退化", 15.5)
    body(doc, f"PatchTST 对0.10标准差高斯噪声和10%随机缺失的平均MSE退化为{old_claims['robustness_patchtst_gaussian_mean_degradation_pct']:.2f}%和{old_claims['robustness_patchtst_random_missing_mean_degradation_pct']:.2f}%，但对2%尖峰和末段12步连续缺失分别退化{old_claims['robustness_patchtst_spikes_mean_degradation_pct']:.2f}%和{old_claims['robustness_patchtst_tail_missing_mean_degradation_pct']:.2f}%。系统应用时应增加尖峰检测，并在窗口尾部连续缺失时降低深度预测置信度。")
    heading(doc, "8.2 分布漂移与失败案例", 2)
    figure(doc, FIG / "fig08_security_distribution_shift.png", "图 8 Security目标在时间切分后的分布漂移", 15.4)
    body(doc, "Security 测试段相对训练段发生明显尺度漂移，标准化测试误差远高于其他领域；文本候选不能稳定修正这种漂移。该现象说明冻结语义编码与线性残差并不具备因果适配能力，报告不能把相关文本当作导致目标变化的证据。")
    heading(doc, "8.3 文本增量归因与置换敏感性", 2)
    figure(doc, ROOT / "outputs" / "reviewer_sensitivity" / "fig_v3_matched_control_ci.png", "图 9 文本候选相对匹配容量纯数值残差对照的测试期差异", 16.2)
    body(doc, f"匹配容量对照保持与候选相近的数值残差结构，但移除文本向量、质量特征和频率语义交互。36条候选中，仅Climate H4和H12的语义候选区间完全大于零，共{sensitivity_summary['matched_control_test_ci_positive']}条；{sensitivity_summary['matched_control_test_ci_negative']}条区间完全小于零，其余跨零。Agriculture H12相对数值回退的较大改善在匹配对照下几乎消失，说明此前差异主要来自残差结构而非可识别的文本增量。")
    figure(doc, ROOT / "outputs" / "reviewer_sensitivity" / "fig_v3_circular_shift_gate.png", "图 10 保留时间结构的999次循环移位置换敏感性", 16.0)
    body(doc, f"999次循环移位将原校准得到的Ridge正则系数固定，只改变文本与数值起点的相对位置。没有候选达到p≤0.025，与冻结协议的零放行结论一致。Climate H4频率、Climate H24语义和Traffic H3频率的p值为0.035至0.042，低于0.05但高于预设门槛，只能视为后续数据上的待验证线索。")
    heading(doc, "8.4 消融与边界", 2)
    body(doc, "频率候选只有8/18项获得相对数值回退的正点估计，少于语义候选的13/18项，并在多个任务扩大损失。相对于匹配容量对照，频率候选也没有形成稳定优势。频率交互因此保留为被检验但未启用的候选，不写成有效创新。当前验证段较短，严格门槛降低了假阳性，也可能提高假阴性；未来可在更多独立领域上预注册同一规则，而不是在当前测试集放宽阈值。")

    heading(doc, "9 系统实现与可复现性", 1)
    heading(doc, "9.1 软件结构", 2)
    table(doc, "表 10 项目目录与职责", ["目录", "内容"], [
        ["references/external", "固定提交的 Time-MMD ETT 与参考实现"], ["data_processed", "时点文本索引 句向量与语义缓存"], ["src", "审计 基线 深度模型 文本候选 选择 汇总与检查"], ["outputs", "逐起点预测 指标 协议 表格 图形 日志 模型"], ["paper", "第4周 第10周 最终报告与渲染检查"], ["prototype", "交互式任务结果浏览器"], ["docs", "数据审计 协议冻结 文献核验 最终检查"],
    ], [4.5, 12.5], 8.5)
    heading(doc, "9.2 复现链", 2)
    body(doc, "复现顺序为：数据审计与排序检查、时点语料构建、文本编码与语义缓存、数值专家和文本候选训练、冻结选择、匹配容量对照与循环移位敏感性、结果汇总与制图、报告与原型生成。每一步输出到新目录，不覆盖原始数据。protocol.json记录冻结主实验的划分、种子、模型、置换次数、阈值和Bootstrap设置；reviewer_sensitivity_summary.json单独记录审查后敏感性分析，避免与预注册式主结果混写。")
    heading(doc, "9.3 计算环境", 2)
    body(doc, "实验在 Windows 环境完成，使用 Python、PyTorch、pandas、scikit-learn、sentence-transformers 与 matplotlib；深度训练使用 NVIDIA RTX 5070 Laptop GPU。依赖版本记录在 requirements.txt，README 给出统一命令。源数据不因体积与许可直接嵌入文档，支撑材料记录来源URL与固定提交。")

    heading(doc, "10 模型评价与结论", 1)
    heading(doc, "10.1 主要结论", 2)
    bullets(doc, [
        "在当前六领域18项任务和冻结门槛下，没有文本路径获得部署资格，系统全部回退至验证期选定的数值专家。",
        "相对于匹配容量纯数值残差对照，只有Climate H4和H12的语义候选区间完全支持改善，同时9条候选区间支持变差，不能概括为文本普遍有效。",
        "999次循环移位置换仍无候选达到p≤0.025，说明零放行结论不依赖逐行随机置换这一种反事实构造。",
        "频率语义交互没有形成跨领域稳定增益，作为否定消融结果保留。",
        "数值专家的优胜者随领域变化；PatchTST在ETT六项任务上表现最好，但在Time-MMD并非所有任务最优。",
        "路径决策段在32/36条候选上不足3个完整非重叠块，因此零放行同时反映严格门槛与低统计功效，不能写成文本普遍无效。",
    ])
    heading(doc, "10.2 优点", 2)
    body(doc, "模型直接回答何时启用文本，数据处理链可审计，公式与代码一致，包含简单基线、匹配容量对照、两类错位置换、消融、跨领域确认、外部数据、三随机种子和输入扰动。选择器允许否定结论，使系统价值不依赖于事后挑选成功任务。")
    heading(doc, "10.3 局限", 2)
    body(doc, "第一，Time-MMD 的 end_date 不能证明真实发布时间，时点安全仍依赖代理假设；第二，验证期约5%的两段样本在少样本领域统计功效很低，32/36条候选不足3个完整非重叠块；第三，冻结MiniLM和线性残差只检验轻量语义增量，不能代表所有多模态结构；第四，主协议的99次逐行置换分辨率较粗且不保留时间依赖，虽然999次循环移位得到一致的零放行结论，仍需更多独立时间段复核；第五，任务目标和采样频率不同，跨领域平均改善只适合计数或相对指标；第六，当前结论为预测相关性，不构成因果解释。")
    heading(doc, "10.4 改进方向", 2)
    body(doc, "下一步应优先获得可核验发布时间或引入7天、30天、60天发布滞后敏感性；在新增领域上预注册并复用同一资格规则；用滚动起点评估增加独立决策区间，并预先报告最小可检测效应；将预测精度与错误启用风险组成Pareto选择，而不是在当前测试集放宽阈值；针对尖峰和连续缺失增加异常检测与缺失掩码。只有在新增数据上通过门槛，才考虑使用更大语言模型或端到端跨模态网络。")
    add_references(doc, full=True)
    heading(doc, "附录 A 关键结果复核", 1)
    body(doc, "18项任务的完整MSE、MAE、RMSE、相对改善、移动块区间、路径p值和分段判定分别保存在 safefame_v2_metrics.csv 与 safefame_v2_selection_audit.csv。循环移位、匹配容量对照和功效审计保存在 reviewer_sensitivity_results.csv。正文表格为这些文件的直接汇总。")
    sample = metrics[metrics.model.eq("semantic_residual")][["domain", "horizon", "mse", "mae", "rmse", "improvement_vs_fallback_pct", "block_ci_low", "block_ci_high"]]
    rows = [[r.domain, int(r.horizon), f"{r.mse:.4f}", f"{r.mae:.4f}", f"{r.rmse:.4f}", f"{r.improvement_vs_fallback_pct:.2f}%", f"[{r.block_ci_low:.4f}, {r.block_ci_high:.4f}]"] for r in sample.itertuples()]
    table(doc, "表 A1 语义候选完整测试指标", ["领域", "H", "MSE", "MAE", "RMSE", "相对改善", "块区间"], rows, [2.8, 1.0, 2.2, 2.2, 2.2, 2.5, 4.0], 7.4)
    heading(doc, "附录 B 课程要求落实清单", 1)
    table(doc, "表 B1 课程设计规范落实", ["课程要求", "交付位置", "状态"], [
        ["完整问题描述", "正文第1章", "完成"], ["系统或实验设计", "第5至6章", "完成"], ["算法设计描述", "第5章公式与步骤", "完成"], ["实验结果及分析图表", "第7至8章", "完成"], ["源代码", "src目录与支撑材料", "完成"], ["详细实验数据和过程资料", "outputs与docs", "完成"], ["个人作业", "所有交付按个人项目整理", "完成"], ["无需PPT展示", "第10周改为书面中期报告", "完成"],
    ], [6.0, 8.0, 3.0], 8.4)
    remove_last_paragraph(doc)
    path = OUT / "SafeFAME-TS_课程设计报告_修订版.docx"
    doc.core_properties.title = "文本何时有助于时间序列预测 多领域严格时点反事实检验与安全回退"
    doc.core_properties.subject = "人工智能课程设计"
    doc.core_properties.author = ""
    apply_font_policy(doc)
    doc.save(path)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (build_week4(), build_week10(), build_final()):
        print(path)


if __name__ == "__main__":
    main()
