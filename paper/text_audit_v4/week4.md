文本增强多领域时间序列预测

**严格时点对齐 反事实检验与安全回退**

第4周课程设计选题与初步技术方案

课程名称 人工智能

作业性质 个人课程设计

姓名学号 \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

提交日期 \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

# 摘要

本课程设计拟研究外部文本在多领域时间序列预测中的真实增益。核心问题不是简单增加文本输入，而是在严格排除未来信息后，判断文本语义何时能稳定改善预测，以及证据不足时如何退回可靠的纯数值模型。项目计划以 Time-MMD 为主数据，在气候、能源、经济、交通、农业和安全六个领域构建18项领域与预测跨度组合，并以 ETTh1、ETTh2 作为纯数值外部基准。候选系统由数值专家池、冻结句向量、语义残差、频率交互、文本置换检验和验证期安全选择组成。最终交付包括源代码、逐步数据审计、实验表图、书面报告和交互式原型。

本方案的可检验目标是：若文本候选不能在预先保留的验证期同时超过数值回退、通过分段稳定性和文本置换门槛，则系统不启用文本。该设计允许最终结论为否定结果，从而避免只报告成功案例。

**关键词** 多模态时间序列 时点对齐 反事实检验 安全融合 负迁移

# 目录

1 选题背景与问题定义

2 文献调研与方案取舍

3 数据与任务设计

4 初步算法方案

5 实验与验证计划

6 进度安排 风险与交付

参考文献

# 1 选题背景与问题定义

## 1.1 研究对象

现实预测通常同时面对历史数值序列与新闻、报告、政策或事件文本。文本可能解释数值突变，但也可能晚于预测时点、偏离主题或重复既有趋势。Time-MMD 提供多领域成对数据，为这一问题提供实验基础[1]。本项目将文本视为受信息可得性约束的外生变量，而不是默认有效的额外模态。

## 1.2 拟解决的具体问题

* 在文本结束时间早于预测起点的约束下，语义特征能否超过纯数值基线。
* 正确对齐文本能否优于随机置换文本，以排除模板结构、文本数量和时间趋势造成的伪增益。
* 频域统计与文本语义交互是否在较长预测跨度更有效。
* 当文本质量或稳定性不足时，能否用冻结规则自动回退到数值专家。
* 数值骨干在独立 ETT 数据与输入扰动下是否保持可复现性能。

## 1.3 输入 输出与边界

输入为长度 L 的历史数值窗口 X 和预测时点前可获得的事实集合 Z；输出为未来 H 步目标序列。数据中的 end\_date 仅作为可得时间代理，不能等同于真实发布时间；包含未来预测内容的 preds 字段不进入模型。

*ŷₜ₊₁:ₜ₊ᴴ = fθ(Xₜ₋ᴸ₊₁:ₜ , Z<t) （1）*

# 2 文献调研与方案取舍

表 1 相关研究 证据等级与本文取舍

| **研究** | **身份** | **主要启发** | **本文使用边界** |
| --- | --- | --- | --- |
| Time-MMD | NeurIPS 2024 | 多领域数值与文本对齐数据 | 提供数据基础；不照抄其结论 |
| DLinear | AAAI 2023 | 分解线性预测与强简单基线 | 作为可解释深度基线 |
| PatchTST | ICLR 2023 | 分块与通道独立 | 作为主要数值专家 |
| TimesNet / FEDformer | ICLR 2023 / ICML 2022 | 多周期与频域建模 | 启发频率统计分支 |
| TimeXer | NeurIPS 2024 | 外生变量建模 | 支持将文本视为外生信息 |
| Time-LLM / TimeCMA | ICLR 2024 / AAAI 2025 | 语言模型重编程与跨模态对齐 | 说明融合方法空间；本文采用轻量实现 |
| T3Time | AAAI 2026 | 时域 频域 提示三分支与跨度条件门控 | 借鉴多模态和跨度条件思想 |
| Spectral Text Fusion | 2026 预印本 | 频谱与文本交互 | 启发候选频率交互；须独立验证 |

数值侧从 Last、季节朴素和 AR-Ridge 起步，再加入 DLinear 与 PatchTST[2-3]。频率分支受 TimesNet 与 FEDformer 的多周期和频域思想启发[4,6]，但只构造低维频率统计和交互，避免复制大型结构。文本侧使用冻结 MiniLM 句向量[10-11]，以受正则化残差模型检验是否存在增量信息。Time-LLM、AAAI 2025 的 TimeCMA 与 AAAI 2026 的 T3Time 提供语言重编程、跨模态对齐和三分支融合依据[9,12-13]；Spectral Text Fusion 仍为预印本[14]，仅作为频谱交互启发。

# 3 数据与任务设计

表 2 Time-MMD 六领域数据审计结果

| **领域** | **行数** | **数值列** | **起始** | **结束** | **重排数** | **OT缺失** |
| --- | --- | --- | --- | --- | --- | --- |
| Agriculture | 532 | 3 | 1980-01 | 2024-04 | 0 | 0 |
| Climate | 1272 | 8 | 2000-01 | 2024-05 | 1272 | 0 |
| Economy | 447 | 3 | 1987-01 | 2024-03 | 447 | 0 |
| Energy | 1622 | 9 | 1993-04 | 2024-04 | 0 | 0 |
| Security | 309 | 1 | 1998-09 | 2024-05 | 0 | 0 |
| Traffic | 651 | 1 | 1970-01 | 2024-03 | 0 | 0 |

气候文件原为倒序，经济文件顺序混乱，建模前必须按解析后的时间稳定升序排序。六领域目标 OT 均无缺失；Security 仅有一个数值变量且样本最少，适合作为分布漂移和少样本压力测试。文本仅使用 fact，按同来源和规范化文本去重，最多保留每来源32条最近事实。

表 3 任务与评价口径

| **数据** | **任务数** | **划分** | **主要指标** | **用途** |
| --- | --- | --- | --- | --- |
| Time-MMD 六领域 | 18 | 70%训练 10%验证 20%测试 | 标准化 MSE MAE RMSE | 主实验 |
| ETTh1 ETTh2 | 6 | 固定时间划分 | 标准化 MSE 与三种子均值 | 外部数值验证 |
| 输入扰动 | 24 | 高斯 缺失 尖峰 尾段缺失 | 相对退化率 | 稳健性 |

# 4 初步算法方案

## 4.1 数值专家池

候选专家包括 Last、SeasonalNaive、AR-Ridge、DLinear-M 和 PatchTST。简单模型提供可解释下限，深度模型固定2026、2027、2028三个随机种子。数值专家仅在验证期前半段比较，测试集不参与选择。

## 4.2 严格时点语义表征

对每个预测起点，仅索引历史窗口内且 end\_date 早于预测起点的 fact。每条文本经冻结 all-MiniLM-L6-v2 编码；同来源向量按时间衰减加权平均并做 L2 归一化。历史窗口长度的四分之一作为半衰期。

*zₜ = Normalize(Σᵢ exp[-ln2·ageᵢ/τ] eᵢ / Σᵢ exp[-ln2·ageᵢ/τ]) （2）*

## 4.3 残差候选与安全选择

语义候选以 Last 预测为锚点，使用 Ridge 学习由数值窗口、文本语义和质量特征共同解释的残差；频率候选再加入历史窗口频谱统计与语义交互。由于候选锚点与最终数值回退可能不同，候选相对回退的改善不能单独归因于文本，必须结合匹配容量纯数值残差对照和文本错位置换解释。验证期再对半：前半选择正则系数和数值专家，后半只作路径资格判定。候选必须同时满足后半总体优于回退、前后两段均获胜、99次文本置换检验经 Bonferroni 校正后 p≤0.025。

*δ̂ = arg minδ ||y - Last(x) - Φ(x,z)δ||²₂ + α||δ||²₂ （3）*

*route = text iff MSEtext<MSEfallback ∧ two-segment wins ∧ pperm≤0.025 （4）*

# 5 实验与验证计划

表 4 实验矩阵与成功标准

| **实验** | **对照** | **验证方式** | **预先判据** |
| --- | --- | --- | --- |
| 数值基线 | 5类专家 | 时间顺序留出 三随机种子 | 报告全部模型 不只报告赢家 |
| 语义增量 | 语义候选 对 数值回退 | 独立路径决策段 | 总体和两分段均更优 |
| 文本反事实 | 对齐文本 对 错位文本 | 99次冻结置换 加循环移位敏感性 | p≤0.025 |
| 测试不确定性 | 逐预测起点配对损失 | 5000次移动块Bootstrap | 报告95%区间 |
| 消融 | 无频率 对 频率交互 | 同一划分同一回退 | 不以单任务胜利概括总体 |
| 稳健性 | 干净输入 对 四类扰动 | 三次重复 | 报告均值和标准差 |

由于滑动预测窗口相互重叠，普通独立同分布Bootstrap会低估不确定性，因此使用移动块Bootstrap保留局部依赖[15]。评价值均在训练期均值与标准差定义的标准化空间计算；不同领域的绝对 MSE 不直接横向比较。

# 6 进度安排 风险与交付

表 5 第4周以后实施计划

| **周次** | **主要工作** | **验收证据** |
| --- | --- | --- |
| 第4至6周 | 数据审计 时点索引 简单基线 | 审计表 自检日志 基线结果 |
| 第7至9周 | DLinear PatchTST 文本编码与开发实验 | 三种子模型 训练日志 开发图表 |
| 第10周 | 书面中期检查 冻结最终协议 | 中期报告 协议JSON 风险清单 |
| 第11至14周 | 六领域确认实验 ETT外部验证 稳健性 | 测试结果 区间估计 消融表 |
| 第15至17周 | 报告 原型 支撑材料 审稿检查 | DOCX PDF 代码 数据清单 校验报告 |

主要风险包括文本真实发布时间缺失、搜索文本主题漂移、验证样本较少、不同领域尺度不一致和深度模型随机性。对应措施是把 end\_date 明确表述为代理变量、保留文本置换与缺失指示、按领域标准化、固定三随机种子并报告限制。少样本任务另报告决策起点数、非重叠块数和近似最小可检测效应。若文本未通过门槛，否定结果仍作为完整结论，不临时放宽标准。

## 最终交付文件

* 第4周选题与初步技术方案；
* 第10周书面中期进展报告；
* 最终课程设计报告 DOCX 与 PDF；
* 完整 Python 源代码、依赖与统一运行说明；
* 原始数据来源和固定提交信息、处理后审计表、实验明细与图表；
* 交互式结果原型、文件哈希与最终检查报告。

# 参考文献

[1] Liu H, Xu S, Zhao Z, et al. Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis. Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2024. DOI: 10.52202/079017-2476.

[2] Nie Y, Nguyen N H, Sinthong P, Kalagnanam J. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. International Conference on Learning Representations, 2023.

[3] Zeng A, Chen M, Zhang L, Xu Q. Are Transformers Effective for Time Series Forecasting? Proceedings of the AAAI Conference on Artificial Intelligence, 2023, 37(9):11121-11128. DOI: 10.1609/aaai.v37i9.26317.

[4] Wu H, Hu T, Liu Y, et al. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. International Conference on Learning Representations, 2023.

[5] Zhang Y, Yan J. Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting. International Conference on Learning Representations, 2023.

[6] Zhou T, Ma Z, Wen Q, et al. FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting. Proceedings of the 39th International Conference on Machine Learning, PMLR 162:27268-27286, 2022.

[7] Liu Y, Hu T, Zhang H, et al. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. International Conference on Learning Representations, 2024.

[8] Wang Y, Wu H, Dong J, et al. TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables. Advances in Neural Information Processing Systems, 2024. DOI: 10.52202/079017-0015.

[9] Jin M, Wang S, Ma L, et al. Time-LLM: Time Series Forecasting by Reprogramming Large Language Models. International Conference on Learning Representations, 2024.

[10] Reimers N, Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP, 2019:3982-3992. DOI: 10.18653/v1/D19-1410.

[11] Wang W, Wei F, Dong L, et al. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. Advances in Neural Information Processing Systems, 2020.

[12] Liu C, Xu Q, Miao H, et al. TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment. Proceedings of the AAAI Conference on Artificial Intelligence, 2025, 39(18):18780-18788. DOI: 10.1609/aaai.v39i18.34067.

[13] Chowdhury A M, Akter R, Arib S H. T3Time: Tri-Modal Time Series Forecasting via Adaptive Multi-Head Alignment and Residual Fusion. Proceedings of the AAAI Conference on Artificial Intelligence, 2026, 40(25):20597-20605. DOI: 10.1609/aaai.v40i25.39196.

[14] Spectral Text Fusion for Multimodal Time Series Forecasting. arXiv:2602.01588, 2026. 预印本.

[15] Künsch H R. The Jackknife and the Bootstrap for General Stationary Observations. The Annals of Statistics, 1989, 17(3):1217-1241. DOI: 10.1214/aos/1176347265.