文本增强多领域时间序列预测

**数据审计 基线复现与严格融合协议**

第10周课程设计书面中期进展报告

课程名称 人工智能

作业性质 个人课程设计

姓名学号 \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

提交日期 \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

# 中期结论摘要

截至第10周计划节点，课程设计已完成数据审计、时间顺序修复、严格时点文本索引、五类数值专家、冻结文本表征、语义与频率残差候选、反事实置换检验和外部数值基准。主实验覆盖六个 Time-MMD 领域与三个预测跨度，共18项任务；测试期共有2764个预测起点。中期检查发现，早期直接拼接或宽松门控容易产生负迁移，因此已将最终协议修订为验证期双分段选择与证据不足回退。

本报告把开发过程中出现的失败结果纳入方法修订依据，不使用测试集调整最终资格门槛。第10周以后将按冻结协议完成一次性汇总、移动块Bootstrap区间、输入扰动、论文与支撑材料。教师已说明个人完成且不再进行PPT展示，因此本节点以书面进展报告替代演示文稿。

# 目录

1 课题与技术路线

2 已完成的数据工作

3 已完成的模型与实现

4 阶段实验发现

5 协议修订与冻结

6 后续计划与风险

参考文献

# 1 课题与技术路线

研究目标保持不变：检验外部事实文本在严格时点约束下是否为多领域时间序列提供可泛化的增量信息，并在证据不足时自动退回验证期选定的数值预测器。项目规模包括六领域18项多模态任务、ETTh1与ETTh2六项长跨度数值任务、四类输入扰动以及完整原型。

![图 1 项目总体技术路线](data:image/png;base64...)

图 1 项目总体技术路线

## 阶段完成度

表 1 第10周任务完成情况

| **模块** | **状态** | **形成文件** | **主要检查** |
| --- | --- | --- | --- |
| 数据审计 | 完成 | 审计CSV JSON | 日期顺序 缺失 重复 单位 |
| 时点文本索引 | 完成 | 语料与样本索引 | fact可得时间严格早于起点 |
| 数值基线 | 完成 | 预测明细与训练日志 | 简单基线 深度模型 三种子 |
| 文本候选 | 完成 | 语义缓存与置换结果 | 对齐文本不使用preds |
| 最终选择协议 | 已冻结 | protocol.json | 校准 决策 测试三重隔离 |
| 报告与原型 | 进行中 | 图表与网页原型 | 一致性 可复现 无PPT |

# 2 已完成的数据工作

表 2 六领域数值数据审计

| **领域** | **行数** | **数值列** | **起始** | **结束** | **重排数** | **OT缺失** |
| --- | --- | --- | --- | --- | --- | --- |
| Agriculture | 532 | 3 | 1980-01 | 2024-04 | 0 | 0 |
| Climate | 1272 | 8 | 2000-01 | 2024-05 | 1272 | 0 |
| Economy | 447 | 3 | 1987-01 | 2024-03 | 447 | 0 |
| Energy | 1622 | 9 | 1993-04 | 2024-04 | 0 | 0 |
| Security | 309 | 1 | 1998-09 | 2024-05 | 0 | 0 |
| Traffic | 651 | 1 | 1970-01 | 2024-03 | 0 | 0 |

原始文件未被覆盖。Climate 的1272行全部因倒序而重排，Economy 的447行全部改变原行位置；如果直接按CSV顺序划分，会把未来样本置于训练段。所有模型统一调用时间排序函数。Health\_AFR、Health\_US、Environment 和 SocialGood 在数据审计中保留，但因目标缺失、重复日期、日频算力或文本质量问题未进入主实验。

![图 2 严格时点约束下的文本覆盖](data:image/png;base64...)

图 2 严格时点约束下的文本覆盖

去重后语料共14281条事实。每个样本最多选择报告和搜索各32条，使用预测起点前的事实结束时间构造索引。end\_date 只是 Time-MMD 提供的区间终点，因此本文只把它称为可得时间代理，不推断其等于真实发布日期。

# 3 已完成的模型与实现

## 3.1 数值专家

数值专家池包括 Last、SeasonalNaive、AR-Ridge、DLinear-M 和 PatchTST。DLinear 与 PatchTST 采用2026、2027、2028三个随机种子；早停轮数由校准段确定，冻结后在训练加验证数据上按同轮数重训。

## 3.2 文本候选

文本由冻结 all-MiniLM-L6-v2 编码，报告与搜索分别以指数时间衰减聚合。两路各384维向量拼接为768维原始语义输入，再仅用训练期数据拟合PCA。语义残差候选使用数值窗口、降维语义向量与覆盖质量特征；频率残差候选额外加入频率统计及其与语义主成分的乘积。两个候选均以 Last 为残差锚点，并用 Ridge 控制容量。由于最终比较对象可能是其他数值专家，后续增加匹配容量纯数值残差对照，避免把结构差异误写成文本贡献。

*L(δ)=Σ||y−Last(x)−Φ(x,z)δ||²₂+α||δ||²₂ （1）*

## 3.3 工程复现

* 数据、代码、输出、图表、报告分目录保存；
* 所有随机过程固定种子；
* 每个模型保存逐预测起点结果，而不只保存平均指标；
* 核心脚本包含最小自检；
* 测试集在选择完成后才计算；
* 所有正式图表由结果CSV自动生成。

# 4 阶段实验发现

开发性实验表明，文本增益对领域和跨度高度敏感。早期直接融合路径在若干任务上优于数值基线，却在经济和安全任务产生明显负迁移；频率全分支也没有稳定胜过无频率版本。这一结果否定了“加入文本或频率模块必然更好”的初始直觉。

![图 3 冻结候选的跨任务点估计收益与负迁移](data:image/png;base64...)

图 3 冻结候选的跨任务点估计收益与负迁移

按当前冻结实现，语义候选在13/18项任务上取得更低测试MSE，但相对于数值回退的移动块Bootstrap只有3项95%区间完全大于零，同时有4项区间完全小于零。这里比较的是完整候选路径与回退专家，不等同于纯文本效应；该测试期汇总只用于说明风险，不用于返回修改门槛。

![图 4 各类数值专家在验证期校准中的入选次数](data:image/png;base64...)

图 4 各类数值专家在验证期校准中的入选次数

PatchTST 在8项任务成为回退专家，但 Last、季节朴素、AR-Ridge 和 DLinear-M 仍分别在不同领域胜出，说明不能按模型知名度预设统一骨干。

# 5 协议修订与冻结

针对开发阶段暴露的数据泄漏和选择偏差风险，最终协议做出四项修订：第一，验证期一分为二，前半校准、后半资格判定；第二，数值回退只能由校准段选择；第三，文本资格同时要求总体胜出、决策段前后两半均胜出和置换检验 p≤0.025；第四，测试不确定性使用5000次移动块Bootstrap，而非把重叠窗口视作独立样本。

表 3 冻结后的数据隔离规则

| **数据段** | **比例** | **允许用途** | **禁止用途** |
| --- | --- | --- | --- |
| 训练 | 70% | 拟合模型与PCA | 查看未来验证和测试目标 |
| 校准 | 约5% | 选数值专家 正则系数 早停轮数 | 判定文本资格 |
| 路径决策 | 约5% | 资格门槛与置换检验 | 再次调参 |
| 测试 | 20% | 冻结后一次评估与区间估计 | 模型或门槛选择 |

![图 5 文本路径启用前的置换检验门槛](data:image/png;base64...)

图 5 文本路径启用前的置换检验门槛

36条候选路径中，6条通过分段稳定性，但没有一条通过 Bonferroni 后的 p≤0.025 门槛。因此若按冻结协议执行，当前18项任务均应回退；这不是系统失败，而是选择器按预先标准拒绝缺乏稳健证据的文本路径。

# 6 后续计划与风险

表 4 第10周以后工作安排

| **工作** | **输出** | **完成判据** |
| --- | --- | --- |
| 冻结协议一次性汇总 | 18任务结果与资格审计 | 结果与protocol.json一致 |
| 移动块不确定性 | 配对损失95%区间 | 逐任务5000次重采样 |
| ETT外部与扰动 | 6项数值任务 4类扰动 | 三随机种子和重复实验 |
| 报告与支撑材料 | DOCX PDF 代码 数据清单 | 数字 图表 代码一致 |
| 原型与最终审稿 | 交互网站 检查报告 哈希 | 无夸大 无缺页 可复算 |

剩余最大风险是验证期样本较少导致路径选择功效不足，以及 end\_date 不能证明真实可得时间。最终报告将补充决策起点数、非重叠块数、近似最小可检测效应、循环移位置换和匹配容量纯数值残差对照，并把“未放行文本”解释为当前数据与门槛下的结论，而非证明文本永远无效。

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