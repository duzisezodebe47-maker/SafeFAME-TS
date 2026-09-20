# SafeFAME-TS v2 冻结协议与结果

## 1. 冻结协议

- 时间划分：70% 训练、10% 验证、20% 测试；验证按时间对半为校准段和路径决策段。
- 数值专家：Last、SeasonalNaive、AR-Ridge、DLinear-M、PatchTST。
- 文本候选：语义残差、频率语义残差。
- 资格条件：候选在路径决策段总体优于校准选出的数值回退，且决策段前后两半均胜出，且 99 次文本置换检验经 Bonferroni 校正后 `p<=0.025`。
- 测试区间：逐预测起点配对损失的 5000 次移动块 Bootstrap。
- 随机种子：2026、2027、2028。

原始冻结协议使用逐行随机置换。外部审查后新增循环移位置换敏感性检查，以保留每个数据段内部的文本时间结构；该检查只用于评估结论稳健性，不反向修改冻结的 v2 路由结果。

机器可读协议：`outputs/safefame_v2/safefame_v2_protocol.json`。

## 2. 关键结果

- 18 项领域—跨度任务、36 条文本候选路径。
- 6 条候选通过决策段前后两半稳定性，但 0 条通过 `p<=0.025` 的置换门槛。
- 最终 18 项任务均选择数值回退。
- 测试期事后诊断：语义候选 13/18 项点估计改善，其中 3 项移动块区间完全大于零，4 项完全小于零。
- 频率候选 8/18 项点估计改善，其中 2 项区间完全大于零，3 项完全小于零。
- 测试期事后优势不得用于返回修改验证门槛。

## 3. 解释边界

该结果说明在当前六领域数据、当前轻量文本表征和预先规则下，没有足够证据部署文本路径；不能解释为文本在所有时间序列任务中无效。原候选以 Last 为残差锚点，而资格判定相对于校准选择的数值回退，因此候选相对回退的点估计同时包含残差结构差异，不能单独归因为文本。文本增量应结合置换检验以及新增的匹配容量纯数值残差对照解释。验证样本较少和 `end_date` 仅为可得时间代理也是主要限制。

## 4. 结果文件

- `outputs/safefame_v2/safefame_v2_metrics.csv`
- `outputs/safefame_v2/safefame_v2_selection_audit.csv`
- `outputs/tables/v2/candidate_test_results.csv`
- `outputs/tables/v2/selection_results.csv`
- `outputs/tables/v2/verified_claims.json`
- `outputs/figures/v2/`
- `outputs/reviewer_sensitivity/reviewer_sensitivity_results.csv`
- `outputs/reviewer_sensitivity/reviewer_sensitivity_summary.json`
