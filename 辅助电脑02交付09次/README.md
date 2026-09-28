# 辅助电脑02交付09次：SocialGood 隔离输入包与退化审计

2026-09-28 修订补全原始到处理后审计。原 v9 Release 保留作历史基线；修订包使用新标签和新 SHA-256，不覆盖旧资产。

任务 `SocialGood_h3_f1`。输入固定为正式 Bundle 签名 `a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38`、冻结 spec SHA256 `a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031`。从历史 24、预测 3、季节周期 12 和四段边界自行计算并检查保留起点。完整重放见 [RUNBOOK.md](RUNBOOK.md)。

本交付是**安全输入子集**：`selection_fit/` 含选择侧的数值历史、频率、目标与文本特征；`test_features/` 只有 origin、时间网格、历史缺失掩码、语义、质量、可用性，不含测试目标或逐起点数值历史/频率。完整重叠测试历史会反推出部分测试真值，因此主动扣留，后续完整数值预测须通过受控、按时点顺序提供历史的接口。本交付不运行 SocialGood 模型，不访问测试评分。

`socialgood_origin_coverage.csv`、`socialgood_segment_coverage.csv` 和 `socialgood_scenario_flip_detail.csv` 给出起点级、分段与覆盖翻转审计。训练段“零文本”定义为：两情景下 340 个保留训练起点均未选中任何 report/search 事实；它**不**意味着质量向量全零。无文本起点的语义为零，两个来源可用掩码及文本可用掩码为假，质量向量是非零缺失哨兵。只有下游模型按掩码关闭文本路径时，才可期待安全退化；这里没有做数值等价的预测检验。

原始来源 URL 与真实发布时间未核验，`end_date` 仅作为文本可得时间代理。来源/日期表保留原有字段和空值，不补造。正式 Bundle 的其他存储位置可能包含测试真值；本隔离仅约束本 Release。

`raw_clean_audit.json` 由 `audit_socialgood_raw_clean.py` 对冻结的 Time-MMD 原始 CSV、数值清洗快照和完整文本血缘表重新计算：数值 924→916 行，8 行因缺失 `OT` 被排除，原始及清洗后起始日期重复数均为 0；生成的清洗 CSV 与冻结快照字节一致。文本原始 4961 行，逐来源列出缺失、重复事实处置和 4724 条保留规范事实。源文件 SHA-256、Time-MMD 提交及完整血缘表 SHA-256 均写入审计。修订包包含复算程序、审计表、选择侧处理数据及测试侧安全特征；原始数值文件、完整清洗序列和测试目标不公开打包，以免泄漏测试真值。持有已锚定原始来源与正式 Bundle 的验收方可按 [RUNBOOK.md](RUNBOOK.md) 复算。

Release ZIP **不进入 Git**。修订版下载地址：[SocialGood v9r2 Release 附件](https://github.com/duzisezodebe47-maker/SafeFAME-TS/releases/download/socialgood-isolated-input-v9r2-20260928/SafeFAME-TS_SocialGood_isolated_v9r2_20260928.zip)，3,353,590 字节，SHA256 `2fe3deeda8999b037b8f0ff50d7083aced93470d5fa0e7ab042ac8091d1952f7`。从最终 ZIP 解压到全新 D 盘目录后，验证器退出码 0，八个负例全部被拒绝；证据在 `clean_replay/revision_20260928/`。原始 v9 历史基线仍在[旧 Release](https://github.com/duzisezodebe47-maker/SafeFAME-TS/releases/download/socialgood-isolated-input-v9-20260927/SafeFAME-TS_SocialGood_isolated_v9_20260927.zip)，3,343,370 字节，SHA256 `f75cbe2dcd6689f3d6a1b6800e366f4bc041d24f9efdf3d341b20baac3154e6f`；旧资产和旧校验记录不改写。

分支内 README/RUNBOOK 是发布后补充的说明，与 Release 内对应文件字节不同；Release 内文件哈希以 ZIP 内 `MANIFEST.json` 为准。`MANIFEST.json`、`EXPECTED_OUTPUTS.json` 和审计表在分支与 Release 中应保持一致。
