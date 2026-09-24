# 辅助电脑02交付08次：Climate 隔离输入包与专项数据审计

以 `3218151885-creator@a06dfe1f2c34c604c6d2b7e374aac44e1783dde9` 为固定起点，正式 Bundle 签名为 `a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38`，冻结 spec SHA256 为 `a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031`。任务 `Climate_h4_f2`：历史 52 周，预测 4 周，分段边界 `[636, 763, 1017, 1144]`。

## 隔离边界

选择包 956 个 train/calibration/decision 起点，包含建模特征和其真值。测试包 124 个起点，仅含 origin、cutoff、日期网格、缺失掩码和文本语义/质量/可用性特征；无测试真值文件，也无逐起点数值历史及频率衍生特征。后两者被主动扣留：滚动窗口间的重叠足以从完整测试历史中重建约 123/127 个测试时间点的真值。因而本包是严格隔离的**安全输入子集**，不是可直接独立完成完整数值预测的全特征包。对外完整预测需要后续受控顺序评测接口。

`truth_commitment.json` 仅含正式 Bundle 目标文件及 test 切片的 SHA256、shape、dtype 和锚点，不含数值。Release 内的验证器检查安全特征与正式 Bundle 全文件哈希及行位置逐字节一致；选择包中的数值特征及真值通过构建时的原 Bundle直接比较和切片承诺约束。由于严格隔离包不含原始完整目标数组，单靠该包不能重新证明选择包切片来自原 Bundle；要做外部追溯须另行核对正式 Bundle。此界限不能写成“绝对防泄漏”。

正式 Bundle 的历史 Release 本身可能另含 test 真值；本轮隔离只防止**此 Release 被无意当作完整测试真值来源**，不能撤销其他地方已经公开的数据访问权限。

## 专项审计

`climate_boundary_audit.json`、`climate_feature_drift.csv`、`climate_source_coverage.csv`、`climate_weekly_axis_audit.csv` 等由包内脚本生成；论文文字见 `paper_ready/Climate数据与边界说明.md`。漂移仅描述历史输入、质量和文本覆盖，不读取或比较 test 真值。原始 URL 与真实发布时间未核验；`end_date` 仅是可得时间代理。

各表的统计单位、分母和无法逐起点重放的测试历史聚合限制见 [table_schema.md](table_schema.md)。

新 Release 的下载、运行与核验命令见 [RUNBOOK.md](RUNBOOK.md)。ZIP 只作为 Release 附件，不放入分支。第七轮及更早目录、正式 Bundle、冻结 spec 和模型结果均未修改。
