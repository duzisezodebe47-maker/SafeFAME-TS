# 主控电脑第八轮任务：Climate 基线与路由封印

执行分支：`team/main-eval-20260921`  
启动条件：收到两台辅助电脑第八次固定提交后。

## 一、主控任务

1. 下载并独立重放 Climate 角色隔离输入包，验证 test 特征包无真值。
2. 从正式 Bundle 独立生成 Last、SeasonalNaive、AR-Ridge 的 calibration/decision 预测；季节周期必须为 52。
3. 固定模型侧提交并核对全部哈希、边界、候选网格、权重重演、测试和运行日志。
4. 只把注册候选加入正式门控；`N+Q` 仅作消融说明。
5. 独立换算两组 999 次零分布，审计每轮 refit 代码路径。
6. 按冻结规则选择数值回退，并判断两个门控候选是否同时满足：决策 MSE 更低、前后半段均改善、经验 p≤0.025。
7. 生成并封印 Climate 唯一路由；封印前不允许任何一方运行 test 预测。

## 二、主控输出

- `Climate_selection_stage/`
- `Climate_numeric_baselines.csv`
- `Climate_null_conversion.jsonl`
- `Climate_refit_provenance_review.json`
- `Climate_raw_route.json`
- `Climate_sealed_route.json` 及 SHA256 sidecar
- `Climate_gate_result.md`

## 三、判断边界

- Agriculture 的成功不能作为 Climate 必然成功的证据。
- Climate 使用周频周期 52、horizon=4 和 fold=2，必须重新拟合和重新门控。
- 若两个文本候选都未过门槛，正式路由应回退到数值模型，不人为放宽 p 值或半段规则。
- test 只在路由封印后的下一轮运行一次。

