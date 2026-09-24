# 辅助电脑02第八次交付指令：Climate 隔离输入包与专项数据审计

执行分支：`3218151885-creator`  
建议目录：`辅助电脑02交付08次/`  
目标：为 Climate 建立可复核、避免意外读取测试真值的模型输入包，同时完成周频数据和文本边界专项审计。

## 一、固定输入

- 从第七次交付固定提交 `a06dfe1f2c34c604c6d2b7e374aac44e1783dde9` 开始。
- 正式 Bundle 签名：`a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38`。
- 冻结 spec SHA256：`a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031`。
- 任务：`Climate_h4_f2`；周期 52；边界 `[636, 763, 1017, 1144]`。
- 不修改正式 Bundle、冻结 spec 和第七轮 Release。

## 二、任务 A：构建 Climate 角色隔离输入包

创建一个新的 Release，至少包含：

1. `selection_fit/`：train、calibration、decision 的特征与真值，不包含 test 行。
2. `test_features/`：只包含 test 的特征、origin、cutoff、时间和网格信息；不得包含 `target`、`targets`、`targets_raw`、`targets_standardized` 或可逆推出真值的字段。
3. `truth_commitment.json`：只记录正式 test 真值文件的 SHA256、shape、dtype、任务和 Bundle 锚，不包含真值。
4. `mapping.csv`：证明两个子包的 origin 与正式 Bundle 一一对应。
5. `MANIFEST.json`：每个文件路径、字节数、shape、列名、SHA256 和来源。

包内必须有独立验证器：确认选择包没有 test 行、测试包没有任何真值字段、特征与正式 Bundle 对应位置逐字节一致、origin 网格满足 horizon=4 的边界。

## 三、任务 B：Climate 专项数据审计

生成以下材料：

- 四段样本量、起点范围、排除原因和缺失率。
- proxy / conservative_lag 在四段的文本可用率、选中文本数分布、来源上限饱和率。
- report/search 两来源的事实数量、日期范围、重复率、URL 缺失率和发布时间代理率。
- train→calibration→decision→test 的数值历史、质量特征、文本可用性分布漂移表；只描述特征，不读取或比较 test 真值。
- 周期 52 的证据：日期间隔、缺周、重复周、时间排序和历史窗口长度检查。

至少输出：

- `climate_boundary_audit.json`
- `climate_feature_drift.csv`
- `climate_source_coverage.csv`
- `climate_weekly_axis_audit.csv`
- `paper_ready/Climate数据与边界说明.md`

## 四、负例与测试

新增至少六个负例：test 包含 target 列、test 包混入真值文件、选择包出现 test 行、origin 重复、origin+horizon 越界、特征哈希与正式 Bundle 不一致。所有负例必须非 0 退出并给出具体原因。

## 五、交付要求

- 新 Release 不能覆盖第七轮 Release。
- 必须在新建 D 盘目录中从 Release 完整重放。
- 提供下载链接、字节数、ZIP SHA256、重放日志和负例日志。
- 不得修改 `辅助电脑02交付07次/` 或更早目录。
- 只提交 `3218151885-creator`，不得合并主控或 main。

## 六、退回条件

- 测试特征包仍含任何 test 真值。
- 只把文件改名，无法证明与正式 Bundle 的特征一致。
- 使用 test 真值计算漂移、相关性或模型指标。
- 把无 URL、代理日期写成已经核验的正式来源。
- Release 仍依赖仓库外缓存。

