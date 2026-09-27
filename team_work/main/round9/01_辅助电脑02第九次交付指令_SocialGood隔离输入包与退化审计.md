# 辅助电脑02第九次交付指令：SocialGood 隔离输入包与覆盖退化审计

执行分支：`3218151885-creator`  
建议交付目录：`辅助电脑02交付09次/`  
预计工期：1—2 天有效工作时间

## 一、任务定位

在不接触 Climate 测试评分的情况下，为下一领域 `SocialGood_h3_f1` 建立可独立下载、可重放、测试真值物理隔离的数据包，并专项解释“训练段文本接近或等于零、proxy 与 conservative_lag 覆盖相差 1 个起点”对模型退化的影响。

本任务是一项完整的数据工程交付，不是写说明文件。需要交出构建程序、验证程序、Release 包、负例测试和起点级审计表。

## 二、冻结输入

- 正式 Bundle 签名：`a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38`
- 冻结 spec SHA256：`a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031`
- 任务：`SocialGood_h3_f1`
- `input_len=24`，`horizon=3`，季节周期 12
- 四段边界：`[366, 458, 641, 732]`
- 预计保留起点：train 340、calibration 90、decision 181、test 89，共 700；选择侧 611，测试侧 89。程序必须自行计算并断言，不能只照抄。

## 三、必须完成的工程

1. 从正式 Bundle 构建 `selection_fit/`：只含 train/calibration/decision 的数值历史、频率、目标、语义、质量、可用性、时间网格、origin 与 metadata。
2. 构建 `test_features/`：允许 origin、时间网格、缺失掩码、语义/质量/可用性；禁止 `targets*`、`numeric_history*`、`frequency*` 和任何可从重叠窗口反推出测试真值的等价文件。
3. 生成 `mapping.csv`，逐起点记录正式 Bundle 行、origin、segment、包内行号和目标窗边界。
4. 分别对 `proxy` 与 `conservative_lag` 生成起点级文本覆盖表，明确：
   - 四段候选数、保留数、文本可用数、实际文本来源数；
   - train 段是否为零文本及其精确定义；
   - 两情景唯一覆盖翻转的 origin、来源 ID、触发规则与质量标记；
   - 无文本时 semantic、quality、mask 的实际值及模型应如何安全退化。
5. 生成来源与时间边界审计：保留原始 URL、来源 ID、开始/结束时间、可得时间代理、是否验证真实发布时间。未核验的字段必须明确标记，禁止把 `end_date` 写成真实发布时间。
6. 编写 `verify_socialgood.py`：逐文件哈希、锚、mapping、四段边界、选择切片承诺、测试目录白名单、两情景覆盖统计都必须失败即退出非 0。
7. 至少 7 个负例：测试真值混入、测试数值历史混入、频率混入、切片篡改、mapping 错位、Bundle 签名错误、冻结 spec 错误。每个负例必须真的运行验证器并被拒绝。
8. 发布 GitHub Release ZIP；重新从远端下载到全新 D 盘目录，核对字节数与 SHA256，并完整重放验证器和负例。

## 四、必须交付的文件

- `prepare_socialgood.py`、`verify_socialgood.py`、`tests/run_negative_cases.py`
- `MANIFEST.json`、`EXPECTED_OUTPUTS.json`、`truth_commitment.json`
- `mapping.csv`、`sample_audit.csv`
- `socialgood_segment_coverage.csv`
- `socialgood_scenario_flip_detail.csv`
- `socialgood_degradation_audit.json`
- `socialgood_source_evidence.csv`
- `paper_ready/SocialGood数据边界与退化说明.md`
- `README.md`、`RUNBOOK.md`、依赖清单
- `clean_replay/remote_download/` 下的命令、退出码、下载件 SHA、验证日志和负例日志

## 五、验收硬条件

- 所有文件只保存在 D 盘工作区与 GitHub；不得写入 C 盘。
- 不修改第 1—8 次历史交付目录。
- Release ZIP 不直接塞进 Git；Git 中保存固定 URL、字节数、SHA256 和远端重放证据。
- test 包出现目标、数值历史或频率即整份退回。
- 覆盖率必须给分子、分母和统计单位；“训练无文本”必须有起点级证据。
- 不运行 SocialGood 模型、不生成预测、不接触任何测试评分。
- 只提交本人分支，不合并 `team/main-eval-20260921` 或 `main`。

## 六、交付回报格式

仅回报：固定提交 SHA、目录名、Release URL、ZIP 字节数与 SHA256、验证器结论、负例通过数、四段起点数、两情景文本覆盖数、已知限制。
