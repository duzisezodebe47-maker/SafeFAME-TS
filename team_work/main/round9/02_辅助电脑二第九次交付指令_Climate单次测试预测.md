# 辅助电脑二第九次交付指令：封印路由后的 Climate 单次测试预测

执行分支：`SHY`  
建议交付目录：`03 辅助电脑二 交付九次/`  
预计工期：0.5—1 天有效工作时间

## 一、唯一授权路线

- 任务：`Climate_h4_f2`
- 情景：`proxy`
- 封印结果：`selected=numeric_fallback`，`fallback=N`
- 唯一允许请求的候选：`N`
- 路由文件：主控分支 `team_work/main/round9/results/sealed_route.json`
- 路由 SHA256：`b3bce4713ff44485c9aa0712c5774ffe449045fb8186b39ef592298e0051e36a`
- Bundle 签名：`a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38`
- 冻结 spec SHA256：`a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031`
- 选择期清单：第八次交付 `evidence/cal_dec_predictions/N_run_manifest.json`

运行前必须从主控分支取得路由并重新计算 SHA256。任何锚不一致都停止，不得自行重建路由。

## 二、唯一允许的主流程

使用第八次交付已经通过 131 项测试的 `model/predict_test.py`，候选参数必须是 `N`。输出目录在首次正式运行前必须不存在。

```powershell
python -B "03 辅助电脑二 交付八次/model/predict_test.py" `
  --bundle <正式Bundle目录> `
  --task Climate_h4_f2 `
  --scenario proxy `
  --signature a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38 `
  --split-spec <主控split_spec_v2.json> `
  --route <主控round9/sealed_route.json> `
  --route-sha256 b3bce4713ff44485c9aa0712c5774ffe449045fb8186b39ef592298e0051e36a `
  --selection-manifest "03 辅助电脑二 交付八次/evidence/cal_dec_predictions/N_run_manifest.json" `
  --candidate N `
  --seed 2026 `
  --output-dir "03 辅助电脑二 交付九次/test_prediction"
```

预计测试起点 124 个、horizon 4、CSV 496 行。程序输出必须自行断言。

## 三、测试隔离与一次性规则

1. 不得读取 test 段 `targets*`，不得计算 MSE、MAE、RMSE、R²、残差或候选排名。
2. 不得生成 Last、SeasonalNaive、AR-Ridge、N+Q、N+S+Q、N+S+Q+SF 的测试预测。
3. alpha、PCA、尺度、数值分支定义全部沿用选择期 `N` 清单；允许按既定流程使用 train+calibration+decision 真值 refit。
4. 不得根据输出形状做人工平滑、裁剪或替换；出现非有限值则失败并保留日志。
5. 正式运行失败时，只能修复环境或契约错误；每次尝试都保留命令、时间、退出码和错误。不得删除失败记录后伪装成首次成功。
6. 允许第二次确定性重放到独立临时目录，只用于比较预测字节与权重哈希；不得用于挑选输出。

## 四、必须交付的证据

- 唯一 `N` 测试预测 CSV，496 行，键唯一且仅含 test 段。
- `run_manifest.json`：任务、候选、四锚、选择期清单 SHA、代码提交、工作区状态、alpha、权重哈希、行数、运行资源。
- `route_acceptance.json`：原样记录路由 SHA、status、selected、fallback、selection/refit 锚。
- `test_isolation_audit.json`：记录 IO 监测方法、读取路径、明确 test 真值访问数为 0。
- `replay_check.json`：第二次确定性重放与正式 CSV 逐字节相同，权重哈希一致。
- `attempt_ledger.json` 与原始日志。
- `MANIFEST.json`、`README.md`、`RUNBOOK.md`。

## 五、测试与负例

保留第八轮 131 项测试，并增加至少以下断言：

1. 路由 SHA 错误时拒绝。
2. `--candidate` 不是 `N` 时拒绝。
3. 路由 `fallback` 不是 `N` 时拒绝。
4. 选择期清单的 alpha、weight_hash 或 candidate 被改动时拒绝。
5. test 真值路径被访问时拒绝。
6. 输出目录已存在时拒绝覆盖。
7. 输出不是 496 行、键重复、origin 越界或有非有限值时拒绝。
8. 交付目录出现第二个候选测试预测时完备性闸门失败。

## 六、分支规则与退回条件

只修改 `03 辅助电脑二 交付九次/`，不修改前八次历史目录；只推送 `SHY`，不合并主控或 main。出现多个测试候选、任何测试评分、路由锚不符、真值读取、代码证据无法固定、预测无法重放，整份退回。

## 七、交付回报格式

仅回报：固定提交 SHA、目录、预测 CSV SHA256、manifest SHA256、测试通过数、测试起点与行数、真值读取数、正式运行与重放退出码。不要报告任何测试性能。
