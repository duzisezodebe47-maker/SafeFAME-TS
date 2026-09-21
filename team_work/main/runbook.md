# 主控技术模块运行说明

## 状态与边界

代码基线为 Git `c63793236d532d2ffdcccd4c7cb0c8e194f96eab`；在本地分支 `team/main-eval-20260921` 开发。协议对应既有 v4 历史滚动回测。新的三机联合实验尚未获得两位队友的样本清单、逐起点预测和置换零分布，所以不能称完成真实重训或全量逐起点复算。所有结果写入 `team_work/main/outputs/`，不覆盖旧 `outputs/safefame_v4/`。

## 环境

独立评测只用 Python 标准库。已用 D 盘 `D:\Miniconda3\python.exe`（3.13.13）运行评测和测试，不安装新依赖。历史模型重训仍须按仓库要求使用兼容 Python 3.12 的环境。

在仓库根目录执行，PowerShell 示例：

```powershell
$env:PYTHONPATH='team_work/main/src'
& 'D:\Miniconda3\python.exe' -B -m unittest discover -s team_work/main/tests -v
& 'D:\Miniconda3\python.exe' -B -m team_eval audit-v4 --repo . --out team_work/main/outputs/v4_audit.json
```

人工烟雾样例（仅检验文件接口，非真实实验）：

```powershell
& 'D:\Miniconda3\python.exe' -B team_work/main/tests/make_smoke.py team_work/main/outputs/smoke_inputs
& 'D:\Miniconda3\python.exe' -B -m team_eval freeze --task team_work/main/outputs/smoke_inputs/task.json --samples team_work/main/outputs/smoke_inputs/selection_samples.jsonl --predictions team_work/main/outputs/smoke_inputs/selection_predictions.jsonl --nulls team_work/main/outputs/smoke_inputs/row_nulls.jsonl --spec team_work/main/protocol/split_spec.json --out team_work/main/outputs/smoke_route.json
& 'D:\Miniconda3\python.exe' -B -m team_eval score --task team_work/main/outputs/smoke_inputs/task.json --samples team_work/main/outputs/smoke_inputs/test_samples.jsonl --predictions team_work/main/outputs/smoke_inputs/test_predictions.jsonl --route team_work/main/outputs/smoke_route.json --spec team_work/main/protocol/split_spec.json --out team_work/main/outputs/smoke_score.json
```

真实接入时：辅助电脑一交付每任务 `task.json`、带来源与时点审计的 `samples.jsonl`；辅助电脑二交付每候选逐起点 `predictions.jsonl` 和在决策段得到的 999 次逐行重拟合 `row_nulls.jsonl`。先单独用校准/决策文件 `freeze`，然后用测试文件 `score`。模型超参数及数值回退必须先按协议在校准段确定。任一接口字段改变都须升级协议版本并重新生成相关预测。

## 目前实际覆盖

- `audit-v4`：对 Git 跟踪的 120 行任务表、240 行候选表进行键覆盖、部分门控与数值一致性检查，并计算路线集中度；状态为 `PARTIAL_PASS`，因为没有完整预测、冻结路由、模型权重。
- 合成烟雾样例：检验跨文件 ID、哈希、时间边界、路由冻结和测试计分的流程，不作为研究结果。
- 未运行：真实三机联合实验、真实逐起点复算、120 项重训、模型权重重放、独立未来数据确认。

## 统计口径

MSE、MAE 在每个起点的 H 个目标上先取均值，再对起点取均值。相对收益为 `100*(1-MSE_selected/MSE_fallback)`。配对损失差为 `MSE_fallback-MSE_selected`。区间是按固定种子的移动块 Bootstrap 诊断；重叠预测窗口并不独立，跨任务与跨候选未做同时区间修正。任务折均值仅是描述统计，不能将 120 折当 120 个独立数据集。

## 资源

当前静态核查与单元测试为秒级、仅 CPU。实际训练资源由辅助电脑二估计；不因为历史 v4 已有 120 折而再次自动重训。
