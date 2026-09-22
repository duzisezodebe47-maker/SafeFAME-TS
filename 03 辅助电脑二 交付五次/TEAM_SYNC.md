# TEAM_SYNC · 模型侧第五轮同步说明

**面向**：主控（Jerry ye）、数据侧（3218151885-creator）
**分支**：`SHY`　**起点**：第四次交付 `51a2031`　**日期**：2026-09-22

---

## 一、状态

**A 部分五项修复完成，66 项合成测试通过；B 部分仍阻塞于正式 Bundle 未交付。**

主控的 `Freeze four audited numerical snapshots` 已完成 —— 解锁链推进了一步，
模型侧已记下冻结 spec 基准 `a0947a5b…f0031`，Bundle 到位后立即用它核对。

---

## 二、A 部分五项：主控指出的问题全部属实

| 编号 | 问题 | 性质 |
|---|---|---|
| **A.1** | `numeric_fallback` 时允许 `GATE_CANDIDATES` 作为请求值却执行回退模型 —— **旁路** | 请求 `N+S+Q` 会拿到 `AR-Ridge` 的预测，入口不报错 |
| **A.2** | 模型侧 `AR-Ridge` 与主控 `numeric_baselines` **是两个不同模型** | 主控选路时用一个、模型出预测用另一个 → **结果系统性错位** |
| **A.3** | 入口对所有模型要求选择期清单，但主控基线**不产出该清单** | 真实回退路径**无法运行** |
| **A.4** | 无锚路由被记录为"缺口"后**继续放行** | 对 smoke 可以，对**正式**不行 |
| **A.5** | 只在注释里声称"不使用测试真值" | 没有强制手段，也没有故障测试 |

### A.1 已修：严格相等

```python
# 第四轮（旁路）
if candidate not in GATE_CANDIDATES and candidate != fallback:
    raise ...
return fallback

# 第五轮
if candidate != fallback:
    raise RouteRejected(...)
```

测试从 `predict_test.main()` **入口**传入门控候选，断言非零退出**且无预测文件产出**。

### A.2 已修：逐行等价移植 + 逐点比对

| | 第四轮（错） | 第五轮（对齐主控） |
|---|---|---|
| 输入 | 末 8 个滞后 | **全部数值历史窗口** |
| α 选择 | 训练段内部 80/20 | **校准段**，7 档 |

测试把主控实现**抄录为参考函数**逐点比对：α、校准 MSE、三个基线的逐起点逐步输出、
拟合行数 —— 全部在相对容差 **1e-10** 内一致。

### A.3 已修：主控基线不虚构模型侧清单

基线（`Last`/`SeasonalNaive`/`AR-Ridge`）按"路由 + 主控固定配置"验证；
`N` 仍是模型候选型回退，保留其选择期清单路径。

### A.4 已修：两锚必填

`check_route` 要求路由**同时**携带 `split_spec_sha256` 与 `bundle_signature`，
任一缺失或不一致即拒绝。

### A.5 已修：真值根本传不进去

`to_inference_bundle` 构建仅含特征的视图。故障测试：**改掉磁盘上的测试真值并重新签名**
（不重新签名的话 `verify_bundle` 会先拦下，测到的就只是签名校验），
预测仍**逐位不变**；删除真值文件后同样不变。

### 复核任务书后补的两处

- **A.3**：`N` 的**独立路径**此前没有单独测（它与主控基线走不同分支）。
  新增端到端测试：有清单时成功、缺清单时拒绝、标签为 `numeric_fallback_candidate`。
  顺带修了一个标签 bug —— `N` 原先被硬编码标成 `"gate_candidate"`。
- **A.5**：原先只测"预测不变"，而模型是篡改前就拟合好的，**等于没测到选路是否依赖真值**。
  新增：用篡改真值后的 Bundle **重新跑完整选路**，断言 α 与权重都不变。
  第一次写这个测试时把**整段数组**替换了（train/cal 的真值也被改掉），
  诊断显示四段全变 —— 改为**只改 test 段的行**后通过。

### 再审一次：两处收紧

- **A.2 改为分段比对**：任务书点名"在 calibration/decision/test 比对"，原先只报全量。
  全量通过不排除某一段整体偏移。现按段分别出结论 + 保留全量结论。
- **候选合法性改为读冻结 spec**：`check_route` 原先用硬编码的 `GATE_CANDIDATES`。
  若主控改了 spec 的 `gate_candidates`，入口会放行协议里已不是候选的名字 ——
  **静默分歧**。现从 spec 读 `gate_candidates` / `numeric_fallback_candidates`，
  任一为空即拒绝。

另核对了 A.2 参考实现的**抄录忠实度**（与主控源码逐行比对），确认无误 ——
否则测试只是"自洽"，两边都错也看不出来。

### 读主控代码后确认的交接链路与一处修复

链路：`模型 predictions.csv` → `run_bridge.py export-stage` → `stage 目录(JSONL)`
→ `seal_route.py` → 封存路由。

兼容性核对**全部通过**：`PREDICTION_COLUMNS`（14 列）、`SCALE`、
`SEGMENTS`（全名）、`config_sha256`（64 hex）、`code_commit`（40 hex）
都与本侧一致。

**但发现并修复了一处**：`load_prediction_grid` 的最终校验是
`grid != expected_grid` —— **列表相等，顺序也必须一致**，
而 `expected_grid` 按 Bundle `samples` 的**原始行序**生成。
本侧原先按**段名顺序**拼接，若 samples 里段序不同会被判 `order_bad`。
现改为按 Bundle 行序导出。**这一点建议主控确认**：数据侧产出的
`samples.csv` 是否按 train/cal/dec/test 连续排列。

### 最终检查：选择期首次导出此前**从未被真正执行过**

三个入口此前只跑过 `--help` 与拒绝分支。补上完整路径端到端后，立刻抓到一个
**会让真实链路卡死的 bug**：`predictions.csv` 的 `segment` 列装的是整个数组
转成的字符串，而不是逐行段名 —— 主控 `load_prediction_grid` 会整表拒绝。

根因是一次**静默失败的字符串替换**（搜索串带了行首空格，实际不在行首），
`str.replace` 没匹配到就返回原串，脚本也没加断言。

现已修复并加入 15 项长期端到端检查。**请主控在收到选择期预测时，先用
`run_bridge.py validate-predictions` 单独验一次列与段名** —— 这条路径是刚被打通的。

---

## 三、测试：66 项全部通过

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付五次/model/tests/test_round5.py"
# 全部通过（66 项检查），退出码 0
```

---

## 四、B 部分：等正式 Bundle

```
主控冻结四份快照 ✅ → 数据侧生成正式 Bundle → 模型侧真实运行
                            ↑ 当前卡在这里
```

模型侧已就绪：用冻结 spec SHA256 `a0947a5b…f0031` 核对 Bundle 后即可开跑
`Agriculture_h12_f1/proxy` 的 N / N+Q / N+S+Q / N+S+Q+SF 选择期预测与两候选 999 次置换。
**只先交选择期证据**，等你冻结并哈希路由后再生成测试预测。

本轮**没有**任何真实预测、零分布或 p 值。

---

## 五、两处需要主控确认

1. **`SeasonalNaive` 的季节周期**：模型侧由 `--seasonal-period` 传入（默认 12）。
   若四任务各有规定周期（如 Climate 52、Environment 7），请告知，以免与主控选路不一致。
2. **`AR-Ridge` 的 α 网格来源**：本侧沿用协议 `alpha_grid`（7 档）。
   若主控 `numeric_baselines` 实际用的是别的网格，两边会选出不同的 α。
