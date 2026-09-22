# 03 辅助电脑二 · 交付六次

**依据**：主控 [`team_work/main/round6/02_辅助电脑二第六次交付指令_真实Agriculture门控.md`](../team_work/main/round6/02_辅助电脑二第六次交付指令_真实Agriculture门控.md)
**起点**：第五次交付 `828e7b4`　**分支**：`SHY`　**日期**：2026-09-22

> **本轮完成了真实 Agriculture 的选择期预测与 999 次置换。**
> 只交选择期证据 —— 测试段预测等主控封印路由后按公布 SHA 执行。

---

## 一、固定输入核对

| 项 | 值 | 核对 |
|---|---|---|
| Bundle Release | `data-bundle-a69821be115445265cde` | ✅ 从 Release 下载到 D 盘 |
| ZIP SHA256 | `8abde504…0f0e06` | ✅ 实测一致 |
| Bundle 签名 | `a69821be…86d38` | ✅ `read_frozen_bundle` 校验通过 |
| 冻结 spec SHA256 | `a0947a5b…6f0031` | ✅ 实测一致 |
| Bundle `schema.status` | `frozen` | ✅ |

> ⚠️ 冻结 spec 在**主控分支的 `round2/split_spec_v2.json`**。我工作区里的同名文件是**过期草稿**
> （`36c44036…`，`status=draft_not_for_training`）—— 我的 `verify_bundle` **正确拒绝了它**。
> 交付使用从主控分支取出的那份。

---

## 二、交叉验证：本侧基线 == 主控实现（在真 Bundle 上）

这是本轮最强的正确性证据。用真 Bundle 跑本侧 `numeric_baselines`，
与主控 `round6/results/agriculture_proxy_baselines/` 的清单与 CSV 逐项比对：

| 项 | 本侧 | 主控 |
|---|---|---|
| `ridge_alpha` | 10.0 | 10.0 |
| `ridge_calibration_mse` | `0.017618668697762695` | `0.017618668697762695` |
| `fit_rows` / `calibration_rows` | 177 / 43 | 177 / 43 |
| `seasonal_period` | 12 | 12 |

**逐点比对**（主控基线 CSV，4968 行）：

| 候选 | 比对点数 | max_abs_err |
|---|---|---|
| `Last` | 1656 | **0.000e+00** |
| `SeasonalNaive` | 1656 | **0.000e+00** |
| `AR-Ridge` | 1656 | **0.000e+00** |

**逐位相同**，不是"容差内"。第五轮 A.2 的等价移植在真实数据上完全命中。

---

## 三、真实选择期预测

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付六次/model/train.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --signature a69821be…86d38 --split-spec <冻结spec> \
    --segments calibration decision --candidate <候选> --output-dir evidence/cal_dec_predictions
```

| 候选 | 行数 | 说明 |
|---|---|---|
| `N` | 1656 | 纯数值 |
| `N+Q` | 1656 | 独立消融（不入门控决策） |
| `N+S+Q` | 1656 | 门控候选 1 |
| `N+S+Q+SF` | 1656 | 门控候选 2 |

1656 = 138 选择期起点 × 12 步，与主控 `origin_count: 138` 一致。
每候选产出 `*_predictions.csv` + `*_run_manifest.json`（含 α、权重哈希、分支宽度、丢弃列）。

---

## 四、🎯 真实 999 次逐行错位置换

两门控候选各 **999 次成功、0 失败**，每次重建 PCA / 尺度 / 交互 / α：

| 候选 | 观察决策损失 | **经验 p 值** | 门槛 0.025 | 循环移位 p（诊断） | 耗时 |
|---|---|---|---|---|---|
| `N+S+Q` | 0.185302 | **0.761** | ❌ 不通过 | 0.347 | 68.9 s |
| **`N+S+Q+SF`** | **0.155100** | **0.014** | ✅ **通过** | — | 136.2 s |

**判别性结果**：语义 × 频谱**显式交互**候选通过置换门槛，而纯分支相加版不通过。
这与第四轮归因审计的方向一致 —— 关键不在"加了文本"，而在**文本与频率的交互结构**。

**决策半段**（`middle=(cal_end+dec_end)//2=319`，跨界窗口剔除）：

| 半段 | 起点数 | 平均损失 |
|---|---|---|
| 前半 | 42 | 0.127755 |
| 后半 | 42 | 0.158387 |

> **路由判定权在主控**。本侧只交观察 MSE 与零分布；"总体 + 前后半段均优于回退"的完整门控由主控独立复算。

逐次日志格式：`iteration,seed,status,loss,error`，成功与失败各自留痕，
**种子绝不错配**（第三轮修复的故障注入测试覆盖）。

---

## 五、三项代码修复

| # | 要求 | 状态 |
|---|---|---|
| 1 | 只接受主控**封印**路由（`status=frozen` + 两锚 + 来源哈希） | ✅ `check_route` 要求 12 个封印字段；`status` 必须为 `frozen`；三个来源哈希必须是合法 64 位十六进制。**自造路由不能解锁测试** |
| 2 | 数值回退接口里**不存在**测试真值字段 | ✅ `numeric_baselines(features, segments, fit_targets)` —— 特征与真值**在签名上分离**；`features.targets` 必须为 None；`fit_targets` 在 train/cal 之外含真值即**硬失败** |
| 3 | 季节周期从冻结 spec 取，不默认 12 | ✅ `task_seasonal_period` 读 spec **顶层** `seasonal_periods`（Climate=52、Environment=7，默认 12 是错的）；CLI 若给出必须与注册值相等 |

---

## 六、测试：30 项全部通过

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付六次/model/tests/test_round6.py"
# 全部通过（30 项检查），退出码 0
```

覆盖：三种任务的周期取值与两种缺登记负例；接口含真值即拒绝；`fit_targets` 越界含真值即拒绝；
改测试真值后输出逐位不变；封印路由的 5 个字段逐个缺失均拒绝、`status` 非 frozen 拒绝、
哈希格式非法拒绝；数值回退旁路保持拒绝；三个入口端到端（CSV 列 == 主控 `PREDICTION_COLUMNS`、
segment 逐行、每起点每步恰一行）。

---

## 七、交付清单

| 要求 | 状态 |
|---|---|
| 三项代码修复 + 测试 | ✅ `COMPLETE` |
| 真实 `Agriculture_h12_f1/proxy` 四段读取与边界检查 | ✅ `COMPLETE` |
| 四候选 calibration/decision 逐步预测 + run manifest | ✅ `COMPLETE`（各 1656 行） |
| 两门控候选各 999 次成功置换 + 逐次日志 | ✅ `COMPLETE`（0 失败） |
| 基线交叉验证（本侧 == 主控） | ✅ `COMPLETE`（逐位相同） |
| 测试段预测 | `NOT_RUN` — 等主控封印路由并公布 SHA |
| Climate / SocialGood / Environment | `NOT_RUN` — Agriculture 闭环后按序扩展 |

**证据体积 2.1 MB，已随仓库入库**（不需另找交接位置）。

---

## 八、目录

```
03 辅助电脑二 交付六次/
├── README.md / RUNBOOK.md / TEAM_SYNC.md
├── MANIFEST.json
├── evidence/                          ← 真实证据（2.1 MB）
│   ├── cal_dec_predictions/           4 候选 × (CSV + run_manifest)
│   ├── row_null/<候选>/               逐次 CSV + 循环移位 + 两份 summary
│   └── permutation_run.log
└── model/
    ├── bundle_reader.py       + task_seasonal_period（修复 3）
    ├── numeric_fallbacks.py   ★ 特征/真值签名分离（修复 2）
    ├── predict_test.py        ★ 封印路由要求（修复 1）
    ├── ...（其余同第五轮）
    └── tests/test_round6.py   30 项检查
```

前五轮目录完整保留，未改动。
