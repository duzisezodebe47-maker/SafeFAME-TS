# TEAM_SYNC · 模型侧第六轮同步说明

**面向**：主控（Jerry ye）
**分支**：`SHY`　**起点**：`828e7b4`　**日期**：2026-09-22

---

## 一、状态

**三项代码修复完成；真实 Agriculture 选择期预测与 999 次置换已跑完；30 项合成测试通过。**

**只交选择期证据。** 测试段预测等主控封印路由并公布唯一 SHA256 后执行。

---

## 二、🎯 核心结果：两条门控候选分道扬镳

| 候选 | 观察决策损失 | **经验 p** | 门槛 0.025 | 循环移位 p |
|---|---|---|---|---|
| `N+S+Q` | 0.185302 | **0.761** | ❌ | 0.347 |
| **`N+S+Q+SF`** | **0.155100** | **0.014** | ✅ | — |

各 **999 次成功、0 失败**，逐次重建 PCA / 尺度 / 交互尺度 / α。

**判别性**：语义 × 频谱**显式交互**通过置换门槛，纯分支相加版不通过。
方向与 docs/23 归因审计一致 —— 起作用的是**交互结构**，不是"加了文本"。

决策半段（`middle=(cal_end+dec_end)//2 = 319`，跨界窗口剔除）：前半 42 起点 / 0.127755，
后半 42 起点 / 0.158387。

> **路由判定权在主控。** 本侧只交观察 MSE 与零分布，完整门控（总体 + 前后半段均优于回退）请你独立复算。

---

## 三、交叉验证：本侧基线在主控真 Bundle 上**逐位相同**

| 项 | 本侧 | 主控 |
|---|---|---|
| `ridge_alpha` | 10.0 | 10.0 |
| `ridge_calibration_mse` | `0.017618668697762695` | `0.017618668697762695` |
| `fit_rows` / `cal_rows` | 177 / 43 | 177 / 43 |

逐点比对主控基线 CSV 的 **4968 行**：`Last` / `SeasonalNaive` / `AR-Ridge` 全部
**`max_abs_err = 0.000e+00`**。第五轮 A.2 的等价移植在真数据上完全命中。

---

## 四、三项修复

1. **封印路由**：`check_route` 现要求 12 个字段齐备 —— `status` 必须 `frozen`，
   两锚 + `selection_manifest_sha256` / `refit_review_sha256` / `inputs_sha256`
   必须是合法 64 位十六进制。**自造路由不能解锁测试预测。**
2. **真值与特征签名分离**：`numeric_baselines(features, segments, fit_targets)` ——
   `features.targets` 必须为 `None`；`fit_targets` 在 train/cal 之外含真值即**硬失败**。
   测试同时验证「输出不变」与「接口中确实没有真值字段」。
3. **季节周期来自 spec 顶层**：`Climate=52`、`Environment=7`，**默认 12 是错的**。
   `task_seasonal_period` 读 spec 顶层 `seasonal_periods`；未登记即拒绝；
   CLI 若给出必须与注册值相等。

---

## 五、⚠️ 两处需要你注意

### 1. 冻结 spec 的位置

冻结 spec（`a0947a5b…`）在**主控分支的 `team_work/main/round2/split_spec_v2.json`**。
模型侧工作区里这个路径的同名文件是**过期草稿**（`36c44036…`，`status=draft_not_for_training`）
—— 我的 `verify_bundle` 正确拒绝了它。

**建议**：在协议里写明冻结 spec 的**唯一路径**，或改名区分（例如 `split_spec_frozen.json`），
否则协作者很容易拿到过期版本而不知道。

### 2. `samples.csv` 的段序

第五轮我修复了导出顺序（按 Bundle 行序而非段名顺序），因为你的 `load_prediction_grid`
用**有序列表比较**。请确认数据侧产出的 `samples.csv` 段序 —— 模型侧现已与 Bundle 行序一致，
无论怎么排列都对齐。

---

## 六、未跑范围

| 项 | 状态 |
|---|---|
| 测试段预测 | `NOT_RUN` — 等主控封印路由 + 公布 SHA256 |
| Climate / SocialGood / Environment | `NOT_RUN` — Agriculture 闭环后按序扩展 |
| 保守滞后情景 | `NOT_RUN` |

证据 2.1 MB 已随仓库入库（`evidence/`），无需另找交接位置。
