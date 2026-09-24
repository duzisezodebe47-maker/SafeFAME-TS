# TEAM_SYNC · 模型侧第六轮同步说明

**面向**：主控　**分支**：`SHY`　**起点**：`828e7b4`　**日期**：2026-09-22
**本轮证据 `code_commit`**：`40293c7`（工作区干净）

---

## 一、状态

**三项代码修复完成；真实 Agriculture 选择期预测、999 次置换、四段边界审计、
基线逐点交叉验证、权重重演全部跑完；49 项合成测试通过。**

**只交选择期证据。** 测试段预测等主控封印路由并公布唯一 SHA256 后执行。

---

## 二、⚠️ 请先看：相对首次提交 `ce1efcc` 的三处修正

首次提交有三处会被直接退回的问题，**已在固定提交上重跑**：

1. **`code_commit` 记的是上一轮的 `828e7b4`**。原因是 `code_commit()` 只取
   `git rev-parse HEAD`，而跑证据时三处修复还没提交 —— 记录下来的提交号**不包含
   实际运行的代码**。你把它当验收锚点，这一条必然对不上。
   现在 `code_provenance` 同时记录「HEAD + 工作区是否干净 + 脏文件清单」，
   并在脏工作区上警告；本轮全部产物 `worktree_dirty=false`。
2. **缺 CPU/内存实测**（任务书第 3 条明列）。现每份 manifest/summary 带 `runtime`：
   墙钟、CPU 时间、峰值 RSS、核数、平台、Python 版本（`runtime_profile.py`，无新依赖）。
3. **两句「最强证据」只有结论没有产物**：基线逐点相同、四段边界检查。
   现补 `audit_baselines.py` 与 `audit_boundaries.py`，产物在 `evidence/` 下。

---

## 三、核心结果

| 候选 | 观察决策损失 | **经验 p** | 门槛 0.025 | 循环移位 p（诊断） |
|---|---|---|---|---|
| `N+S+Q` | 0.185302 | **0.761** | ❌ | 0.347 |
| `N+S+Q+SF` | 0.155100 | **0.014** | ✅ | 0.031 |

各 **999 次成功、0 失败**，逐次重建 PCA / 尺度 / 交互尺度 / α。
置换损失分布以**完整 999 个值**交付（`null_scores.csv`），不是分位数摘要 ——
你的 `convert_nulls` 正是消费这份 CSV，`p = (1 + #{x ≤ observed})/(999+1)` 可由它独立复算。

决策半段（`middle=(cal_end+dec_end)//2 = 319`）：`N+S+Q` 前半 0.127755 / 后半 0.158387；
`N+S+Q+SF` 前半 0.105248 / 后半 0.111527（各 42 起点）。

> **只是置换判据。** 完整资格（决策 MSE < 回退、两个半段都严格改善）请你这边复算，
> 本侧不替主控下结论。

---

## 四、交叉验证：本侧基线在主控真 Bundle 上**逐位相同**

`audit_baselines.py` 先核你清单自报的 CSV 哈希（`3acfd6d2…bef8`，实测一致），
再逐点对齐 **4968 行**：`Last` / `SeasonalNaive` / `AR-Ridge` 各 1656 点，
`exact = 1656/1656`，`max_abs_err = 0.000e+00`。标量同样一致：
α=10.0、`ridge_calibration_mse` `0.017618668697762695`、fit/cal 行 177/43、周期 12。

Bundle 的 `manifest.json` 实测 `6c889af1…0143`，与你的 `baseline_selection_manifest.json`
记录一致。

---

## 五、四段边界审计（上一轮只有一句 ✅，这轮有产物）

四段全过 `assert_grid`：train 177（24–200）、calibration 43（212–254）、
decision 95（266–360）、test 42（372–413）。选择期合计 **138**，与你的 `origin_count` 一致。
被排除的起点是契约要求：头部 24 个（`origin < input_len=24`）、段间各 11 个（目标窗口跨界）。

---

## 六、权重重演

主控第六轮任务书把「权重重演」列为验收项，且把 `code_commit` 当锚点 ——
这两件事只有一起成立才有意义。`audit_replay.py` 对四个交付候选各跑两次 `train.py`：
两次**逐字节相同**，且与**已交付的** CSV/manifest 一致（`weight_hash`、
`alpha_by_group`、`branch_widths` 全等）。产物 `evidence/replay/replay_check.json`。

---

## 七、⚠️ 两处需要你注意（与前一轮相同）

### 1. 冻结 spec 的位置

冻结 spec（`a0947a5b…`）在**主控分支的 `team_work/main/round2/split_spec_v2.json`**。
模型侧工作区里同名文件是**过期草稿**（`36c44036…`，`status=draft_not_for_training`），
我的 `verify_bundle` 正确拒绝了它。**建议**：在协议里写明唯一路径，或改名区分。

### 2. `samples.csv` 的段序

`load_prediction_grid` 用**有序列表**比较，模型侧已按 Bundle 行序导出，与段序无关地对齐。
请确认数据侧产出的段序 —— 现两侧一致。

---

## 八、⚠️ 本轮新增的两处**你必须知道**的变化

### 1. 预测 CSV 的哈希变了（内容没变）

交付的 `*_proxy_predictions.csv` 与你的烟雾文件**不再逐字节相同**：
`y_pred` 逐位不变，但 `code_commit` 列由 `828e7b4…` 改成 `40293c7…`。
你的 `master_both_gate_grid_audit.json` 里记的两个烟雾 CSV 哈希
（`a13b35ae…d697fe06` / `5f37b80a…cb8a8ed2`）对应上一版；正式交付请以
本目录 `*_run_manifest.json` + `evidence/replay/` 为准。

**可核对**：把两版预测 CSV 逐列相比，**只有 `code_commit` 一列不同**；
零分布 `null_scores*.csv` 与上一版**逐字节相同**（同种子、同代码语义）。

### 2. 模型文件集由 11 个变为 16 个

新增 5 个审计 / 工具模块：`audit_boundaries.py`、`audit_baselines.py`、
`audit_replay.py`、`runtime_profile.py`、`make_manifest.py`。
你 `master_model_reader_compare.json` 里的 `model_delivery_manifest_files: 11`
是上一轮快照，重建审计时请以新 `MANIFEST.json` 为准。

---

## 九、未跑范围

| 项 | 状态 |
|---|---|
| 测试段预测 | `NOT_RUN` — 等主控封印路由 + 公布 SHA256 |
| Climate / SocialGood / Environment | `NOT_RUN` — Agriculture 闭环后按序扩展 |
| 保守滞后情景 | `NOT_RUN` |

证据约 3 MB 已随仓库入库（`evidence/`），附逐文件 SHA256 与两个合并哈希
（`MANIFEST.json`，由 `model/make_manifest.py` 生成，规则写死可复算）。

---

## 十、⚠️ 2026-09-24 补遗（第七轮准备时发现并修复）

### 修的是什么

本目录 `model/predict_test.py` **原本残缺**：359 行，`main()` 在主控基线分支中途断掉，
**门控候选预测路径、`PredictionWriter` 写出、`return 0`、`__main__` 守卫全部缺失**
（第五次同名文件 386 行且完整）。这是**静默失败** —— 文件仍能 `ast.parse`，
且没有 `__main__` 守卫，按第七轮任务书跑那条命令会「退出码 0 但什么都不写」。

**你的验收没抓到，因为**：`tests/test_round6.py` 只从 `predict_test` 导入了
`RouteRejected/check_route/spec_candidate_lists`；端到端 `run_entry` 跑了
train / permutation_entry / audit_boundaries / audit_tables，**唯独没跑
`predict_test.main()`**。你在固定工作树上复跑的正是同一套 49 项。
第七轮已把该入口的端到端 + 6 类负例补进测试。

### 你记录的两个哈希，现在是什么状态

| 你记录的值 | 现在 |
|---|---|
| `evidence_manifest_sha256` = `140dcb255b6c7097…9798dace` | **完全一致** —— 证据文件一个字节都没动 |
| `model_manifest_sha256` = `187eb12b8efff8fd…50a54234` | **已变** → `43fce4ea95436194…9cc219712`，因为模型文件集补齐 1 个 + 新增 4 个（测试与审计模块） |

**证据未动**这一点可独立复核：`evidence/` 下 27 个文件的合并哈希仍等于你记录的值，
说明第六次交付的选择期预测、999 次置换、边界/基线/重放证据**全部原样**
（它们走 `train.py` / `permutation_entry.py`，不经过 `predict_test.py`）。

### 两点提醒

1. 本目录在验收后被改动过（修复提交 `4fd2c0c`）。若你的流程要求"已验收提交不再变动"，
   请把它视为一次**已申报的修复提交**。
2. `03 辅助电脑二 交付七次/model/` 与本目录 `model/` **逐字节相同**（20 个文件，已核对），
   第七轮交付自包含。
