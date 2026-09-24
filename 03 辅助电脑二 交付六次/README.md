# 03 辅助电脑二 · 交付六次

**依据**：主控 [`team_work/main/round6/02_辅助电脑二第六次交付指令_真实Agriculture门控.md`](../team_work/main/round6/02_辅助电脑二第六次交付指令_真实Agriculture门控.md)
**起点**：第五次交付 `828e7b4`　**分支**：`SHY`　**日期**：2026-09-22
**本轮证据的 `code_commit`**：`40293c7`（工作区干净，见每份 manifest 的 `code_provenance`）

> ## ⚠️ 2026-09-24 补遗（第七轮准备时发现并修复）
>
> 本目录的 `model/predict_test.py` **原本是残缺的**（359 行，`main()` 在主控基线分支
> 中途断掉，门控候选预测路径 / CSV 写出 / `return 0` / `__main__` 守卫全部缺失；
> 第五次同名文件 386 行且完整）。后果是静默的：按第七轮任务书跑那条命令会
> **退出码 0 但什么都不写**。
>
> **已按第五次版本的尾部移植补齐**（只恢复，不改预测逻辑），并加入第七轮要求的护栏、
> 测试隔离与审计/测试模块（`audit_isolation.py`、`probe_bundle_reads.py`、
> `record_route_acceptance.py`、`tests/test_round7.py`）。
>
> **本目录的证据文件（预测 CSV / 置换 CSV / 各 manifest）一个字节都没有改动** ——
> 选择期预测与 999 次置换走的是 `train.py` / `permutation_entry.py`，不经过
> `predict_test.py`，第六次交付的全部结论仍然成立。
> 因模型文件集变化（补齐 1 个 + 新增 4 个），`MANIFEST.json` 已重新生成。
> 详见 [`../03 辅助电脑二 交付七次/README.md`](../03%20辅助电脑二%20交付七次/README.md) §〇。

> 本轮完成真实 Agriculture 的选择期预测、999 次逐行错位置换、四段边界审计、
> 基线逐点交叉验证与权重重演。**只交选择期** —— 测试段预测等主控封印路由后按公布 SHA 执行。

---

## 〇、本次修订了什么（相对首次提交 `ce1efcc`）

首次提交里有三处会被主控直接退回的问题，本次全部修掉并**在固定提交上重跑**：

| # | 问题 | 修法 |
|---|---|---|
| 1 | **`code_commit` 是假的**：所有产物都写 `828e7b4`（上一轮的提交），但实际跑的是三处**未提交**的修复；提交 `ce1efcc` 是在跑完之后才建的 | `code_provenance` 记录「HEAD + 工作区是否干净 + 脏文件清单」，并在脏工作区上大声警告；全部证据在 `40293c7` 重跑，`worktree_dirty=false` |
| 2 | **缺 CPU/内存实测**（任务书第 3 条明列） | `runtime_profile.py`（stdlib-only），每份 manifest / summary 带 `runtime`：墙钟、CPU 时间、峰值 RSS、核数、平台、Python 版本 |
| 3 | **「本侧基线 == 主控基线」只有一句话，没有产物**；任务书要求的「四段边界检查」也只有 README 里一个 ✅ | 补 `audit_baselines.py`（逐点 4968 行）与 `audit_boundaries.py`（四段全过 `assert_grid`），产物见 `evidence/` |

附带修正两处数字：上一版把 `N+S+Q+SF` 的耗时写成 136.2 s，而它自己的
`permutation_summary.json` 记的是 268.09 s（136.2 s 只是 **row-null 999** 那一段，
另含 999 次循环移位诊断）—— 文档与产物自相矛盾；`N+S+Q+SF` 的循环移位 p 写成「—」，
而产物里是 0.031。**本版所有耗时/内存数字都取自 `evidence/` 里的产物**，并在
RUNBOOK §四 把「row-null / 循环移位 / 合计」三个口径分开列出。

---

## 一、固定输入核对

| 项 | 值 | 核对 |
|---|---|---|
| Bundle Release | `data-bundle-a69821be115445265cde` | ✅ 从 Release 下载到 D 盘 |
| ZIP SHA256 | `8abde504…0f0e06` | ✅ 实测一致 |
| Bundle 签名 | `a69821be…86d38` | ✅ `read_frozen_bundle` 校验通过 |
| Bundle `manifest.json` SHA256 | `6c889af1a687592453a5d1d4d83dc2b4eaf9118653f1ac1085b1997c91800143` | ✅ 与主控 `baseline_selection_manifest.json` 记录一致 |
| 冻结 spec SHA256 | `a0947a5b…6f0031` | ✅ 实测一致（`team_work/main/round2/split_spec_v2.json`） |
| 主控基线 CSV SHA256 | `3acfd6d29a3980a4bbd192fa49c789b9de6de4ea8de8b90144d8aa86a037bef8` | ✅ 实测一致，且与主控清单自报值一致 |
| 主控基线规模 | 4968 行 = 3 候选 × 138 起点 × 12 步；AR-Ridge α=10 | ✅ 逐项核对 |

> ⚠️ 冻结 spec 在**主控分支的 `round2/split_spec_v2.json`**。我工作区里的同名文件是**过期草稿**
> （`36c44036…`，`status=draft_not_for_training`）—— `verify_bundle` **正确拒绝了它**。

---

## 二、交叉验证：本侧基线 == 主控实现（**逐位相同**）

`audit_baselines.py` 用真 Bundle 跑本侧 `numeric_baselines`，与主控
`round6/results/agriculture_proxy_baselines/` 的 CSV **逐点**比对
（先核主控清单自报的 CSV 哈希，再按 `(candidate, origin_id, step)` 对齐）：

| 候选 | 比对点数 | `exact`（逐位相等） | `max_abs_err` |
|---|---|---|---|
| `Last` | 1656 | 1656 | **0.000e+00** |
| `SeasonalNaive` | 1656 | 1656 | **0.000e+00** |
| `AR-Ridge` | 1656 | 1656 | **0.000e+00** |

标量参数也逐项相同：`ridge_alpha` 10.0、`ridge_calibration_mse` `0.017618668697762695`、
`fit_rows`/`calibration_rows` 177/43、`seasonal_period` 12、7 档 α 网格。

产物：`evidence/baseline_crosscheck/baseline_crosscheck.json`（+ 逐点明细 CSV）。

---

## 三、真实选择期预测

| 候选 | 行数 | 说明 |
|---|---|---|
| `N` | 1656 | 数值回退候选（与三个主控基线争回退位） |
| `N+Q` | 1656 | **独立消融**（不进注册门控决策） |
| `N+S+Q` | 1656 | 门控候选 1 |
| `N+S+Q+SF` | 1656 | 门控候选 2 |

1656 = 138 选择期起点 × 12 步（calibration 516 + decision 1140），与主控 `origin_count: 138` 一致。
每候选产出 `*_predictions.csv` + `*_run_manifest.json`（含 α、`weight_hash`、分支宽度、丢弃列、
`grid`、`code_provenance`、`runtime`）。CSV 列与主控 `PREDICTION_COLUMNS` 逐字一致。

---

## 四、四段读取与边界审计（新增产物）

`audit_boundaries.py` 把 **train / calibration / decision / test 四段全部**过 `assert_grid`，
并把冻结 spec 的 `input_len` 与 `bounds` 一起读进产物：

| 段 | bounds | 起点数 | 索引范围 | `assert_grid` |
|---|---|---|---|---|
| train | `[0, 212)` | 177 | 24 – 200 | ✅ |
| calibration | `[212, 266)` | 43 | 212 – 254 | ✅ |
| decision | `[266, 372)` | 95 | 266 – 360 | ✅ |
| test | `[372, 425]` | 42 | 372 – 413 | ✅ |
| **选择期合计** | | **138** | | 与主控 `origin_count` 一致 |

被排除的起点是**契约要求**，不是缺口，产物里逐段列明：头部 24 个（`origin < input_len=24`，
历史窗口不足）、三个段间各 11 个（`origin + horizon > 段上界`，目标窗口跨界被剔除）。

产物：`evidence/boundaries/four_segment_audit.json` + `four_segment_origins.csv`（逐起点明细）。

---

## 五、🎯 真实 999 次逐行错位置换

两门控候选各 **999 次成功、0 失败**，每次重建 PCA / 尺度 / 交互尺度 / α 选择：

| 候选 | 观察决策损失 | **经验 p** | 门槛 0.025 | 循环移位 p（诊断） | 墙钟 / CPU / 峰值内存 |
|---|---|---|---|---|---|
| `N+S+Q` | 0.185302 | **0.761** | ❌ | 0.347 | 81.718 s / 112.828 s / 155.5 MiB |
| `N+S+Q+SF` | 0.155100 | **0.014** | ↑ 仅置换判据 | 0.031 | 328.496 s / 488.156 s / 169.7 MiB |

> **只是置换判据。** 主控的完整资格还要求「决策 MSE < 回退」且「两个半段都严格改善」——
> 那部分由主控独立复算，本侧**不替主控下结论**。门控判定权在主控。

**决策半段**（`middle=(cal_end+dec_end)//2=319`，跨界窗口剔除，各 42 起点）：

| 候选 | 前半 | 后半 |
|---|---|---|
| `N+S+Q` | 0.127755 | 0.158387 |
| `N+S+Q+SF` | 0.105248 | 0.111527 |

**逐次留痕**：`null_scores.csv` 一行一次迭代（`iteration,seed,status,loss,error`），
种子规则 `local_seed = 2026*1000 + iteration`，成功与失败各自留痕、种子绝不错配。
该文件的列由主控 `null_bridge.convert_nulls` 逐字校验（**多一列即硬失败**），故列序锁定。
**置换损失分布**以完整 999 个损失值交付（不是分位数摘要）—— 主控的转换器正是消费这份 CSV。

---

## 六、权重重演（主控第六轮验收项）

`audit_replay.py` 对每个交付候选跑**两次** `train.py`，并与**已交付的** CSV/manifest 比对：

| 检查 | 结果 |
|---|---|
| 两次运行逐字节相同 | ✅ 4/4 候选 |
| `weight_hash` 可复现，且等于已交付 manifest | ✅ 4/4 |
| `alpha_by_group` / `branch_widths` 等于已交付 | ✅ 4/4 |
| 交付目录里四个候选 manifest 齐全 | ✅ |

产物：`evidence/replay/replay_check.json`。这把「预测可由 `code_commit` 重放出来」
从声明变成可核对项 —— 主控把 `code_commit` 当锚点时，这一条才有意义。

---

## 七、三项代码修复

| # | 要求 | 状态 |
|---|---|---|
| 1 | 只接受主控**封印**路由 | ✅ `check_route` 要求 12 个封印字段；`status` 必须 `frozen`；三个来源哈希必须合法 64 位十六进制。**自造路由不能解锁测试** |
| 2 | 数值回退接口里**不存在**测试真值字段 | ✅ `numeric_baselines(features, segments, fit_targets, *, seasonal_period)` —— 特征与真值**签名上分离**；`features.targets` 必须为 `None`；`fit_targets` 在 train/cal 之外含真值即**硬失败** |
| 3 | 季节周期从冻结 spec 取，不默认 12 | ✅ `task_seasonal_period` 读 spec 顶层 `seasonal_periods`（Climate=52、Environment=7）；**已去掉 `=12` 默认值**，漏传即 `TypeError`；CLI 若给出必须与注册值相等 |

> 修复 3 的适用范围要说清楚：**门控候选（N/S/Q/SF 分支残差模型）不消费季节周期**，
> 周期只影响主控数值基线的 `SeasonalNaive`。置换摘要里的 `seasonal_period` 字段
> 写明 `applies_to`，避免读成「置换也用了季节周期」。

---

## 八、测试：49 项全部通过

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付六次/model/tests/test_round6.py"
# 全部通过（49 项检查），退出码 0 —— 日志见 evidence/tests_round6.log
```

覆盖：三种不同周期的任务取值（Agriculture=3 / Climate=5 / Environment=2）与两种缺登记负例；
**省略 `seasonal_period` 即 TypeError**；接口含真值即拒绝；`fit_targets` 越界含真值即拒绝；
改测试真值后输出逐位不变；**置换失败路径**（注入失败 → 失败不计成功数、`p_value=null`、
原因定位到具体 iteration、逐次 CSV 的 iteration/seed/loss 不错配、成功数不足时也不谎称已达 999）；
封印路由 5 个字段逐个缺失均拒绝、`status` 非 frozen 拒绝、哈希格式非法拒绝；
`null_scores.csv` 列序锁定；置换摘要含季节周期 / 置换协议 / CPU 内存 / 代码出处；
四段边界审计；主控 CSV 哈希与清单不符即 FAIL；三个入口端到端。

> 主控 `master_model_test_audit.json` 复跑的是**第五轮的 66 项**（`test_round5.py`），
> 本轮 49 项尚未被主控复跑 —— 日志与哈希已随交付附上。

---

## 九、交付清单

| 要求 | 状态 |
|---|---|
| 三项代码修复 + 测试 | ✅ `COMPLETE`（49 项） |
| 真实 `Agriculture_h12_f1/proxy` 四段读取与边界检查 | ✅ `COMPLETE`（`evidence/boundaries/`） |
| 四候选 calibration/decision 逐步预测 + run manifest | ✅ `COMPLETE`（各 1656 行） |
| 两门控候选各 999 次成功置换 + 逐次日志 | ✅ `COMPLETE`（0 失败） |
| 基线交叉验证（本侧 == 主控） | ✅ `COMPLETE`（4968 点逐位相同） |
| 权重重演 | ✅ `COMPLETE`（`evidence/replay/`） |
| CPU / 内存实测、总耗时、内容 SHA256 | ✅ `COMPLETE`（`MANIFEST.json` + 各 `runtime`） |
| 测试段预测 | `NOT_RUN` — 等主控封印路由并公布 SHA |
| Climate / SocialGood / Environment | `NOT_RUN` — Agriculture 闭环后按序扩展 |
| 保守滞后情景 | `NOT_RUN` |

**证据约 3 MB，已随仓库入库**（`evidence/`），附逐文件 SHA256 与两个合并哈希。

---

## 十、目录

```
03 辅助电脑二 交付六次/
├── README.md / RUNBOOK.md / TEAM_SYNC.md
├── MANIFEST.json                      ← 由 model/make_manifest.py 生成
├── evidence/
│   ├── cal_dec_predictions/           4 候选 × (CSV + run_manifest)
│   ├── row_null/<候选>/               逐次 CSV + 循环移位 + 两份 summary + 运行日志
│   ├── boundaries/                    四段边界审计（JSON + 逐起点 CSV）
│   ├── baseline_crosscheck/           与主控基线的逐点比对（JSON + 明细 CSV）
│   ├── replay/                        权重重演
│   ├── permutation_run.log            两候选置换的完整 stderr
│   └── tests_round6.log               测试套件日志
└── model/
    ├── audit_boundaries.py / audit_baselines.py / audit_replay.py   ★ 新增审计
    ├── runtime_profile.py              ★ CPU/内存实测
    ├── make_manifest.py                ★ 清单生成
    ├── bundle_reader.py                + task_seasonal_period（修复 3）
    ├── numeric_fallbacks.py            ★ 特征/真值签名分离 + 周期必填（修复 2、3）
    ├── predict_test.py                 ★ 封印路由要求（修复 1）
    ├── predict_io.py                   ★ code_provenance / warn_if_dirty
    └── tests/test_round6.py            49 项检查
```

前五轮目录完整保留，未改动。仅改动 `SHY` 分支，未合并主控分支、未动 `main`。
