# 03 辅助电脑二 · 交付八次

**依据**：主控 [`team_work/main/round8/02_辅助电脑二第八次交付指令_Climate选择期门控实验.md`](../team_work/main/round8/02_辅助电脑二第八次交付指令_Climate选择期门控实验.md)
**分支**：`SHY`　**日期**：2026-09-25
**本轮证据的 `code_commit`**：见各 manifest 的 `code_provenance`（`worktree_dirty=false`）

> 对 **`Climate_h4_f2/proxy`** 完成选择期候选、消融与两组 999 次置换证据。
> **本轮不生成任何 Climate test 预测**（§五），测试段只做索引与特征结构审计。

---

## 一、固定协议（逐项核对）

| 项 | 值 | 核对 |
|---|---|---|
| 任务 / fold | `Climate_h4_f2` / 2 | ✅ |
| input_len / horizon | 52 / 4 | ✅ |
| 季节周期 | **52**（冻结 spec `seasonal_periods.Climate`，非默认 12） | ✅ |
| 边界 | train 636、cal 763、dec 1017、test 1144 | ✅ |
| Bundle 签名 | `a69821be…86d38` | ✅ |
| 冻结 spec | `a0947a5b…6f0031` | ✅ |
| 随机种子 / α 网格 | 2026 / 与 Agriculture 相同的 7 档 | ✅ |
| 选择期起点 | **375 = calibration 124 + decision 251** | ✅ |
| 每候选行数 | **1500 = 375 × 4** | ✅ |

---

## 二、输入：消费辅助电脑02的**选择期隔离包**（§五）

Release `climate-isolated-input-v8-20260924`，ZIP 8,109,693 B，
SHA256 `b028052ed5a09868341024c5b89864490f64c3929747f7f70bcc5ae356db314b`（实测一致）。

| 部分 | 行数 | 内容 |
|---|---|---|
| `selection_fit/` | 956 = train 581 + cal 124 + dec 251 | 建模特征 + **这三段真值** |
| `test_features/` | 124 = test | 只含 origin/网格/缺失掩码与文本特征 |

**隔离是结构性的**：`test_features/` 既没有 `targets*`，也没有
`numeric_history` / `frequency`（数据侧说明：滚动窗口重叠足以从完整测试历史重建
约 123/127 个测试时间点的真值，故主动扣留）。这不是"读完再抹掉"，而是**包里根本没有**。

### 本侧独立复核（不靠数据侧的声称）

| 复核 | 结果 |
|---|---|
| 包 `MANIFEST.json` 逐文件字节哈希 | ✅ 全部相符 |
| 锚：Bundle 签名、冻结 spec 哈希、`anchors/formal_bundle_manifest.json` | ✅ 一致 |
| `selection_fit_commitments.json`：每张切片的形状/dtype/字节哈希 | ✅ 6/6 |
| **与正式 Bundle 逐位对照 18 张数组** | ✅ 全部逐位相同 |
| 两侧 `metadata.csv` 行序 == Bundle 对应段顺序 | ✅ |
| `test_features/` 不含真值/可重建真值数组 | ✅ 断言通过（出现即拒绝） |

> 数据侧 README 说"单靠严格隔离包不能重新证明切片来自原 Bundle"。本侧**持有正式
> Bundle**，所以把这条也验掉了：切片 == Bundle 的 train+cal+dec 行，逐位相同。

**对照时 Bundle 侧一律 `np.load(mmap_mode="r")` 且只索引选中行** —— 否则会把
Bundle 的 test 真值整表读进内存，被入口读文件审计（正确地）判失败。

---

## 三、四候选选择期预测（§二）

| 候选 | 行数 | 说明 |
|---|---|---|
| `N` | 1500 | 数值模型候选 |
| `N+Q` | 1500 | **独立消融**，只用于解释，不进正式门控 |
| `N+S+Q` | 1500 | 注册门控候选一 |
| `N+S+Q+SF` | 1500 | 注册门控候选二 |

每候选产出 `*_proxy_predictions.csv` + `*_run_manifest.json`
（含 `alpha_by_group`、`weight_hash`、分支宽度、`grid`、`seasonal_period`
= 52 及其来源、`test_isolation`、`input_kind=isolated_package`、
`code_provenance`、`runtime`）。

---

## 四、四段边界审计（§六：测试段只查索引与特征结构）

| 段 | bounds | 起点数 | 索引范围 | 结论 |
|---|---|---|---|---|
| train | `[0, 636)` | 581 | 52 – 632 | ✅ |
| calibration | `[636, 763)` | 124 | 636 – 759 | ✅ |
| decision | `[763, 1017)` | 251 | 763 – 1013 | ✅ |
| test | `[1017, 1144]` | 124 | 1017 – 1140 | ✅ 只查索引/特征结构，**不预测、不评分** |
| **选择期合计** | | **375** | | 与任务书一致 |

---

## 五、置换检验（§三）

对两个注册门控候选各执行 **999 次逐行错位置换 + 999 次循环移位**，
**每次重新拟合 PCA、各分支尺度、交互尺度与 alpha**：

| 候选 | 观察决策损失 | **经验 p** | 门槛 0.025 | 循环移位 p（诊断） |
|---|---|---|---|---|
| `N+S+Q` | 0.268329 | **0.983** | ❌ 远未通过 | 0.911 |
| `N+S+Q+SF` | 0.288924 | **0.984** | ❌ 远未通过 | 0.849 |

**决策前后半段**（`middle=(cal_end+dec_end)//2 = 890`，跨界窗口剔除，各 124 起点）：

| 候选 | 前半 | 后半 | 墙钟 / CPU / 峰值内存 |
|---|---|---|---|
| `N+S+Q` | 0.252365 | 0.267640 | 237.71 s / 614.016 s / 177.9 MiB |
| `N+S+Q+SF` | 0.253963 | 0.308103 | 588.51 s / 1057.547 s / 193.2 MiB |

**循环移位的块长由 Climate 的周频结构推导**（§三）：block = horizon = **4 周**。
Climate 是周频（spec `lag_days=7`），horizon=4 即"整段平移整数个月"：与评分窗口同阶、
保留该尺度短期依赖，只破坏文本与目标的配对。各段可用位移数 train 145 / cal 31 / dec 62。
（不取一个完整季节周期 52：decision 段 251 行时 `n//52` 只剩 4 个位移，诊断会退化。）
产物 `circular.block_derivation` 记录 rule / why / source，**只进诊断，不回写正式门控 p**。

> **本侧不下门控结论。** 两个候选的经验 p 都远大于 0.025，按冻结规则应当都不合格；
> 但"决策 MSE 是否低于数值回退、前后半段是否均改善"以及最终路由由**主控独立复算**
> （主控第八轮任务 §一.6）。本侧只交观察损失与零分布。

---

## 六、其余证据

| 要求（§六） | 产物 |
|---|---|
| 权重重演 | `evidence/replay/replay_check.json`：四候选各跑两次 `train.py`，两次逐字节相同、且与已交付 CSV/manifest 一致（`weight_hash`、`alpha_by_group`、`branch_widths` 全等） |
| 输入读取清单 | `evidence/read_isolation/test_read_isolation_audit.json`：包模式 `isolated_package`，74 个路径被接触，**违规 0** |
| CPU/内存与耗时 | 各 `run_manifest.json::runtime` 与各 `permutation_summary.json::runtime`（墙钟/CPU/峰值 RSS/核数/平台） |
| 运行日志 | `evidence/test_run.log` + `evidence/run_ledger.json`（逐条命令、时间、退出码、用时） |

---

## 七、测试（§六：≥85 项 + Climate 专项）

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付八次/model/tests/test_round6.py"   # 49 项
.venv/Scripts/python.exe "03 辅助电脑二 交付八次/model/tests/test_round7.py"   # 85 项
.venv/Scripts/python.exe "03 辅助电脑二 交付八次/model/tests/test_round8.py"   # 131 项
```

第八轮专项覆盖：**Climate 周期 = 52**（非默认 12）、**选择期 375 起点**、
**每候选 1500 行**、**禁止 test**（产物无 test 段、目录无 test 预测文件）、
隔离包载入/行数/占位 NaN/**逐位追溯**、四类包反例（test_features 现真值、
切片被篡改、签名不符、包内自洽但与 Bundle 不符）、**端到端等价**（隔离包与 Bundle
跑出的预测 CSV 逐字节相同）、输入互斥、完备性闸门（产物不完整必须判失败）、
循环移位块长推导（含 `cli_override` 留痕）。

---

## 八、完备性闸门（§四.5）

`model/verify_delivery.py --delivery . --task Climate_h4_f2 --horizon 4 --expected-origins 375`
**按产物判成败**：行数 = 起点数 × horizon、manifest 必需键、两门控候选各 999 次成功、
列序与种子规则、摘要含前后半段与逐次重拟合声明、**不得出现 test 预测**、
`scope` 必须是 `03 辅助电脑二 交付八次/model`、**正式证据必须 `input_kind=isolated_package`**。
结论见 `evidence/verify_delivery.json`。

---

## 九、未做（§五 硬边界）

- **未生成任何 Climate test 预测**，未读取 test 真值，未计算任何 test 指标。
- 未改 alpha、PCA、尺度、交互定义或候选结构。
- **未修改第一至第七次历史交付目录**（§四.1）—— 本轮全部改动只在
  `03 辅助电脑二 交付八次/` 内。
- 未 `merge` `team/main-eval-20260921` 或 `main`。

## 十、目录

```
03 辅助电脑二 交付八次/
├── README.md / RUNBOOK.md / TEAM_SYNC.md
├── MANIFEST.json
├── evidence/
│   ├── cal_dec_predictions/     4 候选 × (CSV + run manifest)
│   ├── row_null/<候选>/         999 逐行 + 999 循环移位 + 两份 summary
│   ├── boundaries/              四段边界审计
│   ├── replay/                  权重重演
│   ├── read_isolation/          入口读文件审计（包模式）
│   ├── _probe_train/            读审计的受监测探针产物
│   ├── run_ledger.json          执行台账（逐条命令）
│   ├── test_run.log             原始记录
│   └── tests_round{6,7,8}.log   三套测试日志
└── model/                       与第八轮全部代码；含 isolated_package.py
```

---

## 十一、证据锚点与「最后一次改动」

| 项 | 值 |
|---|---|
| 预测/置换证据的锚定提交 | **`9af6343`**（各 manifest 的 `code_provenance.code_commit`，`worktree_dirty=false`，`scope=03 辅助电脑二 交付八次/model`） |
| 与当前 HEAD 的差异 | **只有 `model/verify_delivery.py`**（验证脚本，**不在预测路径上**；证据入库后又加了两条判据：清单哈希 vs 已提交字节、自指文件排除） |
| 复核命令 | `git diff --stat 9af6343 HEAD -- "03 辅助电脑二 交付八次/model"` → 只列 `verify_delivery.py` |

预测路径上的文件（`train.py` / `permutation_entry.py` / `permutation.py` /
`candidates.py` / `branches.py` / `bundle_reader.py` / `isolated_package.py` /
`predict_io.py` / `runtime_profile.py` / `numeric_fallbacks.py`）自 `9af6343` 起**逐字节未变**，
所以本轮全部证据可由该提交重放。

**执行台账**（`evidence/run_ledger.json` + `test_run.log`）逐条登记了每条命令、时间与退出码：
正式证据是**消费隔离包**、在 `9af6343` 上、干净工作区里跑出来的那一次；
此前的尝试（先跑 Bundle、后改块长）均已作废，不混入交付。
