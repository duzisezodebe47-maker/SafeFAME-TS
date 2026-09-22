# 03 辅助电脑二 · 交付三次

**依据**：主控 [`team_work/main/round3/02_模型侧第三轮任务_SHY.md`](../team_work/main/round3/02_模型侧第三轮任务_SHY.md)
**分支**：`SHY`　**起点**：第二轮 `84e665b`　**日期**：2026-09-22
**状态**：`MODEL_REAL_EVIDENCE_PENDING`

> **本轮范围 = 任务书 A 部分（收到正式 Bundle 前的代码/文档修复）。**
> B 部分（真实运行）**完全阻塞** —— 正式 Bundle 不存在，原因见 §三。
> 按任务书要求，未跑的一律标 `NOT_RUN`，不用合成样本冒充真实结果。

---

## 一、A 部分六项修复对照

| # | 任务书要求 | 状态 | 位置 |
|---|---|---|---|
| A.1 | 核对**全部且只有**清单文件；`split_spec_sha256` 与冻结协议一致；适配层同样校验 | ✅ | [`model/bundle_reader.py`](model/bundle_reader.py) `verify_signature` —— 缺失/哈希不符/**清单外多余文件**三类都拒绝；`split_spec_sha256` 与 `manifest.inputs` 比对 |
| A.2 | `assert_grid` 核对 `origin_id` 的**领域/H/折**与 `task_id` 一致，并核对**分段边界** | ✅ | 同上 `assert_grid` —— 新增 `TASK_ID_RE`；索引对但领域/跨度/折错也拒绝；`bounds=(lo,hi)` 非空时核对起点落在协议半开区间内 |
| A.3 | 修正 RUNBOOK 命令；`complete_source` 不作为可运行情景 | ✅ | [`RUNBOOK.md`](RUNBOOK.md)；`train.RUNNABLE_SCENARIOS = ("proxy", "conservative_lag")` |
| A.4 | 逐次记录 `iteration → seed → 损失/失败`；`requested` 与 `successful` 均达 999 才算 p | ✅ | [`model/permutation.py`](model/permutation.py) `NullIteration` + `unavailable_reason` |
| A.5 | 决策半段按**目标时间边界**切；`circular-block` 真正使用块长；用 7 档 α | ✅ | `bundle_reader.decision_halves_by_target_time`；`circular_shift_null(circular_block=…)`；`PROTOCOL_ALPHAS` 已是 7 档 |
| A.6 | 新增 `predict_test.py`，拒绝未冻结路由/签名不符/权重不一致 | ✅ | [`model/predict_test.py`](model/predict_test.py) |
| 交付物 | 「无文本与半段审计表」 | ✅ | [`model/audit_tables.py`](model/audit_tables.py) —— `text_availability_audit.csv` + `decision_halves_audit.csv` |

### A.4 是本轮最实质的修复

主控指出：第二轮 `seeds` 含**失败**迭代而 `losses` 只含**成功**迭代，
写 CSV 时 `seeds[i]` 配 `losses[i]` 会**错配** —— 第 2 次失败、第 3 次成功时，
第 2 个损失会被安到第 3 个种子上。**确实如此**，我漏了。

现在每次迭代一条 `NullIteration(iteration, seed, loss|error)`，CSV 一行一次迭代，
`status` 列区分 ok/failed。测试用**故障注入**验证：第 2 次失败、第 3 次成功时，
第 3 行的损失仍是它自己的值，种子也对得上。

**p 值门槛**同步收紧：`requested` 达契约下限 999 **且** 全部 999 次成功才计算；
任何失败返回 `None`，并通过 `unavailable_reason()` 给出可定位原因。

---

## 二、测试：49 项全部通过（合成数据）

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付三次/model/tests/test_round3.py"
# 全部通过（49 项检查）
```

按任务书 A 部分逐项对应：

| 任务书 | 测试 |
|---|---|
| A.1 | 缺失/哈希不符/末项 → 拒绝；**清单外多余文件** → 拒绝；`split_spec_sha256` 不符 → 拒绝 |
| A.2 | **索引正确但领域错** → 拒绝；**索引正确但跨度错** → 拒绝；**起点落在协议边界外** → 拒绝 |
| A.4 | **故障注入**：第 2 次失败第 3 次成功，种子的损失**未错配**；失败行不带损失且带可定位错误；请求不足/有失败时 p 均不可用 |
| A.5 | 块长 7 与 2 产生不同结果（证明真的在用块长）；缺 `target_end_time` 时**硬失败不退回按个数切**；两半段互不重叠 |
| A.6 | 路由未冻结 → 拒绝；路由哈希不符 → 拒绝；权重哈希对 1e-9 改动敏感 |
| 回归 | 无文本行 S/SF 贡献严格为零；H=1 返回 `(n,1)` |

**未覆盖**（需正式 Bundle）：真实 Agriculture 训练、999 次置换、测试段预测。

---

## 三、B 部分为何完全没跑：正式 Bundle 不存在

主控第三轮进度说明已把它列为**阻断**：

> 三机格式对接：**阻断** —— 静态检查证实数据输出 `targets_raw.npy`/`FROZEN`，
> 读取器要求 `targets.npy`/`frozen`

而数据侧自己的声明是（[`辅助电脑02交付03次/formal_bundle_manifest.json`](https://github.com/duzisezodebe47-maker/SafeFAME-TS/tree/3218151885-creator)）：

```json
{
  "status": "DATA_FORMAL_PENDING",
  "formal_bundle": null,
  "reason": "Master split_spec_v2.json is not frozen or approved.",
  "observed_split_spec_status": "draft_not_for_training"
}
```

**根因在主控的协议还没冻结**（`split_spec_v2.json` 的 `status` 是 `draft_not_for_training`，
四个任务的 `numerical_sha256` 全为 `null`）。协议不冻结 → 数据侧拒绝出正式 Bundle →
我无从运行。数据侧拒绝得对 —— 协议里自己写了"核验前不得冻结"。

**依赖链**：

```
主控冻结 split_spec  →  数据侧生成正式 Bundle  →  我跑 Agriculture 训练/置换
     ❌ 当前卡在这
```

三个环节任何人做不了别人的活，**我也不能自行生成**（编码是数据侧的职责，
自行重编码会产生签名不一致的缓存）。

---

## 四、交付清单（按任务书要求逐项标注）

| 要求 | 状态 |
|---|---|
| 修复后的读取/训练/置换/测试入口与测试 | ✅ `COMPLETE` |
| `cal_dec_predictions/` | `NOT_RUN` —— 无 Bundle |
| 逐次 `row_null/` | `NOT_RUN` —— 无 Bundle |
| `run_manifest.json`（真实任务） | `NOT_RUN` —— 无 Bundle |
| `runtime_profile.json` | `PARTIAL` —— 仅有合成数据的单次耗时实测，见下 |
| 无文本与半段审计表 | ✅ 生成器已实现（`audit_tables.py`）；真实表 `NOT_RUN`（无 Bundle） |
| 模型权重 SHA256 / 配置哈希 | ✅ 机制已实现（`weight_hash`），真实值待跑 |
| 正式 Bundle 签名 | `NOT_RUN` —— Bundle 不存在 |
| 固定代码提交 | ✅ 见本次提交 SHA |

**资源实测（合成数据，仅供主控量级参考）**：

| 项 | 实测 |
|---|---|
| 单次候选拟合（含 2 轮 α 坐标下降） | ~0.2 s |
| 单次置换（重建 PCA/尺度/交互/α） | ~0.1 s |
| 平台 | CPU，Python 3.12.10 |

> 任务书明确要求**不得**用合成样本时长外推"分钟量级"。故此处只报单次实测值，
> **不外推 999 次的总时长** —— 真实维度（`semantic=768`）比合成样本大，
> 必须以正式 Bundle 的实测为准。

---

## 五、需要主控推进的事

1. **冻结 `split_spec_v2.json`**（补 `numerical_sha256`、`status` 改 `frozen`）——
   这是解锁整条链的唯一入口
2. **裁定 Bundle 字段名的三机不一致**：数据侧 `targets_raw.npy`/`FROZEN`
   vs 契约 `targets.npy`/`frozen`。我按契约实现，请确认以契约为准
3. **确认 7 档 α**（任务书正文曾写"沿用 RIDGE_ALPHAS"9 档，协议文件是 7 档；
   本轮按协议 7 档实现）

---

## 六、目录

```
03 辅助电脑二 交付三次/
└── model/
    ├── branches.py            ← N/Q/S/F + SF 交互 + 无文本掩码
    ├── candidates.py          ← 门控/诊断分类 + 协议 7 档 α
    ├── bundle_reader.py       ← A.1/A.2/A.5 读取、网格校验、半段切分
    ├── predict_io.py          ← 14 字段契约 + weight_hash
    ├── permutation.py         ← A.4 逐次记录 + A.5 块长移位
    ├── train.py               ← 训练入口（A.3：仅 proxy/conservative_lag）
    ├── permutation_entry.py   ← 置换入口（A.4/A.5）
    ├── predict_test.py        ← A.6 测试段入口
    ├── audit_tables.py        ← 无文本审计表 + 决策半段审计表
    └── tests/test_round3.py   ← 49 项检查
```

第二轮目录 [`03 辅助电脑二 交付二次/`](../03%20辅助电脑二%20交付二次/) 完整保留，未改动。
