# RUNBOOK · 第三轮交接说明

面向主控：修正后的命令、契约字段、以及**实际跑过 / 未跑**的范围。

---

## 一、命令（第三轮 A.3 修正版）

### 测试（无需 Bundle）

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付三次/model/tests/test_round3.py"
# 期望：全部通过（41 项检查），退出码 0
```

### 训练（**需正式 Bundle；当前不可运行**）

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付三次/model/train.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --signature <bundle_signature> --split-spec-sha256 <冻结协议哈希> \
    --segments calibration decision \
    --candidate "N+S+Q" --output-dir <目录>
```

**A.3 修正点**：

- `--signature` **必填**（第二轮写成可选，已改）
- 参数名是 `--segments`（复数；第二轮 RUNBOOK 写成 `--segment`，已改）
- `--scenario` 只接受 **`proxy` / `conservative_lag`** —— `complete_source` 是纯数据审计情景
  （协议注明其真实覆盖为 0），不再作为可运行情景暴露
- 新增 `--split-spec-sha256`，提供时核对 `manifest.inputs.split_spec_sha256`

### 置换（同上，需 Bundle）

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付三次/model/permutation_entry.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --signature <sig> --split-spec-sha256 <sha> \
    --candidate "N+S+Q" --nulls 999 --circular-block 7 --output-dir <目录>
```

`--circular-block` 是**真实的移位块长**（第二轮该参数存在但未被使用，已修）。

### 测试段预测（**需主控冻结路由后**）

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付三次/model/predict_test.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --signature <sig> --split-spec-sha256 <sha> \
    --route <冻结路由.json> --route-sha256 <路由哈希> \
    --selection-manifest <选择期 run_manifest.json> \
    --candidate "N+S+Q" --output-dir <目录>
```

---

## 二、置换产出的字段（供主控无损转换）

`null_scores.csv` —— **一行一次迭代**，成功与失败都在：

| 列 | 说明 |
|---|---|
| `iteration` | 第几次（0-based） |
| `seed` | 该次的种子（`2026*1000 + iteration`） |
| `status` | `ok` / `failed` |
| `loss` | 成功时为决策段 MSE；失败时为空 |
| `error` | 失败原因；成功时为空 |

`null_scores_summary.json` 含 `requested` / `successful` / `failed` / `contract_minimum` /
`p_value` / `p_value_note`。**`requested` 与 `successful` 均达 999 时 `p_value` 才非空。**

需要 JSONL 时按上述列直接映射即可，无需重跑。

---

## 三、预测契约（14 字段，与第二轮一致）

`task_id, fold_id, origin_id, origin_index, segment, scenario, candidate_id, seed,
step, y_pred, target_scale, bundle_signature, config_sha256, code_commit`

预测键 = `(task_id, fold_id, segment, origin_id, step, candidate_id, seed)`，重复即拒绝。
`predictions.csv` 与 `null_scores.csv` 严格分离。

---

## 四、实际运行范围

| 项 | 状态 |
|---|---|
| A 部分六项修复 | ✅ `COMPLETE` |
| 合成故障测试（41 项） | ✅ `COMPLETE` |
| 单次拟合 / 单次置换的实测耗时 | ✅ `PARTIAL`（见 `runtime_profile.json`） |
| 真实 Agriculture 训练 | `NOT_RUN` —— 无正式 Bundle |
| 999 次置换 | `NOT_RUN` —— 无正式 Bundle |
| 测试段预测 | `NOT_RUN` —— 无正式 Bundle，且路由未冻结 |
| 四任务扩展 | `NOT_RUN` |

---

## 五、已知限制

1. **无真实数据验证**：全部测试基于合成 Bundle。读取层的两条路径（数据侧 `read_bundle`
   / 适配层直读）共用同一套 `verify_signature`，但真实 Bundle 未到手。
2. **协议未冻结**：`split_spec_v2.json` 仍是 `draft_not_for_training`，`numerical_sha256` 全空。
3. **三机字段名不一致未决**：数据侧输出 `targets_raw.npy`/`FROZEN`，契约要求
   `targets.npy`/`frozen`。本实现按**契约**实现；请主控裁定。
4. **坐标下降近似**：逐分支 α 用 2 轮坐标下降；主控要求的 1 轮/3 轮敏感性分析
   **`NOT_RUN`**（需真实数据）。
5. **`runtime_profile.json` 不外推**：只报单次实测，按任务书要求不做总时长外推。

---

## 六、给主控的请求

1. **冻结 `split_spec_v2.json`** —— 解锁整条链的唯一入口
2. **裁定 Bundle 字段名**（契约 vs 数据侧现状）
3. **确认 7 档 α**（任务书正文曾写 9 档）
