# RUNBOOK · 第二轮交接说明

面向主控：本轮的契约、命令、实际跑过与未跑的范围。

---

## 一、预测契约（主控按此读取）

字段与 `protocol/prediction_contract_v2.json` 的 `prediction_csv_required` **逐字一致**：

| 字段 | 说明 |
|---|---|
| `task_id` | `Domain_hH_fF` |
| `fold_id` | 折号 |
| `origin_id` | 字符串，格式 `Domain:hH:fF:o<origin_index>` |
| `origin_index` | 整数，与 `origin_id` 尾部一致（错位会被拒绝） |
| `segment` | `train` / `calibration` / `decision` / `test` |
| `scenario` | `proxy` / `conservative_lag` / `complete_source` |
| `candidate_id` | 候选名 |
| `seed` | 随机种子 |
| `step` | 1-based，1..H 每步恰好一次 |
| `y_pred` | 预测值 |
| `target_scale` | 固定 `train_only_standardized_OT` |
| `bundle_signature` | Bundle 签名 |
| `config_sha256` | 配置哈希（键排序序列化） |
| `code_commit` | 生成时 git commit |

**预测键** = `(task_id, fold_id, segment, origin_id, step, candidate_id, seed)`，重复即拒绝。

**`predictions.csv` 与 `null_scores.csv` 严格分离** —— 前者不含真值，后者只含置换损失。

**测试预测不在本次交付范围内**（契约 `required_before_freeze` 明确：冻结路由前只交
calibration / decision 及 999 次 refit null）。测试真值从不进入训练进程。

---

## 二、命令

### 测试（无需 Bundle）

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付二次/model/tests/test_round2.py"
# 期望：全部通过（43 项检查），退出码 0
```

### 正式链（**需主控冻结 Bundle，当前不可运行**）

```bash
# 1) 校准 + 决策预测
.venv/Scripts/python.exe "03 辅助电脑二 交付二次/model/train.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --candidate N+S+Q --segment calibration decision --output-dir evidence/

# 2) 决策段 999 次置换零分布
.venv/Scripts/python.exe "03 辅助电脑二 交付二次/model/permutation_entry.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --candidate N+S+Q --nulls 999 --output-dir evidence/null/
```

> `train.py` / `permutation_entry.py` 的命令行入口尚未接 Bundle 实参
> （原先只接受 `v2smoke`，已按 P1#2 重写读取层但未接 CLI）。
> **Bundle 到位后**补 CLI 接线即可，读取、分支、候选、置换四层均已就绪且经测试。

---

## 三、实际运行范围 vs 未运行范围

### ✅ 实际跑过

| 项 | 范围 |
|---|---|
| 工程测试 | **43 项全部通过**（合成数据） |
| P1#3 掩码 | 无文本行 S / SF 贡献**严格为零**（`atol=0.0`） |
| P1#4 SF 分支 | 已实现并纳入门控候选 |
| P1#5 置换 | 小次数自检通过；**同一置换跑两次逐位一致**，证明每次确实重拟合 |
| 训练段无文本降级 | 标记 `s_degraded`，不造伪信号 |

### ❌ 未跑（及原因）

| 未跑项 | 原因 |
|---|---|
| 真实 Bundle 上的任何训练 | **Bundle 未交付**（见 README §三） |
| 真实 999 次置换与 p 值 | 同上 —— **未伪造任何 p 值** |
| 四任务 × 两情景执行 | 同上 |
| 循环移位诊断的真实运行 | 同上（代码已实现并自检） |
| 代表任务扩展 | 主控指定后方可进行 |

---

## 四、置换离线的可验证性

`permutation.py` 的关键性质，测试已证明：

- **每次置换重建全部数据依赖量**：PCA、各分支 `SafeScaler`、SF 交互尺度、α 坐标下降。
- **无跨次缓存**：同一置换重复运行**逐位一致** —— 若有任何量被跨次复用，结果会漂移。
- **失败留痕**：每次置换的种子、损失、异常分别记录；未跑满 999 时 `p_value` 返回 `None`。
- **循环移位**只作诊断，代码注释明确"不回写主门控"。

---

## 五、已知限制

1. **未接真实数据**：全部测试基于合成 Bundle。契约字段与数据侧 `.npy` 的对应关系
   按 `bundle.py` 源码核对（`semantic` / `quality` / `source_available` / `text_available`），
   但**未在真实 Bundle 上验证过一次**。
2. **CLI 未接线**：读取层已重写，但 `train.py` 的命令行参数尚未加 `--bundle` 等实参。
3. **协议未冻结**：`split_spec_v2.json` 的 `status` 是 `draft_not_for_training`，
   `numerical_sha256` 四项全为 `null`。本实现按草稿结构编写，冻结后若结构变化需同步。
4. **α 网格冲突未决**：见 README §四。
5. **坐标下降近似**：逐分支 α 用 2 轮坐标下降，非全网格；主控要求的 1 轮 / 3 轮
   敏感性分析**尚未运行**（需真实数据）。

---

## 六、给主控的请求

1. **催付冻结 Bundle**（含 `manifest.json`、各任务 `.npy`、`schema.status = frozen`）
2. **确认 α 网格**取 7 档还是 9 档
3. **确认代表任务扩展顺序**（任务书已给四任务，是否按原序）
4. Bundle 到位后，请允许先跑 `Agriculture_h12_f1` 的 calibration/decision + null，
   待你冻结路由后再生成测试预测
