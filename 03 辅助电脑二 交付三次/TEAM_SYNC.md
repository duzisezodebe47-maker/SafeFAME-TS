# TEAM_SYNC · 模型侧第三轮同步说明

**面向**：主控（Jerry ye）、数据侧（3218151885-creator）
**分支**：`SHY`　**交付目录**：`03 辅助电脑二 交付三次/`　**日期**：2026-09-22

---

## 一、状态一句话

**A 部分六项已全部修复，49 项合成测试通过；B 部分整条链被"协议未冻结"这一个动作阻塞。**

---

## 二、A 部分修复对照

逐项见 [`README.md` §一](README.md)。要点：

| # | 要求 | 状态 |
|---|---|---|
| A.1 | 核对**全部且只有**清单文件；`split_spec_sha256` 核对；适配层同样校验、不静默放行 | ✅ |
| A.2 | `assert_grid` 核对领域/H/折/索引 **+ 分段边界** | ✅ |
| A.3 | RUNBOOK 命令修正；`complete_source` 移出可运行情景 | ✅ |
| A.4 | 置换逐次记录；`requested` 与 `successful` 均达 999 才算 p | ✅ |
| A.5 | 半段按目标时间边界切；`circular-block` 真正生效；7 档 α | ✅ |
| A.6 | 新增 `predict_test.py`，拒绝未冻结路由/签名不符/权重不一致 | ✅ |
| 交付物 | 无文本与半段审计表 | ✅ `model/audit_tables.py` |

### 主控在 A.4 指出的是真实缺陷

> `seeds` 包含失败迭代而 `losses` 只含成功迭代，写 CSV 时可能错配。

**确实如此，我漏了。** 第 2 次失败、第 3 次成功时，第 2 个损失会被安到第 3 个种子上。

现已改为 `NullIteration(iteration, seed, loss | error)` 逐条记录，CSV 一行一次迭代
并带 `status` 列。测试用**故障注入**验证：第 2 次失败、第 3 次成功时，
第 3 行的损失仍是它自己的值，种子也对得上。

---

## 三、B 部分为何完全没跑

数据侧的声明（`辅助电脑02交付03次/formal_bundle_manifest.json`）：

```json
{
  "status": "DATA_FORMAL_PENDING",
  "formal_bundle": null,
  "reason": "Master split_spec_v2.json is not frozen or approved.",
  "observed_split_spec_status": "draft_not_for_training"
}
```

### 依赖链是单向的

```
冻结 split_spec_v2.json  →  数据侧生成正式 Bundle  →  模型侧跑训练/置换
        ↑
   整条链唯一的入口，只在主控手里
```

而 `split_spec_v2.json` 的 `notes` 自己写着：

> DRAFT: numerical_sha256 is absent for all four cleaned snapshots.
> **Do not change status to frozen** until the data-side files, counts and
> hashes are independently verified.

**所以数据侧拒绝出 Bundle 是按协议办事，模型侧跑不了也是必然的。**
三个环节谁也替不了谁。

### 需要主控做的事

补四个任务的 `numerical_sha256`，把 `status` 改为 `frozen`。
**协议冻结当天模型侧即可开跑 Agriculture** —— A 部分代码全部就绪，无需再改。

---

## 四、需要主控裁定的三处

| # | 事项 | 现状与请求 |
|---|---|---|
| 1 | **Bundle 字段名三机不一致** | 数据侧输出 `targets_raw.npy` / `FROZEN`；契约（`prediction_contract_v2.json`）要求 `targets.npy` / `frozen`。模型侧**按契约实现**。请确认以契约为准（则数据侧需改），或改契约 |
| 2 | **α 网格** | 第三轮任务书正文写"沿用仓库 `RIDGE_ALPHAS`"（**9 档**），但 `split_spec_v2.json` 的 `alpha_grid` 是 **7 档**。本轮**按机器可读的协议文件**实现（7 档） |
| 3 | **代表任务扩展顺序** | 任务书已给 `Climate_h4_f2` → `SocialGood_h3_f1` → `Environment_h7_f2`，请确认是否按原序 |

---

## 五、资源提示

按任务书要求，**没有**用合成样本时长外推 999 次总时长。
[`runtime_profile.json`](runtime_profile.json) 只报单次实测：

| 项 | 实测 |
|---|---|
| 单次候选拟合（含 2 轮 α 坐标下降） | ~0.17 s |
| 单次置换（重建 PCA/尺度/交互/α） | ~0.16 s |
| 平台 | CPU，Python 3.12.10 |

真实维度（`semantic=768`）与样本量不同，**必须以正式 Bundle 的实测为准**。
若大任务超出可用资源，将先报告可定位瓶颈与部分完成数。

---

## 六、未跑范围（一律标 `NOT_RUN`，不用合成样本冒充）

| 未跑项 | 原因 |
|---|---|
| 真实 Agriculture 训练（`cal_dec_predictions/`） | 无正式 Bundle |
| 999 次逐行错位置换（逐次 `row_null/`） | 无正式 Bundle |
| 测试段预测 | 无正式 Bundle，且路由未冻结 |
| 四任务扩展 | 无正式 Bundle |
| 坐标下降 1 轮/3 轮敏感性 | 需真实数据 |

已跑的：A 部分六项修复、49 项合成测试、单次资源实测。
