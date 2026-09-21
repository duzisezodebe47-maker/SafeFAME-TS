# RUNBOOK · 交接说明

面向主控电脑：怎样读取本交付的预测、怎样复算、哪些跑过哪些没跑、有哪些限制。

---

## 一、环境

```
Windows / Python 3.12.10
虚拟环境：仓库根目录 .venv/
依赖：requirements.txt（本模块只用 numpy / pandas / scipy / scikit-learn，不需要 torch）
```

重建环境（干净 clone 后）：

```bash
py -3.12 -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

---

## 二、输入位置

| 输入 | 路径 | 状态 |
|---|---|---|
| 语义特征（v2 时代） | `data_processed/semantic_features/<领域>.npz` | ✅ 本机有 6 个领域 |
| 文本索引 / 句向量 | `data_processed/text/`、`data_processed/embeddings/` | ✅ 有 |
| 原始数值序列 | `references/external/Time-MMD/numerical/<领域>/<领域>.csv` | ✅ 有 |
| **v4 特征** | `data_processed/v4/` | ❌ **本机没有 —— 正式实验的阻塞项** |
| **主控协议** | `split_spec.json` / `prediction_contract.json` | ❌ **尚未交付** |

**依赖方向**：本模块**只读取**上述特征，从不自行清洗、筛选或重编码文本。
若发现辅助电脑一的特征缺字段，应提交接口变更单，而不是就地复制一份管线。

---

## 三、预测契约（主控按此读取）

每个候选在 `evidence/<候选>/predictions.csv` 输出逐起点预测。字段固定为：

| 字段 | 类型 | 含义 |
|---|---|---|
| `task_id` | str | `"<领域>_h<跨度>"` |
| `fold_id` | int | 滚动折号 |
| `origin_id` | int | 预测起点在原序列中的索引 |
| `candidate_id` | str | 候选名，如 `N+S+Q` |
| `seed` | int | 随机种子（Ridge 家族为确定性，记录用） |
| `horizon` | int | 第几步（1..H） |
| `y_pred` | float | 预测值 |
| `config_hash` | str | 配置哈希，键排序后序列化 |
| `feature_hash` | str | 特征哈希，覆盖 x/report/search/quality/origins |
| `code_commit` | str | 生成时的 git commit |

**关键：本文件不含真实目标值。** 测试期真值由主控评测器关联 ——
训练脚本从不读取测试真值来决定保留哪个候选（任务书 §1.3）。

一行 = 一个 (候选, 折, 起点, 跨度) 组合。复算方式：

```python
import pandas as pd
df = pd.read_csv("evidence/N_S_Q_F/predictions.csv")
# 与主控自己的真值表按 (task_id, fold_id, origin_id, horizon) 关联
pred = df.pivot_table(index=["origin_id", "horizon"], values="y_pred")
```

---

## 四、完整命令

### 4.1 跑测试（无需 v4 数据，约 20 秒）

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付一次/model/tests/test_smoke.py"
# 期望输出：全部通过（87 项检查），退出码 0
```

### 4.2 跑单个候选

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付一次/model/train.py" \
    --candidate "N+S" --domain Climate --horizon 4 --fold 0 --seed 2026 \
    --output-dir <输出目录>
# 输出：predictions.csv + run_manifest.json
```

`--candidate` 可选：`N` `N+Q` `N+S` `N+S+Q` `N+S+Q+F` `N+F`

### 4.3 复现整张消融表

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付一次/model/run_ablation.py" \
    --domain Climate --horizon 4 --output-dir <输出目录>
```

---

## 五、实际运行范围 vs 未运行范围

### ✅ 实际跑过

| 项 | 范围 |
|---|---|
| 工程冒烟测试 | 3 个配置（Climate/H4、**Climate/H1**、Energy/H12）× 6 候选，**87 项检查全部通过** |
| 消融对照 | Climate / H=4 / L=52 / 单折单种子 / 6 候选全部完成 |
| 逐起点预测 | 6 候选 × 1008 行 = 6048 行，已写入 `evidence/` |

### ❌ 未跑（及原因）

| 未跑项 | 原因 |
|---|---|
| v4 全量 120 任务—折 | **本机无 `data_processed/v4/`**，待辅助电脑一 |
| 999 次逐行置换主门控 | 需主控冻结的协议（置换单位与次数） |
| 代表任务扩展（§3 第二步） | 需主控指定任务并冻结候选列表 |
| 深度模型候选 | 任务书允许"轻量残差头与现有骨干组合"；当前先完成轻量族，深度候选待主控批准后另立版本 |

### ⚠️ 全是历史权重回放吗？

**不是。** 本交付的每个候选都是**本次实际拟合**的：
`train.py` 与 `run_ablation.py` 都调用 `BranchResidualCandidate.refit()` 在
"训练+校准"起点上重新解正规方程，并写出新预测。

**没有**复用 `outputs/` 下的任何历史权重或历史预测。历史 v2/v3/v4 正式结果
及其哈希**未被读取、更未被修改**。

---

## 六、失败与跳过记录

| 项 | 情况 |
|---|---|
| 候选运行失败 | **0 个**（6/6 完成） |
| 测试失败 | 0（87/87 通过） |
| 跳过的样本 | 无。文本覆盖为零的样本在当前数据切片中未出现；若出现，`SemanticBranch` 不参与融合时该样本走数值分支，**不静默补零** |
| 非有限值 | 被拒绝并抛出明确错误，不代填 |

失败记录写入每候选的 `run_manifest.json` 的 `failures` 字段与
`evidence/ablation_manifest.json`。

---

## 七、已知限制

1. **冒烟级数据**：单一领域（Climate）、单一跨度（H=4）、单折、单种子。
   **不能外推，不能作为任何候选有效性的证据。**
2. **划分来源临时**：主控协议未冻结，当前用 v2 已冻结的 70/10/20
   （`train_famets._split_origins`，docs/10），证据里标记为 `split_source`。
   协议到位后须替换并重跑。
3. **坐标下降选 α**：逐分支 α 用两轮坐标下降近似，而非全网格（4 分支 × 9 档
   = 6561 次闭式解，在 240 条候选上开销过大）。该近似本身应作为敏感性分析报告。
4. **未做置换交换性验证**：`DESIGN.md §5.2` 计划的合成数据置换均匀性检验尚未实现，
   待候选列表冻结后补。**这是 docs/23 频率候选 p 值塌陷教训的直接对应项，不可省。**
5. **`N+S` 的掩码语义待定**：当前实现是"不启用 Q 分支"，
   而非"启用但对文本覆盖为零的样本置零 `f_S`"。需主控在冻结时确认（见 README §七问题 3）。

---

## 八、实现中发现并修复的三个真实缺陷

这三个都会在正式实验中导致失败，记录供另两机参考。

### 缺陷 1：零方差列导致 Gram 奇异

Climate 的质量特征含**恒定列**，`StandardScaler` 对其输出全 0，使设计矩阵出现
全零列、Gram 秩只 89/96、最小特征值为负，cholesky 直接抛
`LinAlgError: potrf`。

**修法**：`branches.SafeScaler` 在训练段上丢弃零方差列。
**关键**：该判定必须**每次置换重新做** —— 置换会改变哪些列恒定，
沿用上一次判定会破坏置换交换性（docs/23 频率候选 p 从 0.021 塌到 0.085 的同类机制）。

### 缺陷 2：float32 下小 α 时 cholesky 失败

同一设计矩阵，`alpha=1e-4` 在 float32 下 potrf 报非正定、**float64 下成功**。

**修法**：`candidates._group_ridge` 在 float64 下构造并求解正规方程；
设计矩阵保持 float32 以省内存。

> 现有管线用 sklearn 的 `Ridge`，它在内部处理了这一点，所以未暴露。
> **自己写求解器的候选必须踩这个坑。**

### 缺陷 3：Windows 文件句柄

`np.load` 返回的对象持有文件句柄，后续删除该文件会 `PermissionError (WinError 32)`。
测试改用上下文管理器。属任务书所指"跨机器环境问题"。

### 另：H=1 广播（历史缺陷）

预测显式 reshape 回 `(n, H)`；`predict_io.extend_grid` **拒绝一维预测**，
在契约层再设一道防线。冒烟测试含 H=1 回归检查。

---

## 九、给主控的下一步请求

1. 冻结候选列表（含 README §七的 4 个问题）
2. 交付 `split_spec.json` 与 `prediction_contract.json`
3. 指定代表任务（§3 第二步）
4. 确认是否批准深度模型候选

特征侧阻塞（`data_processed/v4/`）需辅助电脑一交付后才能启动正式训练。
