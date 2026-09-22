# 03 辅助电脑二 · 交付五次

**依据**：主控 [`team_work/main/round5/02_辅助电脑二第五次交付指令_真实运行前修复.md`](../team_work/main/round5/02_辅助电脑二第五次交付指令_真实运行前修复.md)
**起点**：第四次交付 `51a2031`　**分支**：`SHY`　**日期**：2026-09-22
**状态**：`MODEL_REAL_EVIDENCE_PENDING`

> 本轮 = 任务书 **A 部分**（真实运行前的修复 + 端到端负例）。
> **B 部分**仍阻塞 —— 数据侧正式 Bundle 未交付。未跑项一律 `NOT_RUN`。

---

## 一、A 部分五项修复对照

| # | 任务书要求 | 状态 | 位置 |
|---|---|---|---|
| A.1 | `numeric_fallback` 时 `--candidate` 必须**严格等于** `route.fallback` | ✅ | [`model/predict_test.py`](model/predict_test.py) `check_route` |
| A.2 | `AR-Ridge` 与主控 `v2.numeric_baselines` 统一 | ✅ | [`model/numeric_fallbacks.py`](model/numeric_fallbacks.py) —— 逐行等价移植 |
| A.3 | 主控基线不虚构模型侧清单 | ✅ | `predict_test.py` —— 基线按路由+固定配置验证 |
| A.4 | 路由必须同时携带 `split_spec_sha256` 与 `bundle_signature` 两锚 | ✅ | `check_route` 两锚必填且强制核对 |
| A.5 | 不向 predict 提供测试 `targets` | ✅ | `train.to_inference_bundle` + 故障测试 |

### A.1 主控又抓到一个真旁路

第四轮的判断是：

```python
if candidate not in GATE_CANDIDATES and candidate != fallback:
    raise ...
return fallback          # ← 请求门控候选时条件为假 → 不报错 → 直接跑回退模型
```

**调用方请求 `N+S+Q`，却会拿到 `AR-Ridge` 的预测**，而入口不报错。
现已改为 `if candidate != fallback: raise` —— 请求什么就必须是什么。
测试从 `predict_test.main()` **入口**传入门控候选，断言非零退出**且无预测文件产出**。

### A.2 `AR-Ridge` 曾是「另一个模型」

第四轮实现：**末 8 个滞后**作特征、在**训练段内部 80/20** 选 α。
主控固定在 `team_eval/v2.py::numeric_baselines`：**全部历史窗口**作输入、
**校准段**选 7 档 α、逐 H 直接输出标准化尺度。

**两者是不同的模型** —— 主控选路时用一个、模型侧出预测用另一个，结果会系统性错位。

现已按主控实现**逐行等价移植**，并在测试里把主控实现**抄录为参考函数**逐点比对：

| 比对项 | 结果 |
|---|---|
| α 选择 | ✅ 一致 |
| 校准 MSE | ✅ 相对容差 1e-10 内一致 |
| `Last` / `SeasonalNaive` / `AR-Ridge` 逐起点逐步输出 | ✅ 全部在 1e-10 容差内一致 |
| `fit_rows` / `calibration_rows` | ✅ 一致 |

### A.3 主控基线没有模型侧清单

`Last` / `SeasonalNaive` / `AR-Ridge` 由主控导出，只给预测 CSV + 配置哈希。
第四轮入口却对**所有**模型要求选择期 `run_manifest.json` —— 真实回退**无法供给**。

现在：基线走「路由 + 主控固定配置」验证，**不虚构模型侧清单**；
`N` 仍属模型候选型回退，保留其选择期清单路径，单独测。

### A.4 两锚必填

`check_route` 现在要求路由**同时**携带 `split_spec_sha256` 与 `bundle_signature`，
任一缺失或不一致即拒绝。**旧的无锚 smoke 路由不能作为正式测试许可证。**
（第四轮我把"无锚"记录为缺口并继续放行 —— 那对 smoke 可以，对正式不行。）

### A.5 测试真值不进入预测

`to_inference_bundle` 构建**仅含特征**的视图，`targets` / `targets_standardized`
**根本传不进去** —— 不是靠注释声称。

故障测试：**随机改掉**磁盘上的测试真值并**重新签名**（否则 `verify_bundle` 先拦下来，
测到的就只是签名校验），预测仍**逐位不变**；删除真值文件后同样不变。

---

## 二、测试：35 项全部通过（合成数据）

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付五次/model/tests/test_round5.py"
# 全部通过（35 项检查）
# 退出码 0
```

| 任务书 | 覆盖 |
|---|---|
| A.1 | **请求门控候选但路由选数值回退 → 入口拒绝且无预测文件**；请求值严格等于 `route.fallback` 时通过 |
| A.2 | α / 校准 MSE / 三个基线逐点输出 / 行数，**与主控参考实现逐项比对** |
| A.3 | 主控基线无 selection manifest 也能跑；路径标记与 α 记录正确 |
| A.4 | 缺 `split_spec_sha256` → 拒绝；缺 `bundle_signature` → 拒绝；锚不符 → 拒绝；两锚齐备且一致 → 通过 |
| A.5 | 推理视图无真值；**改真值并重新签名后预测逐位不变**；删除真值后仍不变 |
| 沿用 | 签名 == `signature(inputs)`、四段边界、H 窗口、半段 `middle=37` 与掩码逐位一致 |

### 复核任务书时补的两处遗漏

| 任务书原文 | 缺口 | 修复 |
|---|---|---|
| A.3「`N` 属模型候选型数值回退，保留其选择期清单并**单独测路径**」 | 代码实现了 `N` 走清单路径，但**没有单独测它** —— 它与主控基线走的是**不同分支** | 新增 `N` 端到端测试：有清单时成功、`path=="numeric_fallback_candidate"`、两个权重哈希都记录；**缺清单时必须拒绝**（这正是它与主控基线的差别） |
| A.5「预测字节**和权重/路由选择**不变」 | 只测了**预测**不变；模型是篡改**前**就拟合好的，权重当然不变 —— **等于没测到"选路过程是否依赖真值"** | 新增：用篡改真值后的 Bundle **重新跑一遍** `fit_design + select_alphas + refit`，断言 **α 完全一致、权重逐位一致** |

### 补测过程中又抓到两个自己的 bug

1. **A.5 的故障注入写错了段**：我用 `rng.normal(size=arr.shape)` 替换了**整个数组**，
   把 train / calibration 的真值**一起改掉了**。诊断输出显示**四段的真值全都变了** ——
   那样 α 会（正确地）变化，测到的是"训练数据被毁"，不是"是否偷看测试真值"。
   改为**只改 test 段的行**后通过。

2. **`N` 的路径标签错误**：`path` 被硬编码成 `"gate_candidate"`，但 `N` 既不是门控候选
   也不是主控基线，而是**模型候选型数值回退**。现分三个标签：
   `gate_candidate` / `numeric_fallback_candidate` / `master_numeric_baseline`。
   同时 `model_config.branches` 也从 `[]` 修正为 `["N"]`。

---

## 三、B 部分仍阻塞

主控已 `Freeze four audited numerical snapshots`（第五轮进展），但**正式 Bundle 仍未交付**。
任务书给出冻结 spec SHA256 基准：

```
a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031
```

模型侧将在 Bundle 到位后用该值核对。本轮**没有**任何真实预测、零分布或 p 值。

---

## 四、交付清单

| 要求 | 状态 |
|---|---|
| A 部分五项修复 + 端到端负例 | ✅ `COMPLETE` |
| 测试运行命令 / 退出码 / 固定提交 | ✅ 见 §二与提交信息 |
| 真实 Agriculture 选择期预测 | `NOT_RUN` — 无 Bundle |
| 999 次置换 | `NOT_RUN` — 无 Bundle |
| 测试段预测 | `NOT_RUN` — 无 Bundle 且路由未冻结 |
| 资源实测 | `PARTIAL` — 见 `runtime_profile.json`，**不外推** |

---

## 五、目录

```
03 辅助电脑二 交付五次/
├── README.md / RUNBOOK.md / TEAM_SYNC.md
├── MANIFEST.json
├── runtime_profile.json
└── model/
    ├── branches.py            N/Q/S/F + SF 交互 + 无文本掩码
    ├── candidates.py          门控/诊断分类 + 协议 7 档 α
    ├── bundle_reader.py       verify_bundle + 边界 + 半段
    ├── numeric_fallbacks.py   ★ A.2 与主控 numeric_baselines 逐行等价
    ├── predict_io.py          14 字段契约 + weight_hash
    ├── permutation.py         逐次记录 + 块长循环移位
    ├── train.py               训练入口 + to_inference_bundle（A.5）
    ├── permutation_entry.py   置换入口
    ├── predict_test.py        ★ A.1/A.3/A.4 修复
    ├── audit_tables.py        无文本 + 半段审计表
    └── tests/test_round5.py   35 项检查
```

前四轮目录完整保留，未改动。
