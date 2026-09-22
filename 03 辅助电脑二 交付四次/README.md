# 03 辅助电脑二 · 交付四次

**依据**：主控 [`team_work/main/round4/02_辅助电脑二第四轮任务_模型侧.md`](../team_work/main/round4/02_辅助电脑二第四轮任务_模型侧.md)
**起点**：第三次交付 `fc8fc39`　**分支**：`SHY`　**日期**：2026-09-22
**状态**：`MODEL_REAL_EVIDENCE_PENDING`

> 本轮 = 任务书 **A 部分**（代码返修 + 故障测试）。**B 部分（真实运行）仍阻塞** —— 正式 Bundle 未交付。
> 未跑项一律标 `NOT_RUN`，不用合成 31 项测试冒充真实结果。

---

## 一、A 部分六项返修对照

| # | 任务书要求 | 状态 | 位置 |
|---|---|---|---|
| A.1 | 路由以**文件字节 SHA256** 为锚；核对任务/折/`selection_data_segments=[cal,dec]`/`route.selected`；数值回退走独立路径 | ✅ | [`model/predict_test.py`](model/predict_test.py) `load_route` + `check_route`；[`model/numeric_fallbacks.py`](model/numeric_fallbacks.py) |
| A.2 | 先按 train+calibration 重演核对选择期哈希，再扩至 decision 另存 `test_fit_weight_hash` | ✅ | `predict_test.py` 两阶段重拟合 |
| A.3 | 重算 `signature(manifest.inputs)`；`--split-spec` 变路径必填、自己算字节哈希 | ✅ | [`model/bundle_reader.py`](model/bundle_reader.py) `verify_bundle` |
| A.4 | 从冻结 spec 读四段边界传入 `assert_grid`；跨段窗口拒绝 | ✅ | `segment_bounds` + `assert_grid(..., bounds, horizon)` |
| A.5 | 半段规则与主控 `core.py` 一致 | ✅ | `bundle_reader.decision_halves` |
| A.6 | 更新 README/RUNBOOK/TEAM_SYNC | ✅ | 本目录 |

### 主控在第三次验收抓到的 5 个问题

| 编号 | 问题 | 确认 | 修复 |
|---|---|---|---|
| **P0-1** | 测试入口核对**两个不同训练集合**的权重哈希 —— 选择期是 train+cal，却在 train+cal+dec 上重拟合后要求哈希相同，**真实测试会被错误拒绝** | ✅ 属实 | 两阶段：先重演核对旧哈希，再扩展并**另存** `test_fit_weight_hash` |
| **P0-2** | 入口要求主控不产出的 `status`/`route_sha256` 字段，**不能直接交接**；且**未比对 `route.selected` 与 `--candidate`**，可用未中选候选出测试预测 | ✅ 属实 | 改为文件字节 SHA256 为锚；强制 `selected` 与 `--candidate` 一致；`numeric_fallback` 走独立回退路径 |
| **P1-3** | 只比对 `manifest.signature`，**未独立重算** `signature(manifest.inputs)`，清单输入锚可伪造 | ✅ 属实 | 用与主控**逐字一致**的 `signature()` 重算，并核对 spec 字节哈希与内嵌 spec 对象 |
| **P1-4** | `assert_grid` 的边界参数在三个正式入口**均未传入** —— 边界测试通过 ≠ 正式入口执行了检查 | ✅ 属实 | 边界与 H 改为**必填参数**，三个入口都从冻结 spec 读取后传入 |
| **P1-5** | 半段按**保留起点的时间中位数**切，主控按 `middle=(cal_end+dec_end)//2` 的**行索引**中点切 | ✅ 属实 | 改用主控规则 |

> 五个问题**全部属实**，其中 P0-1 会让真实测试**必然失败**。这是本轮的实质修复。

---

## 二、测试：31 项全部通过（合成数据）

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付四次/model/tests/test_round4.py"
# 全部通过（31 项检查）
# 退出码 0
```

覆盖任务书「必交验证」列出的**全部负例**：

| 必交负例 | 测试 |
|---|---|
| 伪造 JSON 内路由哈希但文件字节改变 | ✅ `load_route` 拒绝 |
| 外部哈希缺失 | ✅ 拒绝（`--route-sha256` 必填） |
| 任务/候选不符 | ✅ 分别拒绝 |
| 数值回退被误当门控 | ✅ 拒绝 |
| **选择期与扩展期权重混比** | ✅ 断言两者**必然不同**，且都可独立记录 |
| 改动嵌入 spec 而文件哈希不变 | ✅ 两类都被拒绝（字节哈希 + 内嵌对象比对） |
| 正式入口跨段窗口 | ✅ `origin + h > 段末` 拒绝 |

另覆盖：四段边界与 spec 一致、`middle = (cal_end+dec_end)//2`、半段掩码逐位等于
`origin+h<=middle` / `origin>=middle`、半段不重叠、未实现回退模型明确报错。

---

## 三、B 部分仍阻塞

第三次交付验收已确认：**四份清洗 CSV 与正式 Bundle 均未到达主控/模型侧**。
主控 `split_spec_v2.json` 保持 `draft_not_for_training` 与空哈希。

```
主控收到并核对四 CSV → 冻结 spec → 数据侧生成正式 Bundle → 模型侧真实运行
        ↑ 当前卡在这里
```

本轮**没有**任何真实预测、真实零分布或真实 p 值。

---

## 四、交付清单（逐项标注）

| 要求 | 状态 |
|---|---|
| 修复后的读取/训练/置换/测试入口与测试 | ✅ `COMPLETE` |
| 全部负例测试与运行命令/退出码 | ✅ `COMPLETE`（见 §二与 RUNBOOK） |
| 真实 Agriculture 的 `cal_dec_predictions/` | `NOT_RUN` — 无 Bundle |
| 逐次 `row_null/` | `NOT_RUN` — 无 Bundle |
| `run_manifest.json`（真实任务） | `NOT_RUN` — 无 Bundle |
| `runtime_profile.json` | `PARTIAL` — 仅单次实测，**不外推** |
| 无文本与半段审计表 | ✅ 生成器 `audit_tables.py`；真实表 `NOT_RUN` |
| 模型权重 SHA256 / 配置哈希 | ✅ 机制 `weight_hash`；真实值待跑 |
| 正式 Bundle 签名 | `NOT_RUN` — Bundle 不存在 |
| 固定代码提交 | ✅ 见提交信息 |

---

## 五、目录

```
03 辅助电脑二 交付四次/
├── README.md / RUNBOOK.md / TEAM_SYNC.md
├── MANIFEST.json         model/ 全部文件的逐文件 + 合并 SHA256
├── runtime_profile.json  单次实测（不外推）
└── model/
    ├── branches.py            N/Q/S/F + SF 交互 + 无文本掩码
    ├── candidates.py          门控/诊断分类 + 协议 7 档 α
    ├── bundle_reader.py       verify_bundle（对齐主控规则）+ 边界 + 半段
    ├── numeric_fallbacks.py   Last / SeasonalNaive / AR-Ridge（A.1 回退路径）
    ├── predict_io.py          14 字段契约 + weight_hash
    ├── permutation.py         逐次记录 + 块长循环移位
    ├── train.py               训练入口
    ├── permutation_entry.py   置换入口
    ├── predict_test.py        测试段入口（路由字节锚 + 两阶段重拟合）
    ├── audit_tables.py        无文本 + 半段审计表
    └── tests/test_round4.py   31 项检查
```

前三轮目录完整保留，未改动。
