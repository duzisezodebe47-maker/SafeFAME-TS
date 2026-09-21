# 03 辅助电脑二 · 交付二次

**依据**：主控 [`team_work/main/round2/03_模型侧第二轮任务_SHY.md`](../team_work/main/round2/03_模型侧第二轮任务_SHY.md)
**分支**：`SHY`　**日期**：2026-09-21

> ## ⚠️ 本次交付**未完成正式实验链**，原因见 §三
>
> 主控冻结 Bundle **尚未交付**，本机也无法生成它。因此本轮交付的是
> **面向冻结契约实现并已通过合成数据验证的代码**，以及**可定位的阻塞报告**。
> 按任务书要求：「若算力不足…**禁止把半途文件写为完成**」——
> 真实数据的预测、null 分布与 p 值**一个都没有伪造**。

---

## 一、第一轮 P1 阻断项的修复情况

主控独立验收列了 5 条 P1。逐条对照：

| # | 阻断项 | 状态 | 位置 |
|---|---|---|---|
| 1 | **协议不一致**（字段/ID 类型/段） | ✅ 已修 | [`model/predict_io.py`](model/predict_io.py) —— 14 字段逐字对齐契约；`origin_id` 改为字符串 `Domain:hH:fF:o<idx>`；段从 1 个改为 4 个 |
| 2 | **入口锁在旧 v2 数据** | ✅ 已修 | [`model/bundle_reader.py`](model/bundle_reader.py) —— 按契约读 Bundle，优先调用数据侧 `read_bundle`；`schema.status != frozen` 一律拒收 |
| 3 | **无文本时语义分支无掩码** | ✅ 已修 | [`model/branches.py`](model/branches.py) `SemanticBranch` —— 掩码从 Bundle 传入，无文本起点**贡献严格为零**；PCA 只用有文本的行拟合 |
| 4 | **频率含义被混淆** | ✅ 已修 | [`model/branches.py`](model/branches.py) `SpectralInteractionBranch` —— SF = PCA(≤24) × 10 维谱统计的外积；`F` 明确标注为诊断消融 |
| 5 | **正式门控未实现** | ✅ 已实现 | [`model/permutation.py`](model/permutation.py) —— 999 次逐行错位，**每次重建 PCA / 尺度 / 交互 / α**；另含循环移位诊断 |

**候选分类按协议收紧**：

```
门控候选（≤2 条）  N+S+Q        N+S+Q+SF
诊断消融（不进门槛） N  N+Q  N+S  N+F  N+S+Q+F
```

`N+S+Q+F` 已按主控口径降为诊断消融，**不得称为交互版本**。

---

## 二、工程测试：51 项全部通过

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付二次/model/tests/test_round2.py"
# 全部通过（51 项检查）
```

用**合成数据**（真 Bundle 未到位），覆盖主控验收门槛中不依赖真实数据的全部条目：

| 验收门槛 | 覆盖 |
|---|---|
| 零文本行语义/交互贡献严格为零 | ✅ 掩码传到贡献层，`atol=0.0` 严格比较 |
| H=1 仍返回 N×1 | ✅ |
| 关闭某分支时扰动该分支不影响输出 | ✅ `N+S` 中扰动质量特征输出逐位不变 |
| 同种子重复结果一致 | ✅ |
| 配置哈希可核验 | ✅ |
| 逐段每样本每步恰有一行 | ✅ 契约完整性校验 |
| 一维预测被拒绝 | ✅ H=1 广播防护 |
| 训练段完全无文本时安全降级 | ✅ 标记 `s_degraded`，不造伪信号 |
| **置换每次重拟合** | ✅ **同一置换跑两次逐位一致 ⇒ 无跨次缓存** |
| **Bundle 读取器的安全校验** | ✅ 签名不符 / 未冻结 / 哈希被篡改 / origin_id 重复 / origin_id 与 index 错位，逐条拒绝 |
| 未跑满 999 时 p 为 `None` | ✅ 不伪填 |

**未覆盖**（需真实 Bundle）：冻结签名下的真实训练、真实 999 次置换与 p 值。

---

## 三、阻塞：主控冻结 Bundle 未交付

### 现象（可定位）

```
$ python .../model/train.py --bundle <路径> --task Agriculture_h12_f1
BundleUnavailable: 找不到 Bundle 清单: <路径>/manifest.json
```

### 根因

数据侧交付目录 [`辅助电脑02交付01次/`](https://github.com/duzisezodebe47-maker/SafeFAME-TS/tree/3218151885-creator)
只有**生成器**（`bundle.py` / `run.py`），**没有已生成的 Bundle**。生成它需要：

```
raw Time-MMD  →  build_text_index.py  →  encode_text.py  →  bundle.py
                                            ↑
                              需要 data_processed/v4/embeddings/
                              （MiniLM 句向量缓存，本机不存在）
```

查证：

| 依赖 | 本机状态 |
|---|---|
| `references/external/Time-MMD/`（原始数据） | ✅ 有 |
| `data_processed/v4/embeddings/`（句向量缓存） | ❌ **没有** |
| HuggingFace MiniLM 本地缓存 | ❌ **没有**（缓存目录仅 13K） |
| 已生成的 Bundle | ❌ 数据侧交付里也没有 |

### 为什么我没有自行生成

1. **越界**：编码与特征缓存是辅助电脑一的任务（其任务书明确），主控第二轮任务书也写"不复制旧 v2 的清洗/编码"。
2. **风险**：本机重编码会产出一个与数据侧**不同**的缓存，签名必然不一致，反而制造不一致证据。
3. **成本**：需联网下载 MiniLM（revision `1110a243…`），本机无缓存。

### 需要谁做什么

| 需要 | 责任方 |
|---|---|
| 生成并交付冻结 Bundle（含 `manifest.json` + 各任务 `.npy`） | 辅助电脑一 |
| 补齐 `numerical_sha256` 并把 `schema.status` 置为 `frozen` | 主控 |

**Bundle 到位后**，本目录的代码可直接运行，无需改动：

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付二次/model/train.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --candidate N+S+Q --segment calibration decision --output-dir evidence/
```

---

## 四、需要主控澄清的一处协议冲突

| 来源 | α 网格 |
|---|---|
| 任务书正文 | 「沿用仓库 `RIDGE_ALPHAS`」→ **9 档**（`1e-4` … `1e4`） |
| `split_spec_v2.json` 的 `alpha_grid` | **7 档**（`0.01` … `10000`，无 `1e-4`/`1e-3`） |

**本实现以机器可读的协议文件为准**（7 档）。请确认以哪个为准。

---

## 五、资源估算（供主控决策）

Bundle 到位后，跑完本轮所需资源的**实测折算**（基于合成数据的单次耗时外推）：

| 项 | 单次 | 本轮总量 | 估算 |
|---|---|---|---|
| 一次候选拟合（含 α 坐标下降 2 轮） | ~0.2 s | — | — |
| 一次置换（重建 PCA/尺度/交互/α） | ~0.1 s | 999 × 2 门控候选 | **约 3–4 分钟** |
| 循环移位诊断 | ~0.1 s | 999 × 2 | 约 3–4 分钟 |
| 四任务 × 两情景 × 全部候选 | — | — | **分钟量级，纯 CPU** |

**结论**：算力不是瓶颈。真正的瓶颈是 Bundle。

---

## 六、目录

```
03 辅助电脑二 交付二次/
├── README.md                       ← 本文件
├── RUNBOOK.md                      ← 交接说明（契约字段、复算命令、未跑范围）
├── protocol/                       ← 从主控分支复制的冻结协议（只读引用）
└── model/
    ├── branches.py                 ← P1#3 掩码 + P1#4 SF 交互
    ├── candidates.py               ← 门控/诊断分类 + 协议 α 网格
    ├── bundle_reader.py            ← P1#2 冻结 Bundle 读取与网格校验（含适配层回退）
    ├── predict_io.py               ← P1#1 14 字段契约
    ├── permutation.py              ← P1#5 999 次置换，每次重拟合
    ├── train.py                    ← 训练入口（消费 Bundle，输出校准/决策预测）
    ├── permutation_entry.py        ← 置换入口（含决策段两半段损失）
    └── tests/test_round2.py        ← 51 项检查
```

**第一轮文件完整保留**在 [`03 辅助电脑二 交付一次/`](../03%20辅助电脑二%20交付一次/)，未改动。
