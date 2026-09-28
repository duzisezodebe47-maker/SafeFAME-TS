# 辅助电脑二交付十次

**依据**：主控 [`team_work/main/round10/02_辅助电脑二第十次交付指令.md`](../team_work/main/round10/02_辅助电脑二第十次交付指令.md)
**分支**：`SHY`　**日期**：2026-09-28

> 两项工程工作：**A** 修复 Climate 第九次交付的可验证证据链（**不重跑、不改写预测**）；
> **B** 为 SocialGood 建立零文本覆盖时的安全退化契约与测试（**不生成正式预测**）。

---

## A. Climate 证据链修复

主控第九轮验收把交付判为「有条件通过」，保留项是：随交付提交的
`evidence/verify_test_delivery.json` 标着 `INCOMPLETE`（清单哈希与最终 Git blob 有差异），
而干净检出里重跑验证器其实得到 `COMPLETE` —— **交付证据没有做到"克隆后无需解释即可自证"**。

### 根因两条

| 编号 | 问题 | 修法 |
|---|---|---|
| A.2 | 清单对**工作区字节**取 sha256；`.gitattributes` 把文本归一成 LF 后，磁盘上的 CRLF 与检出字节不符 | 清单改记 **git 会存储的 blob 字节**（`git hash-object --path` 后 `cat-file blob`）→ 提交后必然一致 |
| A.3 | 门控回执是「读取清单之后」写出的派生产物，**写它就在改变它自己校验的清单**；必须迭代到不动点，一旦漏写最后一次就自相矛盾（第九次正是如此） | receipt 一律**排除在清单之外**（与 `MANIFEST.json` 自身同理）→「跑门控 → 写回执 → 生成清单 → 提交」不再互相影响 |
| — | 回执曾用 shell 重定向写出 → 本机 **cp936** 中文，检出后无法按 UTF-8 读取（修复中第一次尝试就踩到） | 回执一律由 Python 以 **UTF-8 + LF** 写出 |

### 验收对照

| 任务书要求 | 结果 |
|---|---|
| A1/A6 冻结预测字节，任何变化判失败 | ✅ `f2200192…d6b122` 在**第九轮验收提交、修复后工作区、干净解包**三处完全相同 |
| A2 清单记录最终 Git blob 字节 | ✅ 生成器改为读 blob 字节；本轮 `files_normalized_on_commit = []` |
| A3 门控输出不得改变它校验的清单 | ✅ receipt 排除；回执内容可由重跑门控复现 |
| A4 全新 `git archive` 解包目录里验证必须 `COMPLETE` | ✅ **exit 0 / COMPLETE**，且检出里的回执也是 `COMPLETE` |
| A5 同时记录工作树 SHA-256 与 `git hash-object` | ✅ `evidence/round9_hash_registry.json` 逐文件登记 `worktree_sha256` / `git_blob_sha1` / `bytes_*` / `normalized_on_commit`，并说明用途不同 |

> **改动范围**：第九次目录只动了 **生成器 + 清单 + 回执** 三个文件；
> 预测 CSV、run manifest、隔离审计、重放与路由接受证据**一个字节未改**。

**两个哈希的用途别混**（A5）：
`worktree_sha256` = 磁盘字节 → 用于发现"提交时会被改写"的文件（本轮 0 个）；
`git_blob_sha1` = 仓库真正存储的对象 → 与 `git rev-parse HEAD:<path>` 可直接比对，用于跨机器核对。

---

## B. SocialGood 安全退化契约

任务 `SocialGood_h3_f1`（input_len 24、horizon 3、周期 12、边界 `[366,458,641,732]`）。
**先写契约与测试，不生成任何 SocialGood 正式预测**，等主控验收数据包后再启动训练。

`model/degradation_contract.py` 把「覆盖为零或过低时执行安全退化，不以多模态名义参赛」
写成**可执行契约**：

| 规则 | 内容 |
|---|---|
| **R1 退化义务** | 训练区可用文本行 < **2**（与 `branches.SemanticBranch` 的不可训练阈值**同一个**下限）时，含 `S`/`SF` 的候选必须退化为**注册表内**的数值或质量感知候选 |
| **R2 证据自洽** | `requested_candidate` / `effective_candidate` / `degraded` / `features_actually_used` / `reason` / `modality` 必须一致；`effective` 不含 `S`/`SF`；`modality` 只能是 `numeric` 或 `quality_aware`，**不得**是 `multimodal` |
| **R3 只用训练区统计** | `resolve_candidate()` 的签名里**没有**测试段参数 —— 退化判定在结构上读不到测试文本或测试真值 |
| **R4 非法输入拒绝** | 边界非严格递增 / 首段非正 / 段容不下 H 窗口 / 注册表为空 / `requested` 不在注册表 / 无可用退化目标 → 一律 `ContractViolation`，不猜、不兜底 |

**质量特征为常量时判为不可用** —— 常量不携带信息，让它进模型只会得到 0 系数，
却会让证据把它记成"质量感知"，那是名不副实。

`expected_origins()` 可按 spec 语义独立复算起点数，SocialGood 得：
train 340 / cal 90 / dec 181 / test 89 → **all 700、selection_region 611、gate 271**，
与主控第十轮口径的 **700 / 611 / 89** 一致。

---

## 测试

```bash
.venv/Scripts/python.exe "辅助电脑二交付十次/model/tests/test_round10.py"   # 195 项
```

| 套件 | 项数 |
|---|---|
| 第六轮 | 49 |
| 第七轮 | 36 |
| 第八轮 | 46 |
| 第九轮 | 29 |
| **第十轮（本轮新增）** | **35** |
| **合计** | **195**（任务书要求 ≥170） |

第十轮 35 项覆盖 §B4 的全部合成情形：全零文本、稀疏文本（阈值上下各一例）、
时间错位（文本只出现在测试区 → 只把训练区切片交给契约即退化）、质量全缺失 / 常量 /
正常、非法边界四例、注册表边界三例、证据篡改六例、扰动测试真值后退化决策不变，
以及「交付目录里没有任何 SocialGood 预测产物」（B5）。

---

## 验收对照（第十轮任务书）

| 判据 | 结果 |
|---|---|
| 第九次预测字节完全不变 | ✅ `f2200192f4b555a3e56fb8702f59361ecf54f42d4bbc04643b8a130597d6b122` |
| 干净目录验证为 `COMPLETE` | ✅ 见 `evidence/partA_chain_repair.json` |
| 自动测试 ≥170 且全部通过 | ✅ **195** |
| 新增 SocialGood 测试 ≥10 | ✅ **35** |
| 交付含说明、清单、测试日志、SHA-256 | ✅ 本文件 + `MANIFEST.json` + `evidence/tests_round*.log` + 哈希登记 |
| 只写 `SHY`、目录名 `辅助电脑二交付十次` | ✅ |

## 目录

```
辅助电脑二交付十次/
├── README.md / RUNBOOK.md
├── MANIFEST.json
├── evidence/
│   ├── partA_chain_repair.json       A 的修复与干净解包验证（含预测字节冻结证明）
│   ├── round9_hash_registry.json     第九次交付逐文件双哈希（A.5）
│   ├── tests_round{6,7,8,9,10}.log   五套测试日志
│   └── run.log
└── model/
    ├── make_manifest.py              ★ A 修复后的生成器（记 blob 字节 + 排除 receipt）
    ├── degradation_contract.py       ★ B：SocialGood 安全退化契约
    ├── tests/test_round10.py         ★ B：35 项契约测试
    └── （其余与第九次一致）
```
