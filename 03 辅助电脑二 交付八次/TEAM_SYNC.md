# TEAM_SYNC · 模型侧第八轮同步说明

**面向**：主控　**分支**：`SHY`　**日期**：2026-09-25
**本轮证据的 `code_commit`**：见各 manifest 的 `code_provenance`（`worktree_dirty=false`）

---

## 一、状态

**Climate 选择期门控实验已完成：四候选各 1500 行、两门控候选各 999 次逐行置换
+ 999 次循环移位、四段边界审计、权重重演、入口读文件审计、131 项测试。**

**本轮不生成任何 Climate test 预测**（§五）。测试段只做索引与特征结构审计。

---

## 二、输入：消费了你们的隔离包（§五）

Release `climate-isolated-input-v8-20260924`（ZIP SHA256 `b028052e…b314b`），
`selection_fit/` 956 行 + `test_features/` 124 行。

**本侧独立复核了包的每一条声称**，没有照信：

- 包 `MANIFEST.json` 逐文件字节哈希 ✅；锚（Bundle 签名、spec 哈希、包内 Bundle 清单）✅；
- `selection_fit_commitments.json` 的 6 张切片形状/dtype/字节哈希 ✅；
- **与正式 Bundle 逐位对照 18 张数组全部相同**、两侧 `metadata.csv` 行序一致 ✅。

你们 README 说"单靠严格隔离包不能重新证明切片来自原 Bundle"—— 本侧**持有正式
Bundle**，所以把这条也验掉了。对照时 Bundle 侧一律 `mmap_mode="r"` + 只索引选中行。

包的结构隔离有效：`test_features/` 无 `targets*`、无 `numeric_history`/`frequency`，
因此"选择期不用测试真值"是**结构性事实**，比第七轮的"读完再抹 NaN"更强一档。
入口里也加了断言：`test_features/` 出现真值数组即拒绝。

---

## 三、结果（选择期，本侧不下门控结论）

| 候选 | 观察决策损失 | 经验 p | 门槛 0.025 | 循环移位 p（诊断） |
|---|---|---|---|---|
| `N+S+Q` | 0.268329 | **0.983** | ❌ 远未通过 | 0.911 |
| `N+S+Q+SF` | 0.288924 | **0.984** | ❌ 远未通过 | 0.849 |

决策前后半段（各 124 起点，`middle=890`）：`N+S+Q` 0.252365 / 0.267640；
`N+S+Q+SF` 0.253963 / 0.308103。

**两个文本候选的经验 p 都远大于 0.025** —— 按冻结规则应当都不合格。
但「决策 MSE 是否低于数值回退、前后半段是否均改善」与最终路由由你独立复算
（你的第八轮任务 §一.6）。本侧**不替主控下结论**，也不因结果不好而放宽任何规则
（你的 §三 已写明：不得人为放宽 p 值或半段规则）。

---

## 四、这轮改掉的三处（都是复核逼出来的）

1. **循环移位块长**：上一版沿用 Agriculture 的默认值 7，没有任何 Climate 依据 ——
   正是 §三 要挡的东西。现由**冻结 spec 推导**：`block = horizon = 4 周`
   （Climate 周频、`lag_days=7`，平移整数个月，与评分窗口同阶）。
   **不取一个完整季节周期 52**：decision 段 251 行时 `n//52` 只剩 4 个位移，
   诊断会退化（各段位移数：train 145 / cal 31 / dec 62）。
   产物 `circular.block_derivation` 记录 rule/why/source，显式覆盖时记 `cli_override`，
   **只进诊断、不回写正式门控 p**。

2. **完备性闸门把诊断候选当成必交**：`verify_delivery.py` 默认候选集原是
   `ALL_CANDIDATES`（含 `N+S`/`N+F`/`N+S+Q+F`），而任务书 §二 只要求 4 个。
   已改为 `DELIVERED_CANDIDATES`，并新增两条：正式证据必须
   `input_kind=isolated_package`；出现任何 test 预测文件即失败。

3. **一处自己引入的顺序错误**：`derive_circular_block(horizon, …)` 曾写在
   `horizon = …` 之前（NameError），被第六轮的端到端测试抓出来 ——
   你 §四.2「入口必须端到端跑」这条确实值钱。

---

## 五、需要你知道的两点

1. **历史交付目录未被改动**（§四.1）：本轮全部改动只在 `03 辅助电脑二 交付八次/`。
   第七轮你指出的"改了第六次目录"的问题，本轮不再发生。
2. **执行台账**：所有命令、时间、退出码、用时逐条记在
   `evidence/run_ledger.json` 与 `evidence/test_run.log`。若有多次重跑，均如实登记。

---

## 六、未做范围

| 项 | 状态 |
|---|---|
| Climate test 预测 | **未做**（§五 禁止） |
| 读取 test 真值 / 任何测试指标 | **未做** |
| Climate 数值基线（Last/SeasonalNaive/AR-Ridge） | **未做** —— 那是你的第八轮任务 §一.2 |
| 改动 alpha / PCA / 尺度 / 交互 / 候选结构 | **未做** |
| SocialGood / Environment | **未启动**（按冻结顺序，待 Climate 闭环） |

未 `merge` `team/main-eval-20260921` 或 `main`。
