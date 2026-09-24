# TEAM_SYNC · 模型侧第七轮同步说明

**面向**：主控　**分支**：`SHY`　**日期**：2026-09-24
**预测产物锚定的代码提交**：`de88307`

---

## 一、状态

**已按封印路由对唯一候选 `N+S+Q+SF` 完成一次 Agriculture 测试段预测：504 行
（42 起点 × 12 步）。** 路由 SHA 核对通过；测试隔离有四重证据；重放逐字节一致。
**本侧不报告任何测试性能** —— 未读取 test 真值、未计算任何指标，评分请你独立完成。

---

## 二、⚠️ 必须先看：第六次交付的 `predict_test.py` 是残缺的（本轮修复）

准备第七轮时发现 `03 辅助电脑二 交付六次/model/predict_test.py` 只有 **359 行**，
`main()` 在主控基线分支中途断掉：**门控候选预测路径、CSV 写出、`return 0`、
`if __name__ == "__main__"` 守卫全部缺失**（第五次同名文件 386 行且完整）。

**这是静默失败**：文件仍能 `ast.parse`，且没有 `__main__` 守卫 —— 按你第七轮任务书
跑那条命令会「**退出码 0 但什么都不写**」，第七轮交付根本产不出来。
三个独立旁证：`merge_bundles`/`PredictionWriter`/`code_commit` 导入未使用、
`_fit` 定义未调用、`expected_weight_hash` 赋值未消费。

**你的验收为什么没抓到**：`tests/test_round6.py` 只从 `predict_test` 导入了
`RouteRejected/check_route/spec_candidate_lists`，端到端 `run_entry` 跑了
train / permutation_entry / audit_boundaries / audit_tables，**唯独没跑
`predict_test.main()`**；你在固定工作树上复跑的正是同一套 49 项，所以同样覆盖不到。
→ 本轮把 `predict_test.py` 端到端纳入测试（第七轮 36 项新增里含 6 类负例），
这类"入口整段丢失"以后会被直接打出来。

**影响面**：**第六次交付的结论不受影响** —— 选择期预测与 999 次置换走的是
`train.py` / `permutation_entry.py`，不经过 `predict_test.py`；你记录的
16 个模型文件哈希、4968 点基线逐位相同、49 项测试、权重重放结论全部仍成立。
本次只**移植补齐尾部**（门控分支自第四轮未变），预测算法一字未改。

**你那边需要知道的哈希变化**：六次目录的 `MANIFEST.json` 因模型文件集变化
（补齐 1 个 + 新增 5 个审计/测试模块）已重新生成，`model_manifest_sha256` 不再等于
你 `refit_provenance_review.json` 里记录的 `187eb12b…`。**证据文件（预测 CSV、
置换 CSV、各 manifest）一个字节都没动**；同时七次 `model/` 与六次 `model/`
**逐字节相同**（20 个文件，可直接 `git show` 复核）。

---

## 三、预测产物

| 项 | 值 |
|---|---|
| 文件 | `test_prediction/N_S_Q_SF_test_predictions.csv` |
| 行数 | **504**（42 起点 × 12 步） |
| SHA256 | `d43f067d7c8d4c616f728e643b950b8cb060374a05df5a24f9e54c19ee6fc28f` |
| 列 | 与你 `PREDICTION_COLUMNS` 十四列逐字相同 |
| 段 / 候选 / seed | 全为 `test` / 全为 `N+S+Q+SF` / 2026 |
| `weight_hash` | `713a94de421421418fa4511cabc5ea99461ed5872c96dd8ca3cf28ccd7a904c0`（**与第六次选择期清单一一致**） |
| `test_fit_weight_hash` | `c1df959654763e61c609c79f87e488878cda8385102919351c1a403197780454`（扩展到 train+cal+dec 的拟合） |
| 输入锚 | 路由 `8b2c47db…` / Bundle `a69821be…` / spec `a0947a5b…` / 选择期清单 `3849e886…` |

---

## 四、测试隔离的四重证据

| # | 证据 | 位置 |
|---|---|---|
| 1 | 读文件清单（五个入口全部包装，逐路径记录） | `evidence/test_isolation_audit.json` |
| 2 | 通道区分：95 个文件被接触，**8 个当数据载入 / 87 个仅完整性哈希** | `evidence/bundle_read_channels.json` |
| 3 | 静态核查：入口无任何误差计算 | 同上 |
| 4 | 扰动实验：改 test 真值后预测**逐字节不变** | `model/tests/test_round7.py` |

**为什么 `targets.npy` 会出现在数据载入清单里**：它是**单表含全部段**
（train/cal/dec/test 同行），清单完整性哈希与本任务 train/cal/dec 的拟合都必须打开它。
入口的做法是 `isolate_test=True`：载入后**立即**把 test 行置 NaN，再交给任何调用方；
`assert_grid` 与 test 视图走 `with_targets=False`，test 真值连切片都不发生。
这一点如果你要求"连打开都不行"，需要数据侧把 `targets*.npy` 按段拆分
（或另出 `targets_test.npy`）—— 现结构下做不到，如实说明。

---

## 五、执行台账（真实 Bundle 上的每一次执行，逐条登记）

预测入口共执行 **7 次**，全部同一授权配置（同候选、同路由、同 seed）；
**另有 4 次不运行预测**的只读调用（通道探针、路由接受记录各 2 次）与 4 次测试套件。
完整台账见 README §四，逐条命令见 `evidence/run_ledger.json`，原始记录见
`evidence/test_run.log`（第二段会话）与 `evidence/test_run.first_session.log`
（第一段会话，从提交 `c1a63f4` 恢复 —— 第二段重跑覆盖了它，如实说明）。

要点：

- **#6（20:05:57，提交 `de88307`）= 交付的那一次**（504 行）；
- **#7（20:06:00，同一提交）= 受监测重放**，与 #6 **逐字节相同**；
- #1–#5 是被取代的中间尝试，原因三类：(a) #2 暴露审计脚本 provenance 作用域写错
  （指向当时未提交的七次 `model/`，`worktree_dirty=true`，**仅元数据**）；
  (b) #3 因"两次运行之间隔了一次提交"导致溯源列 `code_commit` 不同；
  (c) 我复核任务书时又发现 manifest 少四个扁平锚键 + 交付清单漏了 `test_prediction/`，
  修完整轮重跑（#6/#7）。

**没有一次读取 test 真值，没有一次产生评分，没有一次跑别的候选。**

---

## 六、需要你注意的两点

1. **`code_commit` 是溯源列**：它记录每次运行时的 HEAD，所以"整文件逐字节相同"只在
   两次运行处于同一提交时成立。`replay_check.json` 现在同时给出
   `content_identical`（判 PASS 看它）与 `byte_identical`，并列出两次的 `code_commit`
   与差异列计数 —— 避免把"溯源列变了"误读成"预测变了"。
2. **第六次目录被改动过**：为补齐 `predict_test.py` 尾部并加入第七轮的审计/测试模块。
   若你的验收流程要求"已验收提交不再变动"，请把这视为一组**已申报的修复提交**
   （`SHY` 上从 `4fd2c0c` 起的若干次），六次目录下**证据文件未改动**
   （`evidence_combined_sha256` 仍等于你记录的 `140dcb25…`）。

---

## 七、未做范围

| 项 | 状态 |
|---|---|
| 测试评分（MSE/MAE/R²/残差/区间） | **未做** —— 未读 test 真值 |
| 其他候选的测试预测 | **未做** |
| 改动 alpha / PCA / 尺度 / 交互 / 候选结构 | **未做** |
| Climate / SocialGood / Environment | **未启动**（待 Agriculture 测试闭环） |

未 `merge` `team/main-eval-20260921` 或 `main`。
