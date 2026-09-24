# 03 辅助电脑二 · 交付七次

**依据**：主控 [`team_work/main/round7/02_辅助电脑二第七次交付指令_封印路由后一次性测试预测.md`](../team_work/main/round7/02_辅助电脑二第七次交付指令_封印路由后一次性测试预测.md)
**分支**：`SHY`　**日期**：2026-09-24
**预测产物锚定的代码提交**：`b9934da`（`predict_test.py` 与 `bundle_reader.py` 在
`b9934da` 与当前 HEAD 上**逐字节相同**，可用 `git show` 复核）

> 对主控封印的唯一路由 `N+S+Q+SF` 完成**一次** Agriculture 测试段预测：
> **504 行 = 42 起点 × 12 步**。
> **本侧不报告任何测试性能** —— 未计算、未读取 test 真值，评分由主控独立完成。

---

## 〇、⚠️ 先说一件事：第六次交付里的 `predict_test.py` 是**残缺的**（本轮修复）

准备第七轮时发现：`03 辅助电脑二 交付六次/model/predict_test.py` 只有 **359 行**，
`main()` 在**主控基线分支中途**断掉 —— 门控候选预测路径、`PredictionWriter` 写出、
`return 0`、`if __name__ == "__main__"` 守卫**全部缺失**（第五次交付同名文件是 386 行且完整）。

**后果是静默的**：文件仍能 `ast.parse`，且没有 `__main__` 守卫，所以按第七轮任务书
跑那条命令会「**退出码 0 但什么都不写**」—— 第七轮交付根本无法产生。三个独立旁证：
`merge_bundles`/`PredictionWriter`/`code_commit` 导入未使用、`_fit` 定义未调用、
`expected_weight_hash` 赋值未消费。

**为什么验收没抓到**：`tests/test_round6.py` 只从 `predict_test` 导入了
`RouteRejected/check_route/spec_candidate_lists`，端到端 `run_entry` 跑了
train / permutation_entry / audit_boundaries / audit_tables，**唯独没跑
`predict_test.main()`**；主控复跑的是同一套 49 项，所以同样覆盖不到。

**排查范围**：六次 model 的 16 个文件里**只有这一个**受损（其余可 `ast.parse`、
结构完整、末尾函数与五次一致）。截断已固化进 `ce1efcc` 提交与 `MANIFEST.json`
（哈希匹配截断文件），git 与磁盘都没有第六轮完整版，故按**第五次版本的尾部移植**
补齐 —— 只恢复，不改预测逻辑（门控候选分支自第四轮起未变）。

**对第六次交付结论的影响**：**没有影响**。第六次交付的选择期预测与 999 次置换走的是
`train.py` / `permutation_entry.py`，不经过 `predict_test.py`；主控验收所依据的
16 个模型文件哈希、4968 点基线逐位相同、49 项测试、权重重放结论全部仍然成立。
本次只补齐尾部并加了第七轮要求的护栏，**预测路径的算法一字未改**。

---

## 一、唯一授权路由（先核对，后运行）

| 项 | 值 | 核对 |
|---|---|---|
| 任务 / 情景 / 候选 | `Agriculture_h12_f1` / `proxy` / **`N+S+Q+SF`** | ✅ 与路由 `selected` 一致 |
| 路由来源 | 主控分支 `team_work/main/round7/results/sealed_route.json` | ✅ 原样收录于 `evidence/` |
| 路由 SHA256 | `8b2c47db4909a755f409eb6e0ea7123b9664b17dbdc3790908ecdd2802df4d70` | ✅ 实测一致 |
| 主控 `.sha256` sidecar | 同值 | ✅ 三方一致 |
| 冻结 spec | `a0947a5b…6f0031` | ✅ 实测一致 |
| Bundle 签名 | `a69821be…86d38` | ✅ 校验通过 |
| 路由 `status` / 段 | `frozen` / `["cal","dec"]` | ✅ |

`evidence/route_acceptance.json` 记录 11 项一致性检查，全部 PASS。
**任何一项不符都不产出预测**（脚本非 0 退出）。

---

## 二、测试隔离：结构性保证 + 可核对证据

### 做法

入口**硬编码** `read_frozen_bundle(..., isolate_test=True)`（无开关）：载入后
**立即**把 test 段的 `targets`/`targets_standardized` 置 NaN（保持 dtype，避免浮点差异），
再交给任何调用方；`assert_grid` 与 test 视图一律走 `with_targets=False`，
使 test 真值连**切片**都不发生；模型只拿到仅特征视图（`targets=None`）。

> 为什么不是"不打开 `targets.npy`"：该 `.npy` 是**单表含全部段**（train/cal/dec/test 同行），
> 清单完整性哈希与 train/cal/dec 拟合都必须打开它。**可验证**的保证是
> 「进程里不存在可用的 test 真值」，本交付给出四重证据。

### 四重证据

| # | 证据 | 产物 |
|---|---|---|
| 1 | **读文件清单**（包装 `builtins.open`/`io.open`/`Path.open`/`numpy.load`/`pandas.read_csv`，逐路径记录） | `evidence/test_isolation_audit.json` |
| 2 | **通道区分**：95 个文件被接触，其中 **8 个当作数据载入**、**87 个仅做完整性哈希** | `evidence/bundle_read_channels.json` |
| 3 | **静态核查**：入口源码里除选择期校准 MSE 外无任何误差计算（`mse`/`mae`/`r2_score`/`residual` 等） | 同上两份 JSON |
| 4 | **扰动实验**：把 test 真值改成随机数后预测**逐字节不变** | `model/tests/test_round7.py` |

第 2 条的 8 个数据载入文件（本任务）：
`numeric_history.npy`、`origin_index.npy`、`quality.npy`、`semantic.npy`、
`text_available.npy`、`samples.csv`、`targets.npy`、`targets_standardized.npy`。
其中含 test 真值的只有 `targets*.npy` 两张表，原因见上方引用块。

**未做**：本侧从未打开 test 单独的真值文件（不存在）、从未计算 MSE/MAE/R²/残差、
从未生成其他候选的测试预测、未做任何人工剪裁/平滑/替换。

---

## 三、确定性重放

`evidence/replay_check.json`：同一提交、相同输入再跑一次 ——

| 检查 | 结果 |
|---|---|
| 预测 CSV **逐字节相同** | ✅ |
| `weight_hash` / `test_fit_weight_hash` 相同 | ✅ |
| `alpha_by_group` 相同 | ✅ |
| 行数相同（504） | ✅ |

> `code_commit` 是**溯源列**（记录每次运行时的 HEAD），不参与"预测内容"判等：
> 只要两次运行之间发生过提交，整文件字节就会不同。报告里同时给出
> `content_identical`（判 PASS 看它）与 `byte_identical`，并把两次的
> `code_commit` 并列，避免把"溯源列变了"误读成"预测变了"。

---

## 四、执行台账（真实 Bundle 上的每一次执行）

真实 Bundle 上**预测入口共执行 5 次**，全部是**同一授权配置**（同一候选、同一路由、
同一 seed），没有任何一次读取 test 真值或产生评分。逐条登记如下（源自
`evidence/test_run.log`）：

| # | 时间 | 目的 | 提交 | 行数 | 结果 |
|---|---|---|---|---|---|
| 1 | 19:56:24 | 交付物 | `4fd2c0c` | 504 | ✅ 后被 #4 取代 |
| 2 | 19:56:43 | 受监测重放 | `4fd2c0c` | — | ✅ 预测逐字节相同；但审计的 provenance 作用域指向当时**尚未提交**的七次 `model/`，报告的 `worktree_dirty=true`（仅元数据） |
| 3 | 19:57:30 | 修正作用域后重放 | `4588e79` | — | 内容相同；但 #1 与 #3 之间发生过提交 → `code_commit` 列不同 → 整文件字节不同 |
| 4 | 19:58:28 | **交付物（最终）** | **`b9934da`** | **504** | ✅ **本目录交付的就是这一次** |
| 5 | 19:58:40 | **受监测重放（最终）** | **`b9934da`** | — | ✅ **逐字节相同**（与 #4 同提交） |

另有两次**不运行预测**的调用：通道探针（`probe_bundle_reads.py`）与路由接受记录
（`record_route_acceptance.py`）—— 它们只读 Bundle/路由，不影响任何锚点。

**为什么会有 2–3 次**：#2 暴露了审计脚本 provenance 作用域写错（指向未提交的目录），
修好后又因"两次运行之间隔了一次提交"导致溯源列不同 —— 于是把两次都放到**同一提交**
（`b9934da`）重跑，得到 #4 与 #5 的逐字节一致。原因如实记录，供主控判断。

---

## 五、交付物

| 要求（任务书 §四） | 状态 |
|---|---|
| 唯一测试预测 CSV（42 × 12 = 504 行） | ✅ `test_prediction/N_S_Q_SF_test_predictions.csv` |
| 运行 manifest（候选/任务/情景/四锚/代码提交/工作区状态/α/权重哈希/行数/列名/运行时间/峰值内存） | ✅ `test_prediction/N_S_Q_SF_test_manifest.json` |
| `test_isolation_audit.json`（监测方法 + 读文件清单） | ✅ `evidence/` |
| `route_acceptance.json`（路由 SHA / selected / status / selection·refit 锚） | ✅ `evidence/` |
| `replay_check.json`（逐字节一致 + 权重哈希一致） | ✅ `evidence/` |
| `MANIFEST.json`（本轮全部文件 SHA256） | ✅ |
| `README.md` / `RUNBOOK.md` | ✅ |

**预测 CSV**：`sha256 = 39ff28868b78f990ef791e548e7014670e1c617aa5e99a6000c6980b7737065a`
**manifest 关键值**：`n_rows=504`、`n_test_origins=42`、`code_commit=b9934da`、
`weight_hash=713a94de…`（与第六次选择期清单一致）、`test_fit_weight_hash=c1df9596…`

---

## 六、测试

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付七次/model/tests/test_round6.py"   # 49 项
.venv/Scripts/python.exe "03 辅助电脑二 交付七次/model/tests/test_round7.py"   # 85 项 = 49 + 36
```

第七轮新增 36 项覆盖任务书 §五 的 6 类负例：路由 SHA 错 → 拒绝；候选不是
`N+S+Q+SF` → 拒绝；选择期 manifest 的 `alpha_by_group` 或 `weight_hash` 被改 → 拒绝；
test 真值不可用（隔离断言 + 扰动后预测逐字节不变）；输出目录非空 → 拒绝覆盖；
行数/步数不符、键重复、origin 越界、目标窗口跨界、非有限值 → 拒绝（内存校验与
**回读文件校验**两条路径各测一遍）。

> 主控复跑过的只有第六轮的 49 项。本轮 85 项的日志已入库
> （`evidence/tests_round6.log`、`evidence/tests_round7.log`）。

---

## 七、未做（任务书硬边界）

- **未计算任何测试指标**（MSE / MAE / R² / 残差 / 区间 / 排序），未读取 test 真值。
- **未生成其他候选**的测试预测（`N`、`N+Q`、`N+S+Q`、AR-Ridge 等一概未跑）。
- **未改** alpha、PCA、尺度、交互定义或候选结构 —— 全部沿用选择期冻结清单。
- **未回看测试结果调整任何东西**（本侧根本看不到）。
- Climate / SocialGood / Environment 未启动（按主控顺序，待 Agriculture 测试闭环）。
- 未 `merge` `team/main-eval-20260921` 或 `main`。

## 八、目录

```
03 辅助电脑二 交付七次/
├── README.md / RUNBOOK.md / TEAM_SYNC.md
├── MANIFEST.json
├── test_prediction/
│   ├── N_S_Q_SF_test_predictions.csv      504 行（交付预测）
│   └── N_S_Q_SF_test_manifest.json
├── evidence/
│   ├── sealed_route.json + .sha256        主控路由（原样收录，字节一致）
│   ├── route_acceptance.json              路由接受记录（11 项检查）
│   ├── test_isolation_audit.json          监测方法 + 读文件清单
│   ├── bundle_read_channels.json          通道区分（8 数据载入 / 87 仅哈希）
│   ├── replay_check.json                  确定性重放
│   ├── test_run.log                       执行台账（原样 stderr）
│   ├── tests_round6.log / tests_round7.log
└── model/                                 与六次 model 逐字节相同（18 个文件，已核对）
```

`model/` 与 `03 辅助电脑二 交付六次/model/` **逐字节相同**，可直接核对。
仅改动 `SHY` 分支。
