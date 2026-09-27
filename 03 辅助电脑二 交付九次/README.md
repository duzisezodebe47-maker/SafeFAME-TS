# 03 辅助电脑二 · 交付九次

**依据**：主控 [`team_work/main/round9/02_辅助电脑二第九次交付指令_Climate单次测试预测.md`](../team_work/main/round9/02_辅助电脑二第九次交付指令_Climate单次测试预测.md)
**分支**：`SHY`　**日期**：2026-09-27

> 按主控封印路由，对 **`Climate_h4_f2/proxy`** 跑**一次** `N` 的测试预测：
> **496 行 = 124 起点 × 4 步**。
> **本侧不报告任何测试性能** —— 未计算 MSE/MAE/RMSE/R²/残差/排名。

---

## 一、唯一授权路线（先核对，后运行）

| 项 | 值 | 核对 |
|---|---|---|
| 任务 / 情景 | `Climate_h4_f2` / `proxy` | ✅ |
| 封印结果 | `selected=numeric_fallback`、**`fallback=N`** | ✅ 本侧请求的候选严格等于 `N` |
| 路由文件 | 主控 `team_work/main/round9/results/sealed_route.json` | ✅ 原样收录于 `evidence/` |
| 路由 SHA256 | `b3bce4713ff44485c9aa0712c5774ffe449045fb8186b39ef592298e0051e36a` | ✅ 实测一致（sidecar 亦一致） |
| Bundle 签名 / 冻结 spec | `a69821be…86d38` / `a0947a5b…6f0031` | ✅ |
| 选择期清单 | 第八次交付 `evidence/cal_dec_predictions/N_run_manifest.json` | ✅ 候选一致 |

> 主控的门禁与选择期数值可由路由文件复核：`N` 的 calibration MSE 0.145452 在四个
> 数值回退里最低；两个文本候选 decision MSE（0.268329 / 0.288924）均高于回退
> 0.246491，且 `eligible=false`。**本侧只按封印路线执行**，不重新选候选、不改参数。

`evidence/route_acceptance.json` 记录 12 项一致性检查，全部 PASS；路由 SHA 与主控
`ARTIFACTS_SHA256.tsv` 里 `results/selection_stage/manifest.json` 的记录也逐项核对。

---

## 二、唯一一次正式运行（§二）

命令（与任务书逐字一致，`--candidate N`）：

```bash
python -B "03 辅助电脑二 交付八次/model/predict_test.py" \
  --bundle <正式Bundle目录> --task Climate_h4_f2 --scenario proxy \
  --signature a69821be…86d38 --split-spec <冻结spec> \
  --route "03 辅助电脑二 交付九次/evidence/sealed_route.json" \
  --route-sha256 b3bce471…e36a \
  --selection-manifest "03 辅助电脑二 交付八次/evidence/cal_dec_predictions/N_run_manifest.json" \
  --candidate N --seed 2026 --output-dir "03 辅助电脑二 交付九次/test_prediction"
```

用的是**第八次那份已过 131 项测试的入口**（逐字节未改）。运行前确认输出目录不存在。
逐次尝试（含失败）全部记在 `attempt_ledger.json` 与 `evidence/run.log`。

| 项 | 值 |
|---|---|
| 预测 CSV | `test_prediction/N_test_predictions.csv` |
| SHA256 | `f2200192f4b555a3e56fb8702f59361ecf54f42d4bbc04643b8a130597d6b122` |
| 行数 / 起点 / 步 | **496** / 124 / 4（origin 1017–1140） |
| 段 / 候选 / seed | 全为 `test` / 全为 `N` / 2026 |
| `weight_hash` | `014474d3…`（**与第八次选择期 `N` 清单一一致** → 冻结 α 重演命中） |
| `test_fit_weight_hash` | `e522b9a2…`（扩展到 train+cal+dec 的拟合） |

---

## 三、测试隔离（§三.1 / §四）

`evidence/test_isolation_audit.json`：监测方法（包装 `builtins.open` / `io.open` /
`pathlib.Path.open` / `numpy.load` / `pandas.read_csv`）+ 完整读清单（99 个路径）。

**两项分开报，不混为一谈**：

| 指标 | 值 | 含义 |
|---|---|---|
| `test_truth_data_access_count` | **0** | test 真值**进入模型或参与任何计算**的次数。模型只拿到仅特征视图（`targets=None`），test 行真值在载入后立即被置为 NaN |
| `truth_file_array_loads` | 2 | 含 test 真值的 `targets*.npy` 被**整表载入**的次数。该 `.npy` 是**单表含全部段**，清单完整性哈希与 train/cal/dec 拟合都必须打开它 —— **如实报出，不冒充为 0** |

这个口径与主控第七轮验收时的记录一致（"并非操作系统级'从未打开真值文件'……
因此本轮评分可以接受"）。支撑证据：推理视图无 targets、扰动 test 真值后预测
逐字节不变（见测试套件）、入口源码静态扫描无任何误差计算。

**未做**：未读 test 真值、未计算任何测试指标、未生成其他候选的测试预测、
未做人工平滑/裁剪/替换。

---

## 四、确定性重放（§三.6）

`evidence/replay_check.json`：相同输入再跑一次到独立临时目录 ——
预测 CSV **逐字节相同**、`weight_hash` / `test_fit_weight_hash` / `alpha` / 行数全部一致。

---

## 五、测试与负例（§五）

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付九次/model/tests/test_round6.py"   # 49
.venv/Scripts/python.exe "03 辅助电脑二 交付九次/model/tests/test_round7.py"   # 85
.venv/Scripts/python.exe "03 辅助电脑二 交付九次/model/tests/test_round8.py"   # 131
.venv/Scripts/python.exe "03 辅助电脑二 交付九次/model/tests/test_round9.py"   # 160
```

第九轮新增 29 项断言，覆盖 §五 的全部 8 类：路由 SHA 错；候选不是 `N`；路由
`fallback` 不是 `N`；选择期清单的 `alpha_by_group` / `weight_hash` / `candidate`
被改；test 真值不可用（隔离断言 + 扰动后预测逐字节不变）；输出目录已存在；
行数不是 496 / 键重复 / origin 越界 / 非有限值；**交付出现第二个候选的测试预测
→ 闸门失败**。合成夹具的 Climate 形状与真实登记一致（test 124 × 4 = 496）。

---

## 六、完备性闸门（§五.8）

`model/verify_test_delivery.py` 按产物判成败：恰好一个测试预测且必须是 `N`、
行数 = 起点 × horizon、键唯一、仅 test 段、manifest 必需字段、路由接受 PASS、
**真值被当数据访问的计数为 0**、重放逐字节一致、`attempt_ledger.json` 存在、
无**数值型**评分字段、清单哈希 == 已提交字节。结论见
`evidence/verify_test_delivery.json`。

---

## 七、未做与分支规则（§六）

- **未生成任何其他候选**的测试预测；未做任何测试评分；未读 test 真值。
- 未改 alpha / PCA / 尺度 / 数值分支定义；只按既定流程用 train+cal+dec refit。
- **只改 `03 辅助电脑二 交付九次/`** —— 第八次及更早的历史目录一字未动。
- 未 `merge` 主控分支或 `main`。

## 八、目录

```
03 辅助电脑二 交付九次/
├── README.md / RUNBOOK.md
├── MANIFEST.json
├── attempt_ledger.json              逐次尝试（命令/时间/退出码/用时）
├── test_prediction/                 N_test_predictions.csv（496 行）+ manifest
├── evidence/
│   ├── sealed_route.json(+.sha256)  主控路由（原样收录）
│   ├── master_ARTIFACTS_SHA256.tsv  主控第九轮产物清单（用于核对路由里的阶段清单哈希）
│   ├── route_acceptance.json        路由接受（12 项）
│   ├── test_isolation_audit.json    监测方法 + 读清单 + 真值访问计数
│   ├── replay_check.json            确定性重放
│   ├── verify_test_delivery.json    闸门结论
│   ├── run.log                      原始记录
│   └── tests_round{6,7,8,9}.log     四套测试日志
└── model/                           第八次 model 的拷贝 + 第九轮审计/闸门/测试
```
