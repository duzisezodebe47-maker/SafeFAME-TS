# RUNBOOK · 第六轮交接说明

面向主控：复现命令、产物、未跑范围。

**本轮证据的 `code_commit` = `40293c7`**（SHY 分支，工作区干净）。
每份 manifest / summary 的 `code_provenance` 都写了 HEAD 与 `worktree_dirty`；
若在脏工作区上重跑，`worktree_dirty` 会是 `true` 且列出脏文件 —— 请据此判退。

---

## 一、固定输入（从 Release 下载并校验）

```bash
gh release download data-bundle-a69821be115445265cde \
  --repo duzisezodebe47-maker/SafeFAME-TS \
  --pattern "SafeFAME-TS_formal_bundle_*.zip" --dir <D盘目录>
sha256sum <zip>   # 应为 8abde5043f7399c96d269b5adbbc35dcbcada0e8760114a41b603031c01f0e06
```

**冻结 spec**（**从主控分支取，不要用工作区同名文件**）：

```bash
git show origin/team/main-eval-20260921:team_work/main/round2/split_spec_v2.json > <spec路径>
sha256sum <spec路径>   # 应为 a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031
```

**主控基线**（交叉验证用，来自主控第五·六轮产物）：

```
numeric_selection_predictions.csv       sha256 3acfd6d29a3980a4bbd192fa49c789b9de6de4ea8de8b90144d8aa86a037bef8
baseline_selection_manifest.json        bundle_manifest_sha256 6c889af1a687592453a5d1d4d83dc2b4eaf9118653f1ac1085b1997c91800143
```

---

## 二、复现命令（按顺序）

```bash
PY=.venv/Scripts/python.exe
M="03 辅助电脑二 交付六次/model"
B=<Bundle目录>
S=<冻结spec路径>
SIG=a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38
```

**1. 四候选选择期预测**

```bash
for C in N N+Q N+S+Q N+S+Q+SF; do
  $PY "$M/train.py" --bundle "$B" --task Agriculture_h12_f1 --scenario proxy \
      --signature $SIG --split-spec "$S" --segments calibration decision \
      --candidate "$C" --output-dir "03 辅助电脑二 交付六次/evidence/cal_dec_predictions"
done
```

**2. 权重重演**（须在第 1 步之后，它比对的是已交付的 CSV）

```bash
$PY "$M/audit_replay.py" --bundle "$B" --task Agriculture_h12_f1 --scenario proxy \
    --signature $SIG --split-spec "$S" \
    --delivered-dir "03 辅助电脑二 交付六次/evidence/cal_dec_predictions" \
    --output-dir "03 辅助电脑二 交付六次/evidence/replay"
```

**3. 四段边界审计**

```bash
$PY "$M/audit_boundaries.py" --bundle "$B" --task Agriculture_h12_f1 --scenario proxy \
    --signature $SIG --split-spec "$S" \
    --output-dir "03 辅助电脑二 交付六次/evidence/boundaries"
```

**4. 主控基线逐点交叉验证**

```bash
$PY "$M/audit_baselines.py" --bundle "$B" --task Agriculture_h12_f1 --scenario proxy \
    --signature $SIG --split-spec "$S" \
    --master-csv <主控CSV> --master-manifest <主控清单> \
    --output-dir "03 辅助电脑二 交付六次/evidence/baseline_crosscheck"
```

**5. 两门控候选各 999 次置换**（各约 1–5 分钟）

```bash
for C in N+S+Q N+S+Q+SF; do
  $PY "$M/permutation_entry.py" --bundle "$B" --task Agriculture_h12_f1 --scenario proxy \
      --signature $SIG --split-spec "$S" --candidate "$C" --nulls 999 --circular-block 7 \
      --output-dir "03 辅助电脑二 交付六次/evidence/row_null/$C"
done
```

**6. 测试套件**

```bash
$PY "$M/tests/test_round6.py"     # 期望：全部通过（49 项检查），退出码 0
```

**7. 生成交付清单**（在全部产物就位之后跑）

```bash
$PY "$M/make_manifest.py"
```

---

## 三、逐次日志与主控转换器的契约

`null_scores.csv` —— **一行一次迭代**，成功与失败都在，种子绝不错配：

| 列 | 说明 |
|---|---|
| `iteration` | 第几次（0-based） |
| `seed` | 该次的种子（`2026*1000 + iteration`） |
| `status` | `ok` / `failed` |
| `loss` | 成功时为决策段 MSE；失败时为空 |
| `error` | 失败原因；成功时为空 |

⚠️ **这五列的顺序是锁死的**：主控 `null_bridge.convert_nulls` 用
`reader.fieldnames != [...]` 做相等判断，**多一列即硬失败**。因此「模型代码提交 /
Bundle 签名 / 置换配置」**不进这张 CSV**，而写在同目录的
`permutation_summary.json`（`convert_nulls` 只按键取值，允许附加键）：

```json
{
  "code_commit": "<40hex>", "bundle_signature": "<64hex>", "config_sha256": "<64hex>",
  "code_provenance": {"code_commit": "...", "worktree_dirty": false, "dirty_files": []},
  "runtime": {"wall_seconds": ..., "cpu_seconds": ..., "peak_rss_bytes": ..., ...},
  "seasonal_period": {"value": 12, "source": "...::seasonal_periods", "applies_to": "..."},
  "permutation_config": {"nulls_requested": 999, "seed_base": 2026,
                          "seed_rule": "local_seed = seed_base * 1000 + iteration",
                          "circular_block": 7, "refit_each_iteration": true, ...}
}
```

`null_scores_summary.json` 含 `requested` / `successful` / `failed` / `contract_minimum` /
`observed_loss` / `p_value`。**`requested` 与 `successful` 均达 999 时 `p_value` 才非空**，
未达则 `p_value=null` 且 `p_value_note` 给出原因。

`permutation_run.log`（候选目录下与 `evidence/` 下各一份）是**完整 stderr**，含每 100 次
的「失败 0」进度行 —— 主控转换器声明它「无法证明每行损失来自独立重拟合，除非有模型执行
日志可审计」，这份日志就是那个证据。

---

## 四、本轮真实结果

| 候选 | 行数 | 观察决策损失 | 经验 p | 循环移位 p（诊断） |
|---|---|---|---|---|
| `N` | 1656 | — | — | — |
| `N+Q` | 1656 | — | — | — |
| `N+S+Q` | 1656 | 0.185302 | **0.761** | 0.347 |
| `N+S+Q+SF` | 1656 | 0.155100 | **0.014** | 0.031 |

**耗时口径要看清**（上一版 README 在这里写错了）：

| 候选 | row-null 999（`null_scores_summary.json`） | 循环移位 999 | 合计（`permutation_summary.json`） |
|---|---|---|---|
| `N+S+Q` | 40.62 s | 40.67 s | **81.72 s** |
| `N+S+Q+SF` | 167.64 s | 160.3 s | **328.5 s** |

CPU / 内存实测（`permutation_summary.json::runtime`）：

| 候选 | 墙钟 | CPU 时间 | 峰值 RSS |
|---|---|---|---|
| `N+S+Q` | 81.718 s | 112.828 s | 155.5 MiB |
| `N+S+Q+SF` | 328.496 s | 488.156 s | 169.7 MiB |

运行环境：Windows 11、Python 3.12.10、32 核。

**主控资格判据只走其中一部分**：本侧交的是观察 MSE 与零分布；「决策 MSE < 回退」
「两个半段都严格改善」「经验 p ≤ 0.025」的完整资格由主控 `core.freeze_route` 复算。

---

## 五、已知限制

1. **测试段未跑**：等主控封印路由并公布唯一 SHA256。
2. **仅 Agriculture**：Climate / SocialGood / Environment 待 Agriculture 闭环后按序扩展。
3. **保守滞后情景未跑**。
4. **坐标下降近似**：逐分支 α 用 2 轮坐标下降；1 轮 / 3 轮敏感性未跑。
5. **冻结 spec 路径有歧义**：主控分支 `round2/split_spec_v2.json` 是冻结版；
   模型侧工作区同名文件是过期草稿。建议协议写明唯一路径（见 `TEAM_SYNC.md`）。
6. **模型文件集变大了**：本轮新增 5 个审计/工具模块（`audit_boundaries.py`、
   `audit_baselines.py`、`audit_replay.py`、`runtime_profile.py`、`make_manifest.py`），
   `MANIFEST.json::model_files` 由 11 个变为 16 个。主控
   `master_model_reader_compare.json` 记录的 `model_delivery_manifest_files: 11` 是上一轮的
   快照，**重建审计时请以新清单为准**。
7. **预测 CSV 与主控烟雾文件不再逐字节相同**：内容（`y_pred`）逐位不变，但
   `code_commit` 列由 `828e7b4…` 改成 `40293c7…`，故文件哈希变了。
   主控 `master_both_gate_grid_audit.json` 里记录的两个烟雾 CSV 哈希
   （`a13b35ae…` / `5f37b80a…`）对应的是**上一版**；正式交付请以本目录的
   `*_run_manifest.json` 与 `audit_replay.py` 的产物为准。
