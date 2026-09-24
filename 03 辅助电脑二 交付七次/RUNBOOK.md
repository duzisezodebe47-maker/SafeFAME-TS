# RUNBOOK · 第七轮复现说明

面向主控：如何从零复现本交付，以及每一步的可核对锚点。

**预测产物锚定的代码提交 = `de88307`**（写在
`test_prediction/N_S_Q_SF_test_manifest.json` 的 `code_commit` 里，`worktree_dirty=false`）。

预测路径上的全部文件在该提交、当前 HEAD、工作区**三者逐字节相同**，可这样复核：

```bash
for f in predict_test.py bundle_reader.py branches.py candidates.py train.py          predict_io.py runtime_profile.py numeric_fallbacks.py; do
  git show de88307:"03 辅助电脑二 交付七次/model/$f" | sha256sum
  sha256sum "03 辅助电脑二 交付七次/model/$f"
done
# 与当前 HEAD 的唯一差异应只有 make_manifest.py（清单生成器，不在预测路径上）：
git diff --stat de88307 HEAD -- "03 辅助电脑二 交付七次/model"
```

---

## 一、固定输入

```bash
PY=.venv/Scripts/python.exe
SEVEN="03 辅助电脑二 交付七次"
BUNDLE=<正式Bundle目录>
SPEC=<主控冻结 split_spec_v2.json>
SIG=a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38
ROUTE_SHA=8b2c47db4909a755f409eb6e0ea7123b9664b17dbdc3790908ecdd2802df4d70
```

| 输入 | 校验值 |
|---|---|
| 路由 | `evidence/sealed_route.json`（主控发布字节原样），SHA256 = `8b2c47db…f4d70` |
| 冻结 spec | `a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031` |
| Bundle 签名 | `a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38` |
| 选择期清单 | `03 辅助电脑二 交付六次/evidence/cal_dec_predictions/N_S_Q_SF_run_manifest.json` |

---

## 二、复现步骤

**0. 路由接受（先核对，后运行；不符即停）**

```bash
$PY "$SEVEN/model/record_route_acceptance.py" \
  --route "$SEVEN/evidence/sealed_route.json" --route-sha256 $ROUTE_SHA \
  --route-source "team_work/main/round7/results/sealed_route.json" \
  --sidecar "$SEVEN/evidence/sealed_route.json.sha256" \
  --split-spec "$SPEC" --signature $SIG --candidate N+S+Q+SF \
  --output-dir "$SEVEN/evidence"
# 期望 11 项检查全 true，verdict=PASS
```

**1. 唯一测试预测**（**输出目录必须先不存在**；已存在且非空会被拒绝）

```bash
$PY -B "03 辅助电脑二 交付六次/model/predict_test.py" \
  --bundle "$BUNDLE" --task Agriculture_h12_f1 --scenario proxy \
  --signature $SIG --split-spec "$SPEC" \
  --route "$SEVEN/evidence/sealed_route.json" --route-sha256 $ROUTE_SHA \
  --selection-manifest "03 辅助电脑二 交付六次/evidence/cal_dec_predictions/N_S_Q_SF_run_manifest.json" \
  --candidate N+S+Q+SF --seed 2026 \
  --output-dir "$SEVEN/test_prediction"
# 期望：{"status":"completed","model":"N+S+Q+SF","rows":504,"path":"gate_candidate",
#        "test_predictions_sha256":"d43f067d…fc28f"}
# 该值与 test_prediction/N_S_Q_SF_test_manifest.json 的 csv_sha256 相同
```

**2. 受监测重放 + 隔离审计**（输出到临时目录，不覆盖交付物）

```bash
$PY "$SEVEN/model/audit_isolation.py" --bundle "$BUNDLE" --task Agriculture_h12_f1 \
  --scenario proxy --signature $SIG --split-spec "$SPEC" \
  --route "$SEVEN/evidence/sealed_route.json" --route-sha256 $ROUTE_SHA \
  --selection-manifest "03 辅助电脑二 交付六次/evidence/cal_dec_predictions/N_S_Q_SF_run_manifest.json" \
  --candidate N+S+Q+SF --seed 2026 \
  --delivered-dir "$SEVEN/test_prediction" --output-dir "$SEVEN/evidence"
# 期望：replay PASS / isolation PASS；replay_check.json 的
#       checks.predictions_byte_identical = true
```

**3. Bundle 读取通道**（不运行预测，只读 Bundle）

```bash
$PY "$SEVEN/model/probe_bundle_reads.py" --bundle "$BUNDLE" --task Agriculture_h12_f1 \
  --scenario proxy --signature $SIG --split-spec "$SPEC" --output-dir "$SEVEN/evidence"
# 期望：files_touched=95, data_loaded=8, hashed_only=87
```

**4. 测试**

```bash
$PY "$SEVEN/model/tests/test_round6.py"    # 期望 49 项，退出码 0
$PY "$SEVEN/model/tests/test_round7.py"    # 期望 85 项，退出码 0
```

**5. 交付清单**

```bash
$PY "$SEVEN/model/make_manifest.py" --out "$SEVEN/MANIFEST.json"
# 交付根目录由 --out 的父目录决定（脚本会被复制进每个交付目录）
# 覆盖三块：model_files / evidence_files / other_files，合起来 = 本目录全部文件
#（除 MANIFEST.json 自身与 __pycache__）
```

---

## 三、产物与判读

### `test_isolation_audit.json`

- `monitoring.method`：包装了哪五个入口（`builtins.open` / `io.open` /
  `pathlib.Path.open` / `numpy.load` / `pandas.read_csv`）。
- `read_manifest`：**每一个被打开的路径**，含读取次数与分类；分类里
  `bundle_truth_table_contains_test(integrity_hash_and_fit_inputs)` 标出含 test 真值的表。
- `isolation_assertions`：隔离模式的断言（test 行 NaN、train/cal/dec 未受影响、
  测试视图不含 targets）。
- `static_scan_no_error_metric`：入口源码里除选择期校准 MSE 外无误差计算。

> 注意：`first_via` **看不出**"只被哈希"还是"被 np.load"，因为清单完整性哈希总是先发生。
> 要判这件事看下一份。

### `bundle_read_channels.json`

逐文件给出**各通道调用次数**，并分成「数据载入清单」（8 个）与「仅被哈希清单」（87 个）。
含 test 真值的 `targets.npy` / `targets_standardized.npy` 会出现在数据载入清单里，
`test_truth_tables_loaded_as_data[].why` 写明原因（单表含全部段；载入后 test 行立即置 NaN）。

### `replay_check.json`

- `checks.predictions_content_identical` —— **判 PASS 看这一项**（除 `code_commit`
  外所有列逐行相等）。
- `checks.predictions_byte_identical` —— 整文件相等；两次运行在同一提交上时为 true。
- `csv_comparison.differing_columns` —— 若整文件不同，这里告诉你差在哪列。
- `code_commit_delivered` / `code_commit_replay` —— 两次运行各自记录的提交。

### `test_prediction/N_S_Q_SF_test_manifest.json`

输入锚**同时以两种形式给出**，读 manifest 的一方按哪种约定取键都不会踩空：

- 分组视图 `input_anchors.{route_file_sha256, bundle_signature, split_spec_sha256,
  selection_manifest_sha256, selection_config_sha256}`
- 扁平键 `route_file_sha256` / `bundle_signature` / `split_spec_sha256` /
  `selection_config_sha256`（与第五次 test manifest、`train.py` 的 run manifest 同约定）

另有 `csv_columns`、`csv_sha256`、`code_provenance`、`runtime`、`test_isolation`。

---

## 四、未做范围（硬边界）

1. **无任何测试评分**：未计算 MSE/MAE/R²/残差/区间，未读取 test 真值。
2. **无其他候选**的测试预测。
3. **未改**模型结构与超参：alpha、PCA、尺度、交互全部沿用选择期冻结清单。
4. Climate / SocialGood / Environment 未启动。
5. 未 `merge` 主控分支或 `main`。
