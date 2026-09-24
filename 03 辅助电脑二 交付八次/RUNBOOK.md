# RUNBOOK · 第八轮复现说明

面向主控：如何从零复现本交付，以及每一步的可核对锚点。

---

## 一、固定输入

```bash
PY=.venv/Scripts/python.exe
EIGHT="03 辅助电脑二 交付八次"
M="$EIGHT/model"
PKG=<隔离包解包目录>            # Release climate-isolated-input-v8-20260924
BUNDLE=<正式Bundle目录>          # 仅用于 --formal-bundle 逐位追溯
SPEC=<主控冻结 split_spec_v2.json>
SIG=a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38
TASK=Climate_h4_f2; SCEN=proxy
INK="--input-package $PKG --formal-bundle $BUNDLE"
```

| 输入 | 校验值 |
|---|---|
| 隔离包 ZIP | 8,109,693 B；SHA256 `b028052ed5a09868341024c5b89864490f64c3929747f7f70bcc5ae356db314b` |
| 冻结 spec | `a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031` |
| Bundle 签名 | `a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38` |

---

## 二、复现步骤

**1. 四候选选择期预测**（每个 1500 行 = 375 × 4）

```bash
for C in N N+Q N+S+Q N+S+Q+SF; do
  $PY "$M/train.py" $INK --task $TASK --scenario $SCEN --signature $SIG \
      --split-spec "$SPEC" --segments calibration decision \
      --candidate "$C" --output-dir "$EIGHT/evidence/cal_dec_predictions"
done
```

**2. 四段边界审计**（测试段只查索引与特征结构，不预测）

```bash
$PY "$M/audit_boundaries.py" $INK --task $TASK --scenario $SCEN --signature $SIG \
    --split-spec "$SPEC" --output-dir "$EIGHT/evidence/boundaries"
# 期望：PASS，selection_origin_count=375，input_kind=isolated_package
```

**3. 权重重演**

```bash
$PY "$M/audit_replay.py" $INK --task $TASK --scenario $SCEN --signature $SIG \
    --split-spec "$SPEC" --delivered-dir "$EIGHT/evidence/cal_dec_predictions" \
    --output-dir "$EIGHT/evidence/replay"
```

**4. 入口读文件审计**（包模式下"无 test 真值"是结构性事实）

```bash
$PY "$M/audit_read_isolation.py" $INK --task $TASK --scenario $SCEN --signature $SIG \
    --split-spec "$SPEC" --candidate N+S+Q \
    --output-dir "$EIGHT/evidence/read_isolation" \
    --probe-output "$EIGHT/evidence/_probe_train"
```

**5. 两门控候选各 999 + 999**（各约 4–12 分钟；块长按 spec 推导，不要传 `--circular-block`）

```bash
for C in N+S+Q N+S+Q+SF; do
  $PY "$M/permutation_entry.py" $INK --task $TASK --scenario $SCEN --signature $SIG \
      --split-spec "$SPEC" --candidate "$C" --nulls 999 \
      --output-dir "$EIGHT/evidence/row_null/$C"
done
```

**6. 测试**

```bash
$PY "$M/tests/test_round6.py"   # 49
$PY "$M/tests/test_round7.py"   # 85
$PY "$M/tests/test_round8.py"   # 131
```

**7. 完备性闸门**

```bash
$PY "$M/verify_delivery.py" --delivery "$EIGHT" --task $TASK --horizon 4 \
    --expected-origins 375
```

**8. 交付清单**

```bash
$PY "$M/make_manifest.py" --out "$EIGHT/MANIFEST.json"
```

---

## 三、隔离包怎么读（`model/isolated_package.py`）

一次载入做六件事，任一件不通过即拒绝（`PackageUnavailable`），**不降级到别的输入**：

1. 按包 `MANIFEST.json` 复核**每个文件的字节哈希**；
2. 锚一致性：Bundle 签名、冻结 spec 字节哈希、`anchors/formal_bundle_manifest.json`；
3. 按 `selection_fit_commitments.json` 复核**每张切片的形状/dtype/字节哈希**；
4. **结构隔离断言**：`test_features/` 下出现 `targets*` / `numeric_history*` /
   `frequency` 即判失败（第一版漏了 `.npy` 后缀，是合成反例试出来的）；
5. **与正式 Bundle 逐位对照**（给了 `--formal-bundle` 时）：18 张数组逐位相同、
   两侧 `metadata.csv` 行序一致。Bundle 侧一律 `mmap_mode="r"` + 只索引选中行；
6. 组包：`samples` = 选择期 956 行 + test 124 行；`arrays` 里 test 的
   `targets*`/`numeric_history`/`frequency` 一律 **NaN 占位**（占位由本模块生成，
   不是任何来源的真值），并断言其为 NaN。

`test` 段因此**结构上无法**被用来预测：包内既无测试数值历史，也无测试真值。
`bundle.isolation["test_prediction_supported"] = False`。

---

## 四、产物判读

### `evidence/row_null/<候选>/permutation_summary.json`

- `p_value`：经验单侧 p，`(1 + #{null ≤ observed}) / (1 + 999)`。
- `decision_halves`：前后半段（`middle = (cal_end + dec_end)//2 = 890`，各 124 起点）。
- `circular.block` + `circular.block_derivation`：块长与其**来源**
  （`spec_derived` = block=horizon=4；显式覆盖时为 `cli_override`）。
- `code_provenance`：HEAD + 工作区是否干净 + 范围（必须是第八次目录）。

### `evidence/read_isolation/test_read_isolation_audit.json`

- `monitoring.method`：包装了哪五个 IO 入口，并记录 `np.load` 是否带 mmap。
- `read_manifest`：逐路径、通道与分类。包模式下 `selection_fit/targets.npy`
  **只含选择期行**，不含 test 真值，因此不计为"含 test 真值的表"。
- `isolation_assertions`：模式、`test_truth_present=False`、test 行 NaN、非 test 行有限。
- `violations`：出现"含 test 真值的文件被 `np.load` 整表读入"即非空并判 FAIL。

### `evidence/replay/replay_check.json`

`candidates.<候选>` 下逐项给出：两次 `train.py` 退出码、两次是否逐字节相同、
`weight_hash` / `alpha_by_group` / `branch_widths` 是否与已交付 manifest 一致。

---

## 五、未做范围（§五 硬边界）

1. **无任何 Climate test 预测**，未读 test 真值、未算任何测试指标。
2. 未修改第一至第七次历史交付目录。
3. 未改模型结构与超参；Climate 全程重新拟合、重新选 α、重新门控。
4. 未 `merge` 主控分支或 `main`。
