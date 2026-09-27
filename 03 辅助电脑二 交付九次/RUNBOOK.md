# RUNBOOK · 第九轮复现说明

面向主控：如何核对与复现本交付。

---

## 一、固定输入

```bash
PY=.venv/Scripts/python.exe
S8="03 辅助电脑二 交付八次"
S9="03 辅助电脑二 交付九次"
BUNDLE=<正式Bundle目录>
SPEC=<主控冻结 split_spec_v2.json>
SIG=a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38
ROUTE_SHA=b3bce4713ff44485c9aa0712c5774ffe449045fb8186b39ef592298e0051e36a
SEL="$S8/evidence/cal_dec_predictions/N_run_manifest.json"
ROUTE="$S9/evidence/sealed_route.json"
```

| 输入 | 校验值 |
|---|---|
| 路由（主控 round9/sealed_route.json） | `b3bce471…e36a`（`.sha256` sidecar 同值） |
| 冻结 spec | `a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031` |
| Bundle 签名 | `a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38` |

---

## 二、复现步骤

**0. 核对路由**（不符即停，不得自行重建路由）

```bash
sha256sum "$ROUTE"     # 必须等于 $ROUTE_SHA
```

**1. 唯一一次正式测试预测**（TWO 前提：候选严格为 `N`；输出目录不存在）

```bash
$PY -B "$S8/model/predict_test.py" \
  --bundle "$BUNDLE" --task Climate_h4_f2 --scenario proxy \
  --signature $SIG --split-spec "$SPEC" \
  --route "$ROUTE" --route-sha256 $ROUTE_SHA \
  --selection-manifest "$SEL" --candidate N --seed 2026 \
  --output-dir "$S9/test_prediction"
# 期望：{"status":"completed","model":"N","rows":496,"path":"numeric_fallback_candidate", ...}
# 期望 CSV SHA256 = f2200192f4b555a3e56fb8702f59361ecf54f42d4bbc04643b8a130597d6b122
```

**2. 确定性重放 + 测试隔离审计**（第二段，只用于比对）

```bash
$PY "$S9/model/audit_test_isolation.py" \
  --bundle "$BUNDLE" --task Climate_h4_f2 --scenario proxy \
  --signature $SIG --split-spec "$SPEC" \
  --route "$ROUTE" --route-sha256 $ROUTE_SHA \
  --selection-manifest "$SEL" --candidate N --seed 2026 \
  --delivered-dir "$S9/test_prediction" --output-dir "$S9/evidence"
# 期望：replay PASS / isolation PASS，test_truth_data_access_count = 0
```

**3. 路由接受**

```bash
$PY "$S9/model/record_route_acceptance.py" --route "$ROUTE" --route-sha256 $ROUTE_SHA \
  --route-source "team_work/main/round9/results/sealed_route.json" \
  --sidecar "$S9/evidence/sealed_route.json.sha256" --split-spec "$SPEC" \
  --bundle-signature $SIG --candidate N --selection-manifest "$SEL" \
  --master-artifacts "$S9/evidence/master_ARTIFACTS_SHA256.tsv" \
  --output-dir "$S9/evidence"
```

**4. 测试**（四套，期望 49 / 85 / 131 / **160**，退出码均 0）

```bash
for t in 6 7 8 9; do $PY "$S9/model/tests/test_round$t.py"; done
```

**5. 闸门 + 清单**

```bash
$PY "$S9/model/verify_test_delivery.py" --delivery "$S9" --task Climate_h4_f2 \
    --horizon 4 --expected-origins 124 --candidate N --route-sha256 $ROUTE_SHA
$PY "$S9/model/make_manifest.py" --out "$S9/MANIFEST.json"
```

---

## 三、判读要点

### `test_isolation_audit.json` —— 两个数字别混

| 字段 | 含义 | 本轮值 |
|---|---|---|
| `test_truth_data_access_count` | test 真值**进入模型或参与计算**的次数（判 PASS 看它） | **0** |
| `truth_file_array_loads` | 含 test 真值的 `targets*.npy` 被**整表载入**的次数（文件层面，如实报出） | 2 |

`targets*.npy` 是单表含全部段：清单完整性哈希与 train/cal/dec 拟合都必须打开它，
载入后 test 行立即置 NaN。**不把它写成 0**，也不用它替代上面那一项。

### `route_acceptance.json`

`recorded` 是路由的**原样**记录。注意 `selection_manifest_sha256` 指的是
**主控自己的** `results/selection_stage/manifest.json`（第八轮同一字段同义），
不是本侧的 run manifest；判据里用主控 `ARTIFACTS_SHA256.tsv` 核这一条，
另用"本侧消费的清单候选 == `N`"核局部一致性。

### `attempt_ledger.json`

每条尝试的命令、时间、退出码、用时、stderr 尾部都在；**失败记录不删**（§三.5）。
本轮 #3/#4 曾失败：路由接受里那条 `selection_manifest` 判据原先写错了语义
（现已修正为核对主控 TSV），#4 是它的连带。修正后 #5/#6 通过。

---

## 四、未做范围

1. 未生成任何其他候选的测试预测；未做任何测试评分；未读 test 真值。
2. 未改模型结构与超参（alpha / PCA / 尺度 / 数值分支定义）。
3. 未修改第八次及更早的历史交付目录。
4. 未 `merge` 主控分支或 `main`。
