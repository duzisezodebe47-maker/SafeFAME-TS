# RUNBOOK · 第四轮交接说明

面向主控：修正后的命令、契约、以及**实际跑过 / 未跑**的范围。

---

## 一、命令（第四轮，参数已变）

### 测试（无需 Bundle）

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付四次/model/tests/test_round4.py"
# 期望输出：全部通过（31 项检查）
# 期望退出码：0
```

### 正式入口（**均需正式 Bundle；当前不可运行**）

```bash
# 1) 选择期预测（calibration + decision）
.venv/Scripts/python.exe "03 辅助电脑二 交付四次/model/train.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --signature <bundle_signature> --split-spec <冻结spec路径> \
    --segments calibration decision --candidate "N+S+Q" --output-dir <目录>

# 2) 决策段 999 次置换
.venv/Scripts/python.exe "03 辅助电脑二 交付四次/model/permutation_entry.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --signature <sig> --split-spec <冻结spec路径> \
    --candidate "N+S+Q" --nulls 999 --circular-block 7 --output-dir <目录>

# 3) 测试段预测（**路由冻结后**）
.venv/Scripts/python.exe "03 辅助电脑二 交付四次/model/predict_test.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --signature <sig> --split-spec <冻结spec路径> \
    --route <主控路由.json> --route-sha256 <路由文件字节SHA256> \
    --selection-manifest <选择期 run_manifest.json> \
    --candidate "N+S+Q" --output-dir <目录>

# 4) 无文本与半段审计表
.venv/Scripts/python.exe "03 辅助电脑二 交付四次/model/audit_tables.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --signature <sig> --split-spec <冻结spec路径> \
    --candidate "N+S+Q" --output-dir <目录>
```

### 相对第三轮的参数变化

| 第三轮 | 第四轮 | 原因 |
|---|---|---|
| `--split-spec-sha256 <哈希字符串>` | **`--split-spec <文件路径>`** | 调用方传"声称的哈希"无法防伪造；改为本入口自己算文件字节哈希（P1-3） |
| `--route-sha256`（可选、比 JSON 内字段） | **必填、比文件字节 SHA256** | JSON 内字段可自洽伪造（P0-2） |
| 入口不比对 `route.selected` | **强制比对** | 否则可用未中选候选出测试预测（P0-2） |

---

## 二、`--route-sha256` 怎么算

```bash
# Linux/macOS
sha256sum <主控路由.json>

# Windows PowerShell
Get-FileHash -Algorithm SHA256 <主控路由.json>
```

**必须是对路由文件字节算的**，不是路由 JSON 里任何自称的字段。

---

## 三、预测契约（14 字段，未变）

`task_id, fold_id, origin_id, origin_index, segment, scenario, candidate_id, seed,
step, y_pred, target_scale, bundle_signature, config_sha256, code_commit`

预测键 = `(task_id, fold_id, segment, origin_id, step, candidate_id, seed)`，重复即拒绝。
`predictions.csv` 与 `null_scores.csv` 严格分离。

---

## 四、两阶段权重哈希（P0-1，重要）

`predict_test.py` 现在分两阶段，**两个哈希含义不同、绝不混比**：

| 阶段 | 训练集合 | 用途 |
|---|---|---|
| 重演 | train + calibration（与选择期相同） | 与选择期 `weight_hash` 核对，**不一致才报错** |
| 扩展 | train + calibration + decision | 生成测试预测，哈希记为 `test_fit_weight_hash` |

`*_test_manifest.json` 里同时给出 `selection_rows` / `extended_rows` / `weight_hash` /
`test_fit_weight_hash` / 各段输入 SHA256，供主控独立核对。

---

## 五、实际运行范围

| 项 | 状态 |
|---|---|
| A 部分六项返修 | ✅ `COMPLETE` |
| 负例测试 31 项 | ✅ `COMPLETE` |
| 单次资源实测 | ✅ `PARTIAL`（见 `runtime_profile.json`） |
| 真实 Agriculture 训练 | `NOT_RUN` — 无正式 Bundle |
| 999 次置换 | `NOT_RUN` — 无正式 Bundle |
| 测试段预测 | `NOT_RUN` — 无正式 Bundle，且路由未冻结 |
| 四任务扩展 | `NOT_RUN` |

---

## 六、已知限制

1. **真实数据未验证**：全部测试基于合成 Bundle。读取层两条路径共用同一套
   `verify_bundle`，但真实 Bundle 未到手。
2. **`AR-Ridge` 回退为自实现**：主控协议只给了名字，本模块按 direct multi-step
   的滞后 Ridge 实现，α 在训练段内部选。若主控对 AR 设计另有规定，需对齐。
3. **`runtime_profile.json` 不外推**：只报单次实测，按任务书要求不做总时长外推。
4. **坐标下降近似**：逐分支 α 用 2 轮坐标下降；1 轮/3 轮敏感性分析 `NOT_RUN`（需真实数据）。
5. **半段按行索引切**：与主控 `core.py` 一致；但本侧不产出 `middle` 之外的分界，
   若主控改了规则需同步。
