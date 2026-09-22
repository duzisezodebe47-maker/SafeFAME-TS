# RUNBOOK · 第五轮交接说明

面向主控：命令、契约、两条测试路径的差异、以及实际跑过 / 未跑的范围。

---

## 一、命令

### 测试（无需 Bundle）

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付五次/model/tests/test_round5.py"
# 期望输出：全部通过（35 项检查）
# 期望退出码：0
```

### 正式入口（**均需正式 Bundle；当前不可运行**）

```bash
# 1) 选择期预测（calibration + decision）
.venv/Scripts/python.exe "03 辅助电脑二 交付五次/model/train.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --signature <sig> --split-spec <冻结spec路径> \
    --segments calibration decision --candidate "N+S+Q" --output-dir <目录>

# 2) 决策段 999 次置换
.venv/Scripts/python.exe "03 辅助电脑二 交付五次/model/permutation_entry.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --signature <sig> --split-spec <冻结spec路径> \
    --candidate "N+S+Q" --nulls 999 --circular-block 7 --output-dir <目录>

# 3) 测试段预测（**路由冻结后**）
.venv/Scripts/python.exe "03 辅助电脑二 交付五次/model/predict_test.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --signature <sig> --split-spec <冻结spec路径> \
    --route <主控路由.json> --route-sha256 <路由文件字节SHA256> \
    --candidate <必须等于 route.selected 或 route.fallback> \
    [--selection-manifest <选择期 run_manifest.json>] \
    --output-dir <目录>

# 4) 无文本与半段审计表
.venv/Scripts/python.exe "03 辅助电脑二 交付五次/model/audit_tables.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --signature <sig> --split-spec <冻结spec路径> \
    --candidate "N+S+Q" --output-dir <目录>
```

---

## 二、`predict_test.py` 的两条路径（重要）

| | 门控候选（`N+S+Q` / `N+S+Q+SF`） | 主控数值基线（`Last`/`SeasonalNaive`/`AR-Ridge`） |
|---|---|---|
| `route.selected` | 该候选名 | `numeric_fallback` |
| `--candidate` | **必须等于** `route.selected` | **必须等于** `route.fallback`（A.1 严格相等） |
| `--selection-manifest` | **必填** | **不需要**（A.3：主控只给预测 CSV + 配置哈希） |
| 权重重演 | 两阶段（train+cal 重演 → 扩至 +dec） | 不适用 |
| 实现 | `candidates.BranchResidualCandidate` | `numeric_fallbacks.numeric_baselines`（与主控逐行等价） |
| 输出标记 | `path == "gate_candidate"` | `path == "master_numeric_baseline"` |
| `N` | — | 属**模型候选型**回退，仍走选择期清单路径 |

**A.4**：路由必须**同时**携带 `split_spec_sha256` 与 `bundle_signature`。
缺任一即拒绝 —— 旧的无锚 smoke 路由**不能**作为正式测试许可证。

**A.5**：测试段以**仅含特征**的推理视图喂给模型，`targets`/`targets_standardized`
不传入。评分由主控另一进程读真值完成。

---

## 三、`--route-sha256` 怎么算

```bash
sha256sum <主控路由.json>                              # Linux/macOS
Get-FileHash -Algorithm SHA256 <主控路由.json>         # Windows PowerShell
```

必须是对**路由文件字节**算的，不是路由 JSON 里任何自称的字段。

---

## 四、数值基线的等价性（A.2）

`numeric_fallbacks.numeric_baselines` 是主控 `team_eval/v2.py::numeric_baselines`
的**逐行等价移植**，刻意不做任何"改进"（连 `np.linalg.solve` 与 `(loss, alpha)`
平局规则都保留），否则两边会算出不同结果。

测试把主控实现抄录为参考函数逐点比对：α、校准 MSE、三个基线的逐起点逐步输出、
拟合行数 —— 全部在**相对容差 1e-10** 内一致。

---

## 五、实际运行范围

| 项 | 状态 |
|---|---|
| A 部分五项修复 | ✅ `COMPLETE` |
| 端到端负例 35 项 | ✅ `COMPLETE` |
| 资源实测 | ✅ `PARTIAL`（`runtime_profile.json`，不外推） |
| 真实 Agriculture 选择期预测 | `NOT_RUN` — 无正式 Bundle |
| 999 次置换 | `NOT_RUN` — 无正式 Bundle |
| 测试段预测 | `NOT_RUN` — 无 Bundle，路由未冻结 |
| 四任务扩展 | `NOT_RUN` |

---

## 六、已知限制

1. **真实数据未验证**：全部测试基于合成 Bundle；主控冻结 spec 的 SHA256
   `a0947a5b…f0031` 尚未用于实际核对。
2. **`SeasonalNaive` 的季节周期**由 `--seasonal-period` 传入（默认 12）。
   若主控对四任务各有规定周期，需按任务传入以免与主控选路不一致。
3. **坐标下降近似**：逐分支 α 用 2 轮坐标下降；1 轮/3 轮敏感性分析 `NOT_RUN`。
4. **`runtime_profile.json` 不外推**：只报单次实测。
5. **`N` 的清单路径未在真实数据上跑过**：合成测试覆盖了代码路径，真实清单未到手。
