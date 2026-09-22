# RUNBOOK · 第六轮交接说明

面向主控：复现命令、结果、以及未跑范围。

---

## 一、命令

### 测试（无需 Bundle）

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付六次/model/tests/test_round6.py"
# 期望：全部通过（30 项检查），退出码 0
```

### 本轮真实运行所用的命令（已执行）

**固定输入**（从 Release 下载并校验）：

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

**选择期预测**：

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付六次/model/train.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --signature a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38 \
    --split-spec <冻结spec路径> --segments calibration decision \
    --candidate "<N|N+Q|N+S+Q|N+S+Q+SF>" --output-dir evidence/cal_dec_predictions
```

**999 次置换**：

```bash
.venv/Scripts/python.exe "03 辅助电脑二 交付六次/model/permutation_entry.py" \
    --bundle <Bundle目录> --task Agriculture_h12_f1 --scenario proxy \
    --signature a69821be…86d38 --split-spec <冻结spec路径> \
    --candidate "N+S+Q" --nulls 999 --circular-block 7 \
    --output-dir evidence/row_null/N+S+Q
```

---

## 二、置换逐次日志格式

`null_scores.csv` —— **一行一次迭代**，成功与失败都在，种子绝不错配：

| 列 | 说明 |
|---|---|
| `iteration` | 第几次（0-based） |
| `seed` | 该次的种子（`2026*1000 + iteration`） |
| `status` | `ok` / `failed` |
| `loss` | 成功时为决策段 MSE；失败时为空 |
| `error` | 失败原因；成功时为空 |

`null_scores_summary.json` 含 `requested` / `successful` / `failed` / `contract_minimum` /
`observed_loss` / `p_value` / `p_value_note`。**`requested` 与 `successful` 均达 999 时 `p_value` 才非空。**

`permutation_summary.json` 另含 `decision_halves`（前半/后半起点数与平均损失）、
`decision_halves_middle`、`grid`、`split_spec_sha256`、`code_commit`。

---

## 三、本轮真实结果

| 候选 | 行数 | 观察决策损失 | 经验 p | 循环移位 p |
|---|---|---|---|---|
| `N` | 1656 | — | — | — |
| `N+Q` | 1656 | — | — | — |
| `N+S+Q` | 1656 | 0.185302 | **0.761** | 0.347 |
| `N+S+Q+SF` | 1656 | 0.155100 | **0.014** | 见产物 |

置换耗时实测：`N+S+Q` 68.9 s、`N+S+Q+SF` 136.2 s（各含 999 次 row-null + 999 次循环移位）。

---

## 四、已知限制

1. **测试段未跑**：等主控封印路由并公布唯一 SHA256。
2. **仅 Agriculture**：Climate / SocialGood / Environment 待 Agriculture 闭环后按序扩展。
3. **保守滞后情景未跑**。
4. **坐标下降近似**：逐分支 α 用 2 轮坐标下降；1 轮/3 轮敏感性分析未跑。
5. **冻结 spec 路径有歧义**：主控分支 `round2/split_spec_v2.json` 是冻结版；
   模型侧工作区同名文件是过期草稿。建议协议写明唯一路径（见 `TEAM_SYNC.md` §五）。
