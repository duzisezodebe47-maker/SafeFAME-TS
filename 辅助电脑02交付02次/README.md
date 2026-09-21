# 辅助电脑02交付02次：数据侧冻结前交付

本目录是 SafeFAME-TS 三人协作项目的数据侧第二轮交付。它只处理数据、时点、样本身份和特征，不训练模型、不决定路由、不改写 v2/v3/v4 结果。

## 当前状态

**冻结前工程交付已完成，正式 Bundle 等待主控核对四项清洗哈希后冻结。**

主控分支 `team/main-eval-20260921@bcbbf5d8b780d937e30fb635165e63a8c9c5520a` 已冻结接口、任务、折号、日/周/月滞后规则，但其协议明确标注“逐任务清洗哈希待数据侧回传”。本次已在 `evidence/task_registry.csv` 给出哈希、行数、边界和样本数；`split_spec_freeze_request.json` 仍是 `engineering_preview`，不得改名冒充正式实验。

| 任务 | 行数 | L/H | 折 | 边界 train/cal/dec/test | 工程保留样本 |
|---|---:|---:|---:|---|---:|
| Agriculture_h12_f1 | 532 | 24/12 | 1 | 212/266/372/425 | 357 |
| Climate_h4_f2 | 1272 | 52/4 | 2 | 636/763/1017/1144 | 1080 |
| SocialGood_h3_f1 | 916 | 24/3 | 1 | 366/458/641/732 | 700 |
| Environment_h7_f2 | 15979 | 56/7 | 2 | 7989/9587/12783/14381 | 14301 |

四任务共 16,438 个保留起点，清洗序列的未知目标行数均为 0。边界按 `floor(n_rows * ratio)` 对清洗后行数取整，目标窗口必须整体落入半开分段。

## 情景与字段

- `proxy`：`end_date < cutoff_time`，只是未核验的可得时间代理。
- `conservative_lag`：日频 1 天、周频 7 天、月频 31 天，值来自主控预注册，不按测试收益选择。
- `complete_source`：原文本没有逐条 URL，所以只产生覆盖审计，不产生训练数组。全空只说明来源元数据缺失，不能解释为文本无效。

每个任务的大数组目录包含：`samples.csv`、`sample_audit.csv`、`numeric_fit.json`、N×L 数值历史、N×H 原始/标准化目标、N×768 语义、N×10 质量、N×10 谱统计、缺失和文本覆盖掩码、文本 ID 追踪。`complete_source` 目录按设计不存在。目标单位仍未从上游原始来源独立复核，不在本轮猜测量纲。

## 主控冻结前先看

1. `evidence/task_registry.csv`：四任务的清洗哈希、边界和预期/实际样本数。
2. `evidence/split_spec_freeze_request.json`：候选协议，必须由主控核对后填入 `status=frozen` 和 `approved_by`。
3. `evidence/coverage.csv`：任务×折×情景×分段的起点、文本覆盖、选中次数与剔除数。
4. `evidence/tests_round2.json` 与 `trace_examples.jsonl`：故障拒绝和逐起点紧凑证据。

## 在完整仓库上复跑

本交付分支是精简快照，不可单独克隆后直接运行。它依赖完整项目基线 `c63793236d532d2ffdcccd4c7cb0c8e194f96eab`、第一轮目录、D 盘原始数据和固定 MiniLM 缓存。主控应把 `辅助电脑02交付02次/` 选择性复制到完整仓库根目录，不要对孤立分支做普通整树合并。

```powershell
.\.venv\Scripts\python.exe .\辅助电脑02交付02次\round2_pipeline.py prepare
.\.venv\Scripts\python.exe .\辅助电脑02交付02次\round2_pipeline.py preview
$r = Get-Content .\辅助电脑02交付02次\evidence\preview_result.json -Raw | ConvertFrom-Json
.\.venv\Scripts\python.exe .\辅助电脑02交付02次\test_round2.py $r.bundle_path
```

收到主控冻结文件后，才允许正式构建：

```powershell
.\.venv\Scripts\python.exe .\辅助电脑02交付02次\round2_pipeline.py build --split-spec .\split_spec.frozen.json
```

`build` 会拒绝没有 `status=frozen`、`approved_by`、四任务哈希与边界不匹配的文件。

## 给 SHY 的只读接口

```powershell
.\.venv\Scripts\python.exe .\辅助电脑02交付02次\bundle_read_example.py <Bundle目录> Agriculture_h12_f1 --scenario proxy
```

默认拒绝工程预览；`--allow-preview` 只能用于接口自检。读取器先核对全部 SHA256，再返回相同起点顺序的数值、目标、语义、质量、频率和掩码，并打印原始尺度真值与 `bundle_signature`。本轮仅有 OT 单变量接口；多变量需主控新立协议。

## 大型产物传递（不压缩、不上传 Git）

将已签名 Bundle 复制到 U 盘、共享盘或主控指定的 D 盘目录，复制后立即逐文件复核：

```powershell
.\.venv\Scripts\python.exe .\辅助电脑02交付02次\transfer_bundle.py <Bundle目录> <目标父目录>
```

目标已存在时脚本会拒绝覆盖。主控也可使用 `round2_pipeline.py verify --bundle <目录>` 独立核对。Git 中的 `evidence/bundle_manifest.json` 保存全部 88 个大文件的相对路径和 SHA256，不依赖本机绝对路径。

## 证据边界

- `end_date` 不是已核验发布时间。
- 工程预构建是历史回测接口检查，不是未见未来确证。
- 目前不训练模型、不产生新的预测结论、不对文本有效性作出因果解释。
- 第一轮 19 项测试已重跑通过；第二轮 15 项测试和 12 个追踪类别记录见 `evidence/`。
