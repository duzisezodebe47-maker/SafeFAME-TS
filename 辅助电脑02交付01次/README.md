# 辅助电脑02交付01次：数据时点审计与 FeatureBundle 对接

本目录执行《辅助电脑一任务书：数据时点审计与特征工程》。项目性质仍为个人课程设计；这里的辅助电脑分工不改变课程要求。本轮不训练候选模型、不改评测路由、不修改三份课程报告，不生成 PDF、PPT 或上传压缩包。

## 当前交付状态

**数据审计、管线接口和工程测试已实现；正式特征冻结等待主控协议。**
本机没有找到主控冻结的 `split_spec.json`，也未收到新情景的滞后长度、来源完整性规则和接口确认。
`split_spec.template.json` 是待确认的接口样例，不能直接用来训练。
工程测试使用真实 Agriculture 数据及明确标注的测试划分，不将它作为正式研究方案。

本次交付统一放在仓库根目录 `辅助电脑02交付01次/`，由原 `team_work/data/` 整体迁入，交付分支 `3218151885-creator`（按 GitHub 用户名命名）。目录按用户指定的电脑编号命名，负责的工作仍为任务书中的数据时点审计与特征工程。
大型原始行追踪、数组和缓存只写 `data_processed/team_data`，该目录在 D 盘且由现有 Git 规则忽略。
`evidence` 保存可入库的小型审计表和实际测试输出。主控请先读 [交付与待对接事项](交付与待对接事项.md)，再查本说明中的接口和运行方法。交付分支供审阅与接收，不直接修改主控的 main 分支。

## 已有实现与本轮新增

| 原有能力 | 复用方式 | 本轮补充 |
|---|---|---|
| `src/data_utils.py` | 调用 `numerical_intervals` 统一日期列 | 原始行号到清洗后索引的对应、逐字段缺失与非有限审计 |
| `src/audit_timemmd.py` | 复用 `_valid_text` 的空事实判定 | 所有原始文本行的排除理由、文件哈希与来源标签 |
| `src/build_text_index.py` | 直接调用 `load_corpus`、`normalize_fact`、`stable_id` | 原始行到规范事实 ID 的映射，逐起点可展开的排除追踪 |
| `src/build_semantic_features.py` | 调用原加权归一化函数，沿用 10 维质量字段 | 分离语义、质量和来源掩码，三种代理情景 |
| `src/encode_text.py` 的既有产物 | 校验固定 revision、语料哈希、索引顺序、384 维归一化向量 | 不重新编码；将原始快照、规则、划分、代码、环境一起加入缓存签名 |
| `src/run_famets_selective.py` | 直接调用 `frequency_statistics` | 频率数组独立存储，不自动构造或拟合交互项 |
| `src/prepare_v4.py` 的历史规则 | 只读比较清洗后 OT 和日期 | 为新接口单独生成可追溯的单目标序列；不运行或覆盖历史准备脚本 |

本接口当前为 **OT 单变量历史窗口**；它可接入单变量候选或残差头，但不冒充历史 DLinear-M 的完整多变量输入。全部原始协变量保留于来源快照，逐字段审计标明未用于本接口。若辅助电脑二需要多变量历史，应先由主控确认字段和缺失处理约定。

## 本机输入位置与证据

- 原始 Time-MMD：项目根目录 `references/external/Time-MMD`，固定提交 `00281e2d86058286d5548b15a7670e8eda57ef62`。
- 原始 ETTh1/ETTh2：`references/external/ETDataset/ETT-small`；本轮仅审计数值来源，不创建 ETT 文本或新评测任务。
- 固定 MiniLM 缓存：`data_processed/v4/embeddings`，revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`；25,613×384 维。
- 既有 v4 语义缓存：`data_processed/v4/semantic_features`，用于真实小样本兼容性对照。
- 历史划分背景：`configs/safefame_v4.json`、`docs/19—23`。这些不自动成为本次主控冻结协议。
- 原始数据、旧缓存、源代码、报告和既有结果未覆盖；没有安装依赖或下载新模型。

来源网址为 [Time-MMD](https://github.com/AdityaLab/Time-MMD) 和 [ETDataset](https://github.com/zhouhaoyi/ETDataset)。更详细的固定提交及许可说明沿用项目 `docs/15_数据与代码来源清单.md`。Time-MMD 本地快照没有独立 LICENSE；本轮不声称重新核验了论文许可或原站点许可。
目标变量含义来自本地上游 `DescriptionOfOT.png`；该图部分频率和样本数与 CSV 不一致，例如 Climate 图列月频而实际相邻日期为 7 天，本轮以实际 CSV 计数。图中未明确的量纲保持“未复核”，不猜测美元、百分比等单位。

## 运行方法

在项目根目录打开 PowerShell，使用既有 Python 3.12 环境：

```powershell
.\.venv\Scripts\python.exe .\辅助电脑02交付01次\run.py audit
.\.venv\Scripts\python.exe .\辅助电脑02交付01次\test_pipeline.py
```

第一条执行十序列全量数据审计并校验固定语义缓存；输出路径写在 `evidence/audit_summary.json`。
同时逐一复算既有 v4 的 24,186 个起点的代理时间条件、所选文本 ID 和来源数量，明细保存在审计目录的 `legacy_origin_audit.csv`，领域覆盖汇总见 `evidence/legacy_proxy_coverage.csv`。这是既有索引审计，不是新折划分的正式验收。
第二条运行故障注入和真实数据冒烟测试，结果写在 `evidence/tests.json`；工程预览及独立线程复建路径也在该文件。
缺失原始数据或固定缓存时输出 `MISSING_INPUTS` 和具体路径，不下载替代数据。
普通 Git 克隆按规则不包含这些大型输入，不能在缺失时宣称完成真实数据测试。

正式协议到位后，由主控填写四个分段末端、原始频率对应的日历滞后、清洗序列 SHA256，并确认 `approved_by` 和 `status=frozen`。不要只把模板的状态改为 frozen。然后运行：

```powershell
$auditInfo = Get-Content .\辅助电脑02交付01次\evidence\audit_summary.json -Raw | ConvertFrom-Json
.\.venv\Scripts\python.exe .\辅助电脑02交付01次\run.py build --audit-folder $auditInfo.output --split-spec .\split_spec.json
```

`bounds=[a,b,c,d]` 对应按当前清洗后序列 **0 起算** 的半开区间：训练 `[0,a)`、校准 `[a,b)`、决策 `[b,c)`、测试 `[c,d)`。
每个任务必须提供该清洗 CSV 的 `numerical_sha256`，避免拿原始行号去切已经去重、裁边后的行号。
`origin_index=o` 的输入为 `[o-L,o)`，目标为 `[o,o+H)`；目标跨段、目标未知、历史区间尚未结束或目标区间跨时点边界的窗口剔除并写入 `sample_audit.csv`。
同一段内部允许滑动目标重叠；跨段目标不重叠仍不等于统计独立。后折使用届时已经观测的数据属于滚动训练，不声称折间独立。

## 逐起点追踪

工程样例可以直接检查最后一个起点：

```powershell
$check = Get-Content .\辅助电脑02交付01次\evidence\tests.json -Raw | ConvertFrom-Json
.\.venv\Scripts\python.exe .\辅助电脑02交付01次\run.py trace --audit-folder $check.audit_folder --bundle-folder $check.preview_folder --task-id Agriculture_h3_fsmoke --origin-id Agriculture:h3:fsmoke:o417 --scenario proxy
```

终端给出数值窗口、目标索引和截止时间；展开后的 CSV 在 D 盘 `data_processed/team_data/traces`。
它列出该领域全部原始文本行，包括缺失事实、无日期、倒置区间、重复事实、超出回看、晚于截止、滞后尚未结束、来源缺失和来源数量上限等互斥理由。
同时可由 `*_numeric_lineage.csv` 的 `clean_index` 追溯原始文件行号和原始目标值。
常规 Bundle 保存选中 ID 和排除计数，不重复存储“全部起点×全部文本”的笛卡尔积；按需展开仍能复核每条排除依据。

## 三种代理情景

| 情景 | 判定 | 解释边界 |
|---|---|---|
| `proxy` | `history_start <= end_date < cutoff_time` | 沿用既有代理时间规则，未核验发布时间 |
| `conservative_lag` | 在 proxy 上增加 `end_date + lag_days < cutoff_time` | `lag_days` 由主控按频率预先冻结；不是根据测试收益选择 |
| `complete_source` | 在 proxy 上要求有效日期区间和逐条原始 HTTP(S) URL | URL 存在仅表示元数据较完整，不证明来源真实可靠或发布时间准确 |

当前全部 20 个文本 CSV 只有事实、区间和来源类别，没有逐条 URL，因此本接口提出的 `complete_source` 规则在真实数据中会产生全空文本。这是数据缺口，不是“严格来源文本没有预测价值”的实验结论。主控需确认该规则是否适合后续任务；不能把 `report` 标签当成原始 URL 的替代品。

每来源最多 32 条，按 end_date 降序和 text_id 升序选取。更严格滞后会让较早事实补入 top-32，因此“唯一被选事实数量”可能增加；可得集合和有文本起点覆盖不会因此增加。`coverage.csv` 同时报告可用起点、独特选中事实、跨界排除和相对基准覆盖损失，避免混淆这两种数量。

## FeatureBundle 字段与拟合边界

每个任务目录保存 `samples.csv`、`sample_audit.csv`、`numeric_fit.json` 和下列数组；三个情景分别保存文本数组。所有行按 `samples.csv` 对齐。

| 文件/字段 | 形状 | 含义 |
|---|---|---|
| `origin_id`、`cutoff_time`、`segment` | N 行 | 领域、跨度、折、起点身份；禁止同任务内重复 |
| `numeric_history.npy` | N×L | 使用训练段均值和总体标准差标准化后的 OT 历史 |
| `numeric_history_raw.npy` | N×L | 原始尺度历史，未知值保留 NaN |
| `numeric_missing.npy` | N×L | True 表示该历史值未知、标准化输入使用过填补 |
| `targets.npy` | N×H | 原始尺度真实目标，必须全部有限 |
| `targets_standardized.npy` | N×H | 使用同一训练参数的目标坐标 |
| `target_time.npy`、`target_end_time.npy` | N×H | 每个目标的观测区间起止时间 |
| `frequency.npy` | N×10 | 原项目频谱统计；使用标准化数值历史，不含文本交互 |
| 情景目录 `semantic.npy` | N×768 | report/search 各 384 维，时间衰减后分别 L2 归一化 |
| 情景目录 `quality.npy` | N×10 | 可用/选中数量、缺失、年龄、未来措辞标记；沿用原字段顺序 |
| 情景目录 `source_available.npy` | N×2 | report/search 是否有实际选中文本 |
| 情景目录 `text_available.npy` | N | 是否存在任一来源文本 |
| 情景目录 `text_trace.jsonl` | N 行 | 纳入文本 ID、排除数量和时间代理标记 |

数值缺失只允许历史前向填充；前导缺失用训练段有限目标中位数填补。`targets.npy` 不使用填补值。拟合记录保存训练末端、拟合行索引哈希、均值、标准差和中位数，拒绝训练段外拟合索引。
原始输入与缺失掩码一并交付，不能只保留填好后的值而隐藏异常。
语义不做 PCA，质量不再做额外标准化；后续模型若需要 PCA、质量尺度或交互项，须在自己的允许训练段拟合，并保存参数，不能在本 Bundle 全部行上直接 fit。
无文本语义为零向量，同时显式交付可得性掩码；未来措辞仅作审计标记，不据此判定事实一定泄漏。

## 辅助电脑二读取示例

以下仅适用于真正冻结的 Bundle，工程预览会被读取器拒绝：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('辅助电脑02交付01次').resolve()))
from bundle import read_bundle

samples, arrays, manifest = read_bundle(
    Path('data_processed/team_data/bundle-主控提供的版本号'),
    task_id='Agriculture_h3_f1',
    scenario='proxy',
    expected_signature='主控交付的完整64位签名',
)
train = samples['segment'].eq('train').to_numpy()
x_train = arrays['numeric_history'][train]
semantic_train = arrays['semantic'][train]
quality_train = arrays['quality'][train]
y_train = arrays['targets_standardized'][train]
```

读取器检查全部文件哈希和冻结状态。不得拷贝后自行清洗、删除样本或重新挑文本却沿用原签名。用 `--expected-signature` 可要求构建器拒绝新规则与旧缓存混用。
本地 manifest 是可复核的内容锚点，不是数字签名身份认证；协议审批仍需实际由主控完成。

## 缓存与可复现性

版本签名包含原始 CSV、准备数据、编码器文件/索引/元数据哈希、筛选规则代码、依赖版本及划分配置。
同一输入重用前核验全部产物；输入变化产生新目录，指定旧签名时明确报错。失败后未封存的目录不能作为有效缓存；需保留失败现场并在另行修复后重新生成，不能手工补哈希。
运行时记录耗时和 Windows 进程峰值工作集，包含导入和原生数组内存，不把它当成单模块精确占用。没有 GPU 计算和模型下载。
本轮实测 1/2 线程相同文件哈希，不据此保证所有硬件、Python/BLAS版本逐字节相同。

## 仍需主控提供

1. 正式 `split_spec.json`：各序列清洗版本哈希、任务与折、四个区间末端及审批身份。
2. 各频率保守滞后天数，以及来源完整性情景的操作定义。
3. FeatureBundle 1.0 接口确认，尤其是 OT 单变量是否满足辅助电脑二需求。
4. 若要超越代理情景，需要逐条原始来源 URL 和可核查发布时间；现有数据无法补造。

这些到位后才能生成正式全领域、全折、全跨度的情景覆盖与冻结 Bundle。目前不把单领域软件测试写成该项已经验收。

本次实测结果与范围见 [交付与待对接事项](交付与待对接事项.md)。
