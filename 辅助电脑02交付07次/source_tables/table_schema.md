# 来源与覆盖表口径

所有比例均为 0–1；CSV 的一行只表示表内注明的统计单位。四任务以冻结 spec 为准，`upstream_revision=00281e2d86058286d5548b15a7670e8eda57ef62`。这些表由 `build_tables.py` 从规范事实、血缘记录、正式 Bundle 和逐起点审计结果生成。

## `source_evidence_inventory.csv`

每行 = 一个任务领域 × 一个 Time-MMD 来源类型。`domain` 是领域；`source` 为 `report/search`；`source_file` 为血缘记录所登记的 Time-MMD 相对文件路径；`fact_count` 是该组规范事实数；`start_min/end_max` 是事实区间边界的最小开始日/最大结束日；`original_url_count` 是非空原始 URL 的事实数，分母为同一行的 `fact_count`；`publication_time_status` 标明日期仅为可用时间代理；`upstream_revision` 为本地固定 Time-MMD revision。`source_file` 不能被解读为原始网页网址。

## `missing_source_evidence.csv`

每行 = 一个任务 × 来源。`task_id/domain/source` 定位该组；`fact_count_denominator` 是规范事实数，所有 `*_rate` 均以它为分母；`source_file_missing_count/rate` 指来源文件路径缺失；`original_url_missing_count/rate` 指原始 URL 字段为空；`publication_time_proxy_count/rate` 指缺少独立核验发布时间、只能以 `end_date` 作为代理的事实。三个缺口可在同一事实并存，不得相加为互斥总量。

## `coverage_by_segment_scenario.csv`

每行 = 一个任务 × `train/calibration/decision/test` 分段 × `proxy/conservative_lag` 情景。`origins_denominator` 是保留预测起点数；`text_available_origins` 是至少有一条选中事实的起点数；`text_available_rate` 的分母为该行 `origins_denominator`。`selected_facts_total` 对各起点的选中事实条数求和；同一事实可跨起点重复出现。`selected_facts_q1/median/q3` 是每起点选中事实条数的 25/50/75 百分位，含零事实起点，采用 NumPy 默认线性插值。

## `socialgood_scenario_difference_summary.csv`

每行 = SocialGood 一个分段或 `all`。`origins_denominator` 为配对起点数；`selected_set_changed_origins/rate` 比较两情景选中 `text_id` 集合，分母为配对起点；`coverage_true_to_false_origins/rate` 只数 proxy 可用、保守滞后不可用的起点，分母相同；`changed_but_still_available_origins` 数集合改变后仍有至少一条文本的起点。`all` 是四分段总计，不与分段行再相加。

## `paper_ready/` 三表

`表_数据来源与样本边界.csv` 每行一项任务：`numerical_rows` 是冻结数值 CSV 行数；`retained_origins` 是正式 Bundle 保留起点；`canonical_facts/report_facts/search_facts` 是规范事实数；`registered_original_url_count` 只数非空 URL，并不表示 URL 已逐条核验；`upstream_revision` 为固定输入版本。`表_时间边界与覆盖审计.csv` 每行一项任务 × 情景，`text_available_rate=text_available_origins/origins_denominator`；`selected_facts_total` 对起点求和；不一致计数和质量最大误差来自逐起点复算。`表_来源证据缺口.csv` 与上方缺口表同口径。
