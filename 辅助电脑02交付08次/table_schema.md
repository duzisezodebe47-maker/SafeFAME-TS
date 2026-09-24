# Climate 第八轮审计表口径

全部数据表的时间轴来自冻结的 Climate 清洗数值快照，任务边界为 `[636,763,1017,1144]`，历史窗口为 52 周、预测长度为 4 周。数值真值只存在于 `selection_fit/` 的非测试行；审计程序从不读取测试真值。CSV 均为 UTF-8，比例取 0–1。

## `climate_boundary_audit.json`

每个 `segments` 元素是一段。`candidate_origins` 为该段候选起点数，`retained_origins` 为正式 Bundle 保留数；`origin_min/max` 是保留起点的数值轴索引；`excluded_by_reason` 按原 Bundle `sample_audit.csv` 的原因计数。`historical_cells=retained_origins×52`，`historical_missing_rate=historical_missing_cells/historical_cells`，不使用预测目标。

## `climate_feature_drift.csv`

每行是一段 × 一个**输入特征**。`n/finite_n` 分别是历史输入单元或逐起点特征总数、有限值数；`mean/std/q25/median/q75` 只对有限值计算；`missing_rate=1-finite_n/n`。数值历史 `numeric_history_raw/OT` 在选择三段可由包内输入重算，test 段只保留封包前由历史特征形成的分段聚合，**不能**在隔离包内逐起点重算。`delta_mean_from_train` 是相对 train 均值差；`delta_mean_in_train_sd` 以 train 标准差作分母，分母为零时留空而非伪记为 0。此表不是测试真值误差表，也不支持因果解释。

## `climate_source_coverage.csv`

每行是一段 × 一个文本时点情景。`origins_denominator` 为该段保留起点；`text_available_origins/rate` 统计至少一条选中文本的起点及其占比。`selected_facts_total` 按起点累加，事实可跨起点重复；`selected_facts_q25/median/q75` 是每起点选中事实条数分位数（含零）。`report_cap_32_origins` 和 `search_cap_32_origins` 是各来源选中数达到 32 的起点数；`either_cap_32_rate` 是至少一个来源达到 32 的起点比例。`cap_exceeded_reason_origins` 根据原追踪记录判定有事实被来源上限排除，不等同于前述达到上限比例。

## `climate_source_evidence.csv`

每行一个 `report/search` 来源。`raw_rows_denominator` 为原血缘记录行数，`nonempty_rows_denominator` 排除缺正文行。`duplicate_rate_nonempty=duplicate_fact_rows/nonempty_rows_denominator`；规范事实 `canonical_facts` 来自 retained 行。`original_url_missing_rate` 和 `publication_time_proxy_rate` 的分母均为 `canonical_facts`。`source_file` 是 Time-MMD 文件路径，不是原始网页。`end_date` 仅为可得时间代理，`publication_time_proxy_rate` 不是已核验发布时间的比例。

## `climate_weekly_axis_audit.csv` 与 `climate_weekly_axis_summary.json`

每行是一条清洗后数值时间轴记录，不含 OT。`delta_days_from_previous` 是与前一条周起始日的日差，首行留空；`missing_weeks_before=max(floor(delta/7)-1,0)`；`duplicate_start` 检查起始日重复；`ordered_after_previous` 首行按无前驱约定为 1；`interval_days_inclusive` 是本周起止日期包含两端的日数。汇总中的 `seven_day_steps` 分母是 `rows-1`，`seven_day_intervals` 分母是 `rows`，`history_length` 是每起点历史周数，不是整张表的长度。
