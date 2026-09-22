# 辅助电脑02交付06次：来源与时间边界审计

本目录是对正式 Time-MMD Bundle 的独立、可重放审计。审计器不修改正式 Bundle、冻结 spec、原始 Time-MMD 文件或第五轮历史回执。

## 审计对象

- Bundle 签名：`a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38`
- 四任务：Agriculture、Climate、SocialGood、Environment；两种情景：`proxy`、`conservative_lag`
- 规则：`end_date < cutoff`；保守滞后还要求 `end_date + lag_days < cutoff`；每个来源最多保留 32 条，按 `end_date` 降序、`text_id` 升序截断。
- `proxy` 和 `conservative_lag` 不把 URL 缺失作为门控失败；URL 缺失与 `end_date` 未核验状态仍被逐事实记录。`complete_source` 只在独立负例中测试。

## 运行

在项目根目录、Python 3.12 虚拟环境中运行：

```powershell
.venv\Scripts\python.exe 辅助电脑02交付06次\audit_source_time_boundaries.py `
  --bundle data_processed\team_data\round3-bundle-a69821be115445265cde `
  --audit-dir data_processed\team_data\audit-6266b0a931bd32bb166a `
  --spec data_processed\team_transfer\辅助电脑02交付05次\split_spec_v2.json `
  --output-dir data_processed\team_transfer\辅助电脑02交付06次\time_boundary_audit
```

脚本退出码为 0 才表示通过。逐起点结果和事实目录位于输出目录；本交付目录中的 `coverage_summary.csv`、`socialgood_differences.csv`、`negative_cases.json` 是同一运行的紧凑副本。

## 结果解释

- 四任务所有保留起点的边界、trace、`source_available`、`text_available` 和 `quality` 均通过复算，质量最大绝对误差为 0。
- SocialGood 的 proxy 覆盖为 341/700，保守滞后为 340/700。`socialgood_differences.csv` 记录 341 个发生选中文本集合变化的起点；只有 `SocialGood:h3:f1:o385` 的覆盖布尔值从真变为假，其余集合变化仍保留至少一个来源。
- 事实语料没有已登记的原始 URL，`end_date` 的发布时点状态为 `end_date_proxy_unverified`。这限制了“真实发布时间”的解释，但不改变本轮对现有冻结规则的复算结论。
- 五个独立负例分别覆盖 cutoff 等时点、滞后边界、反向区间、重复事实和无 URL，均得到预期互斥原因。

## 输入重跑

`source_clean_reaudit.json` 记录了原始 Time-MMD 审计和固定清洗缓存核验。四份冻结数值 CSV 的实际字节 SHA256 全部与冻结值一致；固定链使用已验证缓存复核，没有把已有文件复制后冒充新的原始清洗结果。

大体量的逐起点 JSONL 不进入 Git 分支，随第六轮 GitHub Release 附件提供，并在分支回执中记录固定下载地址和 SHA256。

下载：<https://github.com/duzisezodebe47-maker/SafeFAME-TS/releases/download/source-time-boundary-audit-20260923/SafeFAME-TS_source_time_audit_06.zip>
