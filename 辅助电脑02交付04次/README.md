# 辅助电脑02交付04次

本目录执行数据侧第四轮任务。当前只完成第一阶段：把四份清洗 CSV 的实际字节固定到 Git，并从固定提交中的 Git blob 复算哈希、字节数、行数和列名。

## 当前状态

- 第一阶段：`COMPLETE`（以 `github_snapshot_receipt.json` 中的固定快照提交为准）。
- 第二阶段：`PENDING_MASTER_FREEZE`。
- 阻断原因：主控 `team/main-eval-20260921@1933b3c93af7d68dbf54e417457a20d5da62a2a6` 中的 `team_work/main/round2/split_spec_v2.json` 仍为 `draft_not_for_training`，`approved_by` 与四项 `numerical_sha256` 为空。

因此，本目录没有生成正式 Bundle，也没有把旧预览或测试夹具改名为正式数据。第二阶段须等待主控提供冻结 spec 的提交号及原始字节 SHA256 后继续。

## 第一阶段材料

- `clean_snapshots/`：四份实际清洗 CSV，共 531,057 字节。
- `verify_github_snapshots.py`：直接读取固定提交中的 Git blob 并复算证据。
- `github_snapshot_receipt.json`：固定提交与逐文件复核回执。
- `formal_split_spec_status.json`：当前观察到的主控草案状态及字节哈希。
- `formal_bundle_manifest.json`、`transfer_receipt.json`、`reader_comparison.json`：第二阶段的真实等待状态，不含伪造结果。
- `commands.md`：复核及后续执行命令。
- `failure_log.json`：本轮阻断与失败记录。

## 证据边界

第一阶段证明 GitHub 固定提交中存在四份 CSV，且其字节与第二、三轮登记一致；它不证明主控已下载、不证明协议已冻结，也不证明正式 Bundle 已构建或被三方读取。
