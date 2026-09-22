# 辅助电脑02交付05次：正式 Bundle

本次交付严格使用主控冻结提交 `6c05bd60661386c39121cbf079298fa1873d97df` 中 `team_work/main/round2/split_spec_v2.json` 的原始 Git blob。原始字节 SHA256 为 `a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031`。

## 交付结论

- 正式 Bundle：`COMPLETE`（源端构建和验证完成）。
- Bundle 签名：`a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38`。
- 内容：4 个任务 × `proxy` / `conservative_lag`；`complete_source` 仅有审计统计，没有训练数组。
- 完整清单：94 个文件（不含 manifest 自身），含 manifest 共 162,342,003 字节。
- GitHub Release 附件：已上传并从公开地址回读到 D 盘，附件 SHA256 一致。
- Agriculture 三方读取：数据侧、主控、模型侧读取器对 357 个起点中的首、中、末 3 个逐行比较，身份、顺序、原始目标和标准化目标完全一致。
- 接收状态：`MASTER_ACK_PENDING`。发布端回读不能替代主控电脑下载到 D 盘后的独立验收。

## 下载地址

[SafeFAME-TS 正式 Bundle Release](https://github.com/duzisezodebe47-maker/SafeFAME-TS/releases/tag/data-bundle-a69821be115445265cde)

附件：`SafeFAME-TS_formal_bundle_a69821be115445265cde.zip`

SHA256：`8abde5043f7399c96d269b5adbbc35dcbcada0e8760114a41b603031c01f0e06`

## 本目录证据

- `split_spec_v2.json`：从冻结提交导出的原始字节副本。
- `frozen_spec_source.json`：冻结来源、字段门槛与哈希。
- `formal_bundle_manifest.json`：正式 Bundle 原始 manifest。
- `source_verification.json`：协议、清单、快照、数组、边界和标准化全量源端验证。
- `coverage_exclusion_summary.csv`：四任务 × 两情景 × 四段的覆盖与剔除摘要，共 32 行。
- `agriculture_three_reader_compare.json`：真实正式 Bundle 三方读取对照。
- `release_inventory.json`：Release 地址、附件哈希和解包信息。
- `transfer_receipt.json`：传输状态；主控确认前保持 `MASTER_ACK_PENDING`。
- `command_execution.json`：实际执行步骤与退出码。
- `failure_log.json`：两次非数据性命令失败及修正记录。
- `commands.md`：主控下载、解包和复核命令。

## 结论边界

本次证明正式数据接口可被三方一致读取，不证明任何模型有效性、门控结果或测试期性能。模型预测、999 次置换、路由冻结和测试评分仍须按主控协议在后续阶段独立完成。
