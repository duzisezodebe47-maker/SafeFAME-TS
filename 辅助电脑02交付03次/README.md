# 辅助电脑02第三轮数据交付

## 结论先行

本交付当前状态为 **`DATA_FORMAL_PENDING`**，不是正式训练数据包。

已完成且可验收：

1. 四份清洗后数值 CSV 已实际复制到 D 盘共享目录，并逐文件核对 SHA256、字节数、行数与列名；
2. 正式化生成器已修复 `targets.npy`、小写 `schema.status=frozen`、`numeric_features=["OT"]`、冻结协议字节哈希和完整协议对象等契约；
3. 第一轮 19 项和第二轮 15 项回归检查已重新运行通过；
4. 新增 8 类契约/传输故障及 1 项主控冻结门检查均被正确拒绝；
5. 在仅用于接口测试的临时夹具上，数据侧、第一轮读取器、模型侧读取器和主控 `team_eval/v2.py` 对 Agriculture 的 357 个样本身份、顺序和目标尺度一致；
6. 已生成 Agriculture 三个跨段起点及 SocialGood 无文本起点的可追踪证据。

尚未完成且不能假称完成：主控分支 `c608a776336703f8eddeadbd82cb7fe22ce23d8c` 的
`round2/split_spec_v2.json` 仍为 `draft_not_for_training`，`approved_by=null`，四个
`numerical_sha256=null`。因此没有正式 Bundle、没有正式 Bundle 传输回执，也没有任何训练、路由选择或测试收益。

## 文件说明

| 文件 | 用途 |
|---|---|
| `round3_pipeline.py` | 复制清洗快照；冻结后把历史预览晋升为全新正式 Bundle，不改写历史 manifest |
| `bundle_read_example.py` | 数据侧严格读取器，核对冻结 spec、文件清单、起点顺序和目标尺度 |
| `transfer_bundle.py` | D 盘源端到接收端复制，并逐文件比对 SHA256 |
| `contract_probe.py` | 在独立 Python 进程中运行单项契约检查，保留真实退出码和错误信息 |
| `test_round3.py` | 回归、四方互操作及故障注入测试 |
| `make_trace_examples.py` | 生成紧凑起点追踪证据 |
| `clean_snapshot_manifest.csv` | 四份实际清洗 CSV 的来源、路径、哈希、字节数、行数和生成命令 |
| `coverage.csv` | 四任务、两训练情景及 complete-source 审计的分段覆盖、独特文本数和目标统计 |
| `sample_audit_summary.csv` | 每任务、每分段、每种保留/排除原因计数 |
| `formal_split_spec.sha256` | 当前主控协议的实际字节 SHA256；注释明确它仍未冻结 |
| `formal_bundle_manifest.json` | 真实的待冻结记录；不是伪造的正式 manifest |
| `transfer_receipt.json` | 清洗 CSV 交接结果及正式 Bundle 尚未交接的状态 |
| `tests_round3.json` | 真实测试退出状态、拒绝信息和限制 |
| `trace_examples.jsonl` | 三个 Agriculture 起点和一个 SocialGood 无文本起点追踪 |
| `commands.md` | 从当前状态到正式交接的完整命令 |

## D 盘实际交接位置

四份清洗 CSV 位于：

`D:\AAA个人资料\桌面\人工智能大作业\data_processed\team_transfer\辅助电脑02交付03次\clean_snapshots`

该目录由 `.gitignore` 排除，不上传大文件。主控应以 `clean_snapshot_manifest.csv` 独立复算哈希，确认后在自己的分支发布冻结 spec。当前 `receiver_status` 是 `MASTER_ACK_PENDING`，没有填写 `RECEIVED`。

## 正式化安全边界

- `round3_pipeline.py build` 只接受精确小写 `status=frozen`、非空 `approved_by` 且四哈希与本机清洗快照一致的 spec；
- 正式输出会新增与 `targets_raw.npy` 字节完全相同的 `targets.npy`，并全部重新封装、重新计算哈希与签名；
- `manifest.inputs` 同时记录 `split_spec_sha256` 和完整 `split_spec`；
- `complete_source` 只在覆盖表中审计为零，不生成训练数组；
- 接口测试使用的 `TEST_FIXTURE_ONLY` 协议和 Bundle 在测试结束后删除，不能训练、不能提交为正式证据；
- `end_date` 仍只是文本可得时间代理，不能解释为核验过的真实发布时间。

## 主控下一步

1. 从上述 D 盘目录独立计算四个 CSV 的 SHA256，并与 `clean_snapshot_manifest.csv` 比对；
2. 在主控分支把 `split_spec_v2.json` 的四哈希写入、设为 `status=frozen`，填写真实 `approved_by` 并提交；
3. 通知数据侧拉取该冻结提交，再按 `commands.md` 构建并传输正式 Bundle；
4. 正式 Bundle 传输后，再由主控和模型侧重跑同一组读取器检查。

在第 4 步通过之前，状态始终保持 `DATA_FORMAL_PENDING`。
