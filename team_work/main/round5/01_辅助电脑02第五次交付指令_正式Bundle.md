# 辅助电脑02第五次交付指令：构建并交接正式 Bundle

**负责人：**只在 `3218151885-creator` 分支工作；从第四次交付 `e6ea158` 继续，新目录 `辅助电脑02交付05次/`。不要改主控、模型或 `main`；所有生成和传输暂存放 D 盘。

## 已满足的输入门槛

主控已从你在 Git 提交 `2d530ec7ab22fce591767baf1af81f8fe9c6a11d` 固定的四份 CSV 提取实际字节，独立审核通过。主控冻结提交：`team/main-eval-20260921@6c05bd60661386c39121cbf079298fa1873d97df`；`team_work/main/round2/split_spec_v2.json` **原始字节 SHA256**：`a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031`。只接受这个精确文件，不用旧草稿、第三轮测试夹具或人工重打的同内容 JSON。

## 必须完成的工程

1. `git fetch` 主控分支后，用 Git 原始 blob 导出冻结 spec 到 D 盘，并独立核对 SHA256、`status=frozen`、`approved_by`、四任务行数/边界和四个 CSV 哈希。若与上述值不符立刻停下并给失败证据。
2. 用 `round3_pipeline.py build --split-spec <上述原始文件>` 生成**全新正式 Bundle**，覆盖四任务 × `proxy`、`conservative_lag`。`complete_source` 只出审计表，不能伪造可训练数组。禁止改写第三轮预览 manifest 或手填哈希。
3. 对正式 Bundle 全量运行本侧 `verify`：`manifest.signature==signature(inputs)`、`inputs.mode=frozen`、嵌入 spec 和字节哈希一致、全部且只有清单所列文件且逐文件 SHA256 相等。逐任务核对 `targets.npy` 与原始目标、标准化公式、起点 ID/行序、四段无跨界窗口、文本覆盖、剔除原因。
4. 把正式 Bundle 交到主控可从 GitHub 获得的渠道；如果太大，使用仓库 Release 附件并给固定链接、每个附件 SHA256、解包命令和文件清单。数据侧电脑的 D 盘绝对路径**不算跨机交付**。接收端未实测前，回执 `receiver_status=MASTER_ACK_PENDING`。
5. 主控收到后将在 D 盘逐文件核验，并用 `team_eval/v2.py` 实读 Agriculture；再让当前模型侧读取器和你本侧读取器在同一实际 Bundle 对至少 3 个起点逐行核对。测试夹具的四方读取不能充当正式证据。

## 交付内容和时间

`辅助电脑02交付05次/` 应有 `README.md`、冻结 spec 来源提交与 SHA256、正式 Bundle manifest、四任务 × 两情景的覆盖/剔除摘要、打包或传输回执、源/接收端验证命令和真实退出码、Agriculture 三方读取对照及失败日志。先交实际 Bundle 和可访问地址；建议在收到本指令后 48 小时内完成。若资源或 GitHub 限制阻断，给可复现的部分完成清单与具体瓶颈，不能把预览标成正式。
