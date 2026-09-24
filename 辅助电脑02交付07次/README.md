# 辅助电脑02交付07次：可重放来源证据与论文数据表

本交付以 `3218151885-creator@f43c3c287165fb0a2b0fc8c15f85e07350252295` 为固定基点。正式 Bundle 签名 `a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38`，冻结 split spec SHA256 `a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031`。未修改 Bundle、数值 CSV、split 或模型路由。

## 交付结果

- 单个 [第七轮 Release 附件](https://github.com/duzisezodebe47-maker/SafeFAME-TS/releases/tag/source-time-audit-v7-20260924) 包含正式 Bundle、完整规范事实语料、文本血缘记录、冻结 spec、审计代码与输入清单。解包后按 `RUNBOOK.md` 可重放全部 32,876 条起点记录。
- `source_tables/` 给出来源清单、来源证据缺口、分段覆盖和 SocialGood 情景差异。`paper_ready/` 给出可直接用于技术报告的三张数据表及方法限制文字。
- 全量审计按冻结规则复算一致。SocialGood 两情景分别有 341/700 和 340/700 个文本可用起点；341 个起点的选中文本集合发生变化，其中只有 1 个起点由可用变为不可用。
- 规范事实的 `end_date` 只是可用时间代理。现有血缘记录没有原始 URL，也没有独立核验的逐事实发布时间。规则复算通过不等于证明绝无泄漏。

## 文件入口

- `replay.py`：仅从 Release 解包目录读取输入，验证哈希、来源链接与字段后运行第六轮起点级审计。
- `build_tables.py`：由重放结果计算来源与论文表。
- `make_manifest.py`：记录 Release 各输入文件的相对路径、字节数、行列、SHA256、生成链与上游版本。
- `MANIFEST.json`：Release 静态输入清单；自哈希按设计不列入，避免循环引用。
- `EXPECTED_OUTPUTS.json`：重放结果的字节 SHA256。重放入口逐一比对。
- `tests/run_negative_cases.py`：五个规则边界和四个 Release 完整性负例。
- `clean_replay/`：新建 D 盘目录的实际下载、解包和重放记录。

Release ZIP 仅作为 GitHub Release 附件发布；分支不纳入 ZIP 或大体量逐起点证据。输入材料详见 `MANIFEST.json`；从下载到核验的完整步骤见 [RUNBOOK.md](RUNBOOK.md)。
