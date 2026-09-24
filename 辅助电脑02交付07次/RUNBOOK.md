# 第七轮 Release 完整重放

以下命令在 Windows PowerShell 执行。需要 Python 3.12、NumPy 2.5.3、pandas 3.0.5；数据和项目代码均来自单个 Release 附件，无需旧 `audit-*` 缓存或 Git checkout。运行产物只写入指定输出目录。

```powershell
$replayHome = 'D:\SafeFAME-TS-clean-replay-07'
New-Item -ItemType Directory -Force -Path $replayHome | Out-Null
$asset = 'https://github.com/duzisezodebe47-maker/SafeFAME-TS/releases/download/source-time-audit-v7-20260924/SafeFAME-TS_source_time_audit_v7_20260924.zip'
Invoke-WebRequest -Uri $asset -OutFile (Join-Path $replayHome 'release.zip')
Get-FileHash -Algorithm SHA256 (Join-Path $replayHome 'release.zip')
Expand-Archive -LiteralPath (Join-Path $replayHome 'release.zip') -DestinationPath (Join-Path $replayHome 'release')
Set-Location (Join-Path $replayHome 'release')
py -3.12 -m pip install -r requirements-replay.txt
py -3.12 code\replay.py --output-dir replay_output
py -3.12 tests\run_negative_cases.py --output replay_output\negative_cases_v7.json
```

重放入口先校验 `MANIFEST.json` 列出的所有静态文件、正式 Bundle 签名、spec 和语料的固定哈希；然后将规范事实与血缘记录按 `text_id` 一对一合并，执行四任务 × 两情景的全量起点级核验，生成 `audit_summary.json`、`origin_time_audit.jsonl`、`fact_catalog.csv`、`coverage_summary.csv` 与来源/论文数据表。随后逐文件比较 `EXPECTED_OUTPUTS.json` 中的 SHA256。成功时退出码为 0，失败时为 2。

更换输出目录可使用 `--output-dir 其他目录`；重放结果哈希与路径无关。所有输入都从 `release` 根目录读取。附件 SHA256、字节数、真实重放 stdout/stderr、耗时和产物哈希记录在 `clean_replay/`。
