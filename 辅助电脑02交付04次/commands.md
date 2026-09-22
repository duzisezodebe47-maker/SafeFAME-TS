# 第四次交付复核命令

以下命令均在仓库根目录执行。`<固定快照提交>` 取 `github_snapshot_receipt.json` 的 `fixed_snapshot_commit`。

## 1. 从 Git 固定提交复核四份 CSV

```powershell
.venv\Scripts\python.exe "辅助电脑02交付04次\verify_github_snapshots.py" `
  --commit <固定快照提交> `
  --output "辅助电脑02交付04次\github_snapshot_receipt.recheck.json"
```

## 2. 主控提取原始 blob 到 D 盘

`git show` 的输出必须直接写到 D 盘目标目录；不要经 C 盘临时文件中转。

```powershell
New-Item -ItemType Directory -Force -Path "D:\SafeFAME-TS-transfer\clean_snapshots" | Out-Null
git show "<固定快照提交>:辅助电脑02交付04次/clean_snapshots/Agriculture_numerical.csv" > "D:\SafeFAME-TS-transfer\clean_snapshots\Agriculture_numerical.csv"
git show "<固定快照提交>:辅助电脑02交付04次/clean_snapshots/Climate_numerical.csv" > "D:\SafeFAME-TS-transfer\clean_snapshots\Climate_numerical.csv"
git show "<固定快照提交>:辅助电脑02交付04次/clean_snapshots/SocialGood_numerical.csv" > "D:\SafeFAME-TS-transfer\clean_snapshots\SocialGood_numerical.csv"
git show "<固定快照提交>:辅助电脑02交付04次/clean_snapshots/Environment_numerical.csv" > "D:\SafeFAME-TS-transfer\clean_snapshots\Environment_numerical.csv"
Get-FileHash -Algorithm SHA256 "D:\SafeFAME-TS-transfer\clean_snapshots\*_numerical.csv"
```

PowerShell 7 的重定向保持原始字节；若使用 Windows PowerShell 5.1，建议改用 `cmd /c` 或程序化读取 Git blob，避免文本重定向改变字节。

## 3. 第二阶段启动门槛

只有在主控给出冻结 `split_spec_v2.json` 的提交号和原始字节 SHA256，且文件满足 `status=frozen`、`approved_by` 非空、四项数据哈希完整后，才运行第三轮的正式构建入口：

```powershell
.venv\Scripts\python.exe "辅助电脑02交付03次\round3_pipeline.py" build `
  --split-spec "D:\<主控冻结spec原始文件>" `
  --output-root "D:\AAA个人资料\桌面\人工智能大作业\data_processed\team_transfer\辅助电脑02交付04次\formal_bundle"
```

当前不得执行该命令，也不得把 preview 或 `TEST_FIXTURE_ONLY` 结果作为正式 Bundle。
