# Climate 第八次交付独立重放

Windows PowerShell，Python 3.12。将第八轮 Release ZIP 下载到新的 D 盘目录；不要使用第七轮缓存或完整项目目录作为输入。

```powershell
$home08 = 'D:\SafeFAME-TS-Climate-clean-08'
New-Item -ItemType Directory -Path $home08 | Out-Null
$url08 = 'https://github.com/duzisezodebe47-maker/SafeFAME-TS/releases/download/climate-isolated-input-v8-20260924/SafeFAME-TS_Climate_isolated_v8_20260924.zip'
Invoke-WebRequest -Uri $url08 -OutFile (Join-Path $home08 'release.zip')
Expand-Archive -LiteralPath (Join-Path $home08 'release.zip') -DestinationPath (Join-Path $home08 'release')
py -3.12 -m pip install -r (Join-Path $home08 'release\requirements-replay.txt')
py -3.12 (Join-Path $home08 'release\code\verify_climate.py') --package (Join-Path $home08 'release') --out (Join-Path $home08 'output')
py -3.12 (Join-Path $home08 'release\tests\run_negative_cases.py') --package (Join-Path $home08 'release') --out (Join-Path $home08 'negative_cases.json')
```

验证器先逐项核验 Release 清单、正式 Bundle 特征文件哈希、origin 映射、四段边界和测试目录白名单，再重算审计表并与冻结产物哈希比较。输出目录必须位于 Release 目录外。成功退出码为 0；失败退出码为 2。七个负例都应以非 0 退出并给出具体原因，负例汇总程序本身全部通过时退出 0。

此严格隔离版本**不分发测试期逐起点数值历史和由其导出的频率特征**。这些完整滚动特征会反推出大多数测试时间点的真值，因此本包不能单独运行完整数值预测；后续如需评测，应由受控、按时点顺序的数据接口提供当时已观测的历史。`test_history_aggregate.json` 仅为封包前对历史输入的分段聚合，非测试目标统计，且不能在隔离包内逐起点复算。
