# SocialGood 第九次交付独立重放

2026-09-28 修订包在原 v9 安全输入子集上增加 `raw_clean_audit.json` 和复算程序。原 v9 的 3,343,370 字节 / `f75cbe2dcd6689f3d6a1b6800e366f4bc041d24f9efdf3d341b20baac3154e6f` 仅适用于历史 ZIP，不能用于修订包。

在 Windows PowerShell 中使用 Python 3.12、NumPy 2.5.3、pandas 3.0.5。只在 D 盘新目录操作。修订包固定为 3,353,590 字节，SHA256 `2fe3deeda8999b037b8f0ff50d7083aced93470d5fa0e7ab042ac8091d1952f7`；下载后**先核对字节数和 SHA256**，不符立即停止。不要用旧版缓存替代下载件。

```powershell
$replay = 'D:\SafeFAME-TS-SocialGood-clean-09r2'
New-Item -ItemType Directory -Path $replay | Out-Null
$url = 'https://github.com/duzisezodebe47-maker/SafeFAME-TS/releases/download/socialgood-isolated-input-v9r2-20260928/SafeFAME-TS_SocialGood_isolated_v9r2_20260928.zip'
Invoke-WebRequest -Uri $url -OutFile (Join-Path $replay 'release.zip')
$zip = Join-Path $replay 'release.zip'
$expectedBytes = 3353590
$expectedSha256 = '2fe3deeda8999b037b8f0ff50d7083aced93470d5fa0e7ab042ac8091d1952f7'
if ((Get-Item -LiteralPath $zip).Length -ne $expectedBytes) { throw 'Release 字节数不匹配' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant() -ne $expectedSha256) { throw 'Release SHA256 不匹配' }
Expand-Archive -LiteralPath (Join-Path $replay 'release.zip') -DestinationPath (Join-Path $replay 'release')
py -3.12 -m pip install -r (Join-Path $replay 'release\requirements-replay.txt')
py -3.12 (Join-Path $replay 'release\code\verify_socialgood.py') --package (Join-Path $replay 'release') --out (Join-Path $replay 'audit')
py -3.12 (Join-Path $replay 'release\tests\run_negative_cases.py') --package (Join-Path $replay 'release') --out (Join-Path $replay 'negative_cases.json')
```

验证器逐文件检查 Release 清单、正式 Bundle 与 spec 锚、原始到清洗的聚合审计、四段边界、起点 mapping、选择侧切片承诺、测试目录白名单和两情景覆盖，并重算审计表。成功退出 0，失败退出 2。负例汇总程序只有全部案例均被实际验证器拒绝才退出 0。输出目录必须在 Release 目录之外。

持有冻结 Time-MMD 来源时，可独立复算原始到清洗审计：

```powershell
py -3.12 code/audit_socialgood_raw_clean.py --raw-root '<Time-MMD目录>' --clean-numeric '<冻结SocialGood_numerical.csv>' --lineage '<完整text_lineage.csv>' --out '<新目录/raw_clean_audit.json>'
```

复算结果须与包内 `raw_clean_audit.json` 字节一致。公开包不含原始数值与完整清洗序列，以防测试真值泄漏；仅持有包本身时，验证器能检查锚和内部一致性，不能独立重算原始来源。

可从本地冻结 Bundle 构建新包：

```powershell
py -3.12 prepare_socialgood.py --bundle '<formal_bundle目录>' --spec '<split_spec_v2.json>' --numeric '<SocialGood_numerical.csv>' --lineage '<text_lineage.csv>' --raw-root '<Time-MMD目录>' --release '<新的D盘stage目录>' --delivery '<本交付目录>'
py -3.12 make_manifest.py '<stage目录>'
py -3.12 verify_socialgood.py --package '<stage目录>' --out '<新的D盘reference目录>' --reference
# 将 reference 中的六份审计产物复制到 stage 同相对路径后：
py -3.12 make_manifest.py '<stage目录>' --reference '<reference目录>' --zip '<新的D盘ZIP文件>'
py -3.12 verify_socialgood.py --package '<stage目录>' --out '<新的D盘验证目录>'
```

`--reference` 仅用于**首次构建**冻结预期输出；交付验收必须不带该选项。正式包不含测试真值、测试逐起点数值历史或频率；它不能独立完成完整预测。`truth_commitment.json` 仅提供测试目标切片哈希承诺，不能单凭包内数据逆向重建真值，也不能独立证明选择侧数值切片相对于未分发的原 Bundle 的来源；这些需外部核对正式 Bundle。
