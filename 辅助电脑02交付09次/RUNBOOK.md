# SocialGood 第九次交付独立重放

在 Windows PowerShell 中使用 Python 3.12、NumPy 2.5.3、pandas 3.0.5。只在 D 盘新目录操作。先从本分支 README 取得固定 Release URL、字节数与 SHA256；下载后**先核对字节数和 SHA256**，不符立即停止。不要用第七/八轮缓存替代下载件。

```powershell
$replay = 'D:\SafeFAME-TS-SocialGood-clean-09'
New-Item -ItemType Directory -Path $replay | Out-Null
$url = '<以本分支 README 登记的固定 Release URL 替换>'
Invoke-WebRequest -Uri $url -OutFile (Join-Path $replay 'release.zip')
# 对照 README 检查 (Get-Item ...).Length 和 (Get-FileHash -Algorithm SHA256 ...).Hash
Expand-Archive -LiteralPath (Join-Path $replay 'release.zip') -DestinationPath (Join-Path $replay 'release')
py -3.12 -m pip install -r (Join-Path $replay 'release\requirements-replay.txt')
py -3.12 (Join-Path $replay 'release\code\verify_socialgood.py') --package (Join-Path $replay 'release') --out (Join-Path $replay 'audit')
py -3.12 (Join-Path $replay 'release\tests\run_negative_cases.py') --package (Join-Path $replay 'release') --out (Join-Path $replay 'negative_cases.json')
```

验证器逐文件检查 Release 清单、正式 Bundle 与 spec 锚、四段边界、起点 mapping、选择侧切片承诺、测试目录白名单和两情景覆盖，并重算审计表。成功退出 0，失败退出 2。负例汇总程序只有七例均被实际验证器拒绝才退出 0。输出目录必须在 Release 目录之外。

可从本地冻结 Bundle 构建新包：

```powershell
py -3.12 prepare_socialgood.py --bundle '<formal_bundle目录>' --spec '<split_spec_v2.json>' --numeric '<SocialGood_numerical.csv>' --lineage '<text_lineage.csv>' --release '<新的D盘stage目录>' --delivery '<本交付目录>'
py -3.12 make_manifest.py '<stage目录>'
py -3.12 verify_socialgood.py --package '<stage目录>' --out '<新的D盘reference目录>' --reference
# 将 reference 中的五份审计产物复制到 stage 同相对路径后：
py -3.12 make_manifest.py '<stage目录>' --reference '<reference目录>' --zip '<新的D盘ZIP文件>'
py -3.12 verify_socialgood.py --package '<stage目录>' --out '<新的D盘验证目录>'
```

`--reference` 仅用于**首次构建**冻结预期输出；交付验收必须不带该选项。正式包不含测试真值、测试逐起点数值历史或频率；它不能独立完成完整预测。`truth_commitment.json` 仅提供测试目标切片哈希承诺，不能单凭包内数据逆向重建真值，也不能独立证明选择侧数值切片相对于未分发的原 Bundle 的来源；这些需外部核对正式 Bundle。
