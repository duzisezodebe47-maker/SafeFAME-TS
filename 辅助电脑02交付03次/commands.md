# 第三轮实际执行命令

所有命令在仓库根目录用 Windows PowerShell 执行，Python 固定为 3.12 虚拟环境。

## 本轮已执行

```powershell
.\.venv\Scripts\python.exe "辅助电脑02交付03次\round3_pipeline.py" prepare

.\.venv\Scripts\python.exe "辅助电脑02交付03次\test_round3.py" `
  --master-reader "data_processed\team_worktrees\master-c608a77\team_work\main\src\team_eval\v2.py" `
  --model-reader "data_processed\team_worktrees\model-84e665b\03 辅助电脑二 交付二次\model\bundle_reader.py"

.\.venv\Scripts\python.exe "辅助电脑02交付03次\make_trace_examples.py"
```

四份 CSV 的独立复核示例：

```powershell
Get-FileHash -Algorithm SHA256 `
  "data_processed\team_transfer\辅助电脑02交付03次\clean_snapshots\*_numerical.csv"
```

## 主控冻结后执行

先把主控已经提交的冻结文件导出到 D 盘；不得手工把本交付中的待定记录改成 frozen。

```powershell
git fetch origin
git show `
  --output="data_processed\team_transfer\辅助电脑02交付03次\split_spec_v2.frozen.json" `
  origin/team/main-eval-20260921:team_work/main/round2/split_spec_v2.json

.\.venv\Scripts\python.exe "辅助电脑02交付03次\round3_pipeline.py" build `
  --split-spec "data_processed\team_transfer\辅助电脑02交付03次\split_spec_v2.frozen.json"
```

这里使用 Git 的 `--output` 直接导出 blob 字节，不经过 PowerShell 文本管道。随后仍须以主控发布的 SHA256 复核；只有完全一致的文件才能构建。

构建成功后，根据程序输出的 Bundle 路径和签名执行传输：

```powershell
$bundle = "D:\AAA个人资料\桌面\人工智能大作业\data_processed\team_data\round3-bundle-<签名前20位>"
$target = "D:\AAA个人资料\桌面\人工智能大作业\data_processed\team_transfer\辅助电脑02交付03次\formal_bundle\round3-bundle-<签名前20位>"
$specHash = (Get-FileHash -Algorithm SHA256 "data_processed\team_transfer\辅助电脑02交付03次\split_spec_v2.frozen.json").Hash.ToLower()
$sourceCommit = (git rev-parse 3218151885-creator).Trim()

.\.venv\Scripts\python.exe "辅助电脑02交付03次\transfer_bundle.py" `
  $bundle $target "辅助电脑02交付03次\transfer_receipt.json" `
  --source-commit $sourceCommit --split-spec-sha256 $specHash

.\.venv\Scripts\python.exe "辅助电脑02交付03次\round3_pipeline.py" verify --bundle $target
```

完成正式传输后，应重新执行 `test_round3.py`，并让主控在其机器独立执行 `team_eval/v2.py` 读取验证。主控尚未确认前，回执只能写 `MASTER_ACK_PENDING`，不能写 `RECEIVED`。
