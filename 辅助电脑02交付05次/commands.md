# 正式 Bundle 接收与复核命令

以下命令供主控电脑在仓库根目录使用。所有下载和解包路径均在 D 盘。

## 1. 下载附件

```powershell
$url = "https://github.com/duzisezodebe47-maker/SafeFAME-TS/releases/download/data-bundle-a69821be115445265cde/SafeFAME-TS_formal_bundle_a69821be115445265cde.zip"
$zip = "D:\SafeFAME-TS-transfer\SafeFAME-TS_formal_bundle_a69821be115445265cde.zip"
New-Item -ItemType Directory -Force -Path "D:\SafeFAME-TS-transfer" | Out-Null
Invoke-WebRequest -Uri $url -OutFile $zip
Get-FileHash -Algorithm SHA256 -LiteralPath $zip
```

期望 SHA256：

```text
8abde5043f7399c96d269b5adbbc35dcbcada0e8760114a41b603031c01f0e06
```

## 2. 解包到 D 盘

```powershell
New-Item -ItemType Directory -Force -Path "D:\SafeFAME-TS-transfer\received" | Out-Null
tar -xf $zip -C "D:\SafeFAME-TS-transfer\received"
```

解包后的正式目录：

```text
D:\SafeFAME-TS-transfer\received\round3-bundle-a69821be115445265cde
```

## 3. 核对冻结协议原始字节

```powershell
git cat-file blob "6c05bd60661386c39121cbf079298fa1873d97df:team_work/main/round2/split_spec_v2.json" > "D:\SafeFAME-TS-transfer\split_spec_v2.json"
Get-FileHash -Algorithm SHA256 -LiteralPath "D:\SafeFAME-TS-transfer\split_spec_v2.json"
```

期望 SHA256：`a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031`。

## 4. 数据侧完整验证

```powershell
.venv\Scripts\python.exe "辅助电脑02交付03次\round3_pipeline.py" verify `
  --bundle "D:\SafeFAME-TS-transfer\received\round3-bundle-a69821be115445265cde"
```

期望状态：`PASS`；Bundle 签名：`a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38`。

## 5. 主控 Agriculture 实读

主控在自身分支环境中调用：

```python
from pathlib import Path
from team_eval.v2 import load_bundle

value = load_bundle(
    Path(r"D:\SafeFAME-TS-transfer\received\round3-bundle-a69821be115445265cde"),
    "a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38",
    Path(r"D:\SafeFAME-TS-transfer\split_spec_v2.json"),
    "Agriculture_h12_f1",
    "proxy",
)
print(len(value["samples"]))  # 357
```

只有主控完成上述下载、哈希、全清单和 Agriculture 读取后，才可把 `receiver_status` 从 `MASTER_ACK_PENDING` 更新为 `RECEIVED`。
