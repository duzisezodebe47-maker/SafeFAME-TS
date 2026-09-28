# RUNBOOK · 第十轮复现说明

面向主控：A（证据链修复）与 B（退化契约）各自怎么复核。

---

## A. 复核 Climate 第九次交付的证据链

```bash
PY=.venv/Scripts/python.exe
S9="03 辅助电脑二 交付九次"
ROUTE_SHA=b3bce4713ff44485c9aa0712c5774ffe449045fb8186b39ef592298e0051e36a
```

**A1/A6 预测字节冻结**（必须等于公布值）

```bash
sha256sum "$S9/test_prediction/N_test_predictions.csv"
# f2200192f4b555a3e56fb8702f59361ecf54f42d4bbc04643b8a130597d6b122
```

**A4 在全新解包目录里验证**（不留解释空间：不依赖工作区、不依赖 `__pycache__`）

```bash
rm -rf /tmp/verify9 && mkdir -p /tmp/verify9
git archive HEAD "$S9" | tar -x -C /tmp/verify9
cd /tmp/verify9
$PY "$S9/model/verify_test_delivery.py" --delivery "$S9" --task Climate_h4_f2 \
    --horizon 4 --expected-origins 124 --candidate N --route-sha256 $ROUTE_SHA
# 期望 exit 0，{"status":"COMPLETE","problems":[]}
```

**A5 两个哈希的用途**

```bash
# 工作树字节（磁盘）——用于发现"提交时会被改写"的文件
$PY -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <文件>
# 仓库真正存储的对象 —— 与 commit 里的 blob 可直接比对
git hash-object --path <相对路径> <文件>
git rev-parse HEAD:<相对路径>
```

逐文件对照见 `evidence/round9_hash_registry.json`：`worktree_sha256` 与 `git_blob_sha1`
分列，`normalized_on_commit` 标出会被行尾规则改写的文件（本轮为 0 个）。

**A2/A3 为什么不会再坏**

- 生成器 `make_manifest.py` 的每个哈希取自 **git 会存储的 blob 字节**
  （`git hash-object --path` → `cat-file blob`），因此不可能记录"注定对不上"的哈希；
- 门控回执（`evidence/verify_*.json`）被**排除在清单之外** —— 它是读取清单之后写出的
  派生产物，属自指。`MANIFEST.json` 的 `excluded_receipts` 列出被排除的路径与理由。

重新生成清单（任何交付目录通用）:

```bash
$PY "<交付目录>/model/make_manifest.py" --out "<交付目录>/MANIFEST.json" \
    --registry "<交付目录>/evidence/hash_registry.json"
```

---

## B. 复核 SocialGood 安全退化契约

```bash
$PY "辅助电脑二交付十次/model/tests/test_round10.py"    # 期望 195 项、退出码 0
```

契约本身在 `model/degradation_contract.py`，四条规则见 README。要点：

- **阈值只定义一次**：`MIN_TRAIN_TEXT_ROWS = 2`，与 `branches.SemanticBranch.fit`
  的「训练段文本行 < 2 即不可训练」一致 —— 契约与实现不允许有两套标准。
- **接口签名里没有测试段参数**（R3）。想验证这一点可直接看签名：

  ```bash
  $PY -c "import sys,inspect;sys.path.insert(0,'辅助电脑二交付十次/model');
          import degradation_contract as d;print(inspect.signature(d.resolve_candidate))"
  # (requested, *, registry, train_text_rows, train_rows, quality_usable)
  ```

- **起点数可独立复算**：

  ```bash
  $PY -c "import sys;sys.path.insert(0,'辅助电脑二交付十次/model');
          import degradation_contract as d;
          print(d.expected_origins([366,458,641,732], input_len=24, horizon=3))"
  # all=700, selection_region=611, test=89, gating_origins=271
  ```

- **本轮不生成 SocialGood 正式预测**：测试里有一条断言交付目录中不出现任何
  SocialGood 预测产物；主控验收数据包并签发候选注册表后，才启动训练。
  届时 `resolve_candidate()` 的 `registry` 参数应传入**主控签发的注册表**。

---

## 未做范围

1. 未重跑、未改写第九次的任何预测（只修生成器/清单/回执）。
2. 未生成 SocialGood 正式预测，未读取 SocialGood 测试真值。
3. 未修改第八次及更早的历史交付目录。
4. 未 `merge` 主控分支或 `main`。
