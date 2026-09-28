"""生成交付清单 `MANIFEST.json`（第十轮修订版）。

## 相对上一版修了什么（第十轮任务书 A.2 / A.3 / A.5）

**A.2 清单必须记录「最终 Git blob 字节」，而不是磁盘上的临时内容。**
上一版直接对**工作区字节**取 sha256；而 `.gitattributes` 会把 `*.json` / `*.csv`
等归一成 LF，于是一个在磁盘上带 CRLF 的文件，清单里记的哈希与主控检出后拿到的
字节**对不上**（第九次交付就是这么被判 `INCOMPLETE` 的）。
现在每个文件的哈希取自 **`git hash-object --path <rel>` 得到的 blob 内容**
（即 git 实际会存储的字节），并额外记录工作区字节的 sha256，两者不一致就**报出来**。
清单因此**不可能**记录一个注定对不上的哈希。

**A.3 门控输出不得反过来改变它自己校验的清单。**
`evidence/verify_*.json` 这类**派生产物（receipt）**是从"读取清单"这个动作里写出来的，
天然自指：写它就是在改变它。现在生成器把它们**排除在清单之外**（与 `MANIFEST.json`
自身同理），并在产物里列出被排除的路径与理由。这样「门控跑完 → 写回执 → 生成清单
→ 提交」不再需要迭代到不动点。

**A.5 两个哈希用途不同，同时记录。**
`worktree_sha256`：磁盘字节的哈希 —— 用于**本机复现**与发现"检出后会被改写"的文件；
`git_blob_sha1`：`git hash-object` 的结果 —— 即**仓库里真正存的对象**，
与 `git rev-parse HEAD:<path>` 可直接比对，用于**跨机器核对**。

集合范围分三块，合起来 = 本交付目录下除以下两类外的全部文件：
`MANIFEST.json` 自身，以及 receipt（派生产物）。

用法::

    python make_manifest.py --out <交付目录>/MANIFEST.json \\
                            [--registry <交付目录>/evidence/hash_registry.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DELIVERY = HERE.parent

SKIP_DIRS = {"__pycache__"}
# 派生产物（receipt）：由"读取清单/校验交付"这个动作写出，属自指，故不列入清单。
# 与 `MANIFEST.json` 自身同理 —— 无法自哈希。
RECEIPT_PATTERNS = ("evidence/verify_test_delivery.json", "evidence/verify_delivery.json",
                    "evidence/verify_evidence_chain.json")


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stored_bytes(path: Path, rel: str) -> tuple[bytes, str | None]:
    """返回 `(git 会存储的字节, blob id)`。

    用 `git hash-object --path <rel>` 让**同一套 filter**（`.gitattributes` 的 eol 规则）
    作用在该文件上，再 `git cat-file blob` 取回它 —— 这就是提交后主控拿到的字节。
    git 不可用时退回工作区字节，并在返回值里如实说明（blob id 为 None）。
    """
    hashed = _git(["hash-object", "-w", "--path", rel, str(path)], HERE.parents[0])
    if hashed.returncode != 0:
        return path.read_bytes(), None
    blob = hashed.stdout.decode("ascii", errors="replace").strip()
    cat = _git(["cat-file", "blob", blob], HERE.parents[0])
    if cat.returncode != 0:
        return path.read_bytes(), None
    return cat.stdout, blob


def collect(root: Path, delivery: Path, exclude: set[str]) -> tuple[dict, dict]:
    """返回 `(清单 {rel: sha256_of_stored}, 登记 {rel: {...双哈希...}})`。"""
    manifest: dict[str, str] = {}
    registry: dict[str, dict] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = p.relative_to(delivery).as_posix()
        if rel in exclude:
            continue
        raw = p.read_bytes()
        stored, blob = stored_bytes(p, rel)
        manifest[rel] = sha256_bytes(stored)
        registry[rel] = {
            "worktree_sha256": sha256_bytes(raw),
            "git_blob_sha1": blob,
            "bytes_worktree": len(raw),
            "bytes_stored": len(stored),
            "normalized_on_commit": raw != stored,
        }
    return manifest, registry


def combined(mapping: dict[str, str]) -> str:
    text = "\n".join(f"{path}\t{digest}" for path, digest in sorted(mapping.items()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DELIVERY / "MANIFEST.json")
    parser.add_argument("--registry", type=Path, default=None,
                        help="可选：写出逐文件双哈希登记（A.5）")
    args = parser.parse_args()

    delivery = Path(args.out).resolve().parent
    exclude = {args.out.name} | set(RECEIPT_PATTERNS)
    if args.registry is not None:
        try:
            exclude.add(Path(args.registry).resolve().relative_to(delivery).as_posix())
        except ValueError:
            pass

    model_files, model_reg = collect(HERE, delivery, exclude)
    evidence_files, evidence_reg = collect(delivery / "evidence", delivery, exclude)
    covered = {"model", "evidence"}
    other_files, other_reg = {}, {}
    for p in sorted(delivery.rglob("*")):
        if not p.is_file() or any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = p.relative_to(delivery).as_posix()
        if rel in exclude or rel.split("/")[0] in covered:
            continue
        stored, blob = stored_bytes(p, rel)
        other_files[rel] = sha256_bytes(stored)
        other_reg[rel] = {"worktree_sha256": sha256_bytes(p.read_bytes()),
                          "git_blob_sha1": blob, "bytes_worktree": p.stat().st_size,
                          "bytes_stored": len(stored),
                          "normalized_on_commit": p.read_bytes() != stored}

    registry = {**model_reg, **evidence_reg, **other_reg}
    normalized = sorted(rel for rel, r in registry.items() if r["normalized_on_commit"])
    all_files = {**model_files, **evidence_files, **other_files}

    manifest = {
        "rule": "combined = sha256( '\\n'.join( f'{path}\\t{sha256}' 按 path 排序 ) )；"
                "逐文件 sha256 取自 **git 会存储的 blob 字节**（`git hash-object --path` "
                "再 `cat-file blob`），因此提交后必然与仓库字节一致（A.2）",
        "excluded_receipts": sorted(set(RECEIPT_PATTERNS) & exclude),
        "excluded_note": "receipt 由校验动作写出，属自指，无法列入自身校验的清单（A.3）；"
                         "其内容可由重跑同一门控在干净检出中复现",
        "model_files": model_files, "model_files_count": len(model_files),
        "model_combined_sha256": combined(model_files),
        "evidence_files": evidence_files, "evidence_files_count": len(evidence_files),
        "evidence_combined_sha256": combined(evidence_files),
        "other_files": other_files, "other_files_count": len(other_files),
        "other_combined_sha256": combined(other_files),
        "all_files_count": len(all_files), "all_combined_sha256": combined(all_files),
        "files_normalized_on_commit": normalized,
        "hash_algorithms": {
            "sha256": "文件内容（git 归一化后的最终字节）",
            "worktree_sha256": "磁盘上的原始字节 —— 用途不同：发现会被归一化的文件",
            "git_blob_sha1": "git 对象 id，与 `git rev-parse HEAD:<path>` 可直接比对",
        },
    }
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8", newline="\n")

    if args.registry is not None:
        Path(args.registry).write_text(json.dumps({
            "note": "逐文件双哈希（A.5）：worktree_sha256 是磁盘字节，git_blob_sha1 是"
                    "仓库真正存储的对象；两者不同说明该文件在提交时会被改写（行尾等）",
            "files": registry, "normalized_on_commit": normalized,
        }, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    print(json.dumps({"all_files": len(all_files), "normalized_on_commit": normalized,
                      "all_combined_sha256": manifest["all_combined_sha256"],
                      "model_files": len(model_files), "evidence_files": len(evidence_files),
                      "other_files": len(other_files)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
