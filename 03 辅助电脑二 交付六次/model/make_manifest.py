"""生成交付清单 `MANIFEST.json`（逐文件 SHA256 + 两个合并哈希）。

上一轮的 `MANIFEST.json` 没有生成脚本，是手工汇总的 —— 主控无法独立复算
「按路径排序拼接」到底怎么拼。这里把规则写死并随产物一起交付：

  逐文件哈希  ：`sha256(文件字节)`
  合并哈希    ：把 `相对路径` 与 `逐文件哈希` 用 `\t` 连接成一行，按**路径字典序**
                排序，行间用 `\n` 连接（结尾不加换行），整体取 `sha256`

集合范围：`model/`（排除 `__pycache__`）与 `evidence/` 下的全部普通文件。
两个合并哈希分别给出，便于主控用一个字符串核对整棵子树。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DELIVERY = HERE.parent

SKIP_DIRS = {"__pycache__"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root: Path) -> dict[str, str]:
    files = [p for p in sorted(root.rglob("*"))
             if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)]
    return {p.relative_to(DELIVERY).as_posix(): sha256_file(p) for p in files}


def combined(mapping: dict[str, str]) -> str:
    text = "\n".join(f"{path}\t{digest}" for path, digest in sorted(mapping.items()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DELIVERY / "MANIFEST.json")
    args = parser.parse_args()

    model_files = collect(HERE)
    evidence_files = collect(DELIVERY / "evidence")
    total_bytes = sum((DELIVERY / p).stat().st_size
                      for p in list(model_files) + list(evidence_files))

    manifest = {
        "rule": "combined = sha256( '\\n'.join( f'{path}\\t{sha256(file)}' 按 path 排序 ) )",
        "model_files": model_files,
        "model_files_count": len(model_files),
        "model_combined_sha256": combined(model_files),
        "evidence_files": evidence_files,
        "evidence_files_count": len(evidence_files),
        "evidence_combined_sha256": combined(evidence_files),
        "evidence_bytes": total_bytes,
        "note": "model/ 排除 __pycache__；evidence/ 全量。生成脚本 model/make_manifest.py（本文件亦在 model_files 内）。",
    }
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8", newline="\n")
    print(json.dumps({"model_files": len(model_files),
                      "evidence_files": len(evidence_files),
                      "evidence_bytes": total_bytes,
                      "model_combined_sha256": manifest["model_combined_sha256"],
                      "evidence_combined_sha256": manifest["evidence_combined_sha256"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
