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
    # 第三块：交付根下的其余文件（预测产物、文档等）。
    # 为什么必须单独列：第七轮交付的主体产物在 `test_prediction/` 下，既不在 model/
    # 也不在 evidence/ —— 只报前两块的话，交付清单会**漏掉本轮最重要的那个文件**，
    # 而任务书要求的是「本轮所有文件 SHA256」。前两块的算法保持不变，
    # 以免主控已记录的 evidence/model 合并哈希发生变化。
    covered = {"model", "evidence"}
    other_files = {
        p.relative_to(DELIVERY).as_posix(): sha256_file(p)
        for p in sorted(DELIVERY.rglob("*"))
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)
        and p.name != args.out.name
        and p.relative_to(DELIVERY).parts[0] not in covered
    }
    total_bytes = sum((DELIVERY / p).stat().st_size
                      for p in list(model_files) + list(evidence_files) + list(other_files))

    manifest = {
        "rule": "combined = sha256( '\\n'.join( f'{path}\\t{sha256(file)}' 按 path 排序 ) )",
        "model_files": model_files,
        "model_files_count": len(model_files),
        "model_combined_sha256": combined(model_files),
        "evidence_files": evidence_files,
        "evidence_files_count": len(evidence_files),
        "evidence_combined_sha256": combined(evidence_files),
        "other_files": other_files,
        "other_files_count": len(other_files),
        "other_combined_sha256": combined(other_files),
        "all_files_count": len(model_files) + len(evidence_files) + len(other_files),
        "all_files_bytes": total_bytes,
        "note": "三块合并 = 本目录（除 MANIFEST.json 自身与 __pycache__）的全部文件。"
                "MANIFEST.json 无法自哈希，其完整性由 git blob 哈希保证。"
                "生成脚本 model/make_manifest.py（本文件亦在 model_files 内）。",
    }
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8", newline="\n")
    print(json.dumps({"model_files": len(model_files),
                      "evidence_files": len(evidence_files),
                      "other_files": len(other_files),
                      "all_files": manifest["all_files_count"],
                      "all_bytes": total_bytes,
                      "model_combined_sha256": manifest["model_combined_sha256"],
                      "evidence_combined_sha256": manifest["evidence_combined_sha256"],
                      "other_combined_sha256": manifest["other_combined_sha256"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
