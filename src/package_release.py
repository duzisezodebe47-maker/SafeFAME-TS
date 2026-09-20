"""Create a compact, auditable v2 support-material archive."""

from __future__ import annotations

import csv
import hashlib
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "tmp" / "safefame_v2_release"
ARCHIVE = ROOT / "paper" / "final" / "SafeFAME-TS_v2_支撑材料.zip"


FILES = [
    "README.md",
    "requirements.txt",
    "paper/final/第4周_课程设计选题与初步技术方案.docx",
    "paper/final/第4周_课程设计选题与初步技术方案.pdf",
    "paper/final/第10周_课程设计书面中期进展报告.docx",
    "paper/final/第10周_课程设计书面中期进展报告.pdf",
    "paper/final/SafeFAME-TS_课程设计报告_修订版.docx",
    "paper/final/SafeFAME-TS_课程设计报告_修订版.pdf",
    "paper/final/a11y_第4周.json",
    "paper/final/a11y_第10周.json",
    "paper/final/a11y_最终报告.json",
    "paper/final/支撑材料清单_v2.txt",
    "data_processed/text/text_index_audit.json",
    "data_processed/text/point_in_time_coverage.csv",
    "data_processed/semantic_features/manifest.json",
]


def selected_files() -> list[Path]:
    paths = [ROOT / item for item in FILES]
    for folder, patterns in [
        ("src", ("*.py",)),
        ("docs", ("*.md",)),
        ("outputs/safefame_v2", ("*.csv", "*.json")),
        ("outputs/reviewer_sensitivity", ("*.csv", "*.json", "*.png")),
        ("outputs/tables/v2", ("*.csv", "*.json")),
        ("outputs/tables/final", ("*.csv", "*.json")),
        ("outputs/figures", ("*.png",)),
        ("outputs/figures/v2", ("*.png",)),
        ("prototype/dist", ("*.html", "*.css", "*.js")),
    ]:
        base = ROOT / folder
        for pattern in patterns:
            paths.extend(sorted(base.glob(pattern)))
    unique = sorted(set(paths), key=lambda path: path.as_posix())
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing release files: " + ", ".join(str(path) for path in missing))
    return unique


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    records = []
    for source in selected_files():
        relative = source.relative_to(ROOT)
        target = STAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append((relative.as_posix(), target.stat().st_size, digest(target)))
    with (STAGE / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["path", "bytes", "sha256"])
        writer.writerows(records)
    (STAGE / "SHA256SUMS.txt").write_text(
        "\n".join(f"{sha}  {path}" for path, _, sha in records) + "\n", encoding="utf-8"
    )
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(STAGE).as_posix())
    archive_hash = digest(ARCHIVE)
    (ROOT / "paper" / "final" / "SafeFAME-TS_v2_支撑材料.sha256").write_text(
        f"{archive_hash}  {ARCHIVE.name}\n", encoding="utf-8"
    )
    print(f"{ARCHIVE} ({ARCHIVE.stat().st_size} bytes, {len(records) + 2} files)")


if __name__ == "__main__":
    main()
