"""Create a compact, auditable v2 support-material archive."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "SafeFAME-TS_v2_支撑材料.zip"


FILES = [
    "README.md",
    "requirements.txt",
    "submission.json",
    "docs/submission_qa.json",
    "data_processed/text/text_index_audit.json",
    "data_processed/text/point_in_time_coverage.csv",
    "data_processed/semantic_features/manifest.json",
    "data_processed/embeddings/all_minilm_l6_v2/fact_embedding_metadata.json",
    "data_processed/v4/numerical_audit.csv",
    "data_processed/v4/preparation_audit.json",
    "data_processed/v4/embeddings/fact_embedding_metadata.json",
    "data_processed/v4/semantic_features/manifest.json",
]


def selected_files() -> list[Path]:
    paths = [ROOT / item for item in FILES]
    config=json.loads((ROOT/'submission.json').read_text(encoding='utf-8'))
    paths.append(ROOT/config['inventory'])
    paths.append(ROOT/'paper/final'/config['inventory'])
    for report in config['reports']:
        paths.append(ROOT/report['mirror'])
        if config.get('pdf_required', False):
            paths.append(ROOT/report['mirror_pdf'])
    paths.extend((ROOT/'paper/final').glob('a11y_*.json'))
    for folder, patterns in [
        ("src", ("*.py",)),
        ("configs", ("*.json",)),
        ("docs", ("*.md",)),
        ("outputs/safefame_v2", ("*.csv", "*.json")),
        ("outputs/reviewer_sensitivity", ("*.csv", "*.json", "*.png")),
        ("outputs/tables/v2", ("*.csv", "*.json")),
        ("outputs/tables/final", ("*.csv", "*.json")),
        ("outputs/tables", ("*.csv", "*.json")),
        ("outputs/audit", ("*.csv", "*.json")),
        ("outputs/baselines", ("*.csv", "*.json")),
        ("outputs/dlinear_u", ("*metrics*.csv", "*run.json")),
        ("outputs/confirmation_baselines", ("*.csv", "*.json")),
        ("outputs/famets_selective", ("*metrics*.csv", "*audit*.csv", "*.json")),
        ("outputs/famets_confirmation", ("*metrics*.csv", "*audit*.csv", "*.json")),
        ("outputs/figures", ("*.png",)),
        ("outputs/figures/v2", ("*.png",)),
        ("outputs/semantic_probe", ("*metrics.csv", "*config.json")),
        ("outputs/tfidf_probe", ("*metrics.csv", "*config.json")),
        ("outputs/dlinear", ("*metrics*.csv", "*run.json")),
        ("outputs/patchtst", ("*metrics*.csv", "*run.json")),
        ("outputs/ett_benchmark", ("*metrics*.csv", "*run.json")),
        ("outputs/robustness", ("*summary.csv", "*protocol.json")),
        ("prototype/dist", ("*.html", "*.css", "*.js")),
    ]:
        base = ROOT / folder
        for pattern in patterns:
            paths.extend(sorted(base.glob(pattern)))
    for folder in ('safefame_v3', 'reviewer_sensitivity_corrected', 'reviewer_sensitivity_v3', 'revision_comparison', 'safefame_v4', 'attribution_audit_v4'):
        paths.extend(p for p in (ROOT/'outputs'/folder).rglob('*')
                     if p.is_file() and p.suffix in {'.json','.csv','.npz','.pt','.png'})
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
    config=json.loads((ROOT/'submission.json').read_text(encoding='utf-8'))
    shutil.copy2(ROOT/config['inventory'],ROOT/'paper/final'/config['inventory'])
    (ROOT/'tmp').mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='safefame-release-',dir=ROOT/'tmp'))
    records = []
    for source in selected_files():
        relative = source.relative_to(ROOT)
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append((relative.as_posix(), target.stat().st_size, digest(target)))
    with (stage / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["path", "bytes", "sha256"])
        writer.writerows(records)
    (stage / "SHA256SUMS.txt").write_text(
        "\n".join(f"{sha}  {path}" for path, _, sha in records) + "\n", encoding="utf-8"
    )
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(stage).as_posix())
    archive_hash = digest(ARCHIVE)
    ARCHIVE.with_suffix(".sha256").write_text(
        f"{archive_hash}  {ARCHIVE.name}\n", encoding="utf-8"
    )
    config=json.loads((ROOT/'submission.json').read_text(encoding='utf-8'))
    shutil.copy2(ROOT/config['inventory'],ROOT/'paper/final'/config['inventory'])
    print(f"{ARCHIVE} ({ARCHIVE.stat().st_size} bytes, {len(records) + 2} files)")
    shutil.rmtree(stage)


if __name__ == "__main__":
    main()
