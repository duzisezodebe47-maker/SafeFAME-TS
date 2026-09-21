"""Verify model-side 999-row null CSVs and emit the main freeze input JSONL."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from team_eval.null_bridge import convert_nulls, write_conversion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--null-dir", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, report = convert_nulls(args.selection_dir, args.null_dir, args.seed)
    write_conversion(args.out, rows, report)
    print(json.dumps({"status": report["status"], "out": str(args.out),
                      "candidates": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
