from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def values(samples, arrays, indexes: list[int]) -> list[dict]:
    result = []
    for i in indexes:
        row = samples.iloc[i] if hasattr(samples, "iloc") else samples[i]
        result.append({
            "row": i,
            "origin_id": str(row["origin_id"]),
            "origin_index": int(arrays["origin_index"][i]),
            "target_raw": np.asarray(arrays["targets"][i]).astype(float).tolist(),
            "target_standardized": np.asarray(arrays["targets_standardized"][i]).astype(float).tolist(),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--master-package", required=True, type=Path)
    parser.add_argument("--model-reader", required=True, type=Path)
    parser.add_argument("--data-reader", required=True, type=Path)
    parser.add_argument("--master-commit", required=True)
    parser.add_argument("--model-commit", required=True)
    parser.add_argument("--data-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    task, scenario = "Agriculture_h12_f1", "proxy"

    sys.path.insert(0, str(args.master_package.parent))
    master = importlib.import_module(f"{args.master_package.name}.v2")
    model = load_module("round5_model_reader", args.model_reader)
    data = load_module("round5_data_reader", args.data_reader)
    os.environ["DATA_SIDE_DIR"] = str(args.data_reader.parent.parent / "辅助电脑02交付01次")

    data_value = data.read_task(args.bundle, args.spec, args.signature, task, scenario)
    master_value = master.load_bundle(args.bundle, args.signature, args.spec, task, scenario)
    model_value = model.read_frozen_bundle(
        args.bundle, task, scenario, args.signature, args.spec
    )
    n = len(data_value["samples"])
    indexes = [0, n // 2, n - 1]
    data_rows = values(data_value["samples"], data_value["arrays"], indexes)
    master_rows = values(master_value["samples"], master_value["arrays"], indexes)
    model_rows = values(model_value.samples, model_value.arrays, indexes)

    exact = data_rows == master_rows == model_rows
    if not exact:
        raise ValueError("formal Agriculture readers differ")
    report = {
        "status": "PASS",
        "scope": "formal_bundle_agriculture_proxy_three_readers",
        "bundle_signature": args.signature,
        "task_id": task,
        "scenario": scenario,
        "origins_total": n,
        "rows_compared": indexes,
        "readers": {
            "data": {"path": str(args.data_reader), "commit": args.data_commit},
            "master": {"path": str(args.master_package / "v2.py"), "commit": args.master_commit},
            "model": {"path": str(args.model_reader), "commit": args.model_commit},
        },
        "comparison_fields": ["origin_id", "origin_index", "target_raw", "target_standardized"],
        "rows": data_rows,
        "all_exact": exact,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "origins": n, "compared": len(indexes)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
