"""Run one Bundle verification operation; failures intentionally exit nonzero."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("data-read", "master-verify", "master-read", "transfer-verify"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--signature")
    parser.add_argument("--task", default="Agriculture_h12_f1")
    parser.add_argument("--scenario", default="proxy")
    parser.add_argument("--master-reader", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.kind == "data-read":
        reader = load_file("round3_data_reader_probe", HERE / "bundle_read_example.py")
        reader.read_task(args.bundle, args.spec, args.signature, args.task, args.scenario)
    elif args.kind in ("master-verify", "master-read"):
        sys.path.insert(0, str(args.master_reader.resolve().parent.parent))
        master = importlib.import_module("team_eval.v2")
        if args.kind == "master-verify":
            master.verify_bundle(args.bundle, args.signature, args.spec)
        else:
            master.load_bundle(args.bundle, args.signature, args.spec, args.task, args.scenario)
    else:
        transfer = load_file("round3_transfer_probe", HERE / "transfer_bundle.py")
        transfer.verify_transfer(args.source, args.destination)
    print(json.dumps({"status": "PASS", "kind": args.kind}))


if __name__ == "__main__":
    main()
