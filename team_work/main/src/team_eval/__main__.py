"""Command line: audit tracked v4, freeze a new route, then score the held-out test."""
import argparse
import json
import sys
from pathlib import Path

from .core import EvidenceError, evaluate_test, freeze_route, jsonl, sha256, validate
from .legacy import audit


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    old = commands.add_parser("audit-v4")
    old.add_argument("--repo", required=True)
    old.add_argument("--out", required=True)
    for command in ("freeze", "score"):
        item = commands.add_parser(command)
        for name in ("task", "samples", "predictions", "spec", "out"):
            item.add_argument("--" + name, required=True)
        if command == "freeze":
            item.add_argument("--nulls", required=True)
        else:
            item.add_argument("--route", required=True)
    args = parser.parse_args()
    try:
        if args.command == "audit-v4":
            result = audit(args.repo)
        else:
            task = json.loads(Path(args.task).read_text(encoding="utf-8"))
            spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
            samples, predictions, _ = validate(task, jsonl(args.samples), jsonl(args.predictions), require_test=args.command == "score")
            if args.command == "freeze":
                if any(k[0] == "test" for k in samples):
                    raise EvidenceError("freeze must receive cal/dec samples only; no test evidence")
                result = freeze_route(task, samples, predictions, jsonl(args.nulls), spec)
                result["inputs_sha256"] = {name: sha256(getattr(args, name)) for name in ("task", "samples", "predictions", "nulls", "spec")}
            else:
                route = json.loads(Path(args.route).read_text(encoding="utf-8"))
                if route.get("task_id") != task["task_id"] or route.get("fold_id") != task["fold_id"] or route.get("selection_data_segments") != ["cal", "dec"]:
                    raise EvidenceError("route/task mismatch or unverified selection provenance")
                fingerprints = route.get("inputs_sha256", {})
                if fingerprints.get("task") != sha256(args.task) or fingerprints.get("spec") != sha256(args.spec):
                    raise EvidenceError("route was frozen for another task/spec version")
                result = evaluate_test(task, samples, predictions, route, spec)
                result["inputs_sha256"] = {name: sha256(getattr(args, name)) for name in ("task", "samples", "predictions", "route", "spec")}
        write(args.out, result)
        print(json.dumps({"status": result.get("status", "PASS"), "out": args.out, "selected": result.get("selected")}, ensure_ascii=False))
    except (EvidenceError, OSError, ValueError, KeyError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
