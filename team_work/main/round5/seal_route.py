"""CLI for a route after master has reviewed actual refit provenance."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from team_eval.route_seal import seal_route


def main():
    parser = argparse.ArgumentParser()
    for name in ("stage-dir", "nulls", "route", "frozen-spec", "refit-review", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = seal_route(args.stage_dir, args.nulls, args.route,
                        args.frozen_spec, args.refit_review, args.out)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
