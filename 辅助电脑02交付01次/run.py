"""Unified data-audit / frozen-feature / origin-trace entry point."""
import argparse
import json
import time
from pathlib import Path

from audit import ROOT, environment, peak_memory, publish_audit, run_audit, write_json
from bundle import build, trace_origin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'build', 'trace'])
    parser.add_argument('--audit-folder', type=Path)
    parser.add_argument('--split-spec', type=Path)
    parser.add_argument('--bundle-folder', type=Path)
    parser.add_argument('--task-id')
    parser.add_argument('--origin-id')
    parser.add_argument('--scenario', choices=['proxy', 'conservative_lag', 'complete_source'], default='proxy')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--expected-signature')
    args = parser.parse_args()
    start = time.perf_counter()
    evidence = Path(__file__).parent / 'evidence'
    evidence.mkdir(exist_ok=True)
    if args.command == 'audit':
        folder, summary = run_audit()
        publish_audit(folder, summary, evidence)
        print(json.dumps(dict(output=str(folder), **summary), ensure_ascii=False, indent=2))
    elif args.command == 'build':
        if args.audit_folder is None or args.split_spec is None:
            parser.error('build needs --audit-folder and --split-spec from master')
        folder, manifest = build(args.audit_folder.resolve(), args.split_spec.resolve(),
                                 threads=args.threads, expected_signature=args.expected_signature)
        print(json.dumps(dict(output=str(folder), signature=manifest['signature']), indent=2))
    else:
        if not all((args.audit_folder, args.bundle_folder, args.task_id, args.origin_id)):
            parser.error('trace needs --audit-folder, --bundle-folder, --task-id and --origin-id')
        row, lineage = trace_origin(args.audit_folder.resolve(), args.bundle_folder.resolve(),
                                   args.task_id, args.origin_id, args.scenario)
        folder = ROOT / 'data_processed/team_data/traces'
        folder.mkdir(exist_ok=True)
        target = folder / (args.origin_id.replace(':', '_') + '_' + args.scenario + '.csv')
        lineage.to_csv(target, index=False)
        print(json.dumps(dict(sample=row, raw_text_reasons=str(target)), ensure_ascii=False, default=str, indent=2))
    write_json(evidence / 'last_runtime.json', dict(command=args.command, elapsed_seconds=time.perf_counter()-start,
        environment=environment(), memory=peak_memory(), gpu='not used'))


if __name__ == '__main__':
    main()
