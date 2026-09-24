"""权重重演：交付的预测是否可由**已提交的代码 + 冻结配置**逐位重放。

主控第六轮任务书把「权重重演」列为正式门控验收项之一，并且会把交付里的
`code_commit` 当作锚点核对。这两件事只有一起成立才有意义：如果预测无法从
那个提交重放出来，`code_commit` 就只是一个装饰性的字符串。

本模块对每个候选跑**两次** `train.py`（同一 Bundle、同一冻结 spec、同一 seed），
然后比对：

  1. 两次输出的预测 CSV **逐字节相同**（`bytes` 相等，不是「值相同」）；
  2. 两次 run manifest 里的 `weight_hash` 相同，且与交付目录里那份 manifest 相同；
  3. `alpha_by_group` / `branch_widths` 与交付那份一致。

第 3 条把「重放」与「交付物」接起来：不是重放两份临时产物自说自话，而是重放
结果必须**等于已经交付的那份**。任一项不符即 FAIL。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from candidates import ALL_CANDIDATES, GATE_CANDIDATES  # noqa: E402
from isolated_package import add_input_args  # noqa: E402
from predict_io import warn_if_dirty  # noqa: E402
from runtime_profile import cpu_seconds, script_entry_snapshot  # noqa: E402
from train import RUNNABLE_SCENARIOS  # noqa: E402

REPO_ROOT = HERE.parents[1]
TRAIN = HERE / "train.py"

# 只有当轮交付的那几个候选做重演。`ALL_CANDIDATES` 还含 N+S / N+F / N+S+Q+F 等
# 诊断候选，它们**不在本轮交付范围内**（没有对应的已交付 CSV），拿它们去比对
# 「重演 == 交付」必然失败 —— 第一版就这么错了，把不在交付里的候选算成了失败。
DELIVERED_CANDIDATES = ("N", "N+Q", "N+S+Q", "N+S+Q+SF")


def run_train(bundle: Path | None, task: str, scenario: str, signature: str, spec: Path,
              candidate: str, out_dir: Path, seed: int,
              input_package: Path | None = None,
              formal_bundle: Path | None = None) -> int:
    cmd = [sys.executable, str(TRAIN), "--task", task,
           "--scenario", scenario, "--signature", signature, "--split-spec", str(spec),
           "--segments", "calibration", "decision", "--candidate", candidate,
           "--seed", str(seed), "--output-dir", str(out_dir)]
    if input_package is not None:
        cmd += ["--input-package", str(input_package)]
        if formal_bundle is not None:
            cmd += ["--formal-bundle", str(formal_bundle)]
    else:
        cmd += ["--bundle", str(bundle)]
    proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(f"train.py 失败（{candidate}）: {proc.stderr[-500:]}", file=sys.stderr)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    add_input_args(parser)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scenario", required=True, choices=RUNNABLE_SCENARIOS)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--split-spec", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--delivered-dir", type=Path, required=True,
                        help="已交付的 cal_dec_predictions 目录，用于比对")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    cpu_started = cpu_seconds()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    delivered = Path(args.delivered_dir).resolve()

    # 交付范围必须恰好是这四个：门控两候选 + 消融 N+Q + 数值回退 N。
    # 少一个（漏交）或多一个（混入未注册候选）都要在这里暴露。
    missing = [c for c in GATE_CANDIDATES + ("N", "N+Q")
               if not (delivered / f"{c.replace('+', '_')}_run_manifest.json").is_file()]

    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for candidate in DELIVERED_CANDIDATES:
            assert candidate in ALL_CANDIDATES, f"未注册候选: {candidate}"
            stem = candidate.replace("+", "_")
            entry: dict = {"candidate": candidate, "ok": False}
            a, b = tmp / f"{stem}_a", tmp / f"{stem}_b"
            rc_a = run_train(args.bundle, args.task, args.scenario, args.signature,
                             args.split_spec, candidate, a, args.seed,
                             input_package=getattr(args, "input_package", None),
                             formal_bundle=getattr(args, "formal_bundle", None))
            rc_b = run_train(args.bundle, args.task, args.scenario, args.signature,
                             args.split_spec, candidate, b, args.seed,
                             input_package=getattr(args, "input_package", None),
                             formal_bundle=getattr(args, "formal_bundle", None))
            entry["train_exit_codes"] = [rc_a, rc_b]
            if rc_a or rc_b:
                entry["error"] = "train.py 未能成功退出"
                results[candidate] = entry
                continue

            csv_name = f"{stem}_{args.scenario}_predictions.csv"
            man_name = f"{stem}_run_manifest.json"
            bytes_a = (a / csv_name).read_bytes()
            bytes_b = (b / csv_name).read_bytes()
            delivered_path = delivered / csv_name
            man_a = json.loads((a / man_name).read_text(encoding="utf-8"))
            man_b = json.loads((b / man_name).read_text(encoding="utf-8"))
            man_delivered = (json.loads((delivered / man_name).read_text(encoding="utf-8"))
                             if (delivered / man_name).is_file() else None)

            entry.update({
                "two_runs_byte_identical": bytes_a == bytes_b,
                "weight_hash": man_a.get("weight_hash"),
                "weight_hash_reproduced": man_a.get("weight_hash") == man_b.get("weight_hash"),
                "delivered_csv_matches_replay": bool(
                    delivered_path.is_file() and delivered_path.read_bytes() == bytes_a),
                "delivered_weight_hash_matches": bool(
                    man_delivered is not None
                    and man_delivered.get("weight_hash") == man_a.get("weight_hash")),
                "alpha_by_group": man_a.get("alpha_by_group"),
                "alpha_matches_delivered": bool(
                    man_delivered is not None
                    and man_delivered.get("alpha_by_group") == man_a.get("alpha_by_group")),
                "branch_widths": man_a.get("branch_widths"),
                "branch_widths_match_delivered": bool(
                    man_delivered is not None
                    and man_delivered.get("branch_widths") == man_a.get("branch_widths")),
            })
            entry["ok"] = all([
                entry["two_runs_byte_identical"], entry["weight_hash_reproduced"],
                entry["delivered_csv_matches_replay"], entry["delivered_weight_hash_matches"],
                entry["alpha_matches_delivered"], entry["branch_widths_match_delivered"],
            ])
            results[candidate] = entry

    report = {
        "task": args.task, "scenario": args.scenario, "seed": args.seed,
        "bundle_signature": args.signature,
        "delivered_candidates": list(DELIVERED_CANDIDATES),
        "missing_delivered_manifests": missing,
        "candidates": results,
        "verdict": ("PASS" if results and not missing
                    and all(e["ok"] for e in results.values()) else "FAIL"),
        "code_provenance": warn_if_dirty(REPO_ROOT, HERE),
        "runtime": script_entry_snapshot(started, cpu_started),
        # 本模块真正的计算在两个 train.py **子进程**里，父进程只做编排与比对 ——
        # `time.process_time()` 取不到子进程的 CPU。所以这里的 cpu_seconds 会很小，
        # 别读成"这次重演只花了 0.03 s CPU"；有意义的是 wall_seconds。
        "runtime_note": "cpu_seconds / peak_rss_bytes 只覆盖本编排进程；"
                        "train.py 子进程的 CPU 与内存见各自的 run_manifest.json::runtime",
        "note": "每个候选跑两次 train.py：两次逐字节相同，且与已交付的 CSV/manifest 一致",
    }
    (out / "replay_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["verdict"],
                      "candidates": {k: {"ok": v["ok"],
                                         "weight_hash": v.get("weight_hash")}
                                     for k, v in results.items()}}, ensure_ascii=False))
    return 0 if report["verdict"] == "PASS" else 6


if __name__ == "__main__":
    raise SystemExit(main())
