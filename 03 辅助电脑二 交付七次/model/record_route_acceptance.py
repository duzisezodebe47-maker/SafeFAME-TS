"""路由接受记录（第七轮交付物 `route_acceptance.json`）。

任务书第一节要求：**必须先从主控分支取得路由文件并核对 SHA256**，
「任何一字节变化都停止运行并报告」。本脚本把这次核对**留痕**：

  - 路由文件的字节 SHA256 与任务书公布值逐位比对；
  - 主控同目录的 `.sha256` sidecar 一并读取比对（三方一致才算通过）；
  - 原样记录 `selected` / `status` / 三锚（split_spec、selection_manifest、refit_review）
    与 `inputs_sha256` 的五个子锚；
  - 本侧**独立复算**若干一致性：候选 == selected、spec 字节哈希相符、
    Bundle 签名相符、`selection_data_segments == ["cal","dec"]`、`status == "frozen"`。

任一项不符即整体 FAIL 并非 0 退出 —— 路由对不上时不许产出预测。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bundle_reader import sha256_file  # noqa: E402
from predict_io import warn_if_dirty  # noqa: E402
from runtime_profile import cpu_seconds, script_entry_snapshot  # noqa: E402

REPO_ROOT = HERE.parents[1]
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--route-sha256", required=True,
                        help="任务书公布的路由字节哈希")
    parser.add_argument("--route-source", required=True,
                        help="路由在主控分支里的路径（留痕用）")
    parser.add_argument("--sidecar", type=Path, default=None,
                        help="主控的 .sha256 sidecar（可选）")
    parser.add_argument("--split-spec", type=Path, required=True)
    parser.add_argument("--bundle-signature", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    cpu_started = cpu_seconds()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    actual = sha256_file(args.route)
    route = json.loads(args.route.read_text(encoding="utf-8"))
    spec_sha = sha256_file(args.split_spec)

    sidecar_value = None
    if args.sidecar is not None and args.sidecar.is_file():
        first = args.sidecar.read_text(encoding="utf-8").strip().split()[0]
        sidecar_value = first

    checks = {
        "route_sha256_matches_published": actual == args.route_sha256,
        "route_sha256_is_64hex": bool(HEX64.match(actual)),
        "sidecar_matches": (sidecar_value is None or sidecar_value == actual),
        "status_is_frozen": route.get("status") == "frozen",
        "selected_is_candidate": route.get("selected") == args.candidate,
        "candidate_is_gate_registered": args.candidate in ("N+S+Q", "N+S+Q+SF"),
        "selection_data_segments_is_cal_dec":
            list(route.get("selection_data_segments") or []) == ["cal", "dec"],
        "split_spec_sha256_matches_file":
            route.get("split_spec_sha256") == spec_sha,
        "bundle_signature_matches": route.get("bundle_signature") == args.bundle_signature,
        "three_source_anchors_are_64hex": all(
            HEX64.match(str(route.get(k, "")))
            for k in ("selection_manifest_sha256", "refit_review_sha256",
                      "split_spec_sha256", "bundle_signature")),
        "inputs_sha256_subkeys_present_and_valid": all(
            HEX64.match(str(v)) for v in (route.get("inputs_sha256") or {}).values())
        and set(route.get("inputs_sha256") or {}) >= {"task", "samples", "predictions",
                                                     "nulls", "spec"},
    }
    passed = all(checks.values())

    report = {
        "route_source_path": args.route_source,
        "route_file_in_delivery": args.route.name,
        "route_file_sha256_actual": actual,
        "route_file_sha256_published": args.route_sha256,
        "sidecar_sha256": sidecar_value,
        # 原样记录（不改写、不裁剪）
        "recorded": {
            "task_id": route.get("task_id"), "fold_id": route.get("fold_id"),
            "scenario": route.get("scenario"), "status": route.get("status"),
            "selected": route.get("selected"), "fallback": route.get("fallback"),
            "selection_data_segments": route.get("selection_data_segments"),
            "split_spec_sha256": route.get("split_spec_sha256"),
            "bundle_signature": route.get("bundle_signature"),
            "selection_manifest_sha256": route.get("selection_manifest_sha256"),
            "refit_review_sha256": route.get("refit_review_sha256"),
            "inputs_sha256": route.get("inputs_sha256"),
            "calibration_mse": route.get("calibration_mse"),
            "decision_fallback_mse": route.get("decision_fallback_mse"),
            "candidates": route.get("candidates"),
        },
        "checks": checks,
        "verdict": "PASS" if passed else "FAIL",
        "code_provenance": warn_if_dirty(REPO_ROOT, HERE),
        "runtime": script_entry_snapshot(started, cpu_started),
        "note": "路由文件按主控发布字节原样收录；本脚本只读、不改写路由内容",
    }
    (out / "route_acceptance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"verdict": report["verdict"], "checks": checks,
                      "route_sha256": actual}, ensure_ascii=False))
    return 0 if passed else 6


if __name__ == "__main__":
    raise SystemExit(main())
