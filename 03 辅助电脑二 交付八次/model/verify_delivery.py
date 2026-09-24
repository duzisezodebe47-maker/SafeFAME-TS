"""交付完备性闸门（第八轮任务书 §四.5）。

> 测试入口即使退出码为 0 但未生成完整产物，也必须判失败。

为什么需要单独一个闸门：第七轮之前发现 `predict_test.py` 被截断时，它的表现正是
「**退出码 0，什么都不写**」—— 入口不报错，调用方以为成功。只检查退出码的流程
对这种失败完全免疫。所以本脚本按**产物**判成败，而不是按退出码：

  - 期望的候选 CSV 是否存在、行数是否恰为 `起点数 × horizon`；
  - 每个 run manifest 是否存在，且含必需键（候选、四锚、α、权重哈希、行数、列名、
    运行时间、峰值内存、代码出处）；
  - 每个门控候选是否恰有 999 次成功置换、逐次 CSV 列序正确、种子规则正确、
    摘要里的 p 与计数自洽；
  - 是否**混入 test 预测**（文件名含 test_predictions，或任何 CSV 出现 test 段）；
  - `code_provenance.scope` 是否指向**本次**交付目录（第八轮 §四.4），且工作区干净。

任一项不满足 → 非 0 退出并列出具体缺口。退出码 0 = 产物齐全，可以被交付。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from candidates import ALL_CANDIDATES, GATE_CANDIDATES  # noqa: E402
from permutation import ROW_PERMUTATIONS  # noqa: E402
from predict_io import CONTRACT_FIELDS  # noqa: E402

REQUIRED_MANIFEST_KEYS = (
    "task", "scenario", "candidate", "config_sha256", "code_commit",
    "bundle_signature", "alpha_by_group", "weight_hash", "rows_written",
    "wall_seconds", "split_spec_sha256", "grid", "code_provenance", "runtime",
    "seasonal_period", "test_isolation",
)


def _problems_manifest(path: Path) -> list[str]:
    out: list[str] = []
    if not path.is_file():
        return [f"缺少 manifest: {path.name}"]
    doc = json.loads(path.read_text(encoding="utf-8"))
    for key in REQUIRED_MANIFEST_KEYS:
        if key not in doc:
            out.append(f"{path.name} 缺键 {key}")
    prov = doc.get("code_provenance") or {}
    if prov.get("worktree_dirty") is not False:
        out.append(f"{path.name} 工作区不干净: {prov.get('worktree_dirty')}")
    scope = str(prov.get("scope") or "")
    if not scope.endswith("03 辅助电脑二 交付八次/model"):
        out.append(f"{path.name} scope 不是第八次目录: {scope!r}")
    for key in ("wall_seconds", "runtime"):
        if key in doc and doc[key] in (None, ""):
            out.append(f"{path.name} {key} 为空")
    if "runtime" in doc and (doc["runtime"].get("peak_rss_bytes") or 0) <= 0:
        out.append(f"{path.name} 缺峰值内存实测")
    return out


def check(delivery: Path, task: str, horizon: int, expected_origins: int,
          candidates: tuple[str, ...] = ALL_CANDIDATES,
          require_nulls: bool = True) -> list[str]:
    """返回问题列表（空 = 完整）。

    `require_nulls=False` 供**单元测试**在最小合成交付上只验选择期产物；
    正式交付必须用默认 True（§三 要求两门控候选各有完整 999 次零分布）。
    """
    problems: list[str] = []
    pred_dir = delivery / "evidence" / "cal_dec_predictions"

    for candidate in candidates:
        stem = candidate.replace("+", "_")
        csv_path = pred_dir / f"{stem}_proxy_predictions.csv"
        man_path = pred_dir / f"{stem}_run_manifest.json"
        if not csv_path.is_file():
            problems.append(f"缺少预测 CSV: {csv_path.name}")
        else:
            with csv_path.open(encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                cols = reader.fieldnames
                rows = list(reader)
            if list(cols or []) != list(CONTRACT_FIELDS):
                problems.append(f"{csv_path.name} 列与契约不符: {cols}")
            want = expected_origins * horizon
            if len(rows) != want:
                problems.append(f"{csv_path.name} 行数 {len(rows)} != {want}")
            keys = {(r["origin_id"], r["step"]) for r in rows}
            if len(keys) != len(rows):
                problems.append(f"{csv_path.name} 键不唯一")
            segs = {r["segment"] for r in rows}
            if segs - {"calibration", "decision"}:
                problems.append(f"{csv_path.name} 含非选择期段: {segs}")
            if any(r["segment"] == "test" for r in rows):
                problems.append(f"{csv_path.name} 出现 test 段（本轮禁止）")
        problems += _problems_manifest(man_path)

    for candidate in (GATE_CANDIDATES if require_nulls else ()):
        cand_dir = delivery / "evidence" / "row_null" / candidate
        for name, want_rows in (("null_scores.csv", ROW_PERMUTATIONS),
                                ("null_scores_circular.csv", ROW_PERMUTATIONS)):
            f = cand_dir / name
            if not f.is_file():
                problems.append(f"缺少 {candidate}/{name}")
                continue
            with f.open(encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames != ["iteration", "seed", "status", "loss", "error"]:
                    problems.append(f"{candidate}/{name} 列序不符: {reader.fieldnames}")
                rows = list(reader)
            if len(rows) != want_rows:
                problems.append(f"{candidate}/{name} 行数 {len(rows)} != {want_rows}")
            ok = [r for r in rows if r["status"] == "ok"]
            for i, r in enumerate(rows):
                if int(r["iteration"]) != i or int(r["seed"]) != 2026000 + i:
                    problems.append(f"{candidate}/{name} 第 {i} 行 iteration/seed 错配")
                    break
            if len(ok) != want_rows:
                problems.append(f"{candidate}/{name} 成功数 {len(ok)} != {want_rows}")
        summary = cand_dir / "permutation_summary.json"
        if not summary.is_file():
            problems.append(f"缺少 {candidate}/permutation_summary.json")
        else:
            doc = json.loads(summary.read_text(encoding="utf-8"))
            if doc.get("successful") != ROW_PERMUTATIONS or doc.get("failed") != 0:
                problems.append(f"{candidate} 置换计数不符: "
                                f"{doc.get('successful')}/{doc.get('failed')}")
            if doc.get("p_value") is None:
                problems.append(f"{candidate} p_value 为空（未达 999 不得报显著性）")
            if "decision_halves" not in doc:
                problems.append(f"{candidate} 缺 decision 前后半段")
            if not (doc.get("permutation_config") or {}).get("refit_each_iteration"):
                problems.append(f"{candidate} 未声明逐次重拟合")

    # 本轮禁止 test 预测
    for f in delivery.rglob("*test_predictions*"):
        problems.append(f"出现 test 预测文件（本轮禁止）: {f.relative_to(delivery)}")

    for rel in ("README.md", "RUNBOOK.md", "TEAM_SYNC.md", "MANIFEST.json",
                "evidence/boundaries/four_segment_audit.json",
                "evidence/read_isolation/test_read_isolation_audit.json",
                "evidence/replay/replay_check.json"):
        if not (delivery / rel).is_file():
            problems.append(f"缺少交付物: {rel}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--expected-origins", type=int, required=True)
    args = parser.parse_args()

    problems = check(args.delivery.resolve(), args.task, args.horizon,
                     args.expected_origins)
    if problems:
        print(json.dumps({"status": "INCOMPLETE", "problems": problems},
                         ensure_ascii=False, indent=2))
        return 6
    print(json.dumps({"status": "COMPLETE", "task": args.task,
                      "expected_origins": args.expected_origins,
                      "expected_rows_per_candidate": args.expected_origins * args.horizon},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
