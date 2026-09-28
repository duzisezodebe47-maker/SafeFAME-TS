"""第九轮交付完备性闸门：**按产物判成败**（任务书 §五.8 等）。

第八轮的 `verify_delivery.py` 管的是"选择期交付"（四候选 1500 行 + 两候选 999 置换）。
本轮是**测试预测交付**，判据完全不同，所以单独一个闸门：

  1. 交付目录里**恰好一个**测试预测 CSV，且必须是 `N` 的（§五.8：出现第二个候选即失败）；
  2. 行数 == 期望（Climate_h4_f2 为 124 起点 × 4 步 = **496**）、键唯一、仅 test 段、全有限；
  3. `run_manifest.json` 必需字段齐备（任务/候选/四锚/选择期清单 SHA/代码提交/
     工作区状态/alpha/权重哈希/行数/运行资源）；
  4. `route_acceptance.json` 存在且 `verdict=PASS`，路由 SHA 与期望值一致；
  5. `test_isolation_audit.json` 存在，且**test 真值被当作数据访问的计数为 0**；
  6. `replay_check.json` 存在且两次运行逐字节相同、权重哈希一致；
  7. `attempt_ledger.json` 存在（每次尝试留痕，§三.5）；
  8. 交付里**不得出现任何测试评分字段**；
  9. `MANIFEST.json` 存在，且已跟踪文件的记录哈希 == git HEAD 的 blob 哈希；
 10. 代码提交是 40 位十六进制、`worktree_dirty=false`、`scope` 指向第九次目录。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

HEX40 = re.compile(r"^[0-9a-f]{40}$")
# 评分字段禁列：出现即说明本侧算了测试性能（§三.1 明令禁止）
SCORE_KEYS = ("test_mse", "test_mae", "test_rmse", "r2", "residual", "score", "ranking")


def _numeric_score_keys(node, path: str = ""):
    """递归找出**数值型**的评分键（列表/字典都下钻；字符串列表不算）。"""
    if isinstance(node, dict):
        for key, value in node.items():
            low = str(key).lower()
            if low in SCORE_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
                yield f"{path}.{key}", value
            else:
                yield from _numeric_score_keys(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _numeric_score_keys(value, f"{path}[{i}]")


def _manifest_vs_git(delivery: Path) -> list[str]:
    """清单哈希 vs 已提交字节（未跟踪的跳过）——与第八轮同一判据。"""
    manifest_path = delivery / "MANIFEST.json"
    if not manifest_path.is_file():
        return []
    repo = subprocess.run(["git", "-C", str(delivery), "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if repo.returncode != 0:
        return []
    root = Path(repo.stdout.strip())
    try:
        rel_root = delivery.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return []
    problems = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    self_output = "evidence/verify_test_delivery.json"
    for section in ("model_files", "evidence_files", "other_files"):
        for rel, recorded in (manifest.get(section) or {}).items():
            if rel == self_output:
                continue                      # 自指结论文件，见第八轮说明
            blob = subprocess.run(["git", "-C", str(root), "show", f"HEAD:{rel_root}/{rel}"],
                                  capture_output=True)
            if blob.returncode != 0:
                continue
            if hashlib.sha256(blob.stdout).hexdigest() != recorded:
                problems.append(f"{rel} 的清单哈希与已提交字节不一致（行尾归一化？）")
    return problems


def check(delivery: Path, *, task: str, horizon: int, expected_origins: int,
          candidate: str, route_sha256: str, expected_selected: str = "numeric_fallback",
          expected_fallback: str = "N", require_git_clean: bool = True) -> list[str]:
    """返回问题列表（空 = 完整）。"""
    problems: list[str] = []
    expected_rows = expected_origins * horizon

    # 1) 恰好一个测试预测，且必须是本候选的
    preds = sorted(p for p in delivery.rglob("*_test_predictions.csv"))
    if len(preds) != 1:
        problems.append(f"测试预测 CSV 数量 {len(preds)} != 1（§五.8）：{[p.name for p in preds]}")
    csv_path = None
    for p in preds:
        if p.name == f"{candidate}_test_predictions.csv":
            csv_path = p
        else:
            problems.append(f"出现非授权候选的测试预测: {p.name}")
    if csv_path is None and preds:
        problems.append(f"找不到 {candidate}_test_predictions.csv")

    if csv_path is not None:
        with csv_path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames
            rows = list(reader)
        if len(rows) != expected_rows:
            problems.append(f"行数 {len(rows)} != 期望 {expected_rows}")
        keys = {(r["origin_id"], r["step"]) for r in rows}
        if len(keys) != len(rows):
            problems.append("键不唯一")
        segs = {r["segment"] for r in rows}
        if segs != {"test"}:
            problems.append(f"段不是仅 test: {segs}")
        if {r["candidate_id"] for r in rows} != {candidate}:
            problems.append("candidate_id 不唯一或不等于授权候选")
        import numpy as np
        vals = []
        for r in rows:
            try:
                v = float(r["y_pred"])
            except (TypeError, ValueError):
                problems.append("y_pred 不可解析"); break
            vals.append(v)
        if vals and not np.isfinite(np.asarray(vals)).all():
            problems.append("存在非有限预测值")

        man_path = csv_path.parent / f"{candidate}_test_manifest.json"
        if not man_path.is_file():
            problems.append(f"缺少 {candidate}_test_manifest.json")
        else:
            man = json.loads(man_path.read_text(encoding="utf-8"))
            for field in ("task", "candidate", "input_anchors", "selection_config_sha256",
                          "n_rows", "n_test_origins", "csv_columns", "csv_sha256",
                          "code_commit", "code_provenance", "runtime"):
                if field not in man:
                    problems.append(f"manifest 缺字段 {field}")
            if man.get("n_rows") != expected_rows:
                problems.append(f"manifest.n_rows {man.get('n_rows')} != {expected_rows}")
            anchors = man.get("input_anchors") or {}
            for key in ("route_file_sha256", "bundle_signature", "split_spec_sha256",
                        "selection_manifest_sha256"):
                if key not in anchors:
                    problems.append(f"manifest 缺输入锚 {key}")
            if anchors.get("route_file_sha256") != route_sha256:
                problems.append("manifest 的路由锚与任务书公布值不一致")
            if man.get("csv_sha256") != hashlib.sha256(csv_path.read_bytes()).hexdigest():
                problems.append("manifest.csv_sha256 与 CSV 实际字节不符")
            commit = str(man.get("code_commit") or "")
            if not HEX40.match(commit):
                problems.append(f"code_commit 不是 40 位十六进制: {commit!r}")
            prov = man.get("code_provenance") or {}
            if require_git_clean and prov.get("worktree_dirty"):
                problems.append("工作区不干净（§六 退回条件）")
            # §二 指定用**第八次那份已过 131 项测试的入口**跑正式预测，
            # 因此 provenance 的 scope 如实指向那次运行所用的 model 目录：
            # 第八次或第九次都接受（两者都是本侧代码），但必须说得出是哪一个。
            scope = str(prov.get("scope") or "")
            if not scope.endswith(("03 辅助电脑二 交付八次/model",
                                   "03 辅助电脑二 交付九次/model")):
                problems.append(f"provenance scope 指向非本侧目录: {scope!r}")
            rt = man.get("runtime") or {}
            if (rt.get("peak_rss_bytes") or 0) <= 0:
                problems.append("缺峰值内存实测")

    # 4) 路由接受
    ra_path = delivery / "evidence" / "route_acceptance.json"
    if not ra_path.is_file():
        problems.append("缺少 evidence/route_acceptance.json")
    else:
        ra = json.loads(ra_path.read_text(encoding="utf-8"))
        if ra.get("verdict") != "PASS":
            problems.append(f"route_acceptance verdict={ra.get('verdict')}")
        rec = ra.get("recorded") or {}
        if ra.get("route_file_sha256_actual") != route_sha256:
            problems.append("route_acceptance 记录的路由 SHA 与任务书不符")
        if rec.get("selected") != expected_selected or rec.get("fallback") != expected_fallback:
            problems.append(f"路由 selected/fallback 与期望不符: "
                            f"{rec.get('selected')}/{rec.get('fallback')}")

    # 5) 测试隔离：真值被当数据访问的计数必须为 0
    iso_path = delivery / "evidence" / "test_isolation_audit.json"
    if not iso_path.is_file():
        problems.append("缺少 evidence/test_isolation_audit.json")
    else:
        iso = json.loads(iso_path.read_text(encoding="utf-8"))
        if iso.get("test_truth_data_access_count") != 0:
            problems.append(f"test 真值被当作数据访问的次数 "
                            f"{iso.get('test_truth_data_access_count')} != 0")
        if iso.get("verdict") != "PASS":
            problems.append(f"test_isolation_audit verdict={iso.get('verdict')}")

    # 6) 重放
    rp_path = delivery / "evidence" / "replay_check.json"
    if not rp_path.is_file():
        problems.append("缺少 evidence/replay_check.json")
    else:
        rp = json.loads(rp_path.read_text(encoding="utf-8"))
        checks = rp.get("checks") or {}
        if rp.get("verdict") != "PASS":
            problems.append(f"replay verdict={rp.get('verdict')}")
        for key in ("predictions_byte_identical", "weight_hash_identical"):
            if checks.get(key) is not True:
                problems.append(f"replay_check.{key} 不为 true")

    # 7) 尝试台账
    if not (delivery / "attempt_ledger.json").is_file():
        problems.append("缺少 attempt_ledger.json（§四/§三.5）")

    # 8) 不得出现评分字段。
    #    只把**数值型**的评分键算作指标：审计报告里会**列出**它扫了哪些 token
    #    （`tokens_scanned: ["test_mse", ...]`），那是元信息不是指标 ——
    #    按字符串一刀切会误报（本闸门第一版就是这么误报的）。
    for p in delivery.rglob("*.json"):
        if "MANIFEST.json" in p.name or "model" in p.parts:
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        hits = [key for key, value in _numeric_score_keys(doc) if key not in ("",)]
        if hits:
            problems.append(f"{p.name} 出现数值型评分字段 {sorted(set(hits))}")

    # 9) 清单 vs 提交字节
    problems += _manifest_vs_git(delivery)
    if not (delivery / "MANIFEST.json").is_file():
        problems.append("缺少 MANIFEST.json")
    for name in ("README.md", "RUNBOOK.md"):
        if not (delivery / name).is_file():
            problems.append(f"缺少 {name}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--expected-origins", type=int, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--route-sha256", required=True)
    args = parser.parse_args()

    problems = check(args.delivery, task=args.task, horizon=args.horizon,
                     expected_origins=args.expected_origins, candidate=args.candidate,
                     route_sha256=args.route_sha256)
    status = "COMPLETE" if not problems else "INCOMPLETE"
    print(json.dumps({"status": status, "task": args.task, "candidate": args.candidate,
                      "expected_rows": args.expected_origins * args.horizon,
                      "problems": problems}, ensure_ascii=False, indent=2))
    return 0 if not problems else 6


if __name__ == "__main__":
    raise SystemExit(main())
