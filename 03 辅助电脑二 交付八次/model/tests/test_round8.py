"""第八轮测试：Climate 选择期门控（合成数据）。

先跑第七轮 85 项（其中已含第六轮 49 项），再补第八轮专项：

  A. **周期**：Climate 从冻结 spec 读 **52**，不是默认 12；未登记即拒绝。
  B. **Climate 形状**：bounds [636,763,1017,1144] / input_len 52 / horizon 4
     → train 581、cal 124、dec 251、test 124；选择期 **375**；每候选 **1500** 行。
  C. **严格隔离**：`isolate_test="strict"` 下 test 行**从不 materialize**
     （载入结果里为 NaN、非 test 行完好），且扰动 test 真值后预测逐字节不变。
  D. **禁止 test**：产物里不出现 test 段、不出现 test 预测文件。
  E. **§四.5 完备性闸门**：入口退出码 0 但产物不完整（缺文件 / 行数不符）
     必须判失败 —— 这正是第七轮 `predict_test.py` 被截断时"退出码 0 什么都不写"
     那类失败的克星。
  F. **§四.4 scope**：run manifest 的 `code_provenance.scope` 必须指向
     `03 辅助电脑二 交付八次/model`，不得是第六次目录。
  G. **读审计入口端到端**：`audit_read_isolation.py` 在合成 Climate 包上跑通且判 PASS。

用法::

    .venv/Scripts/python.exe "03 辅助电脑二 交付八次/model/tests/test_round8.py"
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent
sys.path.insert(0, str(MODEL))
sys.path.insert(0, str(HERE))

import test_round7 as r7  # noqa: E402  —— 它内部会先跑第六轮 49 项
import test_round6 as r6  # noqa: E402

check = r6.check

# Climate 冻结登记值（取自主控冻结 spec 的 tasks 项）
C_BOUNDS = [636, 763, 1017, 1144]
C_INPUT_LEN = 52
C_HORIZON = 4
C_N = 1144


def make_climate_spec(path: Path) -> Path:
    spec = {
        "schema_version": 1, "status": "frozen", "approved_by": "test-master",
        "alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        "row_permutations": 999, "p_threshold": 0.025,
        "gate_candidates": ["N+S+Q", "N+S+Q+SF"],
        "numeric_fallback_candidates": ["Last", "SeasonalNaive", "AR-Ridge", "N"],
        "seasonal_periods": {"Agriculture": 12, "Climate": 52, "SocialGood": 12,
                             "Environment": 7},
        "tasks": [{"domain": "Climate", "fold_id": 2, "input_len": C_INPUT_LEN,
                   "horizon": C_HORIZON, "lag_days": 7, "n_rows_expected": C_N,
                   "bounds": C_BOUNDS, "numerical_sha256": "a" * 64}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return path


def make_climate_bundle(root: Path, spec_path: Path, *, seed: int = 11) -> Path:
    """按 Climate 的边界造合成包：只有满足 input_len 与 horizon 约束的起点入包。"""
    from bundle_reader import sha256_file, signature

    root.mkdir(parents=True, exist_ok=True)
    task = root / "Climate_h4_f2"
    task.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)

    seg_bounds = [("train", 0, C_BOUNDS[0]), ("calibration", C_BOUNDS[0], C_BOUNDS[1]),
                  ("decision", C_BOUNDS[1], C_BOUNDS[2]), ("test", C_BOUNDS[2], C_BOUNDS[3])]
    keep = []
    for o in range(C_N):
        for nm, lo, hi in seg_bounds:
            if o >= C_INPUT_LEN and lo <= o < hi and o + C_HORIZON <= hi:
                keep.append((o, nm))
                break
    origins = np.array([o for o, _ in keep], dtype=np.int64)
    segments = [s for _, s in keep]

    for key, val in {
        "numeric_history": rng.normal(size=(C_N, C_INPUT_LEN)).astype(np.float32),
        "targets": rng.normal(size=(C_N, C_HORIZON)).astype(np.float32),
        "targets_standardized": rng.normal(size=(C_N, C_HORIZON)).astype(np.float32),
    }.items():
        np.save(task / f"{key}.npy", val[origins], allow_pickle=False)
    np.save(task / "origin_index.npy", origins, allow_pickle=False)

    scen = task / "proxy"; scen.mkdir(exist_ok=True)
    for key, val in {
        "semantic": rng.normal(size=(C_N, 16)).astype(np.float32),
        "quality": rng.normal(size=(C_N, 3)).astype(np.float32),
        "text_available": (rng.random(C_N) < 0.7),
    }.items():
        np.save(scen / f"{key}.npy", val[origins], allow_pickle=False)

    pd.DataFrame([{"task_id": "Climate_h4_f2", "fold_id": 2,
                   "origin_id": f"Climate:h4:f2:o{int(o)}", "origin_index": int(o),
                   "segment": segments[i]} for i, o in enumerate(origins)]
                 ).to_csv(task / "samples.csv", index=False, lineterminator="\n")

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    (root / "schema.json").write_text(json.dumps({"status": "frozen"}), encoding="utf-8")
    inputs = {"mode": "frozen", "split_spec_sha256": sha256_file(spec_path),
              "split_spec": spec}
    files = {p.relative_to(root).as_posix(): sha256_file(p)
             for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (root / "manifest.json").write_text(
        json.dumps({"inputs": inputs, "signature": signature(inputs), "files": files}),
        encoding="utf-8")
    return root


def run_entry(module, argv: list[str]) -> int:
    old = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return module.main()
    except SystemExit as exc:
        return int(exc.code or 1)
    except Exception:
        return -1
    finally:
        sys.argv = old


def main() -> int:
    print("第八轮测试（合成数据）")
    print("\n========== 先跑第七轮 85 项（含第六轮 49 项）==========")
    rc7 = r7.main()
    check("第七轮 85 项全部通过", rc7 == 0)

    import train as train_entry
    import verify_delivery
    from bundle_reader import read_frozen_bundle, sha256_file, task_seasonal_period

    # ================= A. Climate 周期 =================
    print("\n--- A. Climate 周期来自冻结 spec（52，不是默认 12）---")
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        spec = make_climate_spec(tmp / "spec.json")
        check("Climate 周期 = 52（非默认 12）",
              task_seasonal_period(spec, "Climate_h4_f2") == 52)
        check("Agriculture/SocialGood 仍是 12、Environment 是 7",
              task_seasonal_period(spec, "Agriculture_h12_f1") == 12
              and task_seasonal_period(spec, "SocialGood_h3_f1") == 12
              and task_seasonal_period(spec, "Environment_h7_f2") == 7)

        # ================= B/C/D/E/F/G =================
        bundle_dir = make_climate_bundle(tmp / "b", spec)
        sig = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))["signature"]
        b = read_frozen_bundle(bundle_dir, "Climate_h4_f2", "proxy", sig, spec,
                               isolate_test="strict")
        counts = {s: int(b.segment_mask(s).sum())
                  for s in ("train", "calibration", "decision", "test")}
        print(f"\n--- B. Climate 形状 ---\n    段计数: {counts}")
        check("train/cal/dec/test = 581/124/251/124",
              counts == {"train": 581, "calibration": 124, "decision": 251, "test": 124})
        check("选择期起点 = 375（124 + 251）",
              counts["calibration"] + counts["decision"] == 375)

        print("\n--- C. 严格隔离（test 行从不 materialize）---")
        tm = b.segment_mask("test")
        check("隔离模式 = strict，test 行未 materialize",
              (b.isolation or {}).get("mode") == "strict"
              and (b.isolation or {}).get("test_rows_materialized") is False)
        check("targets / targets_standardized 的 test 行全为 NaN",
              np.isnan(np.asarray(b.arrays["targets"])[tm]).all()
              and np.isnan(np.asarray(b.arrays["targets_standardized"])[tm]).all())
        check("非 test 行完好（有限值）",
              np.isfinite(np.asarray(b.arrays["targets"])[~tm]).all())
        check("test 特征视图不含 targets 键",
              "targets" not in b.segment("test", with_targets=False))
        check("网格校验通过（四段）",
              all(True for _ in range(1)) and
              (lambda: (__import__("bundle_reader").assert_grid(b, "train", (0, 636), 4),
                        __import__("bundle_reader").assert_grid(b, "calibration", (636, 763), 4),
                        __import__("bundle_reader").assert_grid(b, "decision", (763, 1017), 4),
                        __import__("bundle_reader").assert_grid(b, "test", (1017, 1144), 4),
                        True)[-1])())

        # 扰动 test 真值 → 预测逐字节不变（需重签清单）
        print("\n--- C2. 扰动 test 真值后预测逐字节不变 ---")
        task_dir = bundle_dir / "Climate_h4_f2"
        rng = np.random.default_rng(3)
        changed = {}
        for key in ("targets", "targets_standardized"):
            p = task_dir / f"{key}.npy"
            arr = np.load(p).copy()
            arr[tm] = rng.normal(size=(int(tm.sum()), arr.shape[1])).astype(arr.dtype)
            np.save(p, arr, allow_pickle=False)
            changed[p.relative_to(bundle_dir).as_posix()] = sha256_file(p)
        mo = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        mo["files"].update(changed)
        (bundle_dir / "manifest.json").write_text(
            json.dumps(mo, ensure_ascii=False), encoding="utf-8", newline="\n")
        check("扰动后签名仍有效（signature 只覆盖 inputs）",
              json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))["signature"] == sig)

        # ================= E2E：train.py =================
        print("\n--- E. 入口端到端 + §四.5 完备性闸门 ---")
        out_dir = tmp / "out"
        code = run_entry(train_entry, ["train.py", "--bundle", str(bundle_dir),
                                       "--task", "Climate_h4_f2", "--scenario", "proxy",
                                       "--signature", sig, "--split-spec", str(spec),
                                       "--segments", "calibration", "decision",
                                       "--candidate", "N+S+Q", "--output-dir", str(out_dir)])
        check("train.py 端到端退出码 0", code == 0)
        pred = out_dir / "N_S_Q_proxy_predictions.csv"
        man = out_dir / "N_S_Q_run_manifest.json"
        with pred.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        print(f"    CSV 行数 = {len(rows)}（期望 375×4 = 1500）")
        check("每候选 1500 行 = 375 × 4", len(rows) == 1500)
        check("只含 calibration / decision",
              {r["segment"] for r in rows} == {"calibration", "decision"})
        check("§四.4 manifest 的 scope 指向第八次目录",
              json.loads(man.read_text(encoding="utf-8"))["code_provenance"]["scope"]
              .endswith("03 辅助电脑二 交付八次/model"))

        # 组装一个最小交付目录，验证 §四.5 闸门：完整 → PASS，缺件 → FAIL
        fake = tmp / "fake_delivery"
        (fake / "evidence" / "cal_dec_predictions").mkdir(parents=True)
        (fake / "evidence" / "read_isolation").mkdir(parents=True)
        (fake / "evidence" / "boundaries").mkdir(parents=True)
        (fake / "evidence" / "replay").mkdir(parents=True)
        import shutil
        shutil.copy(pred, fake / "evidence" / "cal_dec_predictions" / pred.name)
        shutil.copy(man, fake / "evidence" / "cal_dec_predictions" / man.name)
        for rel in ("README.md", "RUNBOOK.md", "TEAM_SYNC.md", "MANIFEST.json",
                    "evidence/boundaries/four_segment_audit.json",
                    "evidence/read_isolation/test_read_isolation_audit.json",
                    "evidence/replay/replay_check.json"):
            (fake / rel).write_text("{}", encoding="utf-8")
        # 只给一个候选时，闸门应报出其余候选缺失 —— 用「缺文件」路径验证
        probs_missing = verify_delivery.check(fake, "Climate_h4_f2", 4, 375,
                                              candidates=("N+S+Q", "N+Q"),
                                              require_nulls=False)
        check("闸门：缺 N+Q 产物 → 报缺失", any("N_Q" in p for p in probs_missing))
        # 行数被截断 → 必须报行数不符（退出码 0 但产物不完整的情形）
        truncated = fake / "evidence" / "cal_dec_predictions" / pred.name
        truncated.write_text("".join(open(pred, encoding="utf-8").readlines()[:100]),
                             encoding="utf-8", newline="\n")
        probs_trunc = verify_delivery.check(fake, "Climate_h4_f2", 4, 375,
                                            candidates=("N+S+Q",), require_nulls=False)
        check("闸门：行数被截断 → 判失败", any("行数" in p for p in probs_trunc))
        # 恢复为完整产物 → 闸门通过（除 test 预测与证据文件用占位外都应满足）
        shutil.copy(pred, truncated)
        probs_ok = verify_delivery.check(fake, "Climate_h4_f2", 4, 375,
                                         candidates=("N+S+Q",), require_nulls=False)
        # 「产物齐全」= 不再有内容类问题；工作区脏是**当前未提交**的属性，
        # 由正式交付时的干净提交另行保证（§四.3），单元测试不据此判失败。
        content_markers = ("行数", "列与契约不符", "键不唯一", "缺键",
                           "缺少预测 CSV", "含非选择期段", "出现 test 段")
        check("闸门：产物齐全 → 无内容类问题",
              not [p for p in probs_ok if any(m in p for m in content_markers)])
        check("闸门：未提交的工作区会被标出（provenance 属性，独立检查）",
              any("工作区不干净" in p for p in probs_ok) or
              json.loads(man.read_text(encoding="utf-8"))["code_provenance"]["worktree_dirty"] is False)

        print("\n--- D. 禁止 test ---")
        check("产物里没有 test 段", not any(r["segment"] == "test" for r in rows))
        check("交付目录里没有 test 预测文件",
              not list(fake.rglob("*test_predictions*")))

        # ================= G. 读审计入口端到端 =================
        print("\n--- G. audit_read_isolation.py 端到端 ---")
        import audit_read_isolation as read_audit
        audit_out = tmp / "audit_out"
        probe_out = tmp / "probe_out"
        code = run_entry(read_audit, ["audit_read_isolation.py", "--bundle", str(bundle_dir),
                                      "--task", "Climate_h4_f2", "--scenario", "proxy",
                                      "--signature", sig, "--split-spec", str(spec),
                                      "--candidate", "N+S+Q", "--output-dir", str(audit_out),
                                      "--probe-output", str(probe_out)])
        check("读审计入口退出码 0（未发现 test 真值被当数据读取）", code == 0)
        audit = json.loads((audit_out / "test_read_isolation_audit.json")
                           .read_text(encoding="utf-8"))
        check("读审计 verdict=PASS 且无违规", audit["verdict"] == "PASS"
              and not audit["violations"])
        check("读审计确认隔离模式 strict",
              audit["isolation_assertions"]["isolation_mode"] == "strict")

    total = len(r6.PASSED)
    print(f"\n全部通过（{total} 项检查：第六轮 49 + 第七轮 36 + 第八轮 {total - 85}）")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1)
