"""第七轮测试：封印路由后一次性测试预测入口（合成数据）。

先跑第六轮 49 项（原样保留），再补任务书 §五 要求的 6 类负例：

  1. 路由 SHA 错误 → 拒绝
  2. 候选不是路由 selected 的 `N+S+Q+SF`（本夹具里是 `N+S+Q`）→ 拒绝
  3. 选择期 manifest 的 `alpha_by_group` 或 `weight_hash` 被改 → 拒绝
  4. test 真值被访问/可用 → 拒绝（隔离模式下 test 行为 NaN；
     且**扰动 test 真值后预测逐字节不变** —— 证明真值无因果影响）
  5. 输出目录已有文件 → 拒绝覆盖
  6. 行数不符 / 键重复 / origin 越界 / 非有限预测 → 拒绝

用法::

    .venv/Scripts/python.exe "03 辅助电脑二 交付六次/model/tests/test_round7.py"
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

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent
sys.path.insert(0, str(MODEL))
sys.path.insert(0, str(HERE))

import test_round6 as r6  # noqa: E402  —— 复用其夹具与 check/PASSED

check = r6.check


def run_entry(module, argv: list[str]) -> int:
    """与第六轮同款：把 argv 临时换掉后调用 module.main()。"""
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


def write_route(path: Path, route: dict) -> str:
    """写出路由文件并返回其**字节**哈希（--route-sha256 的锚）。"""
    from bundle_reader import sha256_file

    path.write_text(json.dumps(route, ensure_ascii=False), encoding="utf-8",
                    newline="\n")
    return sha256_file(path)


def base_argv(bundle: Path, spec: Path, sig: str, route: Path, route_sha: str,
              out: Path, candidate: str, selection: Path | None) -> list[str]:
    argv = ["predict_test.py", "--bundle", str(bundle), "--task", "Agriculture_h3_f1",
            "--scenario", "proxy", "--signature", sig, "--split-spec", str(spec),
            "--route", str(route), "--route-sha256", route_sha,
            "--candidate", candidate, "--seed", "2026", "--output-dir", str(out)]
    if selection is not None:
        argv += ["--selection-manifest", str(selection)]
    return argv


def main() -> int:
    print("第七轮测试（合成数据）")
    print("\n========== 先跑第六轮 49 项 ==========")
    rc6 = r6.main()
    check("第六轮 49 项全部通过", rc6 == 0)

    import predict_test
    from bundle_reader import read_frozen_bundle, sha256_file
    from predict_io import CONTRACT_FIELDS

    print("\n========== 第七轮：测试预测入口端到端 ==========")
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        spec = r6.make_spec(tmp / "spec.json")
        bundle_dir = r6.make_bundle(tmp / "b", spec)
        spec_sha = sha256_file(spec)
        sig = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))["signature"]

        # 选择期 manifest：跑 train.py 生成（含 weight_hash / alpha_by_group）
        import train as train_entry
        sel_dir = tmp / "sel"
        code = run_entry(train_entry, ["train.py", "--bundle", str(bundle_dir),
                                       "--task", "Agriculture_h3_f1", "--scenario", "proxy",
                                       "--signature", sig, "--split-spec", str(spec),
                                       "--segments", "calibration", "decision",
                                       "--candidate", "N+S+Q", "--output-dir", str(sel_dir)])
        check("选择期 train.py 端到端成功", code == 0)
        selection = sel_dir / "N_S_Q_run_manifest.json"
        check("选择期 manifest 含 weight_hash 与 alpha_by_group",
              all(k in json.loads(selection.read_text(encoding="utf-8"))
                  for k in ("weight_hash", "alpha_by_group")))

        route_obj = r6.sealed_route(spec, spec_sha, sig)
        route = tmp / "route.json"
        route_sha = write_route(route, route_obj)

        # ---- 正路：应成功 ----
        out_ok = tmp / "out_ok"
        code = run_entry(predict_test, base_argv(bundle_dir, spec, sig, route, route_sha,
                                                 out_ok, "N+S+Q", selection))
        check("封印路由 + 正确候选 → 端到端成功", code == 0)
        pred_csv = out_ok / "N_S_Q_test_predictions.csv"
        check("预测 CSV 已写出", pred_csv.is_file())
        with pred_csv.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        check("CSV 列 == 主控 PREDICTION_COLUMNS", set(rows[0]) == set(CONTRACT_FIELDS))
        check("只有 test 段", {r["segment"] for r in rows} == {"test"})
        # 合成夹具：test bounds [45,60)、H=3、input_len=6 → 起点 45..57 共 13 个
        origins = {r["origin_id"] for r in rows}
        check("每起点每步恰一行（13 起点 × 3 步）",
              len(rows) == 39 and len(origins) == 13)
        check("预测全为有限值", all(np.isfinite(float(r["y_pred"])) for r in rows))

        man = json.loads((out_ok / "N_S_Q_test_manifest.json").read_text(encoding="utf-8"))
        check("manifest 含四个输入锚 + 选择期 config 哈希",
              all(k in man["input_anchors"] for k in
                  ("route_file_sha256", "bundle_signature", "split_spec_sha256",
                   "selection_manifest_sha256", "selection_config_sha256"))
              and man["input_anchors"]["route_file_sha256"] == route_sha
              and man["input_anchors"]["bundle_signature"] == sig
              and man["input_anchors"]["split_spec_sha256"] == spec_sha)
        check("manifest 含列名 / 运行时间 / 峰值内存 / 代码出处",
              man["csv_columns"] == list(CONTRACT_FIELDS)
              and man["runtime"]["peak_rss_bytes"] is not None
              and len(man["code_provenance"]["code_commit"]) == 40
              and man["test_isolation"]["isolate_test"] is True)
        check("manifest 不含任何 test 误差字段",
              not any(k for k in man if any(t in k.lower()
                                            for t in ("mse", "mae", "r2", "residual", "score"))
                      and k not in ("ridge_calibration_mse",)))
        body_ok = pred_csv.read_bytes()

        # ---- 负例 5：输出目录非空 → 拒绝覆盖 ----
        code = run_entry(predict_test, base_argv(bundle_dir, spec, sig, route, route_sha,
                                                 out_ok, "N+S+Q", selection))
        check("输出目录非空 → 拒绝覆盖", code != 0)

        # ---- 负例 1：路由 SHA 错 → 拒绝 ----
        out_bad = tmp / "out_bad_sha"
        code = run_entry(predict_test, base_argv(bundle_dir, spec, sig, route, "0" * 64,
                                                 out_bad, "N+S+Q", selection))
        check("路由 SHA 不符 → 拒绝", code != 0 and not out_bad.exists())

        # ---- 负例 2：候选与路由 selected 不符 → 拒绝 ----
        out_bad2 = tmp / "out_bad_cand"
        code = run_entry(predict_test, base_argv(bundle_dir, spec, sig, route, route_sha,
                                                 out_bad2, "N+S+Q+SF", selection))
        check("候选不是路由 selected → 拒绝", code != 0)

        # ---- 负例 3：选择期 manifest 被篡改 → 拒绝 ----
        original = json.loads(selection.read_text(encoding="utf-8"))
        for field, bad_value in (("alpha_by_group", {**original["alpha_by_group"],
                                                     "N": original["alpha_by_group"]["N"] * 7}),
                                 ("weight_hash", "f" * 64)):
            tampered = tmp / f"tampered_{field}.json"
            tampered.write_text(json.dumps({**original, field: bad_value}, ensure_ascii=False),
                                encoding="utf-8", newline="\n")
            out_t = tmp / f"out_tampered_{field}"
            code = run_entry(predict_test, base_argv(bundle_dir, spec, sig, route, route_sha,
                                                     out_t, "N+S+Q", tampered))
            check(f"选择期 manifest 的 {field} 被改 → 拒绝", code != 0)

        # ---- 负例 4：test 真值不可用 + 扰动后预测不变 ----
        b = read_frozen_bundle(bundle_dir, "Agriculture_h3_f1", "proxy", sig, spec,
                               isolate_test=True)
        tm = b.segment_mask("test")
        check("隔离模式：test 段 targets 全为 NaN",
              np.isnan(np.asarray(b.arrays["targets"])[tm]).all()
              and np.isnan(np.asarray(b.arrays["targets_standardized"])[tm]).all())
        check("隔离模式：train/cal/dec 真值未被波及",
              np.isfinite(np.asarray(b.arrays["targets"])[~tm]).all())
        check("隔离模式：test 特征切片不含 targets 键",
              "targets" not in b.segment("test", with_targets=False)
              and "targets" in b.segment("test"))

        # 扰动磁盘上的 test 真值并重签清单 → 预测必须逐字节不变
        task_dir = bundle_dir / "Agriculture_h3_f1"
        rng = np.random.default_rng(7)
        changed = {}
        for key in ("targets", "targets_standardized", "targets_raw"):
            p = task_dir / f"{key}.npy"
            if not p.is_file():
                continue
            arr = np.load(p).copy()
            arr[tm] = rng.normal(size=(int(tm.sum()), arr.shape[1])).astype(arr.dtype)
            np.save(p, arr, allow_pickle=False)
            changed[p.relative_to(bundle_dir).as_posix()] = sha256_file(p)
        manifest_obj = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest_obj["files"].update(changed)
        (bundle_dir / "manifest.json").write_text(
            json.dumps(manifest_obj, ensure_ascii=False), encoding="utf-8", newline="\n")
        check("扰动 test 真值后清单已重签（signature 只覆盖 inputs，不变）",
              json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))["signature"] == sig)

        out_pert = tmp / "out_perturbed"
        code = run_entry(predict_test, base_argv(bundle_dir, spec, sig, route, route_sha,
                                                 out_pert, "N+S+Q", selection))
        perturbed_ok = (code == 0
                        and (out_pert / "N_S_Q_test_predictions.csv").read_bytes() == body_ok)
        check("扰动 test 真值后预测逐字节不变（真值无因果影响）", perturbed_ok)

        # ---- 负例 6：输出校验（纯函数，直接构造坏输入）----
        bounds = (45, 60)
        idx = np.arange(45, 58, dtype=np.int64)      # 13 个 test 起点
        check("合法预测 → 无问题",
              predict_test.validate_test_predictions(np.zeros((13, 3)), idx, bounds, 3) == [])
        # 行数不符：13 个起点却只给 12×3（少一行就少一个键）
        check("行数与起点数不符（13 起点只给 12×3）→ 报错",
              predict_test.validate_test_predictions(np.zeros((12, 3)), idx, bounds, 3) != [])
        # 步数不符：给 13×2
        check("步数不符（13×2 而非 13×3）→ 报错",
              predict_test.validate_test_predictions(np.zeros((13, 2)), idx, bounds, 3) != [])
        dup = idx.copy(); dup[5] = dup[4]
        check("起点重复 → 报错",
              predict_test.validate_test_predictions(np.zeros((13, 3)), dup, bounds, 3) != [])
        oob = idx.copy(); oob[0] = 99
        check("origin 越界 → 报错",
              predict_test.validate_test_predictions(np.zeros((13, 3)), oob, bounds, 3) != [])
        spanning = idx.copy(); spanning[0] = 58   # 58 + 3 > 60
        check("目标窗口跨界 → 报错",
              predict_test.validate_test_predictions(np.zeros((13, 3)), spanning, bounds, 3) != [])
        nonfinite = np.zeros((13, 3)); nonfinite[2, 1] = np.nan
        check("含非有限预测 → 报错",
              predict_test.validate_test_predictions(nonfinite, idx, bounds, 3) != [])
        check("非二维预测 → 报错",
              predict_test.validate_test_predictions(np.zeros(13), idx, bounds, 3) != [])

        # 同一套判据也要作用在**文件**上（CSV 才是交给主控的产物）
        def write_csv(path: Path, rows: list[dict]) -> Path:
            cols = ["task_id", "fold_id", "origin_id", "origin_index", "segment",
                    "scenario", "candidate_id", "seed", "step", "y_pred",
                    "target_scale", "bundle_signature", "config_sha256", "code_commit"]
            with path.open("w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
                w.writeheader()
                for r in rows:
                    w.writerow({**{c: "" for c in cols}, **r})
            return path

        def good_rows(n_origins: int = 13) -> list[dict]:
            out = []
            for o in range(45, 45 + n_origins):
                for step in range(1, 4):
                    out.append({"origin_id": f"Agriculture:h3:f1:o{o}", "origin_index": o,
                                "segment": "test", "step": step, "y_pred": 0.5})
            return out

        base_rows = good_rows()
        check("文件校验：合法 CSV → 无问题",
              predict_test.verify_written_csv(
                  write_csv(tmp / "ok.csv", base_rows), bounds, 3,
                  expected_rows=39) == [])
        bad_key = base_rows + [base_rows[0]]
        check("文件校验：键重复 → 报错",
              predict_test.verify_written_csv(
                  write_csv(tmp / "dup.csv", bad_key), bounds, 3) != [])
        bad_seg = [dict(r, segment="decision") for r in base_rows]
        check("文件校验：segment 不是 test → 报错",
              predict_test.verify_written_csv(
                  write_csv(tmp / "seg.csv", bad_seg), bounds, 3) != [])
        bad_val = [dict(r, y_pred="nan") for r in base_rows]
        check("文件校验：y_pred 非有限 → 报错",
              predict_test.verify_written_csv(
                  write_csv(tmp / "nan.csv", bad_val), bounds, 3) != [])
        bad_idx = base_rows[:-1]
        check("文件校验：行数少一行 → 报错（须给期望行数才查得出）",
              predict_test.verify_written_csv(
                  write_csv(tmp / "short.csv", bad_idx), bounds, 3,
                  expected_rows=39) != [])
        check("文件校验：不给期望行数时少一行查不出（说明为何必须传期望值）",
              predict_test.verify_written_csv(
                  write_csv(tmp / "short2.csv", bad_idx), bounds, 3) == [])

    print(f"\n全部通过（{len(r6.PASSED)} 项检查：第六轮 49 + 第七轮 {len(r6.PASSED) - 49}）")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1)
