"""第九轮测试：封印路由后的 Climate 单次测试预测（合成数据）。

先跑第八轮 131 项（其内已含第六、七轮 49 + 36），再补任务书 §五 要求的断言：

  1. 路由 SHA 错误 → 拒绝
  2. `--candidate` 不是 `N` → 拒绝
  3. 路由 `fallback` 不是 `N` → 拒绝
  4. 选择期清单的 alpha / weight_hash / candidate 被改动 → 拒绝
  5. test 真值路径被访问 → 拒绝（隔离断言 + 扰动 test 真值后预测逐字节不变）
  6. 输出目录已存在 → 拒绝覆盖
  7. 输出不是 496 行 / 键重复 / origin 越界 / 非有限值 → 拒绝
  8. 交付目录出现第二个候选测试预测 → 完备性闸门失败

合成夹具的 Climate 形状与真实登记一致（test 124 起点 × horizon 4 = **496 行**）。

用法::

    .venv/Scripts/python.exe "03 辅助电脑二 交付九次/model/tests/test_round9.py"
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

import test_round8 as r8  # noqa: E402  —— 它内部会先跑第六、七轮
import test_round6 as r6  # noqa: E402
import test_round7 as r7  # noqa: E402

check = r6.check
C_HORIZON = r8.C_HORIZON
C_BOUNDS = r8.C_BOUNDS
N_TEST = C_BOUNDS[3] - C_HORIZON - C_BOUNDS[2] + 1          # 1017..1140 → 124 起点
EXPECTED_ROWS = N_TEST * C_HORIZON                          # 496


def main() -> int:
    print("第九轮测试（合成数据）")
    print("\n========== 先跑第八轮 131 项 ==========")
    rc8 = r8.main()
    check("第八轮 131 项全部通过", rc8 == 0)

    import predict_test
    import verify_test_delivery as gate9
    from bundle_reader import sha256_file

    print("\n========== 第九轮：numeric_fallback → N 的测试预测 ==========")
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        spec = r8.make_climate_spec(tmp / "spec.json")
        bundle_dir = r8.make_climate_bundle(tmp / "bundle", spec)
        spec_sha = sha256_file(spec)
        sig = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))["signature"]

        # 选择期清单：跑 train.py --candidate N 生成（含 weight_hash / alpha_by_group）
        import train as train_entry
        sel_dir = tmp / "sel"
        code = r7.run_entry(train_entry, ["train.py", "--bundle", str(bundle_dir),
                                          "--task", "Climate_h4_f2", "--scenario", "proxy",
                                          "--signature", sig, "--split-spec", str(spec),
                                          "--segments", "calibration", "decision",
                                          "--candidate", "N", "--output-dir", str(sel_dir)])
        check("选择期 train.py --candidate N 端到端成功", code == 0)
        selection = sel_dir / "N_run_manifest.json"
        check("选择期清单含 weight_hash / alpha_by_group",
              all(k in json.loads(selection.read_text(encoding="utf-8"))
                  for k in ("weight_hash", "alpha_by_group")))

        # 第九轮的封印路由形态：selected=numeric_fallback, fallback=N
        # 合成夹具的 sealed_route 默认写的是 Agriculture 任务号，这里必须覆盖成 Climate
        route_obj = r6.sealed_route(spec, spec_sha, sig, task_id="Climate_h4_f2", fold_id=2,
                                    selected="numeric_fallback", fallback="N")
        route = tmp / "route9.json"
        route_sha = r7.write_route(route, route_obj)

        def argv_for(out: Path, cand: str = "N", rsha: str | None = None,
                     rt: Path | None = None, sel: Path | None = None) -> list[str]:
            return ["predict_test.py", "--bundle", str(bundle_dir), "--task", "Climate_h4_f2",
                    "--scenario", "proxy", "--signature", sig, "--split-spec", str(spec),
                    "--route", str(rt or route), "--route-sha256", rsha or route_sha,
                    "--selection-manifest", str(sel or selection),
                    "--candidate", cand, "--seed", "2026", "--output-dir", str(out)]

        # ---- 正路 ----
        out_ok = tmp / "out_ok"
        code = r7.run_entry(predict_test, argv_for(out_ok))
        check("numeric_fallback + 候选 N → 端到端成功", code == 0)
        csv_path = out_ok / "N_test_predictions.csv"
        check("写出 N_test_predictions.csv", csv_path.is_file())
        with csv_path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        check(f"行数 = {EXPECTED_ROWS}（{N_TEST} 起点 × {C_HORIZON} 步）", len(rows) == EXPECTED_ROWS)
        check("键唯一", len({(r["origin_id"], r["step"]) for r in rows}) == len(rows))
        check("只含 test 段", {r["segment"] for r in rows} == {"test"})
        check("候选唯一 = N", {r["candidate_id"] for r in rows} == {"N"})
        check("预测全为有限值",
              bool(np.isfinite(np.array([float(r["y_pred"]) for r in rows])).all()))
        man = json.loads((out_ok / "N_test_manifest.json").read_text(encoding="utf-8"))
        check("manifest 记录四锚 + 列名 + 运行资源",
              all(k in man["input_anchors"] for k in
                  ("route_file_sha256", "bundle_signature", "split_spec_sha256",
                   "selection_manifest_sha256"))
              and man["n_rows"] == EXPECTED_ROWS
              and (man["runtime"].get("peak_rss_bytes") or 0) > 0)
        body_ok = csv_path.read_bytes()

        # ---- 1) 路由 SHA 错 ----
        out1 = tmp / "out1"
        code = r7.run_entry(predict_test, argv_for(out1, rsha="0" * 64))
        check("1) 路由 SHA 不符 → 拒绝", code != 0 and not out1.exists())

        # ---- 2) 候选不是 N ----
        out2 = tmp / "out2"
        code = r7.run_entry(predict_test, argv_for(out2, cand="N+S+Q"))
        check("2) 候选不是 N（路由 fallback=N）→ 拒绝", code != 0)

        # ---- 3) 路由 fallback 不是 N ----
        route_bad = tmp / "route_bad.json"
        rsha_bad = r7.write_route(route_bad, r6.sealed_route(
            spec, spec_sha, sig, task_id="Climate_h4_f2", fold_id=2,
            selected="numeric_fallback", fallback="AR-Ridge"))
        out3 = tmp / "out3"
        code = r7.run_entry(predict_test, argv_for(out3, rsha=rsha_bad, rt=route_bad))
        check("3) 路由 fallback 不是 N → 拒绝", code != 0)

        # ---- 4) 选择期清单被篡改（alpha / weight_hash / candidate）----
        original = json.loads(selection.read_text(encoding="utf-8"))
        for field, bad in (("alpha_by_group",
                            {**original["alpha_by_group"], "N": original["alpha_by_group"]["N"] * 7}),
                           ("weight_hash", "f" * 64),
                           ("candidate", "N+Q")):
            tampered = tmp / f"tampered_{field}.json"
            tampered.write_text(json.dumps({**original, field: bad}, ensure_ascii=False),
                                encoding="utf-8", newline="\n")
            out_t = tmp / f"out_tampered_{field}"
            code = r7.run_entry(predict_test, argv_for(out_t, sel=tampered))
            check(f"4) 选择期清单的 {field} 被改 → 拒绝", code != 0)

        # ---- 5) test 真值不可用 + 扰动后预测逐字节不变 ----
        from bundle_reader import read_frozen_bundle
        b = read_frozen_bundle(bundle_dir, "Climate_h4_f2", "proxy", sig, spec,
                               isolate_test=True)
        tm = b.segment_mask("test")
        check("5) 隔离模式：test 段 targets 全为 NaN",
              bool(np.isnan(np.asarray(b.arrays["targets"])[tm]).all()))
        check("5) 非 test 行完好", bool(np.isfinite(np.asarray(b.arrays["targets"])[~tm]).all()))
        check("5) 推理视图不含 targets 键",
              "targets" not in b.segment("test", with_targets=False))
        task_dir = bundle_dir / "Climate_h4_f2"
        rng = np.random.default_rng(3)
        changed = {}
        for key in ("targets", "targets_standardized", "targets_raw"):
            p = task_dir / f"{key}.npy"
            if not p.is_file():
                continue
            arr = np.load(p).copy()
            arr[tm] = rng.normal(size=(int(tm.sum()), arr.shape[1])).astype(arr.dtype)
            np.save(p, arr, allow_pickle=False)
            changed[p.relative_to(bundle_dir).as_posix()] = sha256_file(p)
        mo = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        mo["files"].update(changed)
        (bundle_dir / "manifest.json").write_text(json.dumps(mo, ensure_ascii=False),
                                                  encoding="utf-8", newline="\n")
        out_pert = tmp / "out_pert"
        code = r7.run_entry(predict_test, argv_for(out_pert))
        check("5) 扰动 test 真值后预测逐字节不变（真值无因果影响）",
              code == 0 and (out_pert / "N_test_predictions.csv").read_bytes() == body_ok)

        # ---- 6) 输出目录已存在 ----
        code = r7.run_entry(predict_test, argv_for(out_ok))
        check("6) 输出目录已存在且非空 → 拒绝覆盖", code != 0)

        # ---- 7) 输出校验（纯函数，直接构造坏输入）----
        idx = np.arange(C_BOUNDS[2], C_BOUNDS[2] + N_TEST, dtype=np.int64)
        good = np.zeros((N_TEST, C_HORIZON))
        check("7) 合法预测 → 无问题",
              predict_test.validate_test_predictions(good, idx, (C_BOUNDS[2], C_BOUNDS[3]),
                                                     C_HORIZON) == [])
        check("7) 行数不是 496 → 报错",
              predict_test.validate_test_predictions(
                  np.zeros((N_TEST - 1, C_HORIZON)), idx, (C_BOUNDS[2], C_BOUNDS[3]),
                  C_HORIZON) != [])
        dup = idx.copy(); dup[3] = dup[2]
        check("7) 键重复（origin 重复）→ 报错",
              predict_test.validate_test_predictions(good, dup, (C_BOUNDS[2], C_BOUNDS[3]),
                                                     C_HORIZON) != [])
        oob = idx.copy(); oob[0] = 99
        check("7) origin 越界 → 报错",
              predict_test.validate_test_predictions(good, oob, (C_BOUNDS[2], C_BOUNDS[3]),
                                                     C_HORIZON) != [])
        nonfinite = good.copy(); nonfinite[1, 1] = np.nan
        check("7) 非有限值 → 报错",
              predict_test.validate_test_predictions(nonfinite, idx,
                                                     (C_BOUNDS[2], C_BOUNDS[3]),
                                                     C_HORIZON) != [])

        # ---- 8) 闸门：出现第二个候选的测试预测即失败 ----
        fake = tmp / "fake_delivery"
        (fake / "evidence").mkdir(parents=True, exist_ok=True)
        (fake / "test_prediction").mkdir(parents=True, exist_ok=True)
        (fake / "test_prediction" / "N_test_predictions.csv").write_bytes(body_ok)
        (fake / "test_prediction" / "N_S_Q_test_predictions.csv").write_bytes(body_ok)
        problems = gate9.check(fake, task="Climate_h4_f2", horizon=C_HORIZON,
                               expected_origins=N_TEST, candidate="N",
                               route_sha256=route_sha, require_git_clean=False)
        check("8) 闸门：出现第二个候选测试预测 → 判失败",
              any("数量" in p for p in problems) and any("非授权候选" in p for p in problems))
        only_one = tmp / "one_delivery"
        (only_one / "test_prediction").mkdir(parents=True, exist_ok=True)
        (only_one / "test_prediction" / "N_test_predictions.csv").write_bytes(body_ok)
        problems2 = gate9.check(only_one, task="Climate_h4_f2", horizon=C_HORIZON,
                                expected_origins=N_TEST, candidate="N",
                                route_sha256=route_sha, require_git_clean=False)
        check("8) 闸门：只有一个授权候选时不再报数量问题",
              not any("数量" in p for p in problems2))

    total = len(r6.PASSED)
    print(f"\n全部通过（{total} 项检查：第六轮 49 + 第七轮 36 + 第八轮 46 + 第九轮 {total - 131}）")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1)
