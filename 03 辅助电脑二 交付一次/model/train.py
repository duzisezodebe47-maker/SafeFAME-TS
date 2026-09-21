"""统一训练入口。

任务书 §4.2 要求：按统一 ``FeatureBundle`` 读取，输出统一预测契约，
支持指定任务、折、种子与独立输出目录。

用法::

    python train.py --candidate N+S --domain Climate --horizon 4 --fold 0 \\
                    --seed 2026 --output-dir <目录> [--data-source v2smoke]

**测试真值从不参与保留决策**：候选的 α 只在校准段选，测试段只在最后算一次指标，
且该指标只写入证据文件，不影响任何选择。

**数据划分**：本入口不自建划分。协议未冻结前（主控尚未交付 `split_spec.json`），
使用 v2 已冻结的 70/10/20 划分（`train_famets._split_origins`，见 docs/10），
并在证据里标记 ``split_source``。协议冻结后应改为读取主控的划分文件。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import load_time_ordered_frame  # noqa: E402

from branches import FeatureBundle  # noqa: E402
from candidates import CANDIDATES, BranchResidualCandidate  # noqa: E402
from predict_io import (  # noqa: E402
    PredictionSet, code_commit, hash_config, hash_features, validate_alignment,
)

# 与 train_famets.py 一致：v2 冻结划分（70% 训练 / 10% 校准 / 20% 测试）
TRAIN_FRACTION, VALIDATION_FRACTION = 0.7, 0.8

SEMANTIC_ROOT = ROOT / "data_processed" / "semantic_features"
NUMERIC_ROOT = ROOT / "references" / "external" / "Time-MMD" / "numerical"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_bundle(
    domain: str, input_len: int, horizon: int, origin_range: range
) -> tuple[FeatureBundle, np.ndarray]:
    """按给定起点区间构造 FeatureBundle。真实数据，不合成、不代填。"""
    frame, _ = load_time_ordered_frame(NUMERIC_ROOT / domain / f"{domain}.csv")
    raw = pd.to_numeric(frame["OT"], errors="coerce").to_numpy(float)
    if np.isnan(raw).any():
        raise ValueError(f"{domain} 的 OT 含缺失值，需先由辅助电脑一审计")

    cache = np.load(SEMANTIC_ROOT / f"{domain}.npz")
    positions = {int(v): i for i, v in enumerate(cache["origin_index"].ravel())}
    origins = np.array([o for o in origin_range], dtype=np.int64)
    absent = [int(o) for o in origins if int(o) not in positions]
    if absent:
        raise ValueError(f"{domain} 缺少起点 {absent[:5]} 的语义特征")

    windows = np.stack([raw[int(o) - input_len : int(o)] for o in origins]).astype(np.float32)
    targets = np.stack([raw[int(o) : int(o) + horizon] for o in origins]).astype(np.float32)
    index = [positions[int(o)] for o in origins]

    bundle = FeatureBundle(
        x=windows,
        report=cache["report_embedding"][index].astype(np.float32),
        search=cache["search_embedding"][index].astype(np.float32),
        quality=cache["quality"][index].astype(np.float32),
        origins=origins,
    )
    return bundle, targets


def split_origins(n: int, input_len: int, horizon: int) -> dict[str, range]:
    """v2 冻结划分。**不自行发明新划分。**"""
    train_end, validation_end = int(n * TRAIN_FRACTION), int(n * VALIDATION_FRACTION)
    return {
        "train": range(input_len, train_end - horizon + 1),
        "calibration": range(train_end, validation_end - horizon + 1),
        "test": range(validation_end, n - horizon + 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--domain", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--input-len", type=int, default=52)
    parser.add_argument("--fold", type=int, default=0, help="滚动折号（当前仅 0）")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-source", default="v2smoke",
                        choices=["v2smoke"], help="v2smoke=本机 v2 特征，仅用于工程冒烟")
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    failures: list[dict] = []

    config = {
        "candidate": args.candidate,
        "branches": list(CANDIDATES[args.candidate]),
        "domain": args.domain,
        "horizon": args.horizon,
        "input_len": args.input_len,
        "fold": args.fold,
        "seed": args.seed,
        "data_source": args.data_source,
        "split_source": "v2 frozen 70/10/20 (docs/10) — 主控协议未冻结前的临时取值",
        "solver": "group-penalized ridge, cholesky, float64 normal equations",
    }
    config_hash = hash_config(config)

    try:
        frame, _ = load_time_ordered_frame(NUMERIC_ROOT / args.domain / f"{args.domain}.csv")
        n = len(frame)
        parts = split_origins(n, args.input_len, args.horizon)

        train, y_train = build_bundle(args.domain, args.input_len, args.horizon, parts["train"])
        cal, y_cal = build_bundle(args.domain, args.input_len, args.horizon, parts["calibration"])
        test, y_test = build_bundle(args.domain, args.input_len, args.horizon, parts["test"])

        # 错位检查：起点集合必须与协议声明的区间逐位一致
        for name, part in parts.items():
            bundle = {"train": train, "calibration": cal, "test": test}[name]
            validate_alignment(bundle.origins, np.array(list(part), dtype=np.int64), name)

        feature_hash = hash_features(train)

        model = BranchResidualCandidate(CANDIDATES[args.candidate])
        model.fit_design(train)
        model.select_alphas(train, cal, y_train, y_cal)
        fit_bundle = FeatureBundle(
            x=np.r_[train.x, cal.x], report=np.r_[train.report, cal.report],
            search=np.r_[train.search, cal.search], quality=np.r_[train.quality, cal.quality],
            origins=np.r_[train.origins, cal.origins],
        )
        model.refit(fit_bundle, np.r_[y_train, y_cal])

        predictions = model.predict(test)
        if predictions.ndim != 2 or predictions.shape != y_test.shape:
            raise AssertionError(f"预测形状 {predictions.shape} != 目标形状 {y_test.shape}")

        # 测试段指标只写入证据，不参与任何选择
        test_mse = float(np.mean((predictions - y_test) ** 2))
        last_mse = float(np.mean((test.x[:, -1, None] - y_test) ** 2))

        ps = PredictionSet()
        ps.extend_grid(
            test.origins, predictions,
            task_id=f"{args.domain}_h{args.horizon}", fold_id=args.fold,
            candidate_id=args.candidate, seed=args.seed,
            config_hash=config_hash, feature_hash=feature_hash,
            commit=code_commit(ROOT),
        )
        n_rows = ps.write(out / "predictions.csv")

        elapsed = time.perf_counter() - started
        manifest = {
            "config": config, "config_hash": config_hash, "feature_hash": feature_hash,
            "status": "completed", "utc": utc(), "rows_written": n_rows,
            "n_parameters": model.n_parameters,
            "alpha_by_group": model.alpha_by_group,
            "branch_widths": {b.name: b.width for b in model.branches},
            "dropped_zero_variance": {
                b.name: getattr(b.scaler, "n_dropped", None) for b in model.branches
                if hasattr(b, "scaler")
            },
            "resources": {
                "wall_seconds": round(elapsed, 3),
                "cpu_count": __import__("os").cpu_count(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "device": "cpu",
            },
            "test_metrics_diagnostic_only": {
                "note": "仅诊断，不参与任何候选保留决策；测试真值从不进入拟合",
                "model_mse": test_mse, "last_anchor_mse": last_mse,
            },
            "failures": failures,
        }
    except Exception as exc:  # noqa: BLE001 — 失败必须留证据，不静默
        failures.append({"stage": "train", "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc()})
        manifest = {"config": config, "config_hash": config_hash, "status": "failed",
                    "utc": utc(), "failures": failures}
        (out / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8", newline="\n")   # 固定 LF，使产出字节 == 提交字节
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    (out / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "candidate": args.candidate,
                      "rows": n_rows, "seconds": manifest["resources"]["wall_seconds"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
