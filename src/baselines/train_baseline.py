"""Unified CLI for baseline comparison experiments.

Example:
    # ML baseline
    python src/baselines/train_baseline.py --model ridge_logreg --task FMA_UE --fold 1

    # DL baseline
    python src/baselines/train_baseline.py --model bilstm_attn --task hand_tone --fold 0

    # Local smoke test (any model, finishes in seconds)
    python src/baselines/train_baseline.py --model xgboost --task FMA_UE --fold 1 --smoke-test

Run ``python src/baselines/train_baseline.py --list`` to see all registered models.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src/` importable so we can `from train import ...`, `from baselines... import ...`.
SRC = Path(__file__).resolve().parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bjh_io.bjh_loader import EEG_FS_DEFAULT  # noqa: E402
from task_config import TASK_CONFIGS  # noqa: E402

# Trigger registry population via package imports.
import baselines.ml_models  # noqa: F401, E402
import baselines.dl_models  # noqa: F401, E402
from baselines.registry import REGISTRY, list_models  # noqa: E402
from baselines.runner import apply_smoke_overrides, run_folds  # noqa: E402


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train a single baseline (traditional ML or DL) on one task / fold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Outputs land in results/baselines/<task>/<model>/ unless --out-dir is set.",
    )
    ap.add_argument("--model", required=False, default=None,
                    help=f"Baseline name. One of: {', '.join(list_models())}")
    ap.add_argument("--list", action="store_true",
                    help="Print all registered baselines and exit.")
    ap.add_argument("--task", required=False, default=None,
                    choices=list(TASK_CONFIGS),
                    help="Task: FMA_UE | BI | hand_tone | hand_function.")
    ap.add_argument("--fold", type=int, default=1,
                    help="Fold id to train (default: 1). Use 0 or -1 to train all folds.")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                    help="Project root (defaults to repo root).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Override the output directory (default: results/baselines/<task>/<model>/).")
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/samples_manifest_real_archive.csv"),
                    help="Manifest CSV (relative paths resolved against --root).")
    ap.add_argument("--split-json", type=Path,
                    default=Path("splits/split_patient_3fold.json"),
                    help="Patient-level split JSON.")
    ap.add_argument("--smoke-test", dest="smoke_test", action="store_true",
                    help="Tiny run for local verification (overrides epochs/bags/subjects).")

    # Data / alignment knobs (kept compatible with train.py).
    ap.add_argument("--alignment-mode", default="adk")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--dtw-length", type=int, default=32)
    ap.add_argument("--eeg-channels", type=int, default=30)
    ap.add_argument("--emg-channels", type=int, default=4)
    ap.add_argument("--imu-channels", type=int, default=24)
    ap.add_argument("--eeg-fs", type=float, default=EEG_FS_DEFAULT)
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="Directory for disk-cached aligned trial arrays (.npz). "
                         "Shared with main train.py cache when pointing to the same path.")

    # DL-only knobs (ignored by ML baselines).
    ap.add_argument("--bag-size", type=int, default=4)
    ap.add_argument("--eval-bag-size", type=int, default=4)
    ap.add_argument("--train-bags", type=int, default=15)
    ap.add_argument("--eval-bags", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--tolerance", type=float, default=None,
                    help="Override the task's score_tolerance (raw-error threshold).")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--device", default="")
    return ap


def main() -> None:
    ap = _build_argparser()
    args = ap.parse_args()

    if args.list:
        for name in list_models():
            print(name)
        return

    if not args.model:
        ap.error("--model is required (use --list to see options)")
    if args.model not in REGISTRY:
        ap.error(f"Unknown model {args.model!r}. Registered: {list_models()}")
    if not args.task:
        ap.error("--task is required")

    if args.smoke_test:
        apply_smoke_overrides(args)
        print(f"[smoke-test] overrides applied: "
              f"epochs={args.epochs} bag_size={args.bag_size} "
              f"train_bags={args.train_bags} eval_bags={args.eval_bags} "
              f"max_train={args.smoke_max_train} max_val={args.smoke_max_val}")

    builder = REGISTRY[args.model]
    run_folds(args, builder)


if __name__ == "__main__":
    main()
