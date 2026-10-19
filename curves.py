#!/usr/bin/env python3
"""Plot learning curves recorded by train.py.

Reads the per-fold CSV files that train.py writes to a checkpoint directory
(val_auroc_curve_f*.csv or loss_curve_f*.csv) and plots the mean over folds
with a +/- 1 standard deviation band. Only recorded values are plotted.

Because of early stopping, folds can stop at different epochs. At each epoch
the mean is taken over the folds that reached it, and the curve ends at the
last epoch reached by at least --min-folds folds.

Examples:
    python curves.py --run checkpoints/readmission --metric val_auroc
    python curves.py --run checkpoints/mortality/full --label "literature and UMLS" \
                     --run checkpoints/mortality/umls --label "UMLS" \
                     --metric val_auroc --title "Mortality" --out mortality_auroc.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FILE_PATTERN = {"val_auroc": "val_auroc_curve_f*.csv", "loss": "loss_curve_f*.csv"}
Y_LABEL = {"val_auroc": "Validation AUROC", "loss": "Training loss"}


def load_folds(run_dir: Path, metric: str) -> list:
    files = sorted(run_dir.glob(FILE_PATTERN[metric]))
    if not files:
        raise FileNotFoundError(f"No {FILE_PATTERN[metric]} in {run_dir}")
    folds = []
    for f in files:
        data = np.loadtxt(f, delimiter=",", comments="#", ndmin=2)
        folds.append(dict(zip(data[:, 0].astype(int), data[:, 1])))
    return folds


def aggregate(folds: list, min_folds: int):
    last = max(max(f) for f in folds)
    epochs, means, stds, counts = [], [], [], []
    for ep in range(1, last + 1):
        vals = np.array([f[ep] for f in folds if ep in f], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) < min_folds:
            break
        epochs.append(ep)
        means.append(vals.mean())
        stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        counts.append(len(vals))
    return np.array(epochs), np.array(means), np.array(stds), counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="checkpoint directory written by train.py (repeatable)")
    ap.add_argument("--label", action="append",
                    help="legend label for each --run (same order)")
    ap.add_argument("--metric", choices=FILE_PATTERN, default="val_auroc")
    ap.add_argument("--min-folds", type=int, default=2)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="curves.png")
    args = ap.parse_args()

    labels = args.label or [Path(r).name for r in args.run]
    if len(labels) != len(args.run):
        ap.error("--label must be given once per --run")

    plt.figure(figsize=(7.8, 4.8))
    for run, label in zip(args.run, labels):
        folds = load_folds(Path(run), args.metric)
        ep, mean, std, counts = aggregate(folds, args.min_folds)
        line, = plt.plot(ep, mean, linewidth=2.0,
                         label=f"{label} (n={len(folds)} folds)")
        plt.fill_between(ep, mean - std, mean + std,
                         color=line.get_color(), alpha=0.2, linewidth=0)
        print(f"{label}: {len(folds)} folds, plotted epochs 1-{ep[-1]}, "
              f"folds per epoch {min(counts)}-{max(counts)}")

    plt.xlabel("Epoch")
    plt.ylabel(Y_LABEL[args.metric])
    if args.title:
        plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
