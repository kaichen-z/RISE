#!/usr/bin/env python3
"""Score predicted maps against the ground truth -- the numbers reported in the paper.

    # object detection (paper Table 1): IoU / Dice
    python evaluate.py --task obs --pred eval_output/obs

    # layout reconstruction (paper Fig. 7 / Table 2): Chamfer (cm) / F1 @ 15 cm
    python evaluate.py --task wall --pred eval_output/wall

    # score the stage-4 map itself (no diffusion) -- the "RISE w/ G" ablation row
    python evaluate.py --task wall --pred ../4_inversion/result_ours/result_input/image_lowr

    # trajectory-length ablation (paper Fig. 9): group by the ratio in the file name
    python evaluate.py --task wall --pred eval_output/wall --by-ratio

Predictions are 128x128 grayscale PNGs named `<prefix>_<case>_<ratio>.png`
(`ourdiff_50_0.png`, `ours_50_40.png`, `bsl_50_0_mvdr_M6.png`, ...); ground truth
is `data/GT/{obs,wall}/<case>_gt.png`.  Non-zero pixels are the occupied class.

Units.  The maps are a 600 px view of the room at 50 px/m (common/config.py),
downsampled to 128 px, so one evaluation pixel is 9.375 cm.  Chamfer distances
are reported in centimetres and the F1 tolerance defaults to the paper's 15 cm.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import CM_PER_EVAL_PIXEL, EVAL_SIZE, F1_TOLERANCE_CM  # noqa: E402

# `<prefix>_<case>_<ratio>[...].png`; the tail (baseline tag, etc.) is ignored.
NAME_RE = re.compile(r"^[A-Za-z0-9]+_(\d+)_(\d+)")


def load_mask(path, size=EVAL_SIZE):
    """PNG -> boolean occupancy mask, resized to the evaluation grid if needed."""
    image = Image.open(path).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.NEAREST)
    return np.array(image) > 0


def obs_metrics(pred, gt):
    """Object detection: IoU and Dice over the occupied pixels (paper Table 1)."""
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    total = pred.sum() + gt.sum()
    iou = inter / union if union else 0.0
    dice = 2 * inter / total if total else 0.0
    return {"iou": 100 * iou, "dice": 100 * dice}


def wall_metrics(pred, gt, tolerance_cm=F1_TOLERANCE_CM, cm_per_pixel=CM_PER_EVAL_PIXEL):
    """Layout: symmetric Chamfer distance in cm and F1 at a metric tolerance.

    Chamfer follows the definition used for the paper's numbers -- the sum of the
    two mean nearest-neighbour distances (pred->gt and gt->pred), converted from
    pixels to centimetres.
    """
    pred_pts = np.argwhere(pred).astype(float)
    gt_pts = np.argwhere(gt).astype(float)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return {"chamfer_cm": float("nan"), "f1": 0.0, "n_pred": len(pred_pts), "n_gt": len(gt_pts)}
    d_pred = cKDTree(gt_pts).query(pred_pts)[0] * cm_per_pixel
    d_gt = cKDTree(pred_pts).query(gt_pts)[0] * cm_per_pixel
    chamfer = d_pred.mean() + d_gt.mean()
    precision = (d_pred < tolerance_cm).mean()
    recall = (d_gt < tolerance_cm).mean()
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"chamfer_cm": chamfer, "f1": 100 * f1, "n_pred": len(pred_pts), "n_gt": len(gt_pts)}


def collect(pred_dir, gt_dir, task, tolerance_cm, ratio=None):
    """Score every prediction PNG that has a ground-truth case."""
    gt_paths = {int(p.name.split("_")[0]): p for p in sorted(Path(gt_dir).glob("*_gt.png"))}
    if not gt_paths:
        raise FileNotFoundError(f"no <case>_gt.png files in {gt_dir}")
    rows, skipped = [], []
    for path in sorted(Path(pred_dir).glob("*.png")):
        match = NAME_RE.match(path.name)
        if not match:
            skipped.append((path.name, "unparsable name"))
            continue
        case, file_ratio = int(match.group(1)), int(match.group(2))
        if ratio is not None and file_ratio != ratio:
            continue
        if case not in gt_paths:
            skipped.append((path.name, f"no ground truth for case {case}"))
            continue
        pred, gt = load_mask(path), load_mask(gt_paths[case])
        if not gt.any():
            # 24 of the 47 object maps are empty-room configurations with no
            # furniture; IoU/Dice are undefined there, so they are reported
            # separately instead of being averaged in as zeros.
            skipped.append((path.name, f"case {case} has an empty {task} ground truth"))
            continue
        metrics = obs_metrics(pred, gt) if task == "obs" else wall_metrics(pred, gt, tolerance_cm)
        rows.append({"case": case, "ratio": file_ratio, "file": path.name, **metrics})
    n_scorable = sum(1 for p in gt_paths.values() if load_mask(p).any())
    return rows, skipped, n_scorable


def mean_of(rows, key):
    values = [r[key] for r in rows if not np.isnan(r[key])]
    return float(np.mean(values)) if values else float("nan")


def report(rows, task, label):
    keys = ("iou", "dice") if task == "obs" else ("chamfer_cm", "f1")
    header = {"iou": "IoU (%)", "dice": "Dice (%)", "chamfer_cm": "Chamfer (cm)", "f1": "F1 (%)"}
    print(f"\n=== {label} · {len(rows)} maps ===")
    print(f"{'case':>6} {'ratio':>6} " + " ".join(f"{header[k]:>13}" for k in keys))
    for row in sorted(rows, key=lambda r: (r["case"], r["ratio"])):
        print(f"{row['case']:>6} {row['ratio']:>6} " + " ".join(f"{row[k]:>13.2f}" for k in keys))
    print(f"{'AVG':>6} {'':>6} " + " ".join(f"{mean_of(rows, k):>13.2f}" for k in keys))
    return {k: mean_of(rows, k) for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["obs", "wall"], required=True,
                    help="obs = object detection (IoU/Dice), wall = layout (Chamfer/F1)")
    ap.add_argument("--pred", required=True, help="directory of predicted PNGs")
    ap.add_argument("--gt", default=None, help="ground-truth directory (default: data/GT/<task>)")
    ap.add_argument("--tolerance-cm", type=float, default=F1_TOLERANCE_CM,
                    help=f"F1 tolerance in cm (default {F1_TOLERANCE_CM:g}, as in the paper)")
    ap.add_argument("--ratio", type=int, default=None, help="only score this trajectory-length ratio")
    ap.add_argument("--by-ratio", action="store_true", help="report one block per trajectory-length ratio")
    ap.add_argument("--csv", default=None, help="also write the per-case rows here")
    args = ap.parse_args()

    gt_dir = args.gt or (Path(__file__).parent / "data" / "GT" / args.task)
    rows, skipped, n_gt = collect(args.pred, gt_dir, args.task, args.tolerance_cm, args.ratio)
    if not rows:
        print(f"no scorable predictions in {args.pred} (ground truth: {gt_dir})")
        for name, why in skipped[:10]:
            print(f"  skipped {name}: {why}")
        return 1

    print(f"predictions: {args.pred}\nground truth: {gt_dir}"
          f"\npixel = {CM_PER_EVAL_PIXEL:g} cm · F1 tolerance = {args.tolerance_cm:g} cm")
    if args.by_ratio:
        for ratio in sorted({r["ratio"] for r in rows}):
            report([r for r in rows if r["ratio"] == ratio], args.task, f"ratio {ratio}")
    else:
        report(rows, args.task, Path(args.pred).name)

    cases = {r["case"] for r in rows}
    if len(cases) < n_gt:
        print(f"\nWARNING: scored {len(cases)} of the {n_gt} annotated cases -- the paper's "
              f"averages are over all of them.")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (r["case"], r["ratio"])))
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
