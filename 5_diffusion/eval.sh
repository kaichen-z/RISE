#!/usr/bin/env bash
# Two-stage feed-forward DDIM inference on the real-scene maps exported by 4_inversion,
# followed by scoring against the ground truth (paper Table 1 / Fig. 7).
# Run train.sh first -- the checkpoints below are what it writes.
#
#   ./eval.sh                    sample + score
#   RISE_EVAL_EXTRA="--ddim_steps=20" ./eval.sh      faster, slightly coarser
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

OBS=${RISE_EVAL_OBS:-$HERE/../4_inversion/result_ours/result_input/image_lowr}
TRAJ=${RISE_EVAL_TRAJ:-$HERE/../4_inversion/result_ours/result_input/image_traj}
GT_OBS=${RISE_EVAL_GT_OBS:-$HERE/data/GT/obs}
GT_WALL=${RISE_EVAL_GT_WALL:-$HERE/data/GT/wall}
OUT=${RISE_EVAL_OUT:-$HERE/eval_output}
CKPT_S1=${RISE_CKPT_S1:-$HERE/ddim3/logs/stage1_object/last_ckpt.pt}
CKPT_S2=${RISE_CKPT_S2:-$HERE/ddim3/logs/stage2_layout/last_ckpt.pt}

# RISE_EVAL_EXTRA passes extra absl flags through (e.g. --ddim_steps, --batch_size).
(cd ddim3 && python custom_eval.py --flagfile configs/eval_ddim.txt \
  --ckpt_s1 "$CKPT_S1" --ckpt_s2 "$CKPT_S2" \
  --path_obs "$OBS" --path_traj "$TRAJ" \
  --path_gt_obs "$GT_OBS" --path_gt_wall "$GT_WALL" --eval_out "$OUT" ${RISE_EVAL_EXTRA:-})

# ---- score: object detection (IoU / Dice) and layout (Chamfer cm / F1 @ 15 cm)
python evaluate.py --task obs  --pred "$OUT/obs"  --gt "$GT_OBS"  --csv "$OUT/metrics_obs.csv"
python evaluate.py --task wall --pred "$OUT/wall" --gt "$GT_WALL" --csv "$OUT/metrics_wall.csv"
