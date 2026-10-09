#!/usr/bin/env bash
# Train the two-stage hierarchical diffusion model (SRHD) from scratch.
#   stage 1: objects   p(gt_obs | observation)          -> ddim3/logs/stage1_object/
#   stage 2: layout    p(gt_wall | observation, gt_obs) -> ddim3/logs/stage2_layout/
# Data comes from 6_simulator (run_sim.sh -> run_augment.sh). No weights are shipped.
#
# Both stages resume automatically from <logdir>/last_ckpt.pt, so re-running this
# script after an interruption continues instead of restarting.
#
# Useful overrides:
#   RISE_DEVICE=cuda:0|mps|cpu      device
#   RISE_TRAIN_EXTRA="--precision=fp32 --batch_size=8 --accum_steps=3"
set -euo pipefail
cd "$(dirname "$0")/ddim3"
DATA=${RISE_SIM_DATA:-../../6_simulator/out/augmented}
TEST=${RISE_SIM_TEST:-../../6_simulator/out/augmented_test}

python main.py --flagfile configs/nghost_diff1.txt --train_data "$DATA" --test_data "$TEST" ${RISE_TRAIN_EXTRA:-}
python main.py --flagfile configs/nghost_diff2.txt --train_data "$DATA" --test_data "$TEST" ${RISE_TRAIN_EXTRA:-}
