#!/usr/bin/env bash
# Augment the simulated scenes into the .npy bundles that 5_diffusion trains on.
# Input/output folders are set at the top of simulator/envs/data_augmentation.py.
set -euo pipefail
cd "$(dirname "$0")"
python -m simulator.envs.data_augmentation "$@"
