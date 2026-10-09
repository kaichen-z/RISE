#!/usr/bin/env bash
# Simulate radar observations over HouseExpo floorplans.
# Output folders are set at the top of simulator/envs/main.py (image_address / npy_address).
set -euo pipefail
cd "$(dirname "$0")"
python -m simulator.envs.main "$@"
