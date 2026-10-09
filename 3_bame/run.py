#!/usr/bin/env python3
"""Stage 3 -- per-frame radar detection.

    python run.py --case 50                 # BAME (ours)
    python run.py --case 50 --method mvdr   # MVDR baseline, TX subarrays 6/7/8

Reads   $RISE_DATA_ROOT/<scene>/<traj>/preprocess/     (written by 2_preprocess)
Writes  $RISE_DATA_ROOT/<scene>/<traj>/<tag>/image_batch_diagonal_accelerating_torch/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import BAME_TAG, DATA_ROOT, MVDR_TXS, SCENES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=int, required=True, choices=sorted(SCENES),
                    help="scene id (see common/config.py)")
    ap.add_argument("--method", choices=["bame", "mvdr"], default="bame",
                    help="bame = ours (AoA+AoD, ghost re-integration); mvdr = baseline")
    ap.add_argument("--traj", type=int, nargs="*", default=None,
                    help="trajectory indices (default: all of the scene)")
    ap.add_argument("--tag", default=None,
                    help="output sub-directory name (default: 'bame' / 'mvdr_M<tx>')")
    ap.add_argument("--tx", type=int, nargs="*", default=list(MVDR_TXS),
                    help="MVDR only: TX subarrays to beamform")
    ap.add_argument("--batch", type=int, default=1, help="frames averaged per detection")
    ap.add_argument("--visual-freq", type=int, default=10, help="0 disables debug images")
    ap.add_argument("--start-index", type=int, default=0, help="resume from this batch index")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N batches (smoke test)")
    args = ap.parse_args()

    print(f"[rise] data root: {DATA_ROOT}")
    if args.method == "bame":
        import bame
        bame.run(args.case, args.traj, tag=args.tag or BAME_TAG, visual_freq=args.visual_freq,
                 batch=args.batch, start_index=args.start_index, max_frames=args.max_frames)
    else:
        import mvdr
        mvdr.run(args.case, args.traj, txs=args.tx, visual_freq=args.visual_freq,
                 batch=args.batch, start_index=args.start_index, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
