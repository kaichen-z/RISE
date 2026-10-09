#!/usr/bin/env python3
"""Stage 4 -- multipath inversion, temporal accumulation, GMM + RANSAC layout.

    python run.py --case 50 --step associate            # ghost/real pairs (ours)
    python run.py --case 50 --step associate --method mvdr
    python run.py --case 50 --step layout               # layout + diffusion inputs
    python run.py --case 50 --step layout --mode baseline

Reads   the per-frame detections written by 3_bame
Writes  <scene>_<trajspan>/<tag>/final_preprocess_denoise_traj_diagonal/ghost_*.npy
        and, for `--step layout`, result_ours/result_input/{image,image_lowr,image_traj}
        which is exactly what 5_diffusion consumes.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import BAME_TAG, DATA_ROOT, MVDR_TXS, SCENES, mvdr_tag  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=int, required=True, choices=sorted(SCENES))
    ap.add_argument("--step", choices=["associate", "layout"], default="associate")
    ap.add_argument("--method", choices=["bame", "mvdr"], default="bame",
                    help="associate only: which detector's output to consume")
    ap.add_argument("--mode", choices=["export", "lines", "baseline"], default="export",
                    help="layout only: export = full pipeline + diffusion inputs")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="override the detection sub-directory names")
    ap.add_argument("--traj", type=int, nargs="*", default=None)
    ap.add_argument("--strict-level", type=int, default=1, choices=[0, 1])
    ap.add_argument("--no-threed", action="store_true",
                    help="ignore the ghost-re-integrated (AoA != AoD) points")
    ap.add_argument("--out", default=None, help="layout only: output directory")
    ap.add_argument("--ratio", type=int, default=100,
                    help="layout export: percentage of the trajectory to use "
                         "(paper Fig. 9 sweeps 40..100); also the ratio in the output file name")
    ap.add_argument("--export", action="store_true",
                    help="layout baseline: also write the 128x128 diffusion inputs, so the "
                         "baseline layout can be fed through 5_diffusion (paper's EMT + SRHD row)")
    ap.add_argument("--visuals", action="store_true", help="layout export: also write debug plots")
    ap.add_argument("--check", action="store_true", help="associate: write denoise/cluster debug plots")
    args = ap.parse_args()

    print(f"[rise] data root: {DATA_ROOT}")
    use_threed = not args.no_threed
    if args.step == "associate":
        import associate
        associate.run(args.case, method=args.method, tags=args.tags, trajectories=args.traj,
                      strict_level=args.strict_level, use_threed=use_threed, check=args.check)
    else:
        import layout
        if args.mode == "baseline":
            layout.run_baseline(args.case, out_dir=args.out or "result_baseline", tags=args.tags,
                                export=args.export)
        elif args.mode == "lines":
            layout.run_lines(args.case, tags=args.tags, out_dir=args.out,
                             strict_level=args.strict_level, use_threed=use_threed)
        else:
            layout.run_export(args.case, tags=args.tags, out_root=args.out or "result_ours",
                              strict_level=args.strict_level, use_threed=use_threed,
                              visuals=args.visuals, ratio=args.ratio)


if __name__ == "__main__":
    main()
