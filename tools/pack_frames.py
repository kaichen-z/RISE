"""Convert MATLAB preprocess output into the python-native frames the pipeline reads.

2_preprocess (MATLAB) writes, per radar frame:

    original_mat_<interval>_<frame>.mat    complex double (256, 16, 16, 12)   12.04 MB
    original_hyp_<interval>_<frame>.json   FFT window / bin parameters         7.3 KB

Everything downstream is python.  This script rewrites the `.mat` cubes as `.npy`
complex64 next to the JSON files, so a released dataset needs no MATLAB (and no
MATLAB licence) to be consumed:

    python tools/pack_frames.py /path/to/Raw_Data_11/scene11_set1_r1/preprocess

Profiles
--------
--profile full     (default) whole cube as complex64          12.04 MB -> 6.02 MB / frame
--profile bame     doppler bin 0 only, shape (256, 1, 16, 12)          -> 0.38 MB / frame

`bame` is exact, not lossy, for the released pipeline: 3_bame/bame.py slices
`matrix[:, :1]` before the range FFT, and the range FFT + mean removal act along the
range axis independently per doppler column, so the discarded columns can never
influence the result.  The MVDR baseline (3_bame/mvdr.py) builds a covariance across
all 16 doppler bins -- keep `--profile full` if you want to reproduce that row.

Verification: `--check` re-reads each written file and compares it against the source
cube (max absolute error must be at the complex64 rounding level).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io

BYTES_PER_MB = 1024 * 1024


def load_mat_cube(path):
    data = scipy.io.loadmat(str(path))
    keys = [k for k in data.keys() if not k.startswith('__')]
    if not keys:
        raise ValueError(f"{path}: no variables")
    return data[keys[0]]


def convert_one(mat_path, profile, overwrite, check):
    npy_path = mat_path.with_suffix('.npy')
    if npy_path.exists() and not overwrite:
        return npy_path.stat().st_size, True
    cube = load_mat_cube(mat_path)
    if cube.ndim != 4:
        raise ValueError(f"{mat_path}: expected (range, doppler, rx, tx), got {cube.shape}")
    packed = cube[:, :1] if profile == 'bame' else cube
    packed = np.ascontiguousarray(packed.astype(np.complex64))
    np.save(npy_path, packed)
    if check:
        back = np.load(npy_path)
        reference = (cube[:, :1] if profile == 'bame' else cube).astype(np.complex64)
        error = np.abs(back - reference).max()
        if error > 0:
            raise AssertionError(f"{npy_path}: round-trip error {error}")
    return npy_path.stat().st_size, False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directories', nargs='+',
                        help='preprocess/ directories (or any parent -- they are searched)')
    parser.add_argument('--profile', choices=('full', 'bame'), default='full')
    parser.add_argument('--overwrite', action='store_true', help='rewrite existing .npy files')
    parser.add_argument('--check', action='store_true', help='verify every written file')
    parser.add_argument('--delete-mat', action='store_true',
                        help='remove the .mat once the .npy is written and verified')
    parser.add_argument('--limit', type=int, default=None, help='convert at most N frames per directory')
    args = parser.parse_args()

    if args.delete_mat and not args.check:
        parser.error("--delete-mat requires --check (never drop the source without verifying)")

    total_in = total_out = converted = skipped = 0
    for root in args.directories:
        mats = sorted(Path(root).rglob('original_mat_*.mat'))
        if not mats:
            print(f"{root}: no original_mat_*.mat found", file=sys.stderr)
            continue
        if args.limit:
            mats = mats[:args.limit]
        print(f"{root}: {len(mats)} frames -> .npy ({args.profile})")
        for mat_path in mats:
            size_in = mat_path.stat().st_size
            size_out, was_skipped = convert_one(mat_path, args.profile, args.overwrite, args.check)
            total_in += size_in
            total_out += size_out
            converted += 0 if was_skipped else 1
            skipped += 1 if was_skipped else 0
            if args.delete_mat and not was_skipped:
                mat_path.unlink()
        manifest = Path(root) / 'python_frames.json'
        manifest.write_text(json.dumps({'profile': args.profile, 'dtype': 'complex64',
                                        'frames': len(mats)}, indent=2))

    print(f"converted {converted} frames ({skipped} already present): "
          f"{total_in / BYTES_PER_MB:.0f} MB of .mat -> {total_out / BYTES_PER_MB:.0f} MB of .npy")


if __name__ == '__main__':
    main()
