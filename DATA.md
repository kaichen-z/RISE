# Data

What the RISE-Indoor benchmark looks like on disk, what the pipeline actually reads, and what
the release ships. **The released data starts *after* MATLAB**: everything a user needs is
python-native, so no MATLAB (and no MATLAB licence) is required to run the pipeline.

## The two halves of the dataset

```
capture  ──MATLAB (2_preprocess)──►  dictionary  ──python (3_bame …)──►  maps ──► metrics
 .bin                                 .mat/.npy
   ^                                     ^
   |                                     |
 provenance archive                  THIS is what is released
 (only needed to re-derive
  the dictionary)
```

| | raw capture | post-MATLAB dictionary (released) |
|---|---|---|
| files | `master_0000_data.bin`, `slave{1,2,3}_*.bin`, `*_idx.bin` | `preprocess/original_mat_<i>_<n>.npy` + `original_hyp_<i>_<n>.json` |
| needs | mmWave Studio capture + `calibrateResults_high.mat` + MATLAB | numpy only |
| size / trajectory (~350 frames) | 1.28 GiB | 2.06 GiB (`full`) or **132 MiB** (`bame`) |
| read by | nothing in this repository | `common/utils/utils_tch.py::load_matrix` |

`load_matrix` prefers `original_mat_*.npy` and falls back to `original_mat_*.mat`, so a
consumer of the release never touches MATLAB, while anyone holding the raw archive can still
regenerate everything.

## Producing the python-native frames

```bash
# whole cube, complex64            12.04 MB -> 6.02 MB per frame
python tools/pack_frames.py $RISE_DATA_ROOT/Raw_Data_11/scene11_set1_r1/preprocess --check

# BAME profile: doppler bin 0 only  12.04 MB -> 0.38 MB per frame
python tools/pack_frames.py $RISE_DATA_ROOT/Raw_Data_11 --profile bame --check
```

The `bame` profile is **exact**, not lossy, for the released pipeline: `3_bame/bame.py` slices
`matrix[:, :1]` before the range FFT, and the range FFT (with its per-column mean removal) acts
along the range axis independently for every doppler column — the dropped columns cannot
influence the output. The one exception is the **MVDR baseline** (`3_bame/mvdr.py`), which
builds a spatial covariance across all 16 doppler bins; ship `--profile full` if that row of
Table 1 has to be reproducible.

`--delete-mat` (requires `--check`) frees the MATLAB copies after verification.

## Frame format and what is actually read

One frame is a complex array of shape **(256 range, 16 doppler, 16 RX, 12 TX)** —
12.04 MB as MATLAB doubles, 6.02 MB as complex64.

The BAME path reads **1.04 %** of it: one doppler bin (`3_bame/bame.py`) and, inside
`common/function/fft_tch.py`, 8 of 16 RX and 4 of 12 TX — i.e. `(256, 1, 8, 4)` = 8,192 complex
values = 128 KiB per frame.

- The doppler axis is dead by construction (no Doppler FFT anywhere in the pipeline), except
  for the MVDR baseline.
- RX and TX are *not* optional: RX is the AoA axis and TX is the AoD axis — they are what BAME
  is. What goes unused is 160 of the 192 RX × TX channel pairs, selected geometrically so the
  remaining aperture stays uniformly spaced (a gapped aperture would give ambiguous lobes).
- `fft_tch.py::accelerate_power(..., full=True)` restores 16 RX × 9 TX; no call site uses it.

Measured funnel for one 350-frame trajectory:

```
raw .bin 1.28 GiB -> preprocess/ 4.02 GiB (.mat) / 2.06 GiB (.npy full) / 132 MiB (.npy bame)
                  -> read slice 43.8 MiB -> combine_*.pth 44 MB -> ghost_*.npy 88.6 KB
                  -> one 128x128 PNG
```

MATLAB writes the ADC cube as IEEE-754 doubles, which is why `preprocess/` used to be 3.2×
larger than the raw data it came from.

## Directory contract

`RISE_DATA_ROOT` (default `./RISE-data-4scenes`) holds `Raw_Data_3`, `Raw_Data_5`, `Raw_Data_9`
and `Raw_Data_11`. One capture
directory is one continuous recording:

```
$RISE_DATA_ROOT/
  Raw_Data_11/
    scene11_set1_bkg/                       empty-room reference capture
      preprocess/
        original_mat_<i>_<n>.npy            released frame (complex64)
        original_hyp_<i>_<n>.json           FFT window / bin parameters, 7.3 KB
        python_frames.json                  profile manifest written by pack_frames.py
      master_0000_data.bin  slave{1,2,3}_0000_data.bin   (raw archive, optional)
    scene11_set1_r1/ ... _r3/               trajectory captures, ~350 frames each
    set1.ply                                lidar ground truth for the scene
```

Stage outputs are written next to the input:

```
scene11_set1_r1/<tag>/image_batch_diagonal_accelerating_torch/combine_*.pth   (3_bame)
scene11_set1_r1_3/<tag>/final_preprocess_denoise_traj_diagonal/ghost_*.npy    (4_inversion)
```

where `<tag>` is `bame` for the proposed detector and `mvdr_M6/7/8` for the baselines.

Note for anyone re-deriving the dictionary from raw `.bin`:
**`2_preprocess/main/cascade/input/calibrateResults_high.mat` is board-specific and lives in
this repository, not with the data** — a raw-only archive without it is undecodable.

## Size tiers (python release, `--profile bame`)

| tier | goal | size |
|------|------|------|
| T0 | Train and evaluate SRHD (stages 5+6) | **0 B** of radar data — simulated floorplans only |
| T1 | Smoke-test the signal pipeline on one trajectory | ~140 MiB (one trajectory + a 1-in-30 background) |
| T2 | One complete scene configuration (`r1/r2/r3`) | ~400 MiB |
| T3 | **This release**: the 4 scenes of `RISE-data-4scenes.zip` (16 captures, 4,393 frames) | 1.6 GiB |
| T4 | The paper's quantitative tables (Raw_Data_3..12: 127 trajectories = 47,933 frames + 32 backgrounds at 1-in-30) | **~18 GiB** (`bame`) / ~290 GiB (`full`, MVDR included) |

Frame counts were measured on the capture archive (`ls preprocess | grep -c original_mat_`);
bytes = frames × (393,216 B `.npy` + 7.3 KB `.json`). Backgrounds add 4.5 GiB if shipped in full.

Backgrounds are read at 1-in-30 (`3_bame` averages every 30th frame), so a background capture
contributes ~13 frames regardless of its length. A subsampled
capture records `"every": 30` in its `python_frames.json`; `3_bame` reads that key and reduces
its own stride accordingly, so a shipped background is used in full. `tools/pack_bame.pl
--every 30 --out DIR` writes exactly this (it is a dependency-free Perl port of
`pack_frames.py --profile bame` for hosts without numpy; output is byte-identical).

Not included: pipeline outputs, pilot captures, raw `.bin` files and the MATLAB `.mat` cache.

## Ground truth in this repository

- `5_diffusion/data/GT/{obs,wall}/<case>_gt.png` — 47 cases each, 128×128, the evaluation
  ground truth. 24 of the 47 object maps are empty-room configurations with no furniture.
- `common/background/*.png` — reflector ground truth used for plot overlays.
- The scene-to-case mapping for cases outside `common/config.py::SCENES` is not part of this
  release.

## Training data (no radar needed)

`6_simulator` generates the SRHD training set from HouseExpo floorplans; `5_diffusion` reads
bit-packed `.npy` bundles (four 128×128 binary maps each, ~8 KB packed). The whole training set
is cached in RAM by the dataloader (`--cache_data`, on by default).
