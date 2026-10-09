# 4 — Multipath inversion, temporal accumulation, GMM + RANSAC

```bash
python run.py --case 50 --step associate                  # ghost/real pairs (ours)
python run.py --case 50 --step associate --method mvdr    # baseline
python run.py --case 50 --step layout                     # layout + diffusion inputs
python run.py --case 50 --step layout --mode lines        # quick reflector fusion
python run.py --case 50 --step layout --mode baseline     # MVDR baseline layout
python run.py --case 50 --step layout --ratio 60          # first 60% of the trajectory
```

`--ratio` is the trajectory-length ablation of the paper (Fig. 9): it keeps the first N % of
the frame-ordered detections and writes `ours_<case>_<ratio>.png`, which `5_diffusion` reads
back as the `ratio_choice` of each map (`evaluate.py --by-ratio` groups by it).

| file | was |
|------|-----|
| `associate.py` | `signal/working3_all.py` (ours) + `working2_all.py` (baseline) |
| `layout.py`    | `signal/working5_ours.py` (export) + `working4_ours.py` (lines) + `working4_baseline.py` |
| `run.py`       | the four `__main__` blocks, merged behind one CLI |

**`--step associate`** pools every trajectory of the scene (temporal accumulation), denoises
the per-frame clusters, and inverts each first/second-order ghost into a (ghost, real) pair:

```
<scene><t0>_<tN>/<tag>/final_preprocess_denoise_traj_diagonal/
    ghost_ours_threeD_<strict>.npy   (BAME)
    ghost_ours.npy                   (MVDR baseline)
```

**`--step layout`** loads those bundles, turns every pair into a reflector hypothesis, clusters
them (GMM) and fits wall segments (RANSAC), then writes what stage 5 consumes:

```
result_ours/result_input/image/       full-resolution layout map
result_ours/result_input/image_lowr/  128x128 map  -> 5_diffusion --path_obs
result_ours/result_input/image_traj/  trajectory map -> 5_diffusion --path_traj
```

By default the "ours" layout fuses the BAME bundle with the three MVDR bundles, exactly as in
the original scripts; override with `--tags`.
