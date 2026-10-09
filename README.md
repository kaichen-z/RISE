<div align="center">
<h1>RISE: Single Static Radar-based Indoor Scene Understanding</h1>

Indoor layout reconstruction and object detection from **one static mmWave radar**,
by treating multipath ghosts as geometry.

<a href="https://arxiv.org/abs/2511.14019"><img src="https://img.shields.io/badge/arXiv-2511.14019-b31b1b" alt="arXiv"></a>
<a href="https://rise-cvpr.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://huggingface.co/datasets/kaichen-z/RISE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-RISE--data--4scenes-yellow" alt="Dataset"></a>

**[Media Lab, MIT](https://www.media.mit.edu/)**; **Cartesian Systems**

[Kaichen Zhou](https://kaichen-z.github.io/), Laura Dodds, Sayed Saad Afzal, [Fadel Adib](https://www.media.mit.edu/people/fadel/overview/)
</div>

```bibtex
@inproceedings{zhou2026rise,
  title={Rise: single static radar-based indoor scene understanding},
  author={Zhou, Kaichen and Dodds, Laura and Afzal, Sayed Saad and Adib, Fadel},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={32194--32205},
  year={2026}
}
```

## Overview

A static radar sees only a small part of a room directly. RISE turns the rest of the room —
visible only through multipath — into geometry:

- **BAME** (Bi-Angular Multipath Enhancement) resolves each reflection in **both** angle of
  arrival and angle of departure, which makes ghost targets observable where a conventional
  range–angle map loses them, and inverts each ghost/real pair into a reflector hypothesis.
- **SRHD** (Sim2Real Hierarchical Diffusion) decodes the accumulated reflector map into a
  complete scene in two stages — objects first, then layout conditioned on those objects —
  trained purely on simulated floorplans.

Reported results: **16 cm** mean Chamfer distance on layout (−60 % vs the prior mmWave
baseline) and **58 % IoU** on object detection, over 100 real trajectories / 50,000 frames.

## Repository layout

The pipeline is six stages; each is a directory with a single entry point.

| # | Stage | Entry point | Language |
|---|-------|-------------|----------|
| 1 | Data capture | `1_capture/Cascade_Capture.lua` | Lua (mmWave Studio) |
| 2 | Raw → dictionary preprocessing | `2_preprocess/main/cascade/Pre_cascade_MIMO_signalProcessing.m` | MATLAB |
| 3 | **BAME** — Range FFT + AoA/AoD FFT + 3D CFAR + ghost re-integration | `3_bame/run.py` | Python |
| 4 | Multipath inversion + temporal accumulation + GMM/RANSAC | `4_inversion/run.py` | Python |
| 5 | **SRHD** — two-stage feed-forward diffusion + metrics | `5_diffusion/{train,eval}.sh`, `evaluate.py` | Python |
| 6 | Simulator + training-data generation | `6_simulator/run_sim.sh`, `run_augment.sh` | Python |

```
common/     shared library: config.py (paths, scene table, map scale), function/, utils/, background/
tools/      pack_frames.py (MATLAB .mat -> python .npy) + figure utilities (recolouring, video)
```

Data flows `1 → 2 → 3 → 4 → 5`, with **6 feeding 5**: the diffusion model trains on simulated
data and is evaluated on the real maps exported by stage 4.

## Quick start

```bash
conda env create -f environment.yaml       # or: pip install -r requirements.txt
export RISE_DATA_ROOT=/path/to/RISE-data-4scenes   # the released radar package (see Data below)
export RISE_DEVICE=cuda                    # cuda | mps | cpu (default: cuda if available)
```

Train and evaluate SRHD without any radar data at all — it is a closed loop over simulated
floorplans:

```bash
cd 6_simulator && ./run_sim.sh && ./run_augment.sh
cd ../5_diffusion && ./train.sh && ./eval.sh
```

## Data

See [DATA.md](DATA.md) for the capture format, what the pipeline actually reads (1 % of each
stored frame), and the size of the release. Scenes are addressed by a *case id*
from the table in `common/config.py`; the released package contains these four:

| case | scene | trajectories | background capture |
|------|-------|--------------|--------------------|
| 19 | `Raw_Data_3/scene3_set2_r{1,2,3}` | 3 | `scene3_set2_bkg` |
| 26 | `Raw_Data_5/scene5_set1_r{1,2,3}` | 3 | `scene5_set1_bkg` |
| 42 | `Raw_Data_9/scene9_set1_r{1,2,3}` | 3 | `scene9_set1_bkg` |
| 50 | `Raw_Data_11/scene11_set1_r{1,2,3}` | 3 | `scene11_set1_bkg` |

The released radar package is hosted on Hugging Face:
**https://huggingface.co/datasets/kaichen-z/RISE** (`RISE-data-4scenes.zip`, 1.6 GB).

```bash
hf download kaichen-z/RISE RISE-data-4scenes.zip --repo-type dataset --local-dir .
unzip RISE-data-4scenes.zip && export RISE_DATA_ROOT=$PWD/RISE-data-4scenes
```

It contains cases **19, 26, 42, 50**: every
one of them has furniture in the ground truth. Each scene ships its three trajectory captures
in full and its empty-room background capture subsampled 1-in-30 (the `"every": 30` entry in
`preprocess/python_frames.json` tells `3_bame` not to subsample again). Frames are the
`bame` profile described in [DATA.md](DATA.md); they were packed with
`tools/pack_bame.pl`, a dependency-free Perl equivalent of `tools/pack_frames.py --profile bame`
(byte-identical output).

This is a subset of the benchmark; the remaining annotated cases are not part of this release.

## Running the signal pipeline

```bash
# stage 3 -- per-frame detection (ours, then the MVDR baseline)
cd 3_bame
python run.py --case 50                     # writes <traj>/bame/...
python run.py --case 50 --method mvdr       # writes <traj>/mvdr_M6, mvdr_M7, mvdr_M8

# stage 4 -- inversion + layout
cd ../4_inversion
python run.py --case 50 --step associate                 # ghost/real pairs (ours)
python run.py --case 50 --step associate --method mvdr   # baseline
python run.py --case 50 --step layout                    # layout + diffusion inputs
python run.py --case 50 --step layout --ratio 60         # shorter trajectory (paper Fig. 9)
python run.py --case 50 --step layout --mode baseline    # baseline layout
```

Every entry point supports `--help`; stage 3 accepts `--max-frames` and both stages accept
`--traj` for quick smoke tests.

## Generating the training data (simulator)

SRHD is trained on simulated scenes only; no radar data is needed. `6_simulator` builds the
training set from the public HouseExpo floorplans in two steps.

```bash
cd 6_simulator
tar xzf data/HouseExpo/json.tar.gz -C data/HouseExpo/      # once: 35k floorplans -> data/HouseExpo/json/

# step 1 -- simulate: pick a floorplan, add a random obstacle, place the sensor, ray-cast what it sees
RISE_SIM_N=100000 ./run_sim.sh                            # -> out/scenes/dict_<i>_.npy (600x600 maps)

# step 2 -- augment: random rotation / missing segments / noise, downsample to 128x128, bit-pack
RISE_AUG_N=100000 ./run_augment.sh                        # -> out/augmented/dict_<i>_.npy  (training set)
RISE_AUG_SPLIT=test RISE_AUG_N=5000 ./run_augment.sh      # -> out/augmented_test/           (held-out set)
```

| variable | default | meaning |
|----------|---------|---------|
| `RISE_SIM_N` | 100000 | number of floorplans to simulate |
| `RISE_SIM_OUT` / `RISE_SIM_IMG` | `out/scenes` / `out/scenes_img` | simulator output and debug mosaics |
| `RISE_SIM_IMG_FREQ` | 200 | write a debug mosaic every N scenes |
| `RISE_AUG_N` | 100000 (train) / 5000 (test) | number of augmented samples |
| `RISE_AUG_SPLIT` | `train` | `train` or `test` |
| `RISE_AUG_OUT` / `RISE_AUG_IMG` | `out/augmented[_test]` | augmented bundles and preview PNGs |

Sensor and scene parameters (range, field of view, angular resolution, noise, obstacle size)
live in `6_simulator/simulator/envs/config/config_exploration.yaml`.

Each simulated scene stores six bit-packed 600×600 maps (`image_gt_obs`, `image_gt_wall`,
`image_ob_obs`, `image_ob_wall`, `image_ob_whole`, `image_gt_wall_ori`). Each augmented bundle
stores the four 128×128 maps the trainer reads: `image_ob_obs` and `image_aug2_line_` (the
observed objects and the augmented, partial walls — together the model input) and
`image_gt_obs`, `image_gt_wall` (the targets). A few scenes are enough for a smoke test
(`RISE_SIM_N=10 RISE_AUG_N=20`); the paper used 100k.

`5_diffusion/train.sh` reads the augmented folders through `RISE_SIM_DATA` and `RISE_SIM_TEST`
(defaults: `6_simulator/out/augmented` and `6_simulator/out/augmented_test`).

## Training

```bash
cd 5_diffusion && ./train.sh
```

`train.sh` trains stage 1 (objects: `p(objects | observation)`) then stage 2 (layout:
`p(layout | observation, objects)`) into `ddim3/logs/stage1_object/` and
`ddim3/logs/stage2_layout/`. Configs are `ddim3/configs/nghost_diff{1,2}.txt`; the data comes
from `6_simulator`. `T` and `ch_mult` are baked into the weights — never mix configs between
training and evaluation.

## Evaluation

```bash
cd 5_diffusion
./eval.sh                                          # DDIM sampling + scoring (Table 2: "RISE w/ G, D")
RISE_EVAL_EXTRA="--ddim_steps=20" ./eval.sh        # 5x faster, slightly coarser
```

`eval.sh` samples with `ddim3/custom_eval.py` and then scores with `evaluate.py`, which is the
metric implementation behind the paper's numbers:

```bash
python evaluate.py --task obs  --pred eval_output/obs      # IoU / Dice          (Table 1)
python evaluate.py --task wall --pred eval_output/wall     # Chamfer cm / F1@15cm (Fig. 7)
```
