# 6 — Simulator and training-data generation

Generates the sim2real training set for stage 5 by ray-tracing radar observations over
HouseExpo floorplans and augmenting them.

```bash
./run_sim.sh        # simulator/envs/main.py             -> rgb images + point-cloud .npy
./run_augment.sh    # simulator/envs/data_augmentation.py -> packed .npy bundles for training
```

Output folders are set at the top of `simulator/envs/main.py` (`image_address`, `npy_address`)
and `simulator/envs/data_augmentation.py` (`pre_address`, `save_np`, `save_img`). Point
`5_diffusion/train.sh` (`RISE_SIM_DATA`) at the augmented folder.

`data/HouseExpo/` is the public HouseExpo floorplan dataset (35k JSON layouts). Only
`json.tar.gz` is tracked — extract it once before the first run:

```bash
tar xzf data/HouseExpo/json.tar.gz -C data/HouseExpo/
```

It — cite it and
check its licence before publishing. `simulator/envs/simulator/pseudoSlam.py` is the ray-cast
observation model; `config/config_exploration.yaml` holds the sensor parameters.

Each augmented bundle stores four bit-packed 128×128 maps: `image_ob_obs`, `image_aug2_line_`
(observed objects / walls) and `image_gt_obs`, `image_gt_wall` (ground truth) — the exact
fields `5_diffusion/ddim3/data/data.py` unpacks.
