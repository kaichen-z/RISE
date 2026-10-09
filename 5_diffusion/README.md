# 5 — SRHD: two-stage hierarchical diffusion (feed-forward)

No pretrained weights are shipped. Train first, then evaluate:

```bash
./train.sh    # stage 1 (objects) then stage 2 (layout); both resume automatically
./eval.sh     # two-stage DDIM sampling on the real maps from 4_inversion, then scoring
```

Scoring is `evaluate.py` — IoU / Dice for objects, Chamfer (cm) / F1 @ 15 cm for layout:

```bash
python evaluate.py --task obs  --pred eval_output/obs
python evaluate.py --task wall --pred eval_output/wall --by-ratio     # trajectory-length sweep
```

| stage | config | model | writes |
|-------|--------|-------|--------|
| 1 — objects | `ddim3/configs/nghost_diff1.txt` | p(gt_obs \| observation) | `ddim3/logs/stage1_object/{<step>,last}_ckpt.pt` |
| 2 — layout  | `ddim3/configs/nghost_diff2.txt` | p(gt_wall \| observation, gt_obs) | `ddim3/logs/stage2_layout/{<step>,last}_ckpt.pt` |
| eval        | `ddim3/configs/eval_ddim.txt`    | both, chained | `eval_output/{obs,wall}/ourdiff_<case>_<ratio>.png` + `metrics_{obs,wall}.csv` |

Stage 2 is a *different* UNet than stage 1 (it takes a second conditioning image), so each
stage gets its own sampler; feeding `cond2` to the stage-1 model silently drops it.

All three configs use the same UNet (`ch=128`, `ch_mult=1,2,2,2`, `attn=1`, `num_res_blocks=2`,
`T=500`, 128×128). Keep them in step: `T` and `ch_mult` are baked into the weights, so a model
trained with one setting cannot be evaluated with another.

## Files

```
ddim3/common_flags.py   every flag, the device/precision helpers, the checkpoint reader
ddim3/main.py           training (+ IoU/Dice on held-out simulated data via --eval_ddim)
ddim3/custom_eval.py    two-stage feed-forward inference on the real maps
ddim3/models/           UNet, diffusion trainer + DDIM sampler, metrics
ddim3/data/data.py      NumpyDataset (simulated bundles), CustomDataset (real PNG maps)
evaluate.py             the scoring script behind the paper's tables
```

## Speed and memory knobs

| flag | default | note |
|------|---------|------|
| `--precision` | `auto` | bf16 on Ampere+, fp16 on older CUDA, fp32 on CPU/MPS |
| `--batch_size` / `--accum_steps` | 24 / 1 | effective batch = product; use `8 / 3` on a 16 GB GPU |
| `--ddim_steps` | 20 train previews, 100 eval | inference cost is linear in this |
| `--compile` | off | `torch.compile` the UNet (CUDA, PyTorch ≥ 2.1) |
| `--channels_last` | on (CUDA only) | ignored elsewhere |
| `--cache_data` | on | keeps the bit-packed bundles in RAM; workers are then unnecessary |
| `--keep_last` | 3 | how many periodic checkpoints survive; `last_ckpt.pt` is always kept |
| `--parallel` | off | `DataParallel` across the visible GPUs |
| `RISE_DEVICE` | auto | `cuda:0` / `mps` / `cpu` |

Resuming is automatic: `train.sh` picks up `<logdir>/last_ckpt.pt` unless `--noresume`.

## Feed-forward only

The paper's optional inference-time refinement — optimising the latent noise `x_T` so the
reconstruction stops intersecting the human trajectory (Table 2, row "R") — is **not** part of
this release, and neither is the ancestral DDPM sampler. Everything here is a single forward
DDIM chain per stage.

## Paths are flags, not literals

```
--train_data / --test_data      training bundles from 6_simulator   (RISE_SIM_DATA / RISE_SIM_TEST)
--path_obs / --path_traj        real maps from 4_inversion          (RISE_EVAL_OBS / RISE_EVAL_TRAJ)
--path_gt_obs / --path_gt_wall  ground truth, shipped in data/GT/   (RISE_EVAL_GT_OBS / RISE_EVAL_GT_WALL)
--eval_out                      where predictions are written       (RISE_EVAL_OUT)
--ckpt_s1 / --ckpt_s2           trained checkpoints                 (RISE_CKPT_S1 / RISE_CKPT_S2)
```

`main.py` also scores a stage on held-out **simulated** data:

```bash
cd ddim3
python main.py --flagfile configs/nghost_diff1.txt --notrain_ddpm --eval_ddim \
       --test_data <augmented_test> --logdir ./logs/stage1_object --ckpt last_ckpt.pt
```

A real run wants a CUDA GPU; `RISE_DEVICE=cpu|mps` works for smoke tests.
