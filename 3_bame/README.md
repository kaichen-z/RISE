# 3 — BAME: per-frame detection

Range FFT → AoA/AoD FFT → 3D CFAR → ghost re-integration → DBSCAN, in one pass per frame.

```bash
python run.py --case 50                      # BAME (ours)
python run.py --case 50 --method mvdr        # MVDR baseline, TX subarrays 6/7/8
python run.py --case 50 --traj 1 --max-frames 2 --visual-freq 1   # smoke test
```

| file | was |
|------|-----|
| `bame.py` | `signal/working1_all.py` |
| `mvdr.py` | `signal/working0_all.py` |
| `run.py`  | the two `__main__` blocks, merged behind one CLI |

Input  `$RISE_DATA_ROOT/<scene><traj>/preprocess/`
Output `$RISE_DATA_ROOT/<scene><traj>/<tag>/image_batch_diagonal_accelerating_torch/combine_*.pth`
where `<tag>` is `bame` or `mvdr_M<tx>`, plus debug mosaics under `..._img/`.

BAME additionally averages the empty-room capture (`<scene>_bkg`) and subtracts it; the
positive residual feeds the 3D CFAR, the negative residual is kept as the "behind the
reflector" evidence used in stage 4.

**Hardware.** The 3D CFAR convolves a 256³ volume with a 13³ kernel. Set `RISE_DEVICE=cuda`
(or `mps` on Apple silicon, ~21 s/frame). On CPU it needs >100 GB of RAM.
