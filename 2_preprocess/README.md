# 2 — Raw → dictionary preprocessing (MATLAB)

Turns the raw `*.bin` cascade captures from stage 1 into the per-frame "dictionary" files that
the python pipeline consumes.

```matlab
add_paths                                   % put modules/ and utils/ on the MATLAB path
edit main/cascade/input/testList.txt         % one line per capture folder + calibration file
run main/cascade/Pre_cascade_MIMO_signalProcessing.m
```

For every frame this writes into `<capture>/preprocess/`:

```
original_hyp_<file>_<frame>.json    radar hyper-parameters (chirp, bandwidth, antenna layout)
original_mat_<file>_<frame>.mat     calibrated ADC cube [range, doppler, rx, tx]
```

`3_bame` reads exactly this folder (`Scene.frame_dir()` in `common/config.py`).

Derived from Texas Instruments' `4chip_cascade_MIMO_example` — check its licence before
publishing. `cascade_MIMO_antennaCalib.m` / `cascade_TX_Phase_Calibration.m` generate the
calibration file referenced by `testList.txt`.
