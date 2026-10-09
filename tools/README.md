# tools — figure utilities (not part of the pipeline)

| file | purpose |
|------|---------|
| `visualization.py` | recolour a saved viridis heat map into jet for the paper figures |
| `video_make.py`    | assemble the per-frame debug PNGs from stage 3 into a video |
| `assets/`          | figure assets (reflector overlay used in the paper) |

Both scripts still carry the absolute paths of the machine they were written on — edit the
paths at the top before running. `visualization.py` uses `Image.ANTIALIAS`, removed in
Pillow ≥ 10; replace with `Image.LANCZOS` if you run it on a modern Pillow.
