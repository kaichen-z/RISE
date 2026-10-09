"""Single source of truth for dataset paths, scene table and stage output naming.

Every python stage (3_bame, 4_inversion) reads its paths from here, so the data
root only has to be set once -- either by editing DATA_ROOT below or by exporting

    export RISE_DATA_ROOT=/path/to/RISE-data-4scenes

Layout on disk (as captured / preprocessed by 1_capture + 2_preprocess):

    $RISE_DATA_ROOT/
      Raw_Data_11/
        scene11_set1_bkg/preprocess/     empty-room reference frames
        scene11_set1_r1/preprocess/      trajectory 1  (original_mat_*.mat + original_hyp_*.json)
        scene11_set1_r2/preprocess/      trajectory 2
        scene11_set1_r3/preprocess/      trajectory 3

Stage outputs are written back next to the input trajectory:

    scene11_set1_r1/<tag>/image_batch_diagonal_accelerating_torch/combine_*.pth   (3_bame)
    scene11_set1_r1_3/<tag>/final_preprocess_denoise_traj_diagonal/ghost_*.npy    (4_inversion)

where <tag> is "bame" for the proposed pipeline and "mvdr_M6/7/8" for the
single-TX MVDR baselines.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = REPO_ROOT / "common"
BACKGROUND_DIR = COMMON_DIR / "background"

# Root of the released radar package (holds Raw_Data_3, Raw_Data_5, Raw_Data_9, Raw_Data_11).
DATA_ROOT = Path(os.environ.get("RISE_DATA_ROOT", "RISE-data-4scenes"))

# --------------------------------------------------------------------------- #
# Map geometry.  Everything downstream of 3_bame lives on a top-down grid whose
# scale is fixed by these three numbers -- they are what turns a pixel distance
# into the centimetres reported in the paper.
#
#   3_bame/bame.py:65   H = RATIO*TOTAL*RESOLUTION*2*y_axis = 2/3*15*50*2*0.6 = 600
#   4_inversion/associate.py:35  H, W = 600, 600
#   common/utils/utils_tch.py::low_resolution  600 -> 128 for the diffusion maps
#
PIXELS_PER_METER = 50          # 3_bame/bame.py:60  RESOLUTION
MAP_SIZE = 600                 # full-resolution top-down map, in pixels
EVAL_SIZE = 128                # diffusion / evaluation maps, in pixels
CM_PER_MAP_PIXEL = 100.0 / PIXELS_PER_METER                    # 2.0 cm
CM_PER_EVAL_PIXEL = CM_PER_MAP_PIXEL * MAP_SIZE / EVAL_SIZE    # 9.375 cm
F1_TOLERANCE_CM = 15.0         # paper Sec. 3, "F1-score at a 15 cm tolerance"

# Output-directory naming used by 4_inversion (kept from the original code base).
DENOISE_TYPES = ["no_denoise", "denoise_traj", "denoise_flower", "denoise_both"]
INPUT_TYPES = ["diagonal", "virtual_array", "whole"]
DENOISE_TYPE = 1  # fixed: "denoise_traj"
INPUT_TYPE = 0    # fixed: "diagonal"

# Tag of the proposed (ghost-enhanced) detector and of the MVDR baselines.
BAME_TAG = "bame"
MVDR_TXS = (6, 7, 8)          # single-TX subarrays used by the MVDR baseline


def mvdr_tag(tx: int) -> str:
    return f"mvdr_M{tx}"


@dataclass(frozen=True)
class Scene:
    """One recorded scene: N trajectories + one empty-room background capture."""

    case: int
    raw_dir: str                    # trajectory prefix, trajectory index is appended
    bkg_dir: Optional[str]          # empty-room capture (needed by BAME, not by MVDR)
    background_png: str             # reflector ground truth used for plots / evaluation
    trajectories: Tuple[int, ...] = (1, 2, 3)

    # ---------- input paths ----------
    def frame_dir(self, traj: int) -> Path:
        return DATA_ROOT / f"{self.raw_dir}{traj}" / "preprocess"

    def bkg_frame_dir(self) -> Path:
        if self.bkg_dir is None:
            raise ValueError(f"case {self.case} has no background capture registered")
        return DATA_ROOT / self.bkg_dir / "preprocess"

    def background_path(self) -> Path:
        return BACKGROUND_DIR / self.background_png

    # ---------- output paths ----------
    def detect_dir(self, traj: int, tag: str) -> Path:
        """Where 3_bame writes its per-frame detections."""
        return DATA_ROOT / f"{self.raw_dir}{traj}" / tag

    def assoc_dir(self, tag: str) -> Path:
        """Where 4_inversion writes the per-scene ghost/real association."""
        span = f"{self.trajectories[0]}_{self.trajectories[-1]}"
        sub = f"final_preprocess_{DENOISE_TYPES[DENOISE_TYPE]}_{INPUT_TYPES[INPUT_TYPE]}"
        return DATA_ROOT / f"{self.raw_dir}{span}" / tag / sub


SCENES = {
    19: Scene(19, "Raw_Data_3/scene3_set2_r", "Raw_Data_3/scene3_set2_bkg", "scene3_set2.png"),
    26: Scene(26, "Raw_Data_5/scene5_set1_r", "Raw_Data_5/scene5_set1_bkg", "scene5_set1.png"),
    42: Scene(42, "Raw_Data_9/scene9_set1_r", "Raw_Data_9/scene9_set1_bkg", "scene9_set1.png"),
    50: Scene(50, "Raw_Data_11/scene11_set1_r", "Raw_Data_11/scene11_set1_bkg", "scene7_set1.png"),
}


def get_scene(case: int) -> Scene:
    if case not in SCENES:
        raise KeyError(f"unknown case {case}; known cases: {sorted(SCENES)}")
    return SCENES[case]
