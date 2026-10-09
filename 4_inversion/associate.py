"""Ghost/real association -- the first half of stage 4.

Consumes the per-frame detections written by 3_bame and produces, per scene,
one `ghost_*.npy` bundle holding first/second-order ghost pairs together with
their real-object partners.  This is the temporal-accumulation step: detections
from every trajectory of the scene are denoised over time and pooled.

Two detector front-ends are supported:

    method="bame"  (ours)      originally signal/working3_all.py
    method="mvdr"  (baseline)  originally signal/working2_all.py

Everything except the per-frame feature extraction is shared between the two.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.config import BAME_TAG, MVDR_TXS, get_scene, mvdr_tag
from common.function.denoise import *          # DENOISE_SPACE
from common.function.background_tch import bk_case_scene_bkg_refine
from common.function.geometry_tch import geometry_ghost_update
from common.function.geometry import geometry_ghost_update as geometry_ghost_numpy
from common.function.threeDpro_tch import cluster2point, checkSelfOverlap
from common.utils.utils_tch import overlap_check

H, W = 600, 600
MIN_GMM, MAX_GMM = 1, 20


class WORKSPACE:
    def __init__(self,):
        print('--------- denoise ---------')
        # -------------------- initializing geometry
        self.threshold, self.threshold2, self.sample, self.ratio_var = 0.9, 0.5, 30, 0.1
        self.geometry = geometry_ghost_update(self.threshold, self.threshold2, self.sample, self.ratio_var)
        self.geometry_numpy = geometry_ghost_numpy(self.threshold, self.threshold2, self.sample, self.ratio_var)
        self.H, self.W = H, W
        # ----- Dbscan_coord -----
        self.AREA = 2
        self.MIN_COUNT = 6
        self.RESOLUTION = 50


def draw_scatter(points, labels, sizes, address, H, W, alphas):
    plt.figure(figsize=(8, 6))
    for i, point in enumerate(points):
        plt.scatter(points[i][:, 1], points[i][:, 0], label=f'{labels[i]}', alpha=alphas[i], marker='v', s=sizes[i])
    plt.legend()
    plt.xlim(0, H)
    plt.ylim(0, W)
    plt.savefig(address, dpi=500, bbox_inches='tight')
    plt.close()


def list_detections(detect_dir):
    """Return (background dict, ordered list of per-frame .pth files)."""
    folder_path = Path(detect_dir) / 'image_batch_diagonal_accelerating_torch'
    pth_files = [f for f in folder_path.glob('combine_*.pth') if f.name != 'combine_bg.pth']
    sorted_files = sorted(pth_files, key=lambda p: int(p.stem.split('_')[1]))
    bg_file = folder_path / 'combine_bg.pth'
    try: bg_torch = torch.load(bg_file, weights_only=True)
    except Exception: bg_torch = torch.load(bg_file, weights_only=False)
    return bg_torch, sorted_files


def _stack(values):
    return np.concatenate(values, axis=0)


# --------------------------------------------------------------------------- #
# per-trajectory feature extraction                                            #
# --------------------------------------------------------------------------- #
def _features_mvdr(workpace, sorted_files, np_center):
    """MVDR baseline: one cluster set per frame, no ghost re-integration."""
    sor_mags, sor_means, sor_ranges, sor_thetas, sor_shapes, sor_nums, sor_labs = [], [], [], [], [], [], []
    for file in tqdm(sorted_files):
        file_torch = torch.load(file, weights_only=True)
        mvdr_array_clusters_xy = file_torch["mvdr_array_clusters_xy"]
        mvdr_array_labels_xy = file_torch["mvdr_array_labels_xy"]
        mvdr_array_mags = file_torch["mvdr_array_mags"]
        # ------------------- checking the distance
        distances_msk = torch.norm(mvdr_array_clusters_xy - np_center, dim=1) < 400
        workpace.geometry.threshold_select = 0.3
        if distances_msk.sum() > 0:
            sor_mag, sor_mean, sor_range, sor_theta, sor_shape, sor_num, sor_lab, _ = \
                workpace.geometry.frame_feature_analysis(np_center, mvdr_array_labels_xy, mvdr_array_clusters_xy, magnitudes=mvdr_array_mags)
            sor_mags.append(sor_mag.numpy()[None, :]), sor_means.append(sor_mean.numpy()[None, :]), sor_ranges.append(sor_range.numpy()[None, :])
            sor_thetas.append(sor_theta.numpy()[None, :]), sor_shapes.append(sor_shape.numpy()[None, :]), sor_nums.append(sor_num.numpy()[None, :])
            sor_labs.append(sor_lab.numpy()[None, :])
    sor = (sor_mags, sor_means, sor_ranges, sor_thetas, sor_shapes, sor_nums, sor_labs)
    return sor, [], []


def _features_bame(workpace, sorted_files, np_center, use_threed=True):
    """BAME: diagonal clusters + the AoA != AoD points recovered by ghost re-integration."""
    sor_mags, sor_means, sor_ranges, sor_thetas, sor_shapes, sor_nums, sor_labs = [], [], [], [], [], [], []
    sel_mags, sel_means, sel_ranges, sel_thetas, sel_shapes, sel_nums, sel_labs = [], [], [], [], [], [], []
    neg_clusters, neg_mags = [], []
    for file in tqdm(sorted_files):
        file_torch = torch.load(file, weights_only=True)
        # ------------------- reading negative information
        neg_diagonal_clusters_xy, neg_diagonal_labels_xy = file_torch["neg_diagonal_clusters_xy"], file_torch["neg_diagonal_labels_xy"]
        neg_diagonal_mags = file_torch["neg_diagonal_mags"]
        # ------------------- checking the distance
        distances_msk_dia = torch.norm(file_torch["diagonal_clusters_xy"] - np_center, dim=1) < 400
        if distances_msk_dia.sum() > 0:
            # ------------------- loading diagonal information
            diagonal_clusters_xy, diagonal_labels_xy = file_torch["diagonal_clusters_xy"][distances_msk_dia], file_torch["diagonal_labels_xy"][distances_msk_dia]
            diagonal_mags = file_torch["diagonal_mags"][distances_msk_dia]
            # ------------------- loading 3D convolutional based information
            threed_clusters_final_xy, threed_labels_final_xy = file_torch["threed_clusters_final_xy"], file_torch["threed_labels_final_xy"]
            threed_mags_final_3d = file_torch["threed_labels_mags"]
            # ------------------- check the case aoa != aod
            if threed_labels_final_xy is not None:
                aod_msk_fin = threed_labels_final_xy >= 500
                threed_clusters = threed_clusters_final_xy[aod_msk_fin]; threed_labels = threed_labels_final_xy[aod_msk_fin]
                threed_mags = threed_mags_final_3d[aod_msk_fin]
                if len(threed_labels) > 0:
                    select_cluster, select_labels, select_mask = overlap_check(file_torch["diagonal_clusters_xy"], threed_clusters, threed_labels, 10)
                    select_mag = threed_mags[select_mask]
                else: select_mag = []
            else: select_mag = []
            # ------------------- Combining all negative pairs
            neg_means = workpace.geometry.combine(neg_diagonal_labels_xy, neg_diagonal_clusters_xy); neg_clusters += neg_means
            neg_mag_cur = workpace.geometry.combine(neg_diagonal_labels_xy, neg_diagonal_mags); neg_mags += neg_mag_cur
            # ------------------- diagonal
            sor_mag, sor_mean, sor_range, sor_theta, sor_shape, sor_num, sor_lab, _ = \
                    workpace.geometry.frame_feature_analysis(np_center, diagonal_labels_xy, diagonal_clusters_xy, magnitudes=diagonal_mags)
            sor_mags.append(sor_mag.numpy()[None,:]), sor_means.append(sor_mean.numpy()[None,:]), sor_ranges.append(sor_range.numpy()[None,:])
            sor_thetas.append(sor_theta.numpy()[None,:]), sor_shapes.append(sor_shape.numpy()[None,:]), sor_nums.append(sor_num.numpy()[None,:])
            sor_labs.append(sor_lab.numpy()[None,:])
            # ------------------- Data - convert cluster to points, mag - keeping max and delete repeated points
            sel_mag = []
            if len(select_mag) > 0: sel_mag, sel_mean, sel_range, sel_theta, sel_shape, sel_num, sel_lab  = cluster2point(select_cluster, select_labels, select_mag, np_center)
            if len(sel_mag) > 1:
                Ov_msk  = checkSelfOverlap(sel_mean, sel_mag)
                sel_mag, sel_mean, sel_range, sel_theta, sel_shape, sel_num, sel_lab = \
                    sel_mag[Ov_msk], sel_mean[Ov_msk], sel_range[Ov_msk], sel_theta[Ov_msk], sel_shape[Ov_msk], sel_num[Ov_msk], sel_lab[Ov_msk]
            # ------------------- Data - Delete all points sharing same angle but smaller range; keep the closest points.
            if len(sel_mag) > 0:
                sel_mags.append(sel_mag.numpy()[None,:]), sel_means.append(sel_mean.numpy()[None,:]), sel_ranges.append(sel_range.numpy()[None,:]), \
                    sel_thetas.append(sel_theta.numpy()[None,:]), sel_shapes.append(sel_shape.numpy()[None,:]), sel_nums.append(sel_num.numpy()[None,:]), \
                        sel_labs.append(sel_lab.numpy()[None,:])
            else:
                sel_mags.append([]), sel_means.append([]), sel_ranges.append([]), \
                    sel_thetas.append([]), sel_shapes.append([]), sel_nums.append([]), sel_labs.append([])
    # ---------- Dealing with the negative (behind-the-reflector) part
    neg_clusters = np.stack([clu.numpy() for clu in neg_clusters])
    neg_mags = np.stack([clu.numpy() for clu in neg_mags])
    # ---------- Merging the recovered aoa != aod points into the diagonal ones
    sor = [sor_mags, sor_means, sor_ranges, sor_thetas, sor_shapes, sor_nums, sor_labs]
    sel = [sel_mags, sel_means, sel_ranges, sel_thetas, sel_shapes, sel_nums, sel_labs]
    if use_threed:
        for i in range(len(sor_mags)):
            if len(sel[0][i]) > 0:
                for k in range(len(sor)):
                    sor[k][i] = np.concatenate((sor[k][i], sel[k][i]), axis=1)
    return tuple(sor), neg_clusters, neg_mags


# --------------------------------------------------------------------------- #
# multipath inversion over the accumulated frames                              #
# --------------------------------------------------------------------------- #
def _invert(workpace, sor1, method, strict_level):
    """Turn per-frame detections into (ghost, real) pairs -- the multipath inversion."""
    sor_mags1, sor_means1, sor_ranges1, sor_thetas1, sor_shapes1, sor_nums1, sor_labs1 = sor1
    pghost1_means, pghost1_reals, pghost1_magts = [], [], []
    pghost1pr_means, pghost1pr_reals, pghost1pr_magts = [], [], []
    pghost2pr_means, pghost2pr_reals, pghost2pr_magts = [], [], []
    for num in range(len(sor_mags1)):
        args = (sor_mags1[num][0], sor_means1[num][0], sor_ranges1[num][0], sor_thetas1[num][0],
                sor_shapes1[num][0], sor_nums1[num][0], sor_labs1[num][0])
        if method == "mvdr":
            # baseline keeps the torch implementation and always uses strict level 1
            out = workpace.geometry.post_process_strict_level1(*args)
        elif strict_level == 0:
            out = workpace.geometry_numpy.post_process(*args)
        else:
            out = workpace.geometry_numpy.post_process_strict_level1(*args)
        (pghost1_mean, _, _, pghost1_mag_t, _, pghost1_real,
         pghost1pr_mean, _, _, pghost1pr_mag_t, _, pghost1pr_real,
         pghost2pr_mean, _, _, pghost2pr_mag_t, _, pghost2pr_real) = out
        if method == "mvdr":
            if len(pghost1_mean) > 0:
                pghost1_means.append(pghost1_mean); pghost1_reals.append(pghost1_real); pghost1_magts.append(pghost1_mag_t)
                pghost1pr_means.append(pghost1pr_mean); pghost1pr_reals.append(pghost1pr_real); pghost1pr_magts.append(pghost1pr_mag_t)
            if len(pghost2pr_mean) > 0:
                pghost2pr_means.append(pghost2pr_mean); pghost2pr_reals.append(pghost2pr_real); pghost2pr_magts.append(pghost2pr_mag_t)
        else:
            # For one pghost1 with multiple pghost1pr
            pghost1_means.append(pghost1_mean); pghost1_reals.append(pghost1_real); pghost1_magts.append(pghost1_mag_t)
            pghost1pr_means.append(pghost1pr_mean); pghost1pr_reals.append(pghost1pr_real); pghost1pr_magts.append(pghost1pr_mag_t)
            if pghost2pr_mean is not None:
                pghost2pr_means.append(pghost2pr_mean); pghost2pr_reals.append(pghost2pr_real); pghost2pr_magts.append(pghost2pr_mag_t)
    return (pghost1_means, pghost1_reals, pghost1_magts,
            pghost1pr_means, pghost1pr_reals, pghost1pr_magts,
            pghost2pr_means, pghost2pr_reals, pghost2pr_magts)


def run(case, method=BAME_TAG, tags=None, trajectories=None, strict_level=1,
        use_threed=True, check=False):
    """Associate ghosts with real objects for one scene and store the bundle."""
    method = "bame" if method in ("bame", BAME_TAG) else "mvdr"
    scene = get_scene(case)
    trajectories = trajectories or scene.trajectories
    if tags is None:
        tags = [BAME_TAG] if method == "bame" else [mvdr_tag(tx) for tx in MVDR_TXS]

    workpace = WORKSPACE()
    denoise_space = DENOISE_SPACE()
    saved = []

    for tag in tags:
        trajectories_list = []
        pghost1_means_list, pghost1_reals_list, pghost1_magt_list = [], [], []
        pghost1pr_means_list, pghost1pr_reals_list, pghost1pr_magt_list = [], [], []
        pghost2pr_means_list, pghost2pr_reals_list, pghost2pr_magt_list = [], [], []
        neg_clusters_list, neg_mags_list = [], []

        prefix_save = scene.assoc_dir(tag)
        prefix_save.mkdir(parents=True, exist_ok=True)
        try: subprocess.run(f"rm {prefix_save}/*.png", shell=True, check=True)
        except Exception: print('===== Remove all images =====')

        for traj in trajectories:
            detect_dir = scene.detect_dir(traj, tag)
            print('======= Reading detections from =======', detect_dir)
            bg_torch, sorted_files = list_detections(detect_dir)
            np_center = bg_torch["center"]
            workpace.np_center = np_center
            # ----------- Initialize background points (reflector ground truth).
            bk_class = bk_case_scene_bkg_refine(str(scene.background_path()))
            workpace.bg_points = bk_class.bg_points
            # ----------------------------------------------- Loading moving
            if method == "mvdr":
                sor, neg_clusters, neg_mags = _features_mvdr(workpace, sorted_files, np_center)
            else:
                sor, neg_clusters, neg_mags = _features_bame(workpace, sorted_files, np_center, use_threed)
                neg_clusters_list.append(neg_clusters); neg_mags_list.append(neg_mags)
            # ---------- Denoising1 (eliminate object or reordering all cluster)
            sor1 = denoise_space.denoise_current_ndoppler(*sor)
            if check:
                before = np.array([s[0, 0] for s in sor[1]]); after = np.array([s[0, 0] for s in sor1[1]])
                os.makedirs("image4", exist_ok=True)
                draw_scatter([before, after, workpace.bg_points], ['PRE', 'EST', 'BG'], [10, 1, 2],
                             os.path.join("image4", f"case_{case}_range_{traj}_denoise.png"), H, W, [1, 1, 0.1])
            if method == "bame":
                trajectories_list.append(np.array([s[0, 0] for s in sor1[1]]))
            # ---------- Denoising2 / multipath inversion
            (p1_m, p1_r, p1_t, p1pr_m, p1pr_r, p1pr_t, p2pr_m, p2pr_r, p2pr_t) = \
                _invert(workpace, sor1, method, strict_level)
            if len(p1_m) == 0: continue
            pghost1_means_list.append(_stack(p1_m)); pghost1_reals_list.append(_stack(p1_r)); pghost1_magt_list.append(_stack(p1_t))
            pghost1pr_means_list.append(_stack(p1pr_m)); pghost1pr_reals_list.append(_stack(p1pr_r)); pghost1pr_magt_list.append(_stack(p1pr_t))
            if len(p2pr_m) == 0: continue
            pghost2pr_means_list.append(_stack(p2pr_m)); pghost2pr_reals_list.append(_stack(p2pr_r)); pghost2pr_magt_list.append(_stack(p2pr_t))

        if len(pghost1_means_list) == 0:
            raise RuntimeError(
                f"no ghost/real pair survived for case {case} tag '{tag}' -- the detections in "
                f"{[str(scene.detect_dir(t, tag)) for t in trajectories]} are empty or too short. "
                "Run 3_bame over the full trajectories first.")
        pghost1_means_list, pghost1_reals_list = _stack(pghost1_means_list), _stack(pghost1_reals_list)
        pghost1pr_means_list, pghost1pr_reals_list = _stack(pghost1pr_means_list), _stack(pghost1pr_reals_list)
        pghost1_magt_list = _stack(pghost1_magt_list); pghost1pr_magt_list = _stack(pghost1pr_magt_list)
        try:
            pghost2pr_means_list, pghost2pr_reals_list = _stack(pghost2pr_means_list), _stack(pghost2pr_reals_list)
            pghost2pr_magt_list = _stack(pghost2pr_magt_list)
        except Exception:
            pghost2pr_means_list, pghost2pr_reals_list, pghost2pr_magt_list = [], [], []
        if check:
            os.makedirs("image5", exist_ok=True)
            draw_scatter([pghost1pr_means_list, pghost2pr_means_list, workpace.bg_points], ['1pr', '2pr', 'BG'],
                         [1, 1, 2], os.path.join("image5", f"case_{case}_{tag}_cluster.png"), H, W, [1, 0.1, 0.1])

        dictionary = {"center": workpace.np_center.numpy(),
                      "_pghost1_means_list_": pghost1_means_list, "_pghost1pr_means_list_": pghost1pr_means_list,
                      "_pghost1pr_magt_list_": pghost1pr_magt_list, "_pghost2pr_means_list_": pghost2pr_means_list,
                      "_pghost1_reals_list_": pghost1_reals_list, "_pghost1_magt_list_": pghost1_magt_list,
                      "_pghost1pr_reals_list_": pghost1pr_reals_list, "_pghost2pr_reals_list_": pghost2pr_reals_list,
                      "_pghost2pr_magt_list_": pghost2pr_magt_list,
                      "min_gmm": MIN_GMM, "max_gmm": MAX_GMM, "H": H, "W": W,
                      "bg_points": workpace.bg_points, "prefix_save": str(prefix_save)}
        if method == "bame":
            dictionary["_trajectories_list_"] = _stack(trajectories_list)
            dictionary["neg_clusters_list"] = _stack(neg_clusters_list)
            dictionary["neg_mags_list"] = _stack(neg_mags_list)
            name = f'ghost_ours_threeD_{strict_level}.npy' if use_threed else 'ghost_ours.npy'
        else:
            name = 'ghost_ours.npy'
        out = prefix_save / name
        np.save(out, dictionary)
        print('===== saved =====', out)
        saved.append(out)
    return saved
