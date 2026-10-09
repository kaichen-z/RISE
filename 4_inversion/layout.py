"""Reflector optimisation + layout estimation -- the second half of stage 4.

Loads the ghost/real bundles written by `associate.py`, turns every ghost-real
pair into a reflector hypothesis, and fits the room layout with GMM clustering
+ RANSAC line regression.

Three back-ends share the same front half:

    mode="export"    ours, full pipeline; also writes the diffusion inputs
                     (full-res / low-res / trajectory images)   [signal/working5_ours.py]
    mode="lines"     ours, quick GMM+RANSAC reflector fusion    [signal/working4_ours.py]
    mode="baseline"  MVDR baseline layout                       [signal/working4_baseline.py]
"""

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.config import BAME_TAG, MVDR_TXS, get_scene, mvdr_tag
from common.function.optimizer import *          # Optimizer_one_rel, mirror_of_P_over_lines, ...
from common.function.background_tch import bk_case_scene_bkg_refine
from common.function.denoise import *            # geometry_noise
from common.utils.utils_tch import low_resolution

VECTOR_KEYS = ["_pghost1_means_list_", "_pghost1pr_means_list_", "_pghost2pr_means_list_",
               "_pghost1_reals_list_", "_pghost1pr_reals_list_", "_pghost2pr_reals_list_"]
MAG_KEYS = ["_pghost1_magt_list_", "_pghost1pr_magt_list_", "_pghost2pr_magt_list_"]


def bundle_path(scene, tag, strict_level=1, use_threed=True):
    """`ghost_ours_threeD_<level>.npy` for the BAME bundle, `ghost_ours.npy` for MVDR."""
    d = scene.assoc_dir(tag)
    threed = d / f'ghost_ours_threeD_{strict_level}.npy'
    if use_threed and threed.exists():
        return threed
    return d / 'ghost_ours.npy'


def load_bundles(scene, tags, strict_level=1, use_threed=True):
    """Concatenate the per-tag association bundles into one dictionary."""
    stacked = {k: [] for k in VECTOR_KEYS + MAG_KEYS}
    meta = {}
    for tag in tags:
        address = bundle_path(scene, tag, strict_level, use_threed)
        if not address.exists() and tag != BAME_TAG:
            print(f"[rise] no {tag} bundle for case {scene.case} ({address}): skipped")
            continue
        dictionary = np.load(address, allow_pickle=True).item()
        for k in VECTOR_KEYS + MAG_KEYS:
            stacked[k].append(dictionary[k])
        meta.update({k: dictionary[k] for k in ("center", "min_gmm", "max_gmm", "H", "W", "bg_points")})
        if tag == BAME_TAG or "_trajectories_list_" in dictionary:
            # only the BAME bundle carries the trajectory and behind-reflector points
            meta["trajectories"] = dictionary["_trajectories_list_"]
            meta["neg_clusters_list"] = dictionary["neg_clusters_list"]
            meta["neg_mags_list"] = dictionary["neg_mags_list"]
        print("Number of Add -- --:", tag, "Number -- ++:", dictionary["_pghost1_reals_list_"].shape)
    out = dict(meta)
    for k in VECTOR_KEYS:
        out[k] = np.vstack(stacked[k])
    for k in MAG_KEYS:
        out[k] = np.concatenate(stacked[k])
    out.setdefault("trajectories", np.zeros((0, 2)))
    out.setdefault("neg_clusters_list", np.zeros((0, 2)))
    out.setdefault("neg_mags_list", np.zeros((0,)))
    return out


def _density_reference(densities):
    """Reference density for the sparsity threshold: geometric mean of the positive clusters.

    Cluster density is #points / spread, so a near-degenerate cluster can be two orders of
    magnitude denser than the walls.  An arithmetic mean is then dominated by that outlier
    and the threshold silently discards every real reflector (scene11_set3 lost 3 of 6
    clusters and exported a 103-px map).  The geometric mean is unchanged on well-behaved
    scenes and robust to the outlier.
    """
    d = np.maximum(np.asarray(densities, dtype=float), 1e-9)
    return float(np.exp(np.mean(np.log(d))))


def truncate_bundle(b, ratio):
    """Keep only the first `ratio` percent of the trajectory (paper Fig. 9).

    The association arrays are stored in frame order, so taking a prefix is the
    same as having recorded a shorter walk -- this is how the baseline script
    already sub-sampled its detections (`real_point[:length]`).
    """
    if ratio >= 100:
        return b
    if not 0 < ratio < 100:
        raise ValueError(f"--ratio must be in (0, 100]; got {ratio}")
    out = dict(b)
    for key in VECTOR_KEYS + MAG_KEYS + ["trajectories", "neg_clusters_list", "neg_mags_list"]:
        array = np.asarray(b[key])
        out[key] = array[: max(1, int(len(array) * ratio / 100))]
    print(f"[rise] trajectory length {ratio}% -> {len(out['_pghost1pr_reals_list_'])} detections, "
          f"{len(out['trajectories'])} trajectory points")
    return out


def build_optimizer(b):
    return Optimizer_one_rel(b["center"], b["_pghost1_means_list_"], b["_pghost1pr_means_list_"],
                             b["_pghost2pr_means_list_"], b["_pghost1_reals_list_"],
                             b["_pghost1pr_reals_list_"], b["_pghost2pr_reals_list_"],
                             b["min_gmm"], b["max_gmm"], b["H"], b["W"])


def select_first_order_reflectors(optimizer, b, H):
    """choices == 3 in the original scripts: keep only first-order-ghost reflectors."""
    # ----------- Computing reflector based on First order Ghost
    reflector_1pr, mirror_1pr, norm_1pr, dist_1pr = optimizer.compute_reflector_1pr()
    mask_1pr = (np.mean(reflector_1pr < 0, axis=1) <= 0) * (np.mean(reflector_1pr > H, axis=1) <= 0)
    optimizer.reflector_point = reflector_1pr[mask_1pr]
    optimizer.mirror_point = mirror_1pr[mask_1pr]
    optimizer.vector_norm = norm_1pr[mask_1pr]
    optimizer.distance_ratio = dist_1pr[mask_1pr]
    optimizer.real_point = b["_pghost1pr_reals_list_"][mask_1pr]
    optimizer.ghost_point = b["_pghost1pr_means_list_"][mask_1pr]
    optimizer.mag_ratio = b["_pghost1pr_magt_list_"][mask_1pr]
    return optimizer


def _lines_from_segments(line_segment):
    labels, points, line_point_lists = [], [], []
    for number, line in enumerate(line_segment):
        slope, intercept, cluster_points = line
        labels.append(str(number) + "_" + str(len(cluster_points)))
        points.append(cluster_points)
        lines_x = np.linspace(cluster_points[:, 1].min(), cluster_points[:, 1].max(), num=500)
        lines_y = slope * lines_x + intercept
        line_point_lists.append(np.column_stack((lines_y, lines_x)))
    return labels, points, line_point_lists


def _drop_noise_segments(line_point_lists, points, center, dist_thresh):
    point_list, reflector_list = [], []
    for num, individual_line in enumerate(line_point_lists):
        start_point, end_point = individual_line[0], individual_line[-1]
        start_point = start_point[[1, 0]]; end_point = end_point[[1, 0]]
        if geometry_noise(center, dist_thresh, end_point, start_point): continue
        point_list.append([tuple(start_point.astype(int)), tuple(end_point.astype(int))])
        reflector_list.append(points[num])
    return point_list, reflector_list


# --------------------------------------------------------------------------- #
def _export_diffusion_inputs(map_array, trajectories, H, out_root, case, ratio_choice, prefix):
    """Write the 128x128 pair that 5_diffusion consumes for one map."""
    lowr_dir = Path(out_root) / "result_input" / "image_lowr"
    traj_dir = Path(out_root) / "result_input" / "image_traj"
    lowr_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}_{case}_{ratio_choice}.png"
    Image.fromarray(low_resolution(map_array.astype(float)).astype(np.uint8), mode='L').save(lowr_dir / name)
    black_map = np.zeros((H, H), dtype=np.uint8)
    for x, y in trajectories:
        black_map[int(x), int(y)] = 255
    cv2.imwrite(str(traj_dir / name), low_resolution(black_map.astype(float)))
    return lowr_dir / name


def _trajectory_points(scene, strict_level=1, use_threed=True):
    """The human trajectory, which only the BAME bundle carries."""
    try:
        return load_bundles(scene, [BAME_TAG], strict_level, use_threed)["trajectories"]
    except FileNotFoundError:
        print(f"[rise] no {BAME_TAG} bundle for case {scene.case}: writing an empty trajectory map. "
              f"Run `--step associate` first if you need the diffusion inputs.")
        return np.zeros((0, 2))


def run_baseline(case, txs=MVDR_TXS, out_dir="result_baseline", ratio_choice=0, tags=None,
                 export=False):
    """MVDR baseline layout -- one map per TX subarray (signal/working4_baseline.py).

    With `export=True` the fused baseline map is additionally written as the 128x128
    pair that 5_diffusion consumes, which is what the paper's "EMT + SRHD" row needs.
    """
    scene = get_scene(case)
    os.makedirs(out_dir, exist_ok=True)
    trajectories = _trajectory_points(scene) if export else None
    bg_points = bk_case_scene_bkg_refine(str(scene.background_path())).bg_points
    for tag in (tags or [mvdr_tag(tx) for tx in txs]):
        b = load_bundles(scene, [tag])
        H = b["H"]
        print(len(b["_pghost1pr_means_list_"]), '=========', len(b["_pghost2pr_means_list_"]),
              '=========', len(b["_pghost1_means_list_"]))
        optimizer = build_optimizer(b)
        # ----------- drop reflectors that fall outside the map (choices == 4)
        reflector_point_1pr = optimizer.compute_reflector_1pr_()
        mask_1pr = ~(np.mean(reflector_point_1pr < 0, axis=1) > 0) * ~(np.mean(reflector_point_1pr > H, axis=1) > 0)
        optimizer.reflector_point = reflector_point_1pr[mask_1pr]
        optimizer.real_point = b["_pghost1pr_reals_list_"][mask_1pr]
        optimizer.ghost_point = b["_pghost1pr_means_list_"][mask_1pr]
        for ratio in [i * 5 for i in range(1, 21)]:
            length = int(len(optimizer.real_point) / 100 * ratio)
            address = f"{out_dir}/bsl_{case}_{ratio_choice}_trajectory_{tag}_{ratio}.png"
            plt.figure(figsize=(8, 8))
            plt.scatter(optimizer.real_point[:length, 1], optimizer.real_point[:length, 0], marker='.', s=1, alpha=0.8, color='blue')
            plt.scatter(bg_points[:, 1], bg_points[:, 0], marker='.', s=1, alpha=0.1, color='gray')
            plt.legend(); plt.xlim(0, H); plt.ylim(0, H); plt.savefig(address, dpi=500, bbox_inches='tight'); plt.close()
        # ----------- GMM + RANSAC over the reflector hypotheses
        line_segment = optimizer.gmm_baseline(optimizer.reflector_point)
        labels, points, line_point_lists = _lines_from_segments(line_segment)
        point_list, reflector_list = _drop_noise_segments(line_point_lists, points, b["center"], 20)
        map_array = np.zeros((H, H), dtype=np.uint8)
        for a, bb in point_list: cv2.line(map_array, a, bb, 255, 1)
        Image.fromarray(map_array, mode='L').save(f"{out_dir}/bsl_{case}_{ratio_choice}_{tag}.png")
        if export:
            # one prefix per TX subarray, so the maps do not overwrite each other
            written = _export_diffusion_inputs(map_array, trajectories, H, out_dir, case,
                                               ratio_choice, prefix=tag.replace("_", ""))
            print('===== diffusion input written to =====', written)
        address = f"{out_dir}/bsl_{case}_{ratio_choice}_reflector_{tag}.png"
        plt.figure(figsize=(8, 8))
        for individual_points in reflector_list:
            plt.scatter(individual_points[:, 1], individual_points[:, 0], label=f'pred_{len(individual_points)}', marker='.', s=1, alpha=1, color='blue')
        plt.scatter(bg_points[:, 1], bg_points[:, 0], marker='.', s=1, alpha=0.1, color='gray')
        plt.legend(); plt.xlim(0, H); plt.ylim(0, H); plt.savefig(address, dpi=500, bbox_inches='tight'); plt.close()
        print('===== saved =====', address)


def run_lines(case, tags=None, out_dir=None, ratio_choice=0, strict_level=1, use_threed=True):
    """Quick reflector fusion (signal/working4_ours.py): GMM+RANSAC then mirror fusion."""
    scene = get_scene(case)
    tags = tags or [BAME_TAG] + [mvdr_tag(tx) for tx in MVDR_TXS]
    out_dir = out_dir or f"result_ours/image_{case}_{ratio_choice}"
    os.makedirs(out_dir, exist_ok=True)
    b = load_bundles(scene, tags, strict_level, use_threed)
    H = b["H"]
    print('= 1PR =', len(b["_pghost1pr_means_list_"]), '= 2PR =', len(b["_pghost2pr_means_list_"]),
          '= 1 =', len(b["_pghost1_means_list_"]), '= Tra =', len(b["trajectories"]),
          '= NEG =', len(b["neg_clusters_list"]))
    bg_points = bk_case_scene_bkg_refine(str(scene.background_path())).bg_points
    optimizer = select_first_order_reflectors(build_optimizer(b), b, H)

    line_segment = optimizer.gmm_ours(optimizer.reflector_point)
    labels, points, line_point_lists = _lines_from_segments(line_segment)
    point_list, reflector_list = _drop_noise_segments(line_point_lists, points, b["center"], 5)
    mirrors = mirror_of_P_over_lines(point_list, b["center"])
    line_list, sample_list = fuse_lines_from_mirrors(mirrors, reflector_list)

    address = f"{out_dir}/bsl_{case}_{ratio_choice}_reflector0.png"
    plt.figure(figsize=(8, 8))
    plt.scatter(bg_points[:, 1], bg_points[:, 0], label='Ground Truth', marker='s', s=1, facecolors='grey', edgecolors='grey', alpha=0.1)
    for individual_points in sample_list:
        plt.scatter(individual_points[:, 1], individual_points[:, 0], label=f'Reflectors', marker='.', s=5, alpha=0.7, color='#1a80bb')
    for start, end in line_list:
        plt.plot([start[1], end[1]], [start[0], end[0]], color='red', alpha=0.4, linewidth=10)
    plt.legend(); plt.xlim(0, H); plt.ylim(0, H); plt.savefig(address, dpi=500, bbox_inches='tight'); plt.close()
    print('===== saved =====', address)


def run_export(case, tags=None, out_root="result_ours", ratio_choice=0, strict_level=1,
               use_threed=True, threshold_dense=0.25, threshold_mag_min=0.25, visuals=False,
               ratio=100):
    """Full layout pipeline + diffusion input export (signal/working5_ours.py).

    `ratio` is the percentage of the trajectory to use (paper Fig. 9); it is also
    the second number in the exported file name, which is what 5_diffusion reads
    back as `ratio_choice`.
    """
    scene = get_scene(case)
    ratio_choice = ratio_choice if ratio >= 100 else ratio
    tags = tags or [BAME_TAG] + [mvdr_tag(tx) for tx in MVDR_TXS]
    image_save = Path(out_root) / f"image_{case}_{ratio_choice}"
    image_save_input = Path(out_root) / "result_input" / "image"        # For diffusion: full resolution
    image_save_input2 = Path(out_root) / "result_input" / "image_lowr"  # For diffusion: low resolution
    image_save_input3 = Path(out_root) / "result_input" / "image_traj"  # For diffusion: trajectories
    for d in (image_save, image_save_input, image_save_input2, image_save_input3):
        d.mkdir(parents=True, exist_ok=True)

    b = truncate_bundle(load_bundles(scene, tags, strict_level, use_threed), ratio)
    H = b["H"]; center = b["center"]
    trajectories = b["trajectories"]; neg_clusters_list = b["neg_clusters_list"]; neg_mags_list = b["neg_mags_list"]
    print('= 1PR =', len(b["_pghost1pr_means_list_"]), '= 2PR =', len(b["_pghost2pr_means_list_"]),
          '= 1 =', len(b["_pghost1_means_list_"]), '= Tra =', len(trajectories), '= NEG =', len(neg_clusters_list))
    bg_points = bk_case_scene_bkg_refine(str(scene.background_path())).bg_points
    optimizer = select_first_order_reflectors(build_optimizer(b), b, H)
    mag_impro_dis = optimizer.mag_ratio / optimizer.distance_ratio

    # ---------- positive side: cluster the mirror points, keep the matching reflectors ----------
    clusters, final_labels = optimizer.gmm_clustering(optimizer.mirror_point)
    labels, points, reflectors_pos, magnitudes = [], [], [], []
    for label in np.unique(final_labels):
        if label != -1:
            reflectors_pos.append(optimizer.reflector_point[final_labels == label])
            points.append(optimizer.mirror_point[final_labels == label])
            labels.append(str(label) + "_" + str(len(points[-1])))
            magnitudes.append(np.mean(mag_impro_dis[final_labels == label]))
    line_point_lists = optimizer.fit_line_list(reflectors_pos)
    pos_gaus_sample, pos_sampled_lines, pos_sampled_dense = optimizer.gauss_point_sampling(reflectors_pos)
    pos_sampled_lines = optimizer.gauss_point_sampling_impr(reflectors_pos)
    pos_labels, pos_denses, pos_mags = [], [], []
    for lab_num, _ in enumerate(pos_sampled_lines):
        n = len(pos_sampled_lines[lab_num])
        pos_labels.append(lab_num * np.ones(n))
        pos_denses.append(pos_sampled_dense[lab_num] * np.ones(n))
        pos_mags.append(magnitudes[lab_num] * np.ones(n))

    # ---------- negative side (points behind the reflector) ----------
    neg_pre = 1000
    clusters, final_labels = optimizer.gmm_clustering(neg_clusters_list)
    neg_points, neg_mags = [], []
    for label in np.unique(final_labels):
        if label != -1:
            neg_points.append(neg_clusters_list[final_labels == label])
            neg_mags.append(neg_mags_list[final_labels == label])
    neg_gaus_sample, neg_sampled_lines, neg_sampled_dense = optimizer.gauss_point_sampling(neg_points)
    neg_sampled_lines = optimizer.gauss_point_sampling_impr(neg_points)
    neg_labels, neg_denses, neg_mags_line = [], [], []
    for lab_num, _ in enumerate(neg_sampled_lines):
        n = len(neg_sampled_lines[lab_num])
        neg_labels.append((lab_num + neg_pre) * np.ones(n))
        neg_denses.append(neg_sampled_dense[lab_num] * np.ones(n))
        neg_mags_line.append(0.5 * np.ones(n))  # negative points carry no magnitude

    # ---------- fuse positive and negative samples ----------
    whole_cluster_list = [np.round(arr).astype(np.int32) for arr in neg_sampled_lines + pos_sampled_lines]
    whole_label_list = [(i + 1) * 100 for i in neg_labels] + [i + 2 for i in pos_labels]
    whole_mag_list = neg_mags_line + pos_mags
    whole_dense_list = neg_denses + pos_denses
    whole_cluster_list = np.vstack(whole_cluster_list)
    whole_label_list = np.concatenate(whole_label_list, axis=0)
    whole_mag_list = np.concatenate(whole_mag_list, axis=0)
    whole_dense_list = np.concatenate(whole_dense_list, axis=0)
    mask = (whole_dense_list > threshold_dense * _density_reference(pos_sampled_dense)) * \
           (whole_mag_list > threshold_mag_min * np.mean(magnitudes))
    valid_msk = optimizer.filter_points(whole_cluster_list[mask])
    using_whole_cluster_list = whole_cluster_list[mask][valid_msk]
    using_whole_label_list = whole_label_list[mask][valid_msk]

    # ---------- keep the closest line, drop the short ones, straighten ----------
    dialte_whole_cluster, dialte_whole_labels = optimizer.dilate_line_label(using_whole_cluster_list, using_whole_label_list)
    whole_clusters_map_gmm_gauss, _ = optimizer.select_close_point(dialte_whole_cluster, center)
    whole_clusters_coor = np.column_stack(np.nonzero(whole_clusters_map_gmm_gauss))
    matches = np.any(np.all(dialte_whole_cluster[:, None] == whole_clusters_coor, axis=2), axis=1)
    select_cluster = dialte_whole_cluster[np.where(matches)[0]]
    dilate_select_cluster = optimizer.dilate_line(select_cluster)
    whole_clusters_map_gmm_gauss = optimizer.eliminate_short_line_iter(whole_clusters_map_gmm_gauss)
    whole_clusters_coor = np.column_stack(np.nonzero(whole_clusters_map_gmm_gauss))
    points_list = optimizer.create_line(dialte_whole_cluster, dialte_whole_labels, whole_clusters_coor)

    if visuals:
        address = f"{image_save}/1_reflectors.png"
        plt.figure(figsize=(8, 8))
        plt.scatter(optimizer.reflector_point[:, 1], optimizer.reflector_point[:, 0], label='reflector', marker='v', s=1)
        plt.scatter(bg_points[:, 1], bg_points[:, 0], label='bg', marker='o', s=0.1, color="grey", alpha=0.1)
        plt.legend(); plt.xlim(0, H); plt.ylim(0, H)
        plt.savefig(address, dpi=500, bbox_inches='tight', pad_inches=0); plt.close()
        address = f"{image_save}/2_layout.png"
        plt.figure(figsize=(8, 8))
        plt.scatter(dilate_select_cluster[:, 1], dilate_select_cluster[:, 0], label='layout', marker='.', s=1, alpha=1)
        plt.scatter(bg_points[:, 1], bg_points[:, 0], label='bg', marker='o', s=0.1, color="grey", alpha=0.1)
        plt.legend(); plt.xlim(0, H); plt.ylim(0, H)
        plt.savefig(address, dpi=500, bbox_inches='tight'); plt.close()

    # ---------- diffusion inputs ----------
    map_array = np.zeros((H, H), dtype=np.uint8)
    dist_thresh = 20; point_list = []
    for start_point, end_point in points_list:
        if geometry_noise(center, dist_thresh, end_point, start_point): continue
        point_list.append([tuple(start_point.astype(int)), tuple(end_point.astype(int))])
    for a, bb in point_list: cv2.line(map_array, a, bb, 255, 1)
    Image.fromarray(map_array, mode='L').save(f"{image_save_input}/ours_{case}_{ratio_choice}.png")

    resized_image = low_resolution(map_array.astype(float))
    Image.fromarray(resized_image.astype(np.uint8), mode='L').save(f"{image_save_input2}/ours_{case}_{ratio_choice}.png")

    black_map = np.zeros((H, H), dtype=np.uint8)
    for x, y in trajectories: black_map[int(x), int(y)] = 255
    black_map = low_resolution(black_map.astype(float))
    cv2.imwrite(f"{image_save_input3}/ours_{case}_{ratio_choice}.png", black_map)
    print('===== diffusion inputs written to =====', image_save_input.parent)
