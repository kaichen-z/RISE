"""BAME -- Bi-Angular Multipath Enhancement (the proposed detector).

Per radar frame this runs, in one pass:

    Range FFT  ->  AoA / AoD FFT  ->  3D CFAR  ->  ghost re-integration  ->  DBSCAN

and stores the resulting (x, y) clusters + magnitudes for 4_inversion.
Originally `signal/working1_all.py`.
"""

import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from common.config import BAME_TAG, get_scene
from common.frames import batch_indices, ordered_frames
from common.function.background_tch import bk_case_scene_bkg, bk_case_scene_bkg_refine
from common.function.geometry_tch import geometry_ghost_update
from common.function.fft_tch import _range_fft_, _angle_fft_, _angle_mag_eight_
from common.function.post_process_tch import (
    post_cfar_acc_class,
    ghost_enhance_3d,
    post_cfar_acc_3d_conv_class,
    Dbscan_coord,
)
from common.utils.utils_tch import (
    load_hyperparam,
    load_matrix,
    list_json,
    convert_ra_xy,
    generate_diagonal,
    convert_cluster_xy,
    overlap_check,
)

colors = {'R': 'orange', 'G1': 'purple', 'G1R': 'cyan', 'G2R': 'pink', 'CL': 'pink', 'Rel': 'gray'}


# Including the analysis for AOA and AOD
class range_angle:
    def __init__(self,):
        # ----- Normalize -----
        self.max_value = 1
        # ----- CFAR -----
        self.num_guard_cells, self.num_guard_cells_3d = 2, 2
        self.num_reference_cells, self.num_reference_cells_3d = 4, 4
        # For Gaussian Case
        self.threshold_factor, self.threshold_factor_3d = 1., 1.
        self.threshold_factor2, self.threshold_factor2_3d = 1., 1.
        self.threshold_factor3, self.threshold_factor3_3d = 2., 2.
        self.threshold_factor4, self.threshold_factor4_3d = 2.0, 2.0
        # ----- convert to tx-ty (crop)
        self.y_axis = 0.6; self.x_axis = 1.0
        self.RESOLUTION = 50; self.RATIO = 2/3
        self.TOTAL = 15; self.d = 1
        self.centers = [self.RATIO*self.TOTAL*self.RESOLUTION*self.y_axis, 0]
        self.np_center = torch.tensor(self.centers)
        self.H, self.W = int(self.RATIO*self.TOTAL*self.RESOLUTION*2*self.y_axis), int(self.RATIO*self.TOTAL*self.RESOLUTION)
        # ----- DOPPLER -----
        self.index_doppler = int(0)
        # ----- Gaussian -----
        self.gau_sigma = 2; self.threshold = 0.9
        self.threshold2 = 0.5; self.sample = 30
        self.ratio_var = 0.1
        # ----- Geometry -----
        self.geometry = geometry_ghost_update(self.threshold, self.threshold2, self.sample, self.ratio_var)
        # ----- Dbscan_coord -----
        self.AREA = 2; self.MIN_COUNT = 6
        # ----- temporal
        self.real_mean_time = []; self.ghost1_mean_time = []
        self.ghost2_mean_time = []; self.reflector_time = []
        # ----- temporal_aggregate condition
        self.mean_distance = 50; self.var_threshold = 100
        # ----- AOA * AOD comparison
        self.d_wave_rx = 0.5; self.d_wave_tx = 2
        self.d_wave_virtual = 0.5
        # ----- Ghost Enhance -----
        self.MIN_NUM = 10; self.NIN_NUM_DIA = 10
        self._AREA_ = 2; self._MIN_COUNT_ = 6
        self.AD_DIFF = 10
        # ------
        self.data_size_3d = (256,256,256)
        self.post_cfar_acc_3d_conv_class = post_cfar_acc_3d_conv_class(self.data_size_3d, self.num_guard_cells_3d, \
            self.num_reference_cells_3d, self.threshold_factor_3d, \
            self.threshold_factor2_3d, self.threshold_factor3_3d, self.threshold_factor4_3d)
        self.data_size_2d = (256,256)
        self.post_cfar_acc_class = post_cfar_acc_class(self.data_size_2d, self.num_guard_cells, self.num_reference_cells, \
                self.threshold_factor, self.threshold_factor2, self.threshold_factor3, self.threshold_factor4)

    def process_bg_fr(self, prefix_bg, int_num, fr_num):
        filename = f'original_hyp_{int_num}_{fr_num}.json'
        file_add_hyper = os.path.join(prefix_bg, filename)
        hyper_dict = load_hyperparam(file_add_hyper)
        filename = f'original_mat_{int_num}_{fr_num}'
        file_add_mat = os.path.join(prefix_bg, filename)
        matrix, size_dict = load_matrix(file_add_mat)
        self.bg_range_fft = _range_fft_(hyper_dict, size_dict)
        self.bg_angle_fft = _angle_fft_(hyper_dict, size_dict)
        self.bg_angle_mag = _angle_mag_eight_(hyper_dict, size_dict) # which requies input: (Dim, N)
        #NumRange, NumDop, NumRx, NumTx = matrix.shape
        #print(" NumRange:", NumRange, " NumDop:", NumDop, " NumRx:", NumRx, " NumTx:", NumTx)
        # -------------------Range
        num_division_rx, num_division_tx, start_angle, end_angle = 256, 256, 0, torch.pi
        angles = torch.linspace(start_angle, end_angle, num_division_rx)
        self.angles = angles; self.plot_angles = - self.angles + torch.pi/2
        d_wavelength = 0.5; matrix = matrix[:, :1]; self.bg_range_fft.apply_fft(matrix)
        self.bg_matrix_zero_power = self.bg_angle_mag.accelerate_power(self.bg_range_fft.output, d_wavelength, angles)

    def process_fr_test(self, prefix, num, interval, frame):
        filename = f'original_hyp_{interval[num]}_{frame[num]}.json'
        file_add_hyper = os.path.join(prefix, filename)
        hyper_dict = load_hyperparam(file_add_hyper)
        filename = f'original_mat_{interval[num]}_{frame[num]}'
        file_add_mat = os.path.join(prefix, filename)
        matrix, size_dict = load_matrix(file_add_mat)
        self.range_fft = _range_fft_(hyper_dict, size_dict)
        self.angle_fft = _angle_fft_(hyper_dict, size_dict)
        self.angle_mag = _angle_mag_eight_(hyper_dict, size_dict) # which requies input: (Dim, N)
        matrix = matrix[:, :1];  self.range_fft.apply_fft(matrix)
        # ---------- processing for (RX, TX)
        num_division_rx, _, start_angle, end_angle = 256, 256, 0, torch.pi
        angles = torch.linspace(start_angle, end_angle, num_division_rx)
        self.angles = angles
        self.plot_angles = - self.angles + torch.pi/2; d_wavelength = 0.5
        self.non_bg_matrix_zero_power = self.angle_mag.accelerate_power(self.range_fft.output, d_wavelength, angles)
        # [numrange, numdop, numrx, numtx]

    def process_xy_power(self, MATRIX):
        # input_size: range, doppler, angle
        MATRIX_xy, _ = convert_ra_xy(MATRIX, self.bg_range_fft.hyper, \
            self.bg_angle_fft.hyper, self.RESOLUTION, self.RATIO, self.y_axis, self.x_axis, self.plot_angles)
        return MATRIX_xy

    def process_xy_cluster_power(self, cluster, diagonal_labels):
        coords, labels, mask = convert_cluster_xy(cluster, diagonal_labels, self.bg_range_fft.hyper, \
            self.bg_angle_fft.hyper, self.RESOLUTION, self.RATIO, self.y_axis, self.x_axis, self.plot_angles)
        return coords, labels, mask

    def bk_case_scene_bkg(self, bkg_address):
        bk_class = bk_case_scene_bkg(self.np_center, self.RESOLUTION, bkg_address)
        self.bg_points = bk_class.bg_points

    def gaus_normalize(self, matrix):
        matrix = gaussian_filter(matrix.numpy(), sigma=self.gau_sigma)
        matrix = torch.from_numpy(matrix)
        return matrix

    def compute_detection(self, matrix):
        cfar_detection = self.post_cfar_acc_class.process(matrix)
        return cfar_detection

    def compute_detection_3d_conv(self, matrix):
        cfar_detection = self.post_cfar_acc_3d_conv_class.process(matrix)
        return cfar_detection

    def compute_cluster(self, cfar_detection):
        try: self.labels, self.clusters = Dbscan_coord(cfar_detection, self.AREA, self.MIN_COUNT)
        except:
            self.labels, self.clusters = torch.zeros((1,), dtype=torch.int), torch.zeros((1,2), dtype=torch.int)
            print('-------- There is no cluster for this frame --------')
        return self.labels, self.clusters

    def enhance_ghost_3d(self, diagonal_cfar_3d_conv):
        # Adding 500 to point being copied. Adding 1500 to point being generated by copying old points.
        labels_final, clusters_final, mags_final = ghost_enhance_3d(diagonal_cfar_3d_conv, self.MIN_NUM, self.AD_DIFF, self._AREA_, self._MIN_COUNT_)
        return labels_final, clusters_final, mags_final

    def check_overlap(self, ori_cluster, new_cluster, new_label):
        select_cluster, select_label, _ = overlap_check(ori_cluster, new_cluster, new_label)
        return select_cluster, select_label

    def plot_heatmap(self, matrix, address, title="Heatmap"):
        plt.imshow(matrix.numpy(), cmap='viridis', origin='lower')
        plt.colorbar(); plt.title(title)
        plt.xlabel('Width (2 CM)'); plt.ylabel('Height (2 CM)')
        plt.xlim(0, self.H); plt.ylim(0, self.H)
        plt.savefig(address, dpi=500); plt.close()

    def plot_heatmap_256(self, matrix, address, title="Heatmap", range_max=None, range_min=None, diagonal=False):
        if range_max is not None: plt.imshow(matrix, cmap='viridis', origin='lower', vmax=range_max, vmin=range_min)
        else: plt.imshow(matrix, cmap='viridis', origin='lower')
        if diagonal: plt.plot(np.arange(256), np.arange(256), color='red', linewidth=2.5)  # from (0,0) to (255,255)
        plt.title(title); plt.colorbar()
        plt.xlabel('Width (2 CM)'); plt.ylabel('Height (2 CM)')
        plt.xlim(0, 256); plt.ylim(0, 256)
        plt.savefig(address, dpi=500); plt.close()

    def plot_cluster_debug(self, labels, cluster, address, title="empty"):
        good_labels = torch.unique(labels)
        plt.figure(figsize=(8, 8))
        for label in good_labels:
            cluster_coords = cluster[labels == label]
            plt.scatter(cluster_coords[:, 1], cluster_coords[:, 0], label=f'Cluter {label}')
        plt.scatter(self.bg_points[:, 1], self.bg_points[:, 0], label=f'Reflector GT', marker='.', alpha=0.9, color='gray', s=2)
        plt.scatter(self.np_center[1], self.np_center[0], label=f'Mirror', marker='+', color='Blue', s=200)
        plt.title(title);plt.legend()
        plt.xlim(0, self.H); plt.ylim(0, self.H)
        plt.savefig(address, dpi=500, bbox_inches='tight'); plt.close()

    def concatenate_image_normal(self, combine_address, address):
        img_list = []
        width = 1200
        height = 1200
        for combine in combine_address:
            img_list.append(cv2.resize(cv2.imread(combine), (width, height)))
        fin_image = np.concatenate(img_list, axis=1)
        cv2.imwrite(address, fin_image)


def run(case, trajectories=None, tag=BAME_TAG, visual_freq=10, batch=1,
        start_index=0, max_frames=None, bkg_stride=30):
    scene = get_scene(case)
    trajectories = trajectories or scene.trajectories
    range_class = range_angle()

    for traj in trajectories:
        print(case, '================ WE RE PROCESSING =================', traj)
        # --------------- Chose the case of background based
        bk_class = bk_case_scene_bkg_refine(str(scene.background_path()))
        range_class.bg_points = bk_class.bg_points
        '''--------- Creating directory for saving ---------'''
        prefix_save = scene.detect_dir(traj, tag)
        prefix_save_cl_img = prefix_save / 'image_batch_diagonal_accelerating_img'
        prefix_save_cl = prefix_save / 'image_batch_diagonal_accelerating_torch'
        prefix_save_cl_img.mkdir(parents=True, exist_ok=True)
        prefix_save_cl.mkdir(parents=True, exist_ok=True)
        prefix_save_cl_img, prefix_save_cl = str(prefix_save_cl_img), str(prefix_save_cl)
        print('================= WE RE SAVING TO =================', prefix_save)
        # --------------- background processing ---------------
        prefix_bg = str(scene.bkg_frame_dir())
        interval, frame = list_json(prefix_bg); bg_zero_paddings = []
        # A released background capture may already be subsampled (tools/pack_frames.py or
        # pack_bame.pl --every N write `"every": N` into python_frames.json); keep the
        # effective 1-in-30 sampling instead of striding over the subsampled list again.
        stride = bkg_stride
        manifest = Path(prefix_bg) / 'python_frames.json'
        if manifest.exists():
            every = int(json.loads(manifest.read_text()).get('every', 1) or 1)
            stride = max(1, bkg_stride // every)
        for index_zip, (int_num, fr_num) in enumerate(zip(interval[::stride], frame[::stride])):
            range_class.process_bg_fr(prefix_bg, int_num, fr_num)
            bg_zero_paddings.append(range_class.bg_matrix_zero_power)
        range_class.bg_zero_paddings = torch.abs(torch.mean(torch.stack(bg_zero_paddings, dim=0), dim=0))
        _, _, bg_vol_dia_avg, _ = generate_diagonal(range_class.gaus_normalize(range_class.bg_zero_paddings)) # For visualization
        _, _, bg_vol_dia_avg_ng, _ = generate_diagonal(range_class.bg_zero_paddings) # Substraction
        bg_diagonal_power_zero_paddings = bg_vol_dia_avg[:, 0]
        bg_diagonal_cfar = range_class.compute_detection(bg_diagonal_power_zero_paddings)
        bg_diagonal_labels, bg_diagonal_clusters = range_class.compute_cluster(bg_diagonal_cfar)
        bg_diagonal_clusters_xy, bg_diagonal_labels_xy, bg_diagonal_mask_xy = range_class.process_xy_cluster_power(bg_diagonal_clusters, bg_diagonal_labels)
        bg_diagonal_mags = bg_vol_dia_avg_ng[:,0][bg_diagonal_clusters[bg_diagonal_mask_xy,0], bg_diagonal_clusters[bg_diagonal_mask_xy,1]]
        bg_tensor_data = {"center": range_class.np_center, "bg_points": range_class.bg_points,
            "bg_diagonal_clusters_xy": bg_diagonal_clusters_xy,
            "bg_diagonal_labels_xy": bg_diagonal_labels_xy, "bg_diagonal_mags": bg_diagonal_mags,}
        ADDRESS_DICT = os.path.join(prefix_save_cl, f'combine_bg.pth')
        torch.save(bg_tensor_data, ADDRESS_DICT)
        # bg_diagonal_clusters_xy, bg_diagonal_labels_xy, bg_diagonal_mags
        prefix = str(scene.frame_dir(traj))
        print(case, traj, '----- Here we are -----', prefix)
        try: interval, frame = ordered_frames(prefix)
        except Exception: print(traj, '----------- This is the one doesnt work'); continue

        '''--------- Batching ---------'''
        batch_lists = batch_indices(len(frame), batch, max_frames)

        '''--------- Iterating ---------'''
        for index, batch_list in tqdm(enumerate(batch_lists[:])):
            if index >= start_index:
                zero_paddings = []
                for num in batch_list:
                    range_class.process_fr_test(prefix, num, interval, frame)
                    zero_paddings.append(range_class.non_bg_matrix_zero_power)
                range_class.zero_paddings = torch.abs(torch.mean(torch.stack(zero_paddings, dim=0), dim=0))
                # Only dealing with the negative part.
                sub_zero_paddings = range_class.zero_paddings - range_class.bg_zero_paddings
                sub_zero_paddings[sub_zero_paddings>0] = 0; sub_zero_paddings = abs(sub_zero_paddings)
                range_class.neg_sub_zero_paddings = sub_zero_paddings
                _, _, neg_vol_dia_avg_ng, _ = generate_diagonal(range_class.neg_sub_zero_paddings) # Need this for saving Magnitude.
                _, _, neg_vol_dia_avg, _ = generate_diagonal(range_class.gaus_normalize(range_class.neg_sub_zero_paddings))
                range_class.neg_sub_diagonal_power = neg_vol_dia_avg[:, 0]
                neg_diagonal_cfar = range_class.compute_detection(range_class.neg_sub_diagonal_power)  # Gaus&Det step 2&3
                # Only dealing with the positive part.
                sub_zero_paddings = range_class.zero_paddings - range_class.bg_zero_paddings
                sub_zero_paddings[sub_zero_paddings<0] = 0; range_class.sub_zero_paddings = sub_zero_paddings
                _, _, vol_dia_avg_ng, _ = generate_diagonal(range_class.sub_zero_paddings)
                _, _, vol_dia_avg, _ = generate_diagonal(range_class.gaus_normalize(range_class.sub_zero_paddings))
                range_class.sub_diagonal_power = vol_dia_avg[:, 0]
                diagonal_cfar = range_class.compute_detection(range_class.sub_diagonal_power)
                range_class.sub_zero_paddings_gauss = range_class.gaus_normalize(range_class.sub_zero_paddings)
                diagonal_cfar_3d_conv = range_class.compute_detection_3d_conv(range_class.sub_zero_paddings_gauss)
                range_class.neg_diagonal_cfar, range_class.diagonal_cfar = neg_diagonal_cfar, diagonal_cfar
                # Using DBScan based on the diagonal result.
                neg_diagonal_labels, neg_diagonal_clusters = range_class.compute_cluster(range_class.neg_diagonal_cfar)
                diagonal_labels, diagonal_clusters = range_class.compute_cluster(range_class.diagonal_cfar)
                try:
                    threed_labels_final, threed_clusters_final_3d, threed_mags_final_3d = range_class.enhance_ghost_3d(diagonal_cfar_3d_conv)
                    FL_three = True
                except Exception: FL_three = False
                if FL_three:
                    added_msk = threed_labels_final >= 500
                    threed_labels_final_add, threed_clusters_final_add = threed_labels_final[added_msk], threed_clusters_final_3d[added_msk,:2]
                # Converting all points to x,y coordiante including magnitude.
                neg_diagonal_clusters_xy, neg_diagonal_labels_xy, neg_diagonal_mask_xy = range_class.process_xy_cluster_power(neg_diagonal_clusters, neg_diagonal_labels)
                diagonal_clusters_xy, diagonal_labels_xy, diagonal_mask_xy = range_class.process_xy_cluster_power(diagonal_clusters, diagonal_labels)
                if FL_three: threed_clusters_final_xy, threed_labels_final_xy, threed_labels_mask_xy = range_class.process_xy_cluster_power(threed_clusters_final_3d[:,:2], threed_labels_final)
                neg_diagonal_mags = neg_vol_dia_avg_ng[:,0][neg_diagonal_clusters[neg_diagonal_mask_xy,0], neg_diagonal_clusters[neg_diagonal_mask_xy,1]]
                diagonal_mags = vol_dia_avg_ng[:,0][diagonal_clusters[diagonal_mask_xy,0], diagonal_clusters[diagonal_mask_xy,1]]
                if FL_three: threed_clusters_final_3d_msk = threed_clusters_final_3d[threed_labels_mask_xy]
                if FL_three: threed_mags_final_3d = range_class.sub_zero_paddings[:,0][threed_clusters_final_3d_msk[:,0], threed_clusters_final_3d_msk[:,1], threed_clusters_final_3d_msk[:,2]]
                # Visualizing generated clusters and points for evaluation.
                if visual_freq and index%visual_freq == 0:
                    try:
                        threed_clusters_final_xy_add, threed_labels_final_xy_add, threed_mask_final_xy_add = range_class.process_xy_cluster_power(threed_clusters_final_add, threed_labels_final_add)
                        threed_clusters_final_xy_sel, threed_labels_final_xy_sel = range_class.check_overlap(diagonal_clusters_xy, threed_clusters_final_xy_add, threed_labels_final_xy_add)
                        whole_clusters_xy, whole_labels_xy = torch.cat([diagonal_clusters_xy, threed_clusters_final_xy_sel], dim=0), torch.cat([diagonal_labels_xy, threed_labels_final_xy_sel+10000], dim=0)
                    except Exception:
                        print("--------- No Aod != Aoa Case ---------Scene: ", scene.raw_dir)
                    try:
                        address1 = os.path.join(prefix_save_cl_img, f'{index}_heatmap.png')
                        address6 = os.path.join(prefix_save_cl_img, f'{index}_diagonal_clusters_xy.png')
                        address8 = os.path.join(prefix_save_cl_img, f'{index}_diagonal_coords_xy.png')
                        address9 = os.path.join(prefix_save_cl_img, f'{index}_diagonal_coords_xy_add.png')
                        address10 = os.path.join(prefix_save_cl_img, f'{index}_diagonal_coords_xy_check.png')
                        address11 = os.path.join(prefix_save_cl_img, f'{index}_diagonal_coords_xy_whole.png')
                        range_class.plot_heatmap(range_class.process_xy_power(vol_dia_avg_ng)[:,0], address1, "Heatmap")
                        range_class.plot_cluster_debug(diagonal_labels_xy, diagonal_clusters_xy, address6, "diagonal")
                        range_class.plot_cluster_debug(threed_labels_final_xy, threed_clusters_final_xy, address8, "three_dim")
                        range_class.plot_cluster_debug(threed_labels_final_xy_add, threed_clusters_final_xy_add, address9, "three_dim_add")
                        range_class.plot_cluster_debug(threed_labels_final_xy_sel, threed_clusters_final_xy_sel, address10, "three_dim_sel")
                        range_class.plot_cluster_debug(whole_labels_xy, whole_clusters_xy, address11, "whole")
                        address0 = os.path.join(prefix_save_cl_img, f'{index}_whole.png')
                        range_class.concatenate_image_normal([address1, address6, address8, address9, address10, address11], address0)
                        os.remove(address1), os.remove(address6), os.remove(address8), os.remove(address9), os.remove(address10), os.remove(address11)
                    except Exception:
                        address1 = os.path.join(prefix_save_cl_img, f'{index}_heatmap.png')
                        address6 = os.path.join(prefix_save_cl_img, f'{index}_diagonal_clusters_xy.png')
                        address8 = os.path.join(prefix_save_cl_img, f'{index}_diagonal_coords_xy.png')
                        range_class.plot_heatmap(range_class.process_xy_power(vol_dia_avg_ng)[:,0], address1, "Heatmap")
                        range_class.plot_cluster_debug(diagonal_labels_xy, diagonal_clusters_xy, address6, "diagonal")
                        if FL_three: range_class.plot_cluster_debug(threed_labels_final_xy, threed_clusters_final_xy, address8, "three_dim")
                        address0 = os.path.join(prefix_save_cl_img, f'{index}_whole.png')
                        if FL_three: range_class.concatenate_image_normal([address1, address6, address8], address0)
                        else: range_class.concatenate_image_normal([address1, address6], address0)
                        if FL_three: os.remove(address1), os.remove(address6), os.remove(address8)
                        else: os.remove(address1), os.remove(address6)
                if not FL_three: threed_clusters_final_3d_msk, threed_clusters_final_xy, threed_labels_final_xy, threed_mags_final_3d = None, None, None, None
                tensor_data = {
                    "neg_diagonal_clusters_xy": neg_diagonal_clusters_xy,
                    "neg_diagonal_labels_xy": neg_diagonal_labels_xy,
                    "neg_diagonal_mags": neg_diagonal_mags,
                    "diagonal_clusters_xy": diagonal_clusters_xy,
                    "diagonal_labels_xy": diagonal_labels_xy,
                    "diagonal_mags": diagonal_mags,
                    "threed_clusters_final_3d": threed_clusters_final_3d_msk, # r, aoa, aod
                    "threed_clusters_final_xy": threed_clusters_final_xy,
                    "threed_labels_final_xy": threed_labels_final_xy,
                    "threed_labels_mags": threed_mags_final_3d,
                    "index": index}
                ADDRESS_DICT = os.path.join(prefix_save_cl, f'combine_{index}.pth')
                torch.save(tensor_data, ADDRESS_DICT)
