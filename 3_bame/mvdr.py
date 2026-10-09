"""MVDR baseline detector (single-TX virtual array).

Range FFT -> MVDR beamforming -> 2D CFAR -> DBSCAN, run once per TX subarray
(M6 / M7 / M8). No AoD dimension, hence no ghost re-integration: this is the
baseline the BAME detector is compared against.
Originally `signal/working0_all.py`.
"""

import os

import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from common.config import MVDR_TXS, get_scene, mvdr_tag
from common.frames import batch_indices, ordered_frames
from common.function.background_tch import bk_case_scene_bkg_refine
from common.function.fft_tch import MVDR, _range_fft_
from common.function.post_process_tch import Dbscan_coord, PostCFAR_1D_2D_v2
from common.utils.utils_tch import load_hyperparam, load_matrix, convert_cluster_xy_mvdr

colors = {'R': 'orange', 'G1': 'purple', 'G1R': 'cyan', 'G2R': 'pink', 'CL': 'pink', 'Rel': 'gray'}


# Including the analysis for AOA and AOD
class range_angle:
    def __init__(self, doppler_dim=64):
        # ----- Normalize -----
        self.max_value = 1
        # ----- CFAR Hyperparameter -----
        self.num_guard_cells = 2; self.num_reference_cells = 4
        self.threshold_factor_fg  = 1.00; self.threshold_factor2_fg = 0.40
        self.threshold_factor3_fg = 0; self.threshold_factor4_fg = 2.25
        # ----- convert to tx-ty (crop)
        self.y_axis = 0.6
        self.x_axis = 1.0
        self.RESOLUTION = 50
        self.RATIO = 2/3
        self.TOTAL = 15
        self.d = 1
        self.centers = [self.RATIO*self.TOTAL*self.RESOLUTION*self.y_axis, 0]
        self.np_center = torch.tensor(self.centers)
        self.H, self.W = int(self.RATIO*self.TOTAL*self.RESOLUTION*2*self.y_axis), int(self.RATIO*self.TOTAL*self.RESOLUTION)
        # ----- DOPPLER -----
        self.index_doppler = int(0)
        # ----- Gaussian -----
        self.gau_sigma = 2  # Standard deviation for Gaussian kernel, small for easy detection, large for denoise
        self.threshold = 0.9
        self.threshold2 = 0.5
        self.sample = 30
        self.ratio_var = 0.1
        # ----- Dbscan_coord -----
        self.AREA = 2
        self.MIN_COUNT = 6
        # ----- temporal
        self.real_mean_time = []
        self.ghost1_mean_time = []
        self.ghost2_mean_time = []
        self.reflector_time = []
        # ----- temporal_aggregate condition
        self.mean_distance = 50
        self.var_threshold = 100
        # ----- AOA * AOD comparison
        self.d_wave_rx = 0.5
        self.d_wave_tx = 2
        self.d_wave_virtual = 0.5
        # ----- Ghost Enhance -----
        self.MIN_NUM = 10
        self.NIN_NUM_DIA = 10
        self._AREA_ = 2
        self._MIN_COUNT_ = 6
        self.AD_DIFF = 10
        # ------
        self.data_size_2d = (256,181)
        self.post_cfar_acc_class_fg = PostCFAR_1D_2D_v2(self.data_size_2d, self.num_guard_cells, self.num_reference_cells, \
                self.threshold_factor_fg, self.threshold_factor2_fg, self.threshold_factor3_fg, self.threshold_factor4_fg)
        self.process_mvdr = MVDR(doppler_dim)

    def process_mvdr_final(self, prefix, num, interval, frame, mvdr_type = 1):
        filename = f'original_hyp_{interval[num]}_{frame[num]}.json'
        file_add_hyper = os.path.join(prefix, filename)
        hyper_dict = load_hyperparam(file_add_hyper)
        filename = f'original_mat_{interval[num]}_{frame[num]}'
        file_add_mat = os.path.join(prefix, filename)
        matrix, size_dict = load_matrix(file_add_mat)
        self.range_fft = _range_fft_(hyper_dict, size_dict)
        if mvdr_type == 6: matrix = matrix[:, :, 8:, 1:2]
        if mvdr_type == 7: matrix = matrix[:, :, 8:, 3:4]
        if mvdr_type == 8: matrix = matrix[:, :, 8:, 5:6]
        self.range_fft.apply_fft(matrix)
        self.mvdrresult = self.process_mvdr.acc_process_2(self.range_fft.output.numpy())
        self.mvdrresult = torch.from_numpy(self.mvdrresult)
        self.plot_angles = torch.from_numpy(np.radians(np.arange(90, -91, -1)))

    def process_xy_cluster_power(self, cluster, diagonal_labels):
        coords, labels, mask = convert_cluster_xy_mvdr(cluster, diagonal_labels, self.range_fft.hyper, \
            self.RESOLUTION, self.RATIO, self.y_axis, self.x_axis, self.plot_angles)
        return coords, labels, mask

    def compute_detection(self, matrix):
        cfar_detection = self.post_cfar_acc_class_fg.process(matrix)
        return cfar_detection

    def compute_cluster(self, cfar_detection):
        try: self.labels, self.clusters = Dbscan_coord(cfar_detection, self.AREA, self.MIN_COUNT)
        except:
            self.labels, self.clusters = torch.zeros((1,), dtype=torch.int), torch.zeros((1,2), dtype=torch.int)
            print('-------- There is no cluster for this frame --------')
        return self.labels, self.clusters

    def gaus_normalize(self, matrix):
        matrix = gaussian_filter(matrix.numpy(), sigma=self.gau_sigma)
        matrix = torch.from_numpy(matrix)
        return matrix

    def plot_range(self, range_angle, address):
        plt.figure(figsize=(10, 6)); #plt.pcolormesh(angles, ranges, range_angle[1:-1, 1:-1], shading='auto', cmap='jet')
        plt.imshow(range_angle, cmap='jet', aspect='auto')
        plt.xlabel("Angle (degrees)")#; plt.colorbar(label='Power')
        plt.ylabel("Range (cm)"); plt.title("MVDR DOA Estimation")
        plt.savefig(address, dpi=300, bbox_inches='tight'); plt.close()

    def plot_cluster_debug(self, labels, cluster, mags, address, title="empty"):
        good_labels = torch.unique(labels)
        plt.figure(figsize=(8, 8))
        for label in good_labels:
            cluster_coords = cluster[labels == label]
            mags_coords = torch.max(mags[labels == label])/torch.max(mags)
            plt.scatter(cluster_coords[:, 1], cluster_coords[:, 0], label=f'Cluter {label}_{mags_coords:01.3f}')
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


def run(case, trajectories=None, txs=MVDR_TXS, visual_freq=10, batch=1,
        start_index=0, max_frames=None, doppler_dim=16):
    scene = get_scene(case)
    trajectories = trajectories or scene.trajectories

    for mvdr_type in txs:
        range_class = range_angle(doppler_dim)
        bk_class = bk_case_scene_bkg_refine(str(scene.background_path()))
        range_class.bg_points = bk_class.bg_points
        for traj in trajectories:
            # --------------- Chose the case of background based
            print(case, '================= WE RE PROCESSING =================', traj, "+++++", mvdr_type)
            '''--------- Creating directory for saving ---------'''
            prefix_save = scene.detect_dir(traj, mvdr_tag(mvdr_type))
            print('================= WE RE SAVING TO =================', prefix_save)
            prefix_save_cl_img = prefix_save / 'image_batch_diagonal_accelerating_img'   # Where we save image.
            prefix_save_cl = prefix_save / 'image_batch_diagonal_accelerating_torch'     # Where we save torch.
            prefix_save_cl_img.mkdir(parents=True, exist_ok=True)
            prefix_save_cl.mkdir(parents=True, exist_ok=True)
            prefix_save_cl_img, prefix_save_cl = str(prefix_save_cl_img), str(prefix_save_cl)
            bg_tensor_data = {"center": range_class.np_center, "bg_points": range_class.bg_points,}
            ADDRESS_DICT = os.path.join(prefix_save_cl, f'combine_bg.pth'); torch.save(bg_tensor_data, ADDRESS_DICT)
            prefix = str(scene.frame_dir(traj))
            try: interval, frame = ordered_frames(prefix)
            except Exception: print(traj, '----------- This is the one doesnt work'); continue
            '''--------- Batching ---------'''
            batch_lists = batch_indices(len(frame), batch, max_frames)
            '''--------- Processing Based On Batch ---------'''
            for index, batch_list in tqdm(enumerate(batch_lists[:])):
                if index >= start_index:
                    mvdrresult = []
                    # Applying mvdr to processing data
                    for num in batch_list:
                        range_class.process_mvdr_final(prefix, num, interval, frame, mvdr_type)
                        mvdrresult.append(range_class.mvdrresult)
                    range_class.mvdrresult = torch.abs(torch.mean(torch.stack(mvdrresult, dim=0), dim=0))
                    # Applying CFAR to detection
                    mvdrresult_cfar = range_class.compute_detection(range_class.gaus_normalize(range_class.mvdrresult.float()))  # Gaus&Det step 2&3
                    range_class.mvdrresult_cfar = mvdrresult_cfar
                    # Applying DBSCAN for clustering
                    mvdrresult_labels, mvdrresult_clusters = range_class.compute_cluster(range_class.mvdrresult_cfar)
                    # Converting
                    mvdr_array_clusters_xy, mvdr_array_labels_xy, mvdr_array_mask_xy = range_class.process_xy_cluster_power(mvdrresult_clusters, mvdrresult_labels)
                    mvdr_array_mags = range_class.mvdrresult[mvdrresult_clusters[mvdr_array_mask_xy,0], mvdrresult_clusters[mvdr_array_mask_xy,1]]
                    if visual_freq and index%visual_freq == 0:
                        address0 = os.path.join(prefix_save_cl_img, f'{index}_whole.png')
                        address1 = os.path.join(prefix_save_cl_img, f'{index}_heatmap.png')
                        range_class.plot_range(range_class.mvdrresult, address1)
                        address2 = os.path.join(prefix_save_cl_img, f'{index}_clusters.png')
                        range_class.plot_cluster_debug(mvdr_array_labels_xy, mvdr_array_clusters_xy, mvdr_array_mags, address2, "virtual_array")
                        range_class.concatenate_image_normal([address1, address2], address0)
                        os.remove(address1), os.remove(address2)
                    tensor_data = { "mvdr_array_clusters_xy": mvdr_array_clusters_xy,
                        "mvdr_array_labels_xy": mvdr_array_labels_xy, "mvdr_array_mags": mvdr_array_mags,}
                    ADDRESS_DICT = os.path.join(prefix_save_cl, f'combine_{index}.pth')
                    torch.save(tensor_data, ADDRESS_DICT)
