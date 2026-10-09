import math
import numpy as np
import numpy as np
import pdb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
import time

def normalize(matrix):
    return (matrix - matrix.min()) / (matrix.max() - matrix.min())


def default_device():
    """Where the CFAR convolutions run.

    The 3D CFAR convolves a 256^3 volume with a 13^3 kernel -- that needs a GPU
    (or >100 GB of RAM on CPU).  Override with RISE_DEVICE=cpu|cuda|mps.
    """
    import os
    forced = os.environ.get("RISE_DEVICE")
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"

class PostCFAR_1D_2D_v2:
    def __init__(self, data_size, num_guard_cells, num_reference_cells, threshold_factor, threshold_factor2, threshold_factor3, threshold_factor4, device=None):
        self.num_guard_cells = num_guard_cells
        self.num_reference_cells = num_reference_cells
        self.threshold_factor = threshold_factor; self.threshold_factor2 = threshold_factor2
        self.threshold_factor3 = threshold_factor3; self.threshold_factor4 = threshold_factor4
        device = self.device = device or default_device()
        window_size = num_reference_cells * 2 + num_guard_cells * 2 + 1
        self.kernel = torch.ones((1, 1, window_size), dtype=torch.float32, device=device)
        self.kernel[:, :, num_reference_cells:num_reference_cells + num_guard_cells * 2 + 1] = 0
        self.edge_size = num_reference_cells + num_guard_cells
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=window_size, stride=1, padding=self.edge_size, bias=False)
        self.conv1d.weight.data = self.kernel
        self.conv1d.to(device)
        self.reference_cells_count = self.conv1d(torch.ones((1, 1, data_size[0]), device=device)).squeeze(0).squeeze(0)
        self.mask = torch.ones((data_size[0]), device=device)
        self.mask[:self.edge_size] = 0; self.mask[-self.edge_size:] = 0

        self.reference_cells_count_dim1 = self.conv1d(torch.ones((1, 1, data_size[1]), device=device)).squeeze(0).squeeze(0)
        self.mask_dim1 = torch.ones((data_size[1]), device=device)
        self.mask_dim1[:self.edge_size] = 0; self.mask_dim1[-self.edge_size:] = 0

    def process(self, data):
        # (256, 256) (range_dim, angle_dim)
        data = data.to(self.device); data = normalize(data)  # Normalize the data tensor
        # --------- processing range (conv1d along range) --------- 
        general_mean = data.mean()
        data_rg = data.permute(1, 0)  # selected_range, angle -> angle, selected_range
        whole_mean = data_rg.mean(dim=0, keepdim=True) # 1, selected_range
        reference_cells_sum_rg = self.conv1d(data_rg.unsqueeze(1)).squeeze(1)  # (angle, range_dim_processed)
        reference_cells_mean_rg = reference_cells_sum_rg / self.reference_cells_count
        squared_diff_rg = ((data_rg - reference_cells_mean_rg) ** 2 / self.reference_cells_count)
        reference_cells_var_rg = self.conv1d(squared_diff_rg.unsqueeze(1)).squeeze(1).sqrt()  # (angle, range_dim_processed)
        threshold_rg = self.threshold_factor * (reference_cells_mean_rg + reference_cells_var_rg * self.threshold_factor2) + whole_mean * self.threshold_factor3  + general_mean * self.threshold_factor4
        cfar_detection_rg = (data_rg > threshold_rg).float().permute(1, 0) # angle, selected_range -> selected_range, angle
        cfar_detection_rg *= self.mask[:, None] # selected_range, angle
        # --------- processing angle (conv1d along angle) ---------
        #data_an_rg = data[cfar_detection_rg.bool()] # selected_range, angle
        data_an_rg = data  # selected_range, angle
        whole_mean = data_an_rg.mean(dim=0, keepdim=True) # 1, angle
        reference_cells_sum_an_rg = self.conv1d(data_an_rg.unsqueeze(1)).squeeze(1)  # (range_dim_processed, angle)
        reference_cells_mean_an_rg = reference_cells_sum_an_rg / self.reference_cells_count_dim1
        squared_diff_an_rg = ((data_an_rg - reference_cells_mean_an_rg) ** 2 / self.reference_cells_count_dim1)
        reference_cells_var_an_rg = self.conv1d(squared_diff_an_rg.unsqueeze(1)).squeeze(1).sqrt()  # (range_dim_processed, angle)
        threshold_an_rg = self.threshold_factor * (reference_cells_mean_an_rg + reference_cells_var_an_rg * self.threshold_factor2) + whole_mean * self.threshold_factor3  + general_mean * self.threshold_factor4
        cfar_detection_an_rg = (data_an_rg > threshold_an_rg).float()
        cfar_detection_an_rg *= self.mask_dim1[None,:] # selected_range, angle
        # --------- return final prediction ---------
        cfar_detection = cfar_detection_rg * cfar_detection_an_rg
        # If no detection occurs, ensure at least one cell is detected
        if cfar_detection.sum() == 0:
            max_index = data.argmax().cpu()
            row, col = divmod(max_index.item(), data.shape[1])  # Convert to 2D index
            cfar_detection[row-2:row+3, col-2:col+3] = 1
        return cfar_detection.cpu()

class PostCFAR_1D_3D_v2:
    def __init__(self, data_size, num_guard_cells, num_reference_cells, threshold_factor, threshold_factor2, threshold_factor3, threshold_factor4, device=None):
        self.num_guard_cells = num_guard_cells; self.num_reference_cells = num_reference_cells
        self.threshold_factor = threshold_factor; self.threshold_factor2 = threshold_factor2
        self.threshold_factor3 = threshold_factor3; self.threshold_factor4 = threshold_factor4
        device = self.device = device or default_device()
        window_size = num_reference_cells * 2 + num_guard_cells * 2 + 1
        self.kernel = torch.ones((1, 1, window_size), dtype=torch.float32, device=device)
        self.kernel[:, :, num_reference_cells:num_reference_cells + num_guard_cells * 2 + 1] = 0
        self.edge_size = num_reference_cells + num_guard_cells
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=window_size, stride=1, padding=self.edge_size, bias=False)
        self.conv1d.weight.data = self.kernel
        self.conv1d.to(device)
        self.reference_cells_count = self.conv1d(torch.ones((1, 1, data_size[0]), device=device)).squeeze(0).squeeze(0)
        self.mask = torch.ones((data_size[0]), device=device)
        self.mask[:self.edge_size] = 0
        self.mask[-self.edge_size:] = 0
    def process(self, data):
        # data 256, 256, 256 (range, Aoa, Aod)
        data = data[:, 0]; data = normalize(data)  # Normalize the data tensor
        range_dim, aoa_dim, aod_dim = data.size() 
        # --------- along range
        general_mean = data.mean()
        data_rg = data.permute(1,2,0).view(aoa_dim*aod_dim, range_dim) # (range, Aoa, Aod) - > Aoa*Aod, range
        whole_mean = data_rg.mean(dim=0, keepdim=True) # 1, range
        reference_cells_sum_rg = self.conv1d(data_rg.unsqueeze(1)).squeeze(1)  # (Aoa*Aod, range)
        reference_cells_mean_rg = reference_cells_sum_rg / self.reference_cells_count
        squared_diff_rg = ((data_rg - reference_cells_mean_rg) ** 2 / self.reference_cells_count)
        reference_cells_var_rg = self.conv1d(squared_diff_rg.unsqueeze(1)).squeeze(1).sqrt()  # (Aoa*Aod, range)
        threshold_rg = self.threshold_factor * (reference_cells_mean_rg + reference_cells_var_rg * self.threshold_factor2) + whole_mean * self.threshold_factor3  + general_mean * self.threshold_factor4
        cfar_detection_rg = (data_rg > threshold_rg).float() # (Aoa*Aod, range)
        cfar_detection_rg *= self.mask[None, :] # (Aoa*Aod, range)
        cfar_detection_rg = cfar_detection_rg.permute(1, 0).view(range_dim, aoa_dim, aod_dim)
        # --------- along AOA
        data_aoa = data.permute(0,2,1).contiguous().view(range_dim*aod_dim, aoa_dim) # (range, Aoa, Aod) - > range*Aod, Aoa
        whole_mean = data_aoa.mean(dim=0, keepdim=True) # (1, Aoa)
        reference_cells_sum_aoa = self.conv1d(data_aoa.unsqueeze(1)).squeeze(1)  # (range*Aod, Aoa)
        reference_cells_mean_aoa = reference_cells_sum_aoa / self.reference_cells_count
        squared_diff_aoa = ((data_aoa - reference_cells_mean_aoa) ** 2 / self.reference_cells_count)
        reference_cells_var_aoa = self.conv1d(squared_diff_aoa.unsqueeze(1)).squeeze(1).sqrt()  # (range*Aod, Aoa)
        threshold_aoa = self.threshold_factor * (reference_cells_mean_aoa + reference_cells_var_aoa * self.threshold_factor2) + whole_mean * self.threshold_factor3  + general_mean * self.threshold_factor4
        cfar_detection_aoa = (data_aoa > threshold_aoa).float() # (range*Aod, Aoa)
        cfar_detection_aoa *= self.mask[None, :] # (range*Aod, Aoa)
        cfar_detection_aoa = cfar_detection_aoa.view(range_dim, aod_dim, aoa_dim).permute(0, 2, 1) # (range, Aoa, Aod)
        # --------- along AOD
        data_aod = data.view(range_dim*aoa_dim, aod_dim) # (range, Aoa, Aod) - > range*Aoa, Aod
        whole_mean = data_aod.mean(dim=0, keepdim=True) # (1, Aod)
        reference_cells_sum_aod = self.conv1d(data_aod.unsqueeze(1)).squeeze(1)  # (range*Aoa, Aod)
        reference_cells_mean_aod = reference_cells_sum_aod / self.reference_cells_count
        squared_diff_aod = ((data_aod - reference_cells_mean_aod) ** 2 / self.reference_cells_count)
        reference_cells_var_aod = self.conv1d(squared_diff_aod.unsqueeze(1)).squeeze(1).sqrt()  # (range*Aoa, Aod)
        threshold_aod = self.threshold_factor * (reference_cells_mean_aod + reference_cells_var_aod * self.threshold_factor2) + whole_mean * self.threshold_factor3  + general_mean * self.threshold_factor4
        cfar_detection_aod = (data_aod > threshold_aod).float() # (range*Aoa, Aod)
        cfar_detection_aod *= self.mask[None, :] # (range*Aoa, Aod)
        cfar_detection_aod = cfar_detection_aod.view(range_dim, aoa_dim, aod_dim)
        cfar_detection = cfar_detection_rg * cfar_detection_aoa * cfar_detection_aod
        return cfar_detection

class post_cfar_acc_class:
    def __init__(self, data_size, num_guard_cells, num_reference_cells, threshold_factor, threshold_factor2, threshold_factor3, threshold_factor4=0, device=None):
        # using convolutional 2D for cfar detection.
        self.num_guard_cells = num_guard_cells; self.num_reference_cells = num_reference_cells
        self.threshold_factor = threshold_factor; self.threshold_factor2 = threshold_factor2 
        self.threshold_factor3 = threshold_factor3; self.threshold_factor4=threshold_factor4
        device = self.device = device or default_device()
        window_size = num_reference_cells * 2 + num_guard_cells * 2 + 1
        self.kernel = torch.ones((window_size, window_size), dtype=torch.float32, device=device)
        self.kernel[num_reference_cells:num_reference_cells + num_guard_cells * 2 + 1,
               num_reference_cells:num_reference_cells + num_guard_cells * 2 + 1] = 0
        self.edge_size = num_reference_cells + num_guard_cells
        self.reference_cells_count = F.conv2d(torch.ones(data_size).unsqueeze(0).unsqueeze(0).to(self.device), self.kernel.unsqueeze(0).unsqueeze(0), stride=1, padding=self.edge_size)
        self.mask = torch.ones(data_size).to(device)
        self.mask[:self.edge_size, :] = 0; self.mask[-self.edge_size:, :] = 0  # Bottom edge
        self.mask[:, :self.edge_size] = 0; self.mask[:, -self.edge_size:] = 0  # Right edge
    def process(self, data):
        data = data.to(self.device)
        data = normalize(data)  # Normalize the data tensor
        reference_cells_sum = F.conv2d(data.unsqueeze(0).unsqueeze(0), self.kernel.unsqueeze(0).unsqueeze(0), stride=1, padding=self.edge_size)
        reference_cells_mean = reference_cells_sum.squeeze(0).squeeze(0) / self.reference_cells_count.squeeze(0).squeeze(0)
        squared_diff = (data.unsqueeze(0).unsqueeze(0) - reference_cells_mean.unsqueeze(0).unsqueeze(0))**2 / self.reference_cells_count.squeeze(0).squeeze(0)
        reference_cells_var = F.conv2d(squared_diff, self.kernel.unsqueeze(0).unsqueeze(0), stride=1, padding=self.edge_size).squeeze(0).squeeze(0)
        noise_level3 = data.mean(dim=1); noise_level4 = data.mean()
        # Perform CFAR detection using thresholds
        threshold = reference_cells_mean * (self.threshold_factor + reference_cells_var * self.threshold_factor2) + noise_level3.unsqueeze(1) * self.threshold_factor3 + noise_level4 * self.threshold_factor4
        cfar_detection = (data > threshold).float()  # Element-wise comparison to threshold
        cfar_detection *= self.mask
        # If no detection occurs, ensure at least one cell is detected
        if cfar_detection.sum() == 0:
            max_index = torch.unravel_index(data.argmax(), data.shape)
            cfar_detection[max_index[0], max_index[1]] = 1
        return cfar_detection.cpu()

def Dbscan_coord(matrix, AREA=5, MIN_COUNT = 10):
    H, W = matrix.shape
    THRESHOLD = 0.5
    coords = np.array(np.nonzero(matrix.numpy() > THRESHOLD)).T
    dbscan = DBSCAN(eps=AREA, min_samples=1)  
    labels = dbscan.fit_predict(coords)
    uniq_label, count = np.unique(labels, return_counts=True)
    mask = count > MIN_COUNT
    good_labels = uniq_label[mask]
    np_good_clusters = []
    np_good_labels = []
    for label in good_labels:
        cluster_coords = coords[labels == label]
        np_good_clusters.append(cluster_coords)
        np_good_labels += [label for i in range(len(cluster_coords))]
    try:
        np_good_clusters = np.concatenate(np_good_clusters, axis=0)
        np_good_labels = np.array(np_good_labels)
    except:
        np_good_clusters = np.zeros((1, 2), dtype=int)
        np_good_labels = np.zeros((1,), dtype=int)
    if len(np_good_clusters) == 0 or len(np_good_labels) == 0:
        np_good_clusters = np.zeros((1, 2), dtype=int)
        np_good_labels = np.zeros((1,), dtype=int)
    return torch.from_numpy(np_good_labels), torch.from_numpy(np_good_clusters)

def Dbscan_coord_3d(matrix, AREA=5, MIN_COUNT = 10):
    THRESHOLD = 0.5
    coords = np.array(np.nonzero(matrix.numpy() > THRESHOLD)).T
    dbscan = DBSCAN(eps=AREA, min_samples=1)  
    labels = dbscan.fit_predict(coords)
    uniq_label, count = np.unique(labels, return_counts=True)
    mask = count > MIN_COUNT
    good_labels = uniq_label[mask]
    np_good_clusters = []
    np_good_labels = []
    for label in good_labels:
        cluster_coords = coords[labels == label]
        np_good_clusters.append(cluster_coords)
        np_good_labels += [label for i in range(len(cluster_coords))]
    try:
        np_good_clusters = np.concatenate(np_good_clusters, axis=0)
        np_good_labels = np.array(np_good_labels)
    except:
        np_good_clusters = np.zeros((1, 3), dtype=int)
        np_good_labels = np.zeros((1,), dtype=int)
    if len(np_good_clusters) == 0 or len(np_good_labels) == 0:
        np_good_clusters = np.zeros((1, 3), dtype=int)
        np_good_labels = np.zeros((1,), dtype=int)
    return torch.from_numpy(np_good_labels), torch.from_numpy(np_good_clusters)

def ghost_enhance_3d(diagonal_cfar_3d_conv, MIN_NUM, AD_DIFF, _AREA_, _MIN_COUNT_):
    # we want to know the position of real object. 
    # this one is used to enhance the ghost capability.
    """
    1. mask > min_num; 2. detecting the cluster of diagonal
    3. they shouldn't touch diagnoal; 4. the aoa of the mask should equal to real-object. (considering that the).
    """
    upper_triangle1 = torch.triu(diagonal_cfar_3d_conv)
    upper_triangle2 = torch.triu(diagonal_cfar_3d_conv.transpose(2,1))
    diagonal_system_msk = upper_triangle1 + upper_triangle2 > 1
    diagoanl_sum =  torch.sum(diagonal_system_msk, dim=(1, 2)) > MIN_NUM
    indices = torch.nonzero(diagoanl_sum).squeeze(1) # Correct
    output = torch.zeros_like(diagonal_system_msk)
    output[indices] = diagonal_system_msk[indices]
    # iterating along range dimension
    np_good_labels, np_good_clusters = Dbscan_coord_3d(output, AREA=_AREA_, MIN_COUNT =_MIN_COUNT_)
    clusters_list, labels_list, mag_list = [], [], []
    for label in torch.unique(np_good_labels):
        cluster_coords = np_good_clusters[np_good_labels == label]
        centeroid = torch.mean(cluster_coords.float(), dim=0)
        mag_cluster = diagonal_cfar_3d_conv[cluster_coords]
        mag_list.append(mag_cluster)
        _, y, z = centeroid[0], centeroid[1], centeroid[2] # range, aoa, aod
        if abs(y-z) >= AD_DIFF:
            clusters_copy = cluster_coords.clone()
            clusters_copy[:, [1, 2]] = clusters_copy[:, [2, 1]]
            mag_cluster2 = diagonal_cfar_3d_conv[clusters_copy]
            labels_copy = torch.tensor([label+1500]).repeat(len(clusters_copy))
            np_good_labels[np_good_labels == label] += 500
            # Adding 500 points being copied, adding 1500 to points be generated by copying.
            clusters_list.append(clusters_copy); labels_list.append(labels_copy); mag_list.append(mag_cluster2)
    if len(clusters_list) > 0:
        clusters_final = torch.cat([np_good_clusters, torch.cat(clusters_list, dim=0)], dim=0)
        labels_final = torch.cat([np_good_labels, torch.cat(labels_list)], dim=0)
        mags_final = torch.cat(mag_list, dim=0)
    else:
        clusters_final = np_good_clusters; labels_final = np_good_labels
        mags_final = torch.cat(mag_list, dim=0)
    return labels_final, clusters_final, mags_final

class post_cfar_acc_3d_conv_class:
    def __init__(self, data_size, num_guard_cells, num_reference_cells, threshold_factor, threshold_factor2, threshold_factor3, threshold_factor4=0, device=None):
        # Using convolutional 3D for CFAR detection
        self.num_guard_cells = num_guard_cells; self.num_reference_cells = num_reference_cells
        self.threshold_factor = threshold_factor; self.threshold_factor2 = threshold_factor2
        self.threshold_factor3 = threshold_factor3; self.threshold_factor4 = threshold_factor4
        device = self.device = device or default_device()
        window_size = num_reference_cells * 2 + num_guard_cells * 2 + 1
        self.kernel = torch.ones((window_size, window_size, window_size), dtype=torch.float32, device=device)
        self.kernel[num_reference_cells:num_reference_cells + num_guard_cells * 2 + 1,
            num_reference_cells:num_reference_cells + num_guard_cells * 2 + 1,
            num_reference_cells:num_reference_cells + num_guard_cells * 2 + 1] = 0
        self.edge_size = num_reference_cells + num_guard_cells
        self.reference_cells_count = F.conv3d(torch.ones(data_size, device=device).unsqueeze(0).unsqueeze(0), self.kernel.unsqueeze(0).unsqueeze(0), stride=1, padding=self.edge_size)
        self.mask = torch.ones(data_size, device=device)
        self.mask[:self.edge_size, :, :] = 0; self.mask[-self.edge_size:, :, :] = 0  # Bottom edge
        self.mask[:, :self.edge_size, :] = 0; self.mask[:, -self.edge_size:, :] = 0  # Right edge
        self.mask[:, :, :self.edge_size] = 0; self.mask[:, :, -self.edge_size:] = 0  # Back edge (depth axis)
    def process(self, data):
        data = data[:, 0].to(self.device)
        data = normalize(data)  # Normalize the data tensor
        reference_cells_sum = F.conv3d(data.unsqueeze(0).unsqueeze(0), self.kernel.unsqueeze(0).unsqueeze(0), stride=1, padding=self.edge_size)
        reference_cells_mean = reference_cells_sum.squeeze(0).squeeze(0) / self.reference_cells_count.squeeze(0).squeeze(0)
        squared_diff = (data.unsqueeze(0).unsqueeze(0) - reference_cells_mean.unsqueeze(0).unsqueeze(0)) ** 2 / self.reference_cells_count.squeeze(0).squeeze(0)
        reference_cells_var = F.conv3d(squared_diff, self.kernel.unsqueeze(0).unsqueeze(0), stride=1, padding=self.edge_size).squeeze(0).squeeze(0)
        noise_level3 = data.mean(dim=(1, 2))  # Take the mean along dimension 1 (height)
        noise_level4 = data.mean()  # Take the mean along dimension 1 (height)
        # Perform CFAR detection using thresholds
        threshold = reference_cells_mean * (self.threshold_factor + reference_cells_var * self.threshold_factor2) + noise_level3[:,None,None] * self.threshold_factor3 + noise_level4 * self.threshold_factor4
        cfar_detection = (data > threshold).float()  # Element-wise comparison to threshold
        cfar_detection *= self.mask
        return cfar_detection.cpu()