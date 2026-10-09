import os
import json
import numpy as np
import scipy.io
import pdb 
import glob
from scipy import interpolate
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

def list_json(directory):
    json_files = glob.glob(os.path.join(directory, '*.json'))
    interval = []
    frame = []
    for file in json_files:
        pre_file = file.split('/')[-1].split('.')[0]
        _, _, cur_int, cur_fr = pre_file.split('_')
        interval.append(cur_int)
        frame.append(int(cur_fr))    
    paired = list(zip(frame, interval))
    sorted_paired = sorted(paired)
    frame, interval = zip(*sorted_paired)
    interval = list(interval)
    frame = list(frame)
    frame = [str(f) for f in frame]
    return interval, frame

def load_hyperparam(file_add):
    with open(file_add, 'r') as file:
        data = json.load(file)
    keys = [key for key in data.keys()]
    for key in keys:
        try:
            new_tensor = torch.tensor(data[key])
            data[key] = new_tensor
        except:
            Flag = True
    return data

def load_matrix(file_name):
    """Load one preprocessed radar frame as a (range, doppler, rx, tx) complex tensor.

    Two on-disk forms are accepted, in this order:

      <file_name>.npy   python-native complex64 cube  (written by tools/pack_frames.py)
      <file_name>.mat   MATLAB v5 complex double cube (written by 2_preprocess)

    The `.npy` form is what the released dataset ships: same array, half the bytes,
    and no MATLAB anywhere in the consumer's pipeline.  See DATA.md.
    """
    npy_path = file_name if file_name.endswith('.npy') else file_name + '.npy'
    if os.path.exists(npy_path):
        array = np.load(npy_path)
        matrix = torch.from_numpy(np.ascontiguousarray(array).astype(np.complex64))
    else:
        mat_data = scipy.io.loadmat(file_name)
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        matrix = torch.from_numpy(mat_data[keys[0]])
    rangesize = matrix.shape[0]
    dopplersize = matrix.shape[1]
    rxsize = matrix.shape[2]
    txsize = matrix.shape[3]
    size_dict = {"range_size": rangesize,
            "doppler_size": dopplersize,
            "rx_size": rxsize,
            "tx_size": txsize}
    return matrix, size_dict

def generate_diagonal(volume):
    # volume: numrange, numdoppler, numrx, numtx
    numrange, numdoppler, numrx, numtx = volume.shape
    volume = volume.permute(2, 3, 0, 1)
    assert numrx == numtx, "Error: last two dimensions are not equals."
    mask = torch.ones((numrx, numtx), dtype=torch.float32)
    mask_dia = torch.tril(torch.triu(mask, 0), 0)
    mask_non_dia = 1 - mask_dia
    volume_dia = volume * mask_dia[:, :, None, None]  # numrx, numtx, numrange, numdoppler
    volume_non_dia = volume * mask_non_dia[:, :, None, None]  # numrx, numtx, numrange, numdoppler
    count_dia = torch.sum(mask_dia, dim=1)  # numrx, numrange, numdoppler
    count_non_dia = torch.sum(mask_non_dia, dim=1)  # numrx, numrange, numdoppler
    volume_dia_avg = torch.sum(volume_dia, dim=1) / count_dia[:, None, None]  # numrx, numrange, numdoppler
    volume_non_dia_avg = torch.sum(volume_non_dia, dim=1) / count_non_dia[:, None, None]  # numrx, numrange, numdoppler
    volume_dia_avg = volume_dia_avg.permute(1, 2, 0)  # numrange, numdoppler, numrx
    volume_non_dia_avg = volume_non_dia_avg.permute(1, 2, 0)  # numrange, numdoppler, numrx
    return volume_dia, volume_non_dia, volume_dia_avg, volume_non_dia_avg

def convert_ra_xy(output_1d, range_hyper, angle_hyper, RESOLUTION, RATIO, y_axis_ratio, x_axis_ratio, angles=None):
    # matrix: Range, Doppler, Angle.
    output_1d = output_1d.permute(0, 2, 1)  # prevent possible bug
    d = 1  # time of half wavelength
    wavelength = 2 * d
    TOTAL = 15
    NAN_VALUE = -999
    rangeBinSize = range_hyper['rangeBinSize']
    n_range_fft_size = range_hyper['rangeFFTSize']
    minRangeBinKeep = range_hyper['minRangeBinKeep']
    rightRangeBinDiscard = range_hyper['rightRangeBinDiscard']
    # Sine and Cosine of the angles (convert to tensor)
    low_indices, up_indices = int(minRangeBinKeep), int(RATIO * (n_range_fft_size - rightRangeBinDiscard - minRangeBinKeep) + minRangeBinKeep)
    indices_1D = torch.arange(low_indices, up_indices)
    if angles is None:
        n_angle_fft_size = angle_hyper['azimuthFFTSize']
        angles = torch.arange(-n_angle_fft_size/2, n_angle_fft_size/2 + 1)  # f_n​= n⋅f_s/N
        sine_theta = - wavelength * (angles / n_angle_fft_size) / d  # reverse the direction
        cos_theta = torch.sqrt(1 - sine_theta**2)
        radar_data_angle_range_Static = output_1d
        _radar_data_angle_range_Static_ = radar_data_angle_range_Static[indices_1D]
        _radar_data_angle_range_Static_ = torch.cat((_radar_data_angle_range_Static_, _radar_data_angle_range_Static_[:,0:1]), dim=1)
    else:
        sine_theta = torch.sin(angles)
        cos_theta = torch.cos(angles)
        radar_data_angle_range_Static = output_1d
        _radar_data_angle_range_Static_ = radar_data_angle_range_Static[indices_1D]
        _radar_data_angle_range_Static_ = _radar_data_angle_range_Static_.permute(1, 0, 2).flip(0) # 256, 154, 5
    # Create meshgrid and compute x and y axis values
    R_mat, sine_theta_mat = torch.meshgrid(indices_1D * rangeBinSize, sine_theta, indexing="ij")
    _, cos_theta_mat = torch.meshgrid(indices_1D.float(), cos_theta, indexing="ij")
    R_mat, sine_theta_mat = R_mat.transpose(1,0), sine_theta_mat.transpose(1,0)
    cos_theta_mat = cos_theta_mat.transpose(1,0)
    x_axis = R_mat * cos_theta_mat
    y_axis = R_mat * sine_theta_mat
    # ------ Convert the unit ------
    min_value = _radar_data_angle_range_Static_.min()
    y_axis = ((y_axis + RATIO * TOTAL) * RESOLUTION).to(torch.int32)  # RATIO*TOTAL due to the center
    x_axis = (x_axis * RESOLUTION).to(torch.int32)
    # ------ Cropping ------
    max_y = int(RATIO * TOTAL * RESOLUTION * y_axis_ratio + RATIO * TOTAL * RESOLUTION)
    min_y = int(RATIO * TOTAL * RESOLUTION - RATIO * TOTAL * RESOLUTION * y_axis_ratio)
    crop_matrix1 = (y_axis < max_y) & (y_axis > min_y)
    y_axis -= min_y
    max_x = int(RATIO * TOTAL * RESOLUTION * x_axis_ratio)
    crop_matrix2 = x_axis < max_x
    crop_matrix = crop_matrix1 & crop_matrix2
    # ------ Assignment ------
    matrix = torch.ones((int(RATIO * TOTAL * RESOLUTION * 2 * y_axis_ratio), int(RATIO * TOTAL * RESOLUTION), _radar_data_angle_range_Static_.shape[2])) * NAN_VALUE
    matrix[y_axis[crop_matrix], x_axis[crop_matrix], :] = _radar_data_angle_range_Static_[crop_matrix]
    # ------ Fill up empty space ------
    matrix1 = interpolate_matrix(matrix)
    matrix1[torch.isnan(matrix1)] = min_value
    matrix1[matrix1 < min_value] = min_value
    # ------ Useless Area ------
    distance = int(minRangeBinKeep) * rangeBinSize * RESOLUTION
    x_center, y_center = RATIO * TOTAL * RESOLUTION, 0
    # Create a grid of coordinates
    x, y = torch.meshgrid(torch.arange(matrix1.shape[0]), torch.arange(matrix1.shape[1]), indexing="ij")
    distances = torch.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    mask = distances <= distance
    matrix1[mask] = min_value
    center = [RATIO * TOTAL * RESOLUTION * y_axis_ratio, 0]
    # output_1d: Range, Angle, Doppler
    matrix1 = matrix1.permute(0, 2, 1)  # prevent possible bug
    return matrix1, center

def convert_cluster_xy_mvdr(cluster_ori, labels_ori, range_hyper, RESOLUTION, RATIO, y_axis_ratio, x_axis_ratio, angles=None):
    # RATIO: 2/3 TOTAL: 15 RESOLUTION: 50
    # y_axis_ratio: 0.6 x_axis_ratio: 1.
    # matrix: cluster: N, 2
    cluster = cluster_ori.clone()
    labels = labels_ori.clone()
    d = 1  # time of half wavelength
    TOTAL = 15
    rangeBinSize = range_hyper['rangeBinSize']
    n_range_fft_size = range_hyper['rangeFFTSize']
    minRangeBinKeep = range_hyper['minRangeBinKeep']
    rightRangeBinDiscard = range_hyper['rightRangeBinDiscard']
    # Sine and Cosine of the angles (convert to tensor)
    sine_theta = torch.sin(angles)
    cos_theta = torch.cos(angles)
    
    low_indices, up_indices = int(minRangeBinKeep), int(RATIO * (n_range_fft_size - rightRangeBinDiscard - minRangeBinKeep) + minRangeBinKeep)
    indices_1D = torch.arange(low_indices, up_indices)

    R_mat, sine_theta_mat = torch.meshgrid(indices_1D.float() * rangeBinSize, sine_theta.float(), indexing="ij")
    _, cos_theta_mat = torch.meshgrid(indices_1D.float(), cos_theta.float(), indexing="ij")
    R_mat, sine_theta_mat = R_mat.transpose(1,0), sine_theta_mat.transpose(1,0)
    cos_theta_mat = cos_theta_mat.transpose(1,0)
    x_axis = R_mat * cos_theta_mat
    y_axis = R_mat * sine_theta_mat
    y_axis = ((y_axis + RATIO * TOTAL) * RESOLUTION).to(torch.int32)  # 把所有y轴调正
    x_axis = (x_axis * RESOLUTION).to(torch.int32)
    min_y = int(RATIO * TOTAL * RESOLUTION - RATIO * TOTAL * RESOLUTION * y_axis_ratio) # 
    y_axis -= min_y
    cluster[:,0] = cluster[:,0] - low_indices        
    mask = (cluster[:, 0] < up_indices-low_indices) & (cluster[:, 0] >= 0)
    cluster = cluster[mask]
    labels = labels[mask]
    x_coord, y_coord  = y_axis[cluster[:, 1], cluster[:, 0]], x_axis[cluster[:, 1], cluster[:, 0]]
    coords = torch.cat([x_coord[:, None], y_coord[:, None]], dim=-1)
    return coords, labels, mask

def convert_cluster_xy(cluster_ori, labels_ori, range_hyper, angle_hyper, RESOLUTION, RATIO, y_axis_ratio, x_axis_ratio, angles=None):
    # RATIO: 2/3 TOTAL: 15 RESOLUTION: 50
    # y_axis_ratio: 0.6 x_axis_ratio: 1.
    # matrix: cluster: N, 2
    cluster = cluster_ori.clone()
    labels = labels_ori.clone()
    d = 1  # time of half wavelength
    wavelength = 2 * d
    TOTAL = 15
    rangeBinSize = range_hyper['rangeBinSize']
    n_range_fft_size = range_hyper['rangeFFTSize']
    minRangeBinKeep = range_hyper['minRangeBinKeep']
    rightRangeBinDiscard = range_hyper['rightRangeBinDiscard']
    # Sine and Cosine of the angles (convert to tensor)
    if angles is None:
        n_angle_fft_size = angle_hyper['azimuthFFTSize']
        angles = torch.arange(-n_angle_fft_size/2, n_angle_fft_size/2 + 1)  # f_n​= n⋅f_s/N
        sine_theta = - wavelength * (angles / n_angle_fft_size) / d  # reverse the direction
        cos_theta = torch.sqrt(1 - sine_theta**2)
    else:
        sine_theta = torch.sin(angles)
        cos_theta = torch.cos(angles)
        cluster[:,1] = 255 - cluster[:,1]
    low_indices, up_indices = int(minRangeBinKeep), int(RATIO * (n_range_fft_size - rightRangeBinDiscard - minRangeBinKeep) + minRangeBinKeep)
    indices_1D = torch.arange(low_indices, up_indices)
    # Create meshgrid and compute x and y axis values
    R_mat, sine_theta_mat = torch.meshgrid(indices_1D * rangeBinSize, sine_theta, indexing="ij")
    _, cos_theta_mat = torch.meshgrid(indices_1D.float(), cos_theta, indexing="ij")
    R_mat, sine_theta_mat = R_mat.transpose(1,0), sine_theta_mat.transpose(1,0)
    cos_theta_mat = cos_theta_mat.transpose(1,0)
    x_axis = R_mat * cos_theta_mat
    y_axis = R_mat * sine_theta_mat
    y_axis = ((y_axis + RATIO * TOTAL) * RESOLUTION).to(torch.int32)  # 把所有y轴调正
    x_axis = (x_axis * RESOLUTION).to(torch.int32)
    min_y = int(RATIO * TOTAL * RESOLUTION - RATIO * TOTAL * RESOLUTION * y_axis_ratio) # 
    y_axis -= min_y
    cluster[:,0] = cluster[:,0] - low_indices        
    mask = (cluster[:, 0] < up_indices-low_indices) & (cluster[:, 0] >= 0)
    cluster = cluster[mask]
    labels = labels[mask]
    x_coord, y_coord  = y_axis[cluster[:, 1], cluster[:, 0]], x_axis[cluster[:, 1], cluster[:, 0]]
    coords = torch.cat([x_coord[:, None], y_coord[:, None]], dim=-1)
    return coords, labels, mask

def overlap_check(ori_cluster, new_cluster, new_label, check_distance=5):
    distance_matrix = torch.sqrt(((new_cluster[:, None] - ori_cluster[None, :]) ** 2).sum(dim=2))
    mask = (distance_matrix < check_distance).any(dim=1) # 找到位置上的overlapping
    new_label_common = torch.unique(new_label[mask]) # 找到共同的label
    mask_2 = torch.isin(new_label, new_label_common) # 找到和公共label匹配的label
    return new_cluster[~mask_2], new_label[~mask_2], ~mask_2

def overlap_check_sel(ori_cluster, new_cluster, new_label, check_distance=5):
    distance_matrix = torch.sqrt(((new_cluster[:, None] - ori_cluster[None, :]) ** 2).sum(dim=2))
    mask = (distance_matrix < check_distance).any(dim=1) # 找到位置上的overlapping
    new_label_common = torch.unique(new_label[mask]) # 找到共同的label
    mask_2 = torch.isin(new_label, new_label_common) # 找到和公共label匹配的label
    return new_cluster[~mask_2], new_label[~mask_2]

def interpolate_matrix(matrix):
    # X, Y, Doppler
    matrix = matrix.numpy()
    matrix[matrix<0] = np.nan
    Doppler = matrix.shape[2]
    x = np.arange(0, matrix.shape[1])
    y = np.arange(0, matrix.shape[0])
    GD1 = np.zeros((matrix.shape[0], matrix.shape[1], Doppler))
    for dop in range(Doppler):
        mat = matrix[:, :, dop]
        mat = np.ma.masked_invalid(mat)
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~mat.mask]
        y1 = yy[~mat.mask]
        newarr = mat[~mat.mask]
        GD1[:, :, dop] = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='cubic')
    return torch.from_numpy(GD1)

def low_resolution(image_clear):
    #cropped_image = image_clear[44:-44, :-128]
    cropped_image = image_clear
    intermediate_size = (cropped_image.shape[1] // 2, cropped_image.shape[0] // 2)
    resized_image = cv2.resize(cropped_image, intermediate_size, interpolation=cv2.INTER_AREA)
    resized_image = cv2.resize(resized_image, (128, 128), interpolation=cv2.INTER_AREA)
    resized_image = (resized_image > 0)*255.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    resized_image = (cv2.dilate(resized_image, kernel, iterations=1)>0).astype(float)*255.
    return resized_image

def cluster_similar_with_index(distances, threshold=0.5):
    # 先对列表进行排序，保存原索引
    indexed_distances = sorted(enumerate(distances), key=lambda x: x[1])
    clusters = []
    current_cluster = [indexed_distances[0][0]]  # 只存索引
    for i in range(1, len(indexed_distances)):
        if indexed_distances[i][1] - indexed_distances[i-1][1] <= threshold:
            current_cluster.append(indexed_distances[i][0])  # 存索引
        else:
            clusters.append(current_cluster)
            current_cluster = [indexed_distances[i][0]]
    clusters.append(current_cluster)  # 加入最后一个分组
    return clusters

def magnitude_compression(threed_clusters, threed_labels, threed_mags, center, threshold=10):
    label_mask = (threed_labels >= 500) * (threed_labels < 1500)
    mags_list = []; distance_list = []; label_list = []
    for labels in torch.unique(threed_labels[label_mask]):
        mags_list.append(torch.max(threed_mags[threed_labels==labels]))
        distance_list.append(torch.mean(torch.norm(threed_clusters[threed_labels==labels] - center[None, :], dim=1)))
        label_list.append(labels)
    clusters_indexes = cluster_similar_with_index(distance_list, threshold=threshold)
    label_list2 = []
    for indexes in clusters_indexes:
        max_idx = torch.argmax(torch.tensor(mags_list)[indexes])
        label_list2.append(label_list[max_idx])
        label_list2.append(label_list[max_idx]+1000)
    return torch.isin(threed_labels, torch.tensor(label_list2))

if __name__ == "__main__":
    distance_list = [1.0, 1.2, 1.3, 3.0, 3.1, 5.0, 5.05, 5.1]
    clusters_indexes = cluster_similar_with_index(distance_list, threshold=0.2)
    print("Clusters (indexes):", clusters_indexes)