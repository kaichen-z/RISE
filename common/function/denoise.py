import numpy as np
import pdb
# get a list of real objects.
# computing the median distance between each points.
# if it is far away from mean of windows, eliminating it. 
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from collections import defaultdict

class DENOISE_SPACE:
    def __init__(self,):
        self.neighbors = 10
        self.distance_threshold = 25
        self.distance_threshold2 = 25
        self.num_threshold = 2
        self.num_threshold2 = 2

    def denoise_current(self, sor_mags, sor_means, sor_ranges, sor_thetas, sor_dopplers, sor_shapes, sor_nums, sor_labs):
        reals_object = [i[0,0] for i in sor_means] 
        self.lof = LocalOutlierFactor(n_neighbors=self.neighbors)
        outliers = self.lof.fit_predict(reals_object)
        self.indices = np.where(outliers < 0)[0]
        self.inv_indices = np.where(outliers > 0)[0]
        sor_mags1, sor_means1, sor_ranges1, sor_thetas1, sor_dopplers1, sor_shapes1, sor_nums1, sor_labs1 = \
            self.copyln(sor_mags), self.copyln(sor_means), self.copyln(sor_ranges), self.copyln(sor_thetas), \
            self.copyln(sor_dopplers), self.copyln(sor_shapes), self.copyln(sor_nums), self.copyln(sor_labs)
        for indice in self.indices:
            whole_indices = [i for i in range(len(sor_mags))]
            previous_frames = whole_indices[:indice]
            previous_frames = [item for item in previous_frames if item not in self.indices]
            next_frames = whole_indices[indice+1:]
            next_frames = [item for item in next_frames if item not in self.indices]
            dist = 0
            if len(previous_frames)>0:
                dist += np.linalg.norm(sor_means[indice][0] - sor_means[previous_frames[-1]][0][0][None], axis=1)
            if len(next_frames)>0:
                dist += np.linalg.norm(sor_means[indice][0] - sor_means[next_frames[0]][0][0][None], axis=1)
            output = self.reorder(dist, sor_mags[indice][0], sor_means[indice][0], sor_ranges[indice][0], \
                        sor_thetas[indice][0], sor_dopplers[indice][0], sor_shapes[indice][0], sor_nums[indice][0], sor_labs[indice][0])
            dist, sor_mags1[indice][0], sor_means1[indice][0], sor_ranges1[indice][0], \
                sor_thetas1[indice][0], sor_dopplers1[indice][0], sor_shapes1[indice][0], sor_nums1[indice][0], sor_labs1[indice][0] = output
        return sor_mags1, sor_means1, sor_ranges1, sor_thetas1, sor_dopplers1, sor_shapes1, \
                sor_nums1, sor_labs1
        #pdb.set_trace()

    def denoise_current_ndoppler(self, sor_mags, sor_means, sor_ranges, sor_thetas, sor_shapes, sor_nums, sor_labs):
        sor_mags1, sor_means1, sor_ranges1, sor_thetas1, sor_shapes1, sor_nums1, sor_labs1 = \
            self.copyln(sor_mags), self.copyln(sor_means), self.copyln(sor_ranges), self.copyln(sor_thetas), \
            self.copyln(sor_shapes), self.copyln(sor_nums), self.copyln(sor_labs)
        reals_object = [i[0,0] for i in sor_means] 
        self.lof = LocalOutlierFactor(n_neighbors=self.neighbors)
        outliers = self.lof.fit_predict(reals_object)
        self.indices = np.where(outliers < 0)[0]
        whole_indices = [i for i in range(len(sor_mags))]
        for indice in self.indices:
            previous_frames = whole_indices[:indice]; next_frames = whole_indices[indice+1:]
            previous_frames = [item for item in previous_frames if item not in self.indices]
            next_frames = [item for item in next_frames if item not in self.indices]; dist = 0
            if len(previous_frames)>0:
                dist += np.linalg.norm(sor_means[indice][0] - sor_means[previous_frames[-1]][0][0][None], axis=1)
            if len(next_frames)>0:
                dist += np.linalg.norm(sor_means[indice][0] - sor_means[next_frames[0]][0][0][None], axis=1)
            output = self.reorder(dist, sor_mags[indice][0], sor_means[indice][0], sor_ranges[indice][0], \
                        sor_thetas[indice][0], sor_shapes[indice][0], sor_nums[indice][0], sor_labs[indice][0])
            dist, sor_mags1[indice][0], sor_means1[indice][0], sor_ranges1[indice][0], \
                sor_thetas1[indice][0], sor_shapes1[indice][0], sor_nums1[indice][0], sor_labs1[indice][0] = output
        return sor_mags1, sor_means1, sor_ranges1, sor_thetas1, sor_shapes1, \
                sor_nums1, sor_labs1
    
    def denoise_current_ndoppler_impr(self, sor_mags, sor_means, sor_ranges, sor_thetas, sor_shapes, sor_nums, sor_labs):
        sor_mags1, sor_means1, sor_ranges1, sor_thetas1, sor_shapes1, sor_nums1, sor_labs1 = \
            self.copyln(sor_mags), self.copyln(sor_means), self.copyln(sor_ranges), self.copyln(sor_thetas), \
            self.copyln(sor_shapes), self.copyln(sor_nums), self.copyln(sor_labs)
        reals_object = [i[0,0] for i in sor_means] 
        self.lof = LocalOutlierFactor(n_neighbors=self.neighbors)
        outliers = self.lof.fit_predict(reals_object)
        self.indices = np.where(outliers < 0)[0]
        whole_indices = [i for i in range(len(sor_mags))]
        for indice in self.indices:
            previous_frames = whole_indices[:indice]; next_frames = whole_indices[indice+1:]
            previous_frames = [item for item in previous_frames if item not in self.indices]
            next_frames = [item for item in next_frames if item not in self.indices]; dist = 0
            if len(previous_frames)>0:
                dist += np.linalg.norm(sor_means[indice][0] - sor_means[previous_frames[-1]][0][0][None], axis=1)
            if len(next_frames)>0:
                dist += np.linalg.norm(sor_means[indice][0] - sor_means[next_frames[0]][0][0][None], axis=1)
            output = self.reorder(dist, sor_mags[indice][0], sor_means[indice][0], sor_ranges[indice][0], \
                        sor_thetas[indice][0], sor_shapes[indice][0], sor_nums[indice][0], sor_labs[indice][0])
            dist, sor_mags1[indice][0], sor_means1[indice][0], sor_ranges1[indice][0], \
                sor_thetas1[indice][0], sor_shapes1[indice][0], sor_nums1[indice][0], sor_labs1[indice][0] = output
        return sor_mags1, sor_means1, sor_ranges1, sor_thetas1, sor_shapes1, \
                sor_nums1, sor_labs1
    
    def reorder(self, *lists):
        copied_lists = [np.array(lst).copy() for lst in lists]
        paired = list(zip(*copied_lists))
        sorted_pairs = sorted(paired, reverse=False)
        sorted_lists = list(zip(*sorted_pairs))
        sorted_lists = [np.array(lst) for lst in sorted_lists]
        return sorted_lists

    def remove_noisy_points(self, points, features, num_neighbors=25, density_threshold=0.1, ratio=0.1):
        nbrs = NearestNeighbors(n_neighbors=num_neighbors).fit(points)
        distances, indices = nbrs.kneighbors(points) # distance: (n, num_nei), indice: (n, num_nei)
        #densities = 1 / np.percentile(distances[:, 1:5], 20, axis=1)
        densities = 1 / np.median(distances[:, 1:6], axis=1)
        mask_pts = densities > density_threshold
        n, m = indices.shape
        indices2 = indices.reshape(-1)
        neig_feature = features[indices2]
        neig_feature = neig_feature.reshape(n, m, 2)
        ft_distance = np.linalg.norm((neig_feature[:, :1, :] - neig_feature[:, 1:, :]), axis=-1)
        _ft_distance_ = np.sort(ft_distance, axis=1)
        #ft_measure = 1/np.percentile(ft_distance, 25, axis=1)
        ft_measure = 1/np.median(_ft_distance_[:, :5], axis=1)
        mask_ft =  ft_measure > density_threshold
        mask_final = mask_pts * mask_ft            
        print(f"{len(points)} -> {len(points[mask_pts])} -> {len(points[mask_ft])} -> {len(points[mask_final])}")
        return mask_final

    def copyln(self, ln):
        return [l.copy() for l in ln]

    def denoise_across(self, ghost, reals):
        print('-------------denoise')

def geometry_noise(center, dist_thresh, end_point, start_point):
    Flag = False
    py, px = center
    ab = end_point - start_point
    ap = np.array([px, py], dtype=float) - start_point
    cross = abs(ab[0]*ap[1] - ab[1]*ap[0])
    line_len = np.linalg.norm(ab)
    if line_len < 1e-6: Flag = True
    dist = cross / line_len
    if dist < dist_thresh: Flag = True
    return Flag

import numpy as np
def slope_intercept(p, q):
    (x0, y0), (x1, y1) = p, q
    if x1 == x0:
        return np.inf, np.nan
    m = (y1 - y0) / (x1 - x0)
    b = y0 - m * x0
    if m == 0:
        return m, np.nan
    x_int = -b / m
    return m, x_int

def merge_line_segments(point_list, theta_thresh=0.3, b_thresh=35.0):
    used = [False]*len(point_list)
    merged = []
    for i, (p0, p1) in enumerate(point_list):
        if used[i]:
            continue
        m0, b0 = slope_intercept(p0, p1)
        print(m0, '==============', b0)
        group_pts = [p0, p1]
        used[i] = True
        for j in range(i+1, len(point_list)):
            if used[j]:
                continue
            q0, q1 = point_list[j]
            m1, b1 = slope_intercept(q0, q1)
            if np.isinf(m0) or np.isinf(m1):
                continue
            if abs(m0 - m1) < theta_thresh and abs(b0 - b1) < b_thresh:
                used[j] = True
                group_pts += [q0, q1]
        pts_arr = np.array(group_pts)
        dists = np.linalg.norm(pts_arr[:,None,:] - pts_arr[None,:,:], axis=2)
        idx_flat = np.argmax(dists)
        i1, i2 = divmod(idx_flat, dists.shape[1])
        new_start = tuple(pts_arr[i1].astype(int))
        new_end   = tuple(pts_arr[i2].astype(int))
        merged.append((new_start, new_end))
    return merged