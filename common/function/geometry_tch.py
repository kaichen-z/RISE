import pdb
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter

class geometry_ghost_update:
    # To dealing with torch tensor.
    def __init__(self, threshold, threshold2, sample, ratio_var):
        self.threshold = threshold
        self.threshold2 = threshold2
        self.sample = sample
        self.ratio_var = ratio_var
        self.threshold_select = 0.3
        self.threshold_range_simi = 0.05 * 250 
        self.threshold_theta_simi = 0.05 * np.pi
        
    def combine(self, labels, coords):
        cluster_means = []
        for label in torch.unique(labels):
            coords_cur = coords[labels==label]
            cluster_means.append(torch.mean(coords_cur.float(), dim=0))
        if len(cluster_means) > 0: return torch.stack(cluster_means)  
        else: return torch.tensor([])  

    def frame_feature_analysis(self, center, labels, coords, slice_matrix=None, magnitudes=None):
        unique_labels, unique_counts = torch.unique(labels, return_counts=True)
        clusters = []
        cluster_means = []
        cluster_ranges = []
        cluster_thetas = []
        cluster_dopplers = []
        cluster_shapes = []
        cluster_nums = []
        cluster_mags = []
        center, labels, coords = center.float(), labels.float(), coords.float()
        for label in unique_labels:
            coords_cur = coords[labels == label]
            clusters.append(coords_cur)
            cluster_means.append(torch.mean(coords_cur, dim=0))
            cluster_ranges.append(torch.norm(cluster_means[-1] - center))
            cluster_thetas.append(self.compute_angle(cluster_means[-1] - center, torch.zeros(2) - center))
            width = torch.max(coords_cur[:, 0]) - torch.min(coords_cur[:, 0])
            length = torch.max(coords_cur[:, 1]) - torch.min(coords_cur[:, 1])
            cluster_shapes.append(length / width if width != 0 else torch.tensor(0))
            cluster_nums.append(len(coords_cur))
            if slice_matrix is not None:
                rounded_coords = coords_cur.round().long()
                mag_mat = slice_matrix[rounded_coords[:, 0], rounded_coords[:, 1]]
            if magnitudes is not None:
                mag_mat = magnitudes[labels == label]
            mag = torch.max(mag_mat) # using the max magnitude of the cluster.
            cluster_mags.append(mag)
        sor_mags, sor_ranges, sor_means, sor_thetas, sor_shapes, sor_nums, sor_labels, clusters = \
            self.reorder_threshold(cluster_mags, cluster_ranges, cluster_means, cluster_thetas, cluster_shapes, cluster_nums, unique_labels, clusters)
        return torch.tensor(sor_mags), torch.stack(sor_means), torch.tensor(sor_ranges), \
                torch.tensor(sor_thetas), torch.tensor(sor_shapes), torch.tensor(sor_nums), torch.tensor(sor_labels), clusters # change order, dont change

    def reorder_threshold(self, *lists):
        magnitudes = lists[0]
        ranges = lists[1]
        threshold = self.threshold_select * max(magnitudes)
        filtered_indices = [i for i, mag in enumerate(magnitudes) if mag > threshold]
        min_range_index = min(filtered_indices, key=lambda i: ranges[i])
        reordered_indices = [min_range_index] + [i for i in range(len(magnitudes)) if i != min_range_index]
        reordered_lists = [[lst[i] for i in reordered_indices] for lst in lists]
        return tuple(reordered_lists)

    def reorder(self, *lists):
        sorted_lists = sorted(zip(*lists), reverse=True)
        #sorted_lists = [torch.stack(lst) if torch.is_tensor(lst[0]) else torch.tensor(lst) for lst in zip(*sorted_lists)]
        sorted_lists = [lst for lst in zip(*sorted_lists)]
        return sorted_lists

    def post_process(self, sor_mags, sor_means, sor_ranges, sor_thetas, sor_shapes, sor_nums, sor_labels):
        self.threshold_range_simi = 0.075 * 250
        self.threshold_theta_simi = 0.075 * torch.pi
        sor_ranges = sor_ranges
        sor_thetas = sor_thetas
        simi_range = (torch.abs(sor_ranges[:, None] - sor_ranges[None, :]) <= self.threshold_range_simi)
        sup_range = (sor_ranges[:, None] - sor_ranges[None, :]) > self.threshold_range_simi
        inf_range = (~simi_range) & (~sup_range)
        simi_theta = (torch.abs(sor_thetas[:, None] - sor_thetas[None, :]) <= self.threshold_theta_simi)
        sup_theta = (sor_thetas[:, None] - sor_thetas[None, :]) > self.threshold_theta_simi
        inf_theta = (~simi_theta) & (~sup_theta)
        preprocess = True
        if preprocess:
            time_mag = sor_mags / sor_mags[0]
            threshold_mag = 0.5
            select = (simi_range[1:, 0]) & (~simi_theta[1:, 0]) & (time_mag[1:] <= threshold_mag)
            inv_s = torch.cat((torch.tensor([True]), ~select))
            sor_mags, sor_means, sor_ranges, sor_thetas, sor_shapes, sor_nums, sor_labels = [
                tensor[inv_s] for tensor in [sor_mags, sor_means, sor_ranges, sor_thetas, sor_shapes, sor_nums, sor_labels]]
            inv_sm = inv_s[:, None] & inv_s[None, :]
            num_s = inv_s.sum()
            simi_range, sup_range, inf_range = [
                tensor[inv_sm].reshape(num_s, num_s) for tensor in [simi_range, sup_range, inf_range]]
            simi_theta, sup_theta, inf_theta = [
                tensor[inv_sm].reshape(num_s, num_s) for tensor in [simi_theta, sup_theta, inf_theta]]
        pghost1_index = (sup_range[1:, 0]) & (simi_theta[1:, 0])
        pghost1pr_index = (sup_range[1:, 0]) & (~simi_theta[1:, 0]) & (torch.sum(sup_range[1:, 1:] * simi_theta[1:, 1:], dim=1) == 0)
        if pghost1pr_index.sum() > 0:
            pghost2pr_index = (sup_range[1:, 0]) & (~simi_theta[1:, 0]) & (
                torch.sum(sup_range[1:, 1:][:, pghost1pr_index] * simi_theta[1:, 1:][:, pghost1pr_index], dim=1) > 0)
        else:
            pghost2pr_index = pghost1pr_index
        results = []
        for idx, indices in enumerate([pghost1_index, pghost1pr_index, pghost2pr_index]):
            means, ranges, mags, nums = [sor_means[1:][indices], sor_ranges[1:][indices], sor_mags[1:][indices], sor_nums[1:][indices]]
            real_means = sor_means[0].unsqueeze(0).expand(len(means), -1)
            results.extend([means, ranges, mags, nums, real_means])
        return results

    def compute_angle(self, u, v):
        dot_product = torch.dot(u, v)
        magnitude_u = torch.norm(u)
        magnitude_v = torch.norm(v)
        cos_theta = dot_product / (magnitude_u * magnitude_v)
        angle_radians = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
        return angle_radians

    def post_process_strict_level1(self, sor_mags, sor_means, sor_ranges, sor_thetas, sor_shapes, sor_nums, sor_labs):
        """
        We've a list of points and a center. We name the first point of this list as the reference point. 
        We want to find all pairs. 
        The first point in this pair have large distance than the reference point to the center, but it lays on the extension of the line from the center to the reference point. 
        The second point in this pair have similar distance to center as the first point in the pair, but it doesn't lays on te extension of the line from  the center to the reference point. 
        """
        # do check before post, n must >= 2
        # (n,); (n, 2); (n,); (n,); (n, 2); (n,); (n,); 
        simi_range = np.abs(sor_ranges[:, None] - sor_ranges[None, :]) <= self.threshold_range_simi
        sup_range = (sor_ranges[:, None] - sor_ranges[None, :]) > self.threshold_range_simi
        simi_theta = np.abs(sor_thetas[:, None] - sor_thetas[None, :]) <= self.threshold_theta_simi
        # real, pghost1, pghost1pr, range, mag, num; need to add frame.
        # ---------- Preprocess for deleting all same range object. ----------
        preprocess = True
        if preprocess == True:
            sor_mag_t = sor_mags[:] / sor_mags[0]
            """# Delete all point at the same range with the real object."""
            select = (simi_range[1:,0]) #& (simi_theta[1:,0]==0) 
            # True for deleting one; False for keeping one
            inv_s = [not value for value in select]; inv_s = [True] + inv_s
            # True for saving one; False for deleting one
            sor_mags, sor_mag_t, sor_means, sor_ranges, sor_thetas, sor_shapes, sor_nums, sor_labs = \
                sor_mags[inv_s], sor_mag_t[inv_s], sor_means[inv_s], sor_ranges[inv_s], sor_thetas[inv_s], \
                sor_shapes[inv_s], sor_nums[inv_s], sor_labs[inv_s]
            inv_sm = np.array(inv_s); inv_sm = inv_sm[:, None] * inv_sm[None, :]; num_s = np.sum(inv_s)
            simi_range, sup_range = simi_range[inv_sm].reshape(num_s,num_s), sup_range[inv_sm].reshape(num_s,num_s)
            simi_theta = simi_theta[inv_sm].reshape(num_s,num_s)
        ''' ---------- pghost1 (M4)
        large than real range; same angle with real range '''
        pghost1_index_pre = (sup_range[1:,0]) & (simi_theta[1:,0])
        pghost1_index_pre = np.where(pghost1_index_pre==1)[0]
        pghost1_index_pre_msk = np.sum(sup_range[1:,1:][pghost1_index_pre], axis=1) == 0
        pghost1_index_pre = pghost1_index_pre[pghost1_index_pre_msk]
        ''' ---------- pghost1pr (M4) Check acgain
        large than real range; different angle with real range; no larger than anything else '''
        a = [i for i in range(simi_range[1:,1:].shape[0])]
        pghost1pr_index_pre = [item for item in a if item not in pghost1_index_pre]
        pghost1pr_index = (simi_range[1:,1:][pghost1pr_index_pre,:][:,pghost1_index_pre]) & (simi_theta[1:,1:][pghost1pr_index_pre,:][:,pghost1_index_pre]==0) # (N-1, M-1)
        '''----------- deleting wrong matching'''
        indices = np.argwhere(pghost1pr_index) 
        pghost1pr_index = np.array([pghost1pr_index_pre[ind[0]] for ind in indices])
        pghost1_index = np.array([pghost1_index_pre[ind[1]] for ind in indices]) # recording all matchings
        check1 = [item in pghost1_index for item in pghost1pr_index]; 
        if np.sum(check1) > 0: print("--------- We found an error here ---------")
        if len(pghost1pr_index) > 1:
            item_counts = Counter(pghost1_index); index_list = []
            for item, count in item_counts.items(): 
                if count>1: 
                    cur_index_list = np.array(np.where(pghost1_index == item)[0].tolist())
                    cur_values = sor_mags[1:][pghost1pr_index[cur_index_list]]
                    max_value = np.max(cur_values)
                    index_list.extend(cur_index_list[cur_values < max_value])
            indices_to_delete = set(index_list)
            pghost1pr_index = [item for idx, item in enumerate(pghost1pr_index) if idx not in indices_to_delete]
            pghost1_index = [item for idx, item in enumerate(pghost1_index) if idx not in indices_to_delete]
        ''' ---------- pghost2pr (M5)
        large than real range; different angle with real range; no larger than anything else '''
        if len(pghost1pr_index) > 0:
            pghost2pr_index = (sup_range[1:,1:][:,pghost1pr_index]) & (simi_theta[1:,1:][:,pghost1pr_index])
            pghost2pr_msk = np.sum(simi_range[1:,1:], axis=1) <2
            pghost2pr_index = np.sum(pghost2pr_index, axis=1) * pghost2pr_msk
            pghost2pr_index = np.where(pghost2pr_index==1)[0]
        ''' ---------- Assign value ----------'''
        if len(pghost1_index) > 0:
            pghost1_mean = sor_means[1:][pghost1_index]; pghost1_range = sor_ranges[1:][pghost1_index]
            pghost1_mag = sor_mags[1:][pghost1_index]; pghost1_mag_t = sor_mag_t[1:][pghost1_index]
            pghost1_num = sor_nums[1:][pghost1_index]
            num = len(pghost1_mean); pghost1_real = np.repeat(sor_means[0][None, :], num, axis=0)
            pghost1pr_mean = sor_means[1:][pghost1pr_index]; pghost1pr_range = sor_ranges[1:][pghost1pr_index]
            pghost1pr_mag = sor_mags[1:][pghost1pr_index]; pghost1pr_mag_t = sor_mag_t[1:][pghost1pr_index]
            pghost1pr_num = sor_nums[1:][pghost1pr_index]
            num = len(pghost1pr_mean); pghost1pr_real = np.repeat(sor_means[0][None, :], num, axis=0)
            if len(pghost2pr_index) > 0:
                pghost2pr_mean = sor_means[1:][pghost2pr_index]; pghost2pr_range = sor_ranges[1:][pghost2pr_index]
                pghost2pr_mag = sor_mags[1:][pghost2pr_index]; pghost2pr_mag_t = sor_mag_t[1:][pghost2pr_index]
                pghost2pr_num = sor_nums[1:][pghost2pr_index]
                num = len(pghost2pr_mean); pghost2pr_real = np.repeat(sor_means[0][None, :], num, axis=0)
            else:
                pghost2pr_mean, pghost2pr_range, pghost2pr_mag = [], [], []
                pghost2pr_mag_t, pghost2pr_num, pghost2pr_real = [], [], []
        else:
            pghost1pr_mean, pghost1pr_range, pghost1pr_mag = [], [], []
            pghost1pr_mag_t, pghost1pr_num, pghost1pr_real = [], [], []
            pghost1_mean, pghost1_range, pghost1_mag = [], [], []
            pghost1_mag_t, pghost1_num, pghost1_real = [], [], []
            pghost2pr_mean, pghost2pr_range, pghost2pr_mag = [], [], []
            pghost2pr_mag_t, pghost2pr_num, pghost2pr_real = [], [], []
        return pghost1_mean, pghost1_range, pghost1_mag, pghost1_mag_t, pghost1_num, pghost1_real, \
             pghost1pr_mean, pghost1pr_range, pghost1pr_mag, pghost1pr_mag_t, pghost1pr_num, pghost1pr_real, \
             pghost2pr_mean, pghost2pr_range, pghost2pr_mag, pghost2pr_mag_t, pghost2pr_num, pghost2pr_real