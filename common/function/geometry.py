import numpy as np
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from collections import Counter

class geometry_ghost_update:
    # To dealing with numpy array.
    def __init__(self, threshold, threshold2, sample, ratio_var):
        self.threshold = threshold
        self.threshold2 = threshold2
        self.sample = sample
        self.ratio_var = ratio_var
        self.threshold_range_simi = 0.05 * 250 
        self.threshold_theta_simi = 0.05 * np.pi
        
    def frame_feature_analysis(self, center, labels, coords, slice_matrix): 
        """
        Position -> (X, Y)
        Position -> (Range, Theta)
        Doppler -> Normalized (Mean, Variance)
        Shape -> Length/Width
        Number -> Points
        Magnitude -> Normalize
        Relationship -> Range + Theta {X(1,2,3), Y(1,2,3)}
        """
        unique_label, unique_counts = np.unique(labels, return_counts=True)
        # feature -> x, y, range, angle
        clusters = []
        cluster_means = []
        cluster_vars = []
        cluster_ranges = []
        cluster_thetas = []
        # feature -> doppler, shape, num, mag
        cluster_dopplers = []
        cluster_shapes = []
        cluster_nums = []
        cluster_mags = []
        # similarity 1: =; 2: >; 3: <
        Length = 15
        for number, labels_cur in enumerate(unique_label):
            coords_cur = coords[labels==labels_cur]
            clusters.append(coords_cur)
            cluster_means.append(np.mean(coords_cur, axis=0))
            cluster_vars.append(np.var(coords_cur, axis=0))
            cluster_ranges.append(np.linalg.norm(cluster_means[-1]-center))
            cluster_thetas.append(self.compute_angle(cluster_means[-1]-center, np.zeros((2,))-center))
            # ---------- shape ----------
            width = np.max(coords_cur[:, 0]) - np.min(coords_cur[:, 0])
            length = np.max(coords_cur[:, 1]) - np.min(coords_cur[:, 1])
            if width != 0:
                cluster_shapes.append(length/width)
            else:
                cluster_shapes.append(0)
            # ---------- number ----------
            cluster_nums.append(len(coords_cur))
            # ---------- mag ----------
            mag_mat = slice_matrix[np.round(coords_cur[:, 0]), np.round(coords_cur[:, 1])] # N, n_d
            mag = np.max(mag_mat) # N, n_d
            cluster_mags.append(mag)
        sor_mags, sor_ranges, sor_means, sor_thetas, sor_shapes, sor_nums, sor_labels = \
            self.reorder(cluster_mags, cluster_ranges, cluster_means, cluster_thetas, cluster_shapes, cluster_nums, unique_label)
        return sor_mags, sor_means, sor_ranges, sor_thetas, sor_shapes, sor_nums, sor_labels

    def reorder(self, *lists):
        paired = list(zip(*lists))
        sorted_pairs = sorted(paired, reverse=True)
        sorted_lists = list(zip(*sorted_pairs))
        sorted_lists = [np.array(lst) for lst in sorted_lists]
        return sorted_lists

    def post_process(self, sor_mags, sor_means, sor_ranges, sor_thetas, sor_shapes, sor_nums, sor_labs):
        # do check before post, n must >= 2
        # (n,); (n, 2); (n,); (n,); (n, 2); (n,); (n,); 
        # All are normalized and reordered. 

        simi_range = np.abs(sor_ranges[:, None] - sor_ranges[None, :]) <= self.threshold_range_simi
        sup_range = (sor_ranges[:, None] - sor_ranges[None, :]) > self.threshold_range_simi
        inf_range = (simi_range == 0) & (sup_range == 0)

        simi_theta = np.abs(sor_thetas[:, None] - sor_thetas[None, :]) <= self.threshold_theta_simi
        sup_theta = sor_thetas[:, None] - sor_thetas[None, :] > self.threshold_theta_simi
        inf_theta = (simi_theta == 0) & (sup_theta == 0)
        # real, pghost1, pghost1pr, range, mag, num; need to add frame.
        # ---------- Preprocess for deleting all same range object. ----------
        preprocess = True
        if preprocess == True:
            sor_mag_t = sor_mags[:] / sor_mags[0]
            """# Delete all point at the same range."""
            #select = (simi_range[1:,0]) & (simi_theta[1:,0]==0) & (time_mag[1:] <= threshold_mag) 
            select = (simi_range[1:,0]) & (simi_theta[1:,0]==0)
            # True for deleting one 
            # False for keeping one
            inv_s = [not value for value in select]
            inv_s = [True] + inv_s
            sor_mags, sor_mag_t, sor_means, sor_ranges, sor_thetas, sor_shapes, sor_nums, sor_labs = \
                sor_mags[inv_s], sor_mag_t[inv_s], sor_means[inv_s], sor_ranges[inv_s], sor_thetas[inv_s], \
                sor_shapes[inv_s], sor_nums[inv_s], sor_labs[inv_s]
            inv_sm = np.array(inv_s)
            inv_sm = inv_sm[:, None] * inv_sm[None, :]
            num_s = np.sum(inv_s)
            simi_range, sup_range, inf_range = \
                simi_range[inv_sm].reshape(num_s,num_s), sup_range[inv_sm].reshape(num_s,num_s), inf_range[inv_sm].reshape(num_s,num_s)
            simi_theta, sup_theta, inf_theta = \
                simi_theta[inv_sm].reshape(num_s,num_s), simi_theta[inv_sm].reshape(num_s,num_s), simi_theta[inv_sm].reshape(num_s,num_s)
        ''' ---------- pghost1 
        large than real range; same angle with real range; no larger than anything else '''
        pghost1_index = (sup_range[1:,0]) & (simi_theta[1:,0]) #& (np.sum(sup_range[1:,1:]*simi_theta[1:,1:],axis=1)==0)
        ''' ---------- pghost1pr
        large than real range; different angle with real range; no larger than anything else '''
        pghost1pr_index = (sup_range[1:,0]) & (simi_theta[1:,0]==0) & (np.sum(sup_range[1:,1:]*simi_theta[1:,1:],axis=1)==0)
        ''' ---------- pghost2pr
        large than real range; different angle with real range; no larger than anything else '''
        if np.sum(pghost1pr_index > 0):
            pghost2pr_index = (sup_range[1:,0]) & (simi_theta[1:,0]==0) & \
                (np.sum(sup_range[1:,1:][:,pghost1pr_index]*simi_theta[1:,1:][:,pghost1pr_index],axis=1)>0)
        else:
            pghost2pr_index = pghost1pr_index
        pghost1_mean = sor_means[1:][pghost1_index]
        pghost1_range = sor_ranges[1:][pghost1_index]
        pghost1_mag = sor_mags[1:][pghost1_index]
        pghost1_mag_t = sor_mag_t[1:][pghost1_index]
        pghost1_num = sor_nums[1:][pghost1_index]
        num = len(pghost1_mean)
        pghost1_real = np.repeat(sor_means[0][None, :], num, axis=0)
        
        pghost1pr_mean = sor_means[1:][pghost1pr_index]
        pghost1pr_range = sor_ranges[1:][pghost1pr_index]
        pghost1pr_mag = sor_mags[1:][pghost1pr_index]
        pghost1pr_mag_t = sor_mag_t[1:][pghost1pr_index]
        pghost1pr_num = sor_nums[1:][pghost1pr_index]
        num = len(pghost1pr_mean)
        pghost1pr_real = np.repeat(sor_means[0][None, :], num, axis=0)

        pghost2pr_mean = sor_means[1:][pghost2pr_index]
        pghost2pr_range = sor_ranges[1:][pghost2pr_index]
        pghost2pr_mag = sor_mags[1:][pghost2pr_index]
        pghost2pr_mag_t = sor_mag_t[1:][pghost2pr_index]
        pghost2pr_num = sor_nums[1:][pghost2pr_index]
        num = len(pghost2pr_mean)
        pghost2pr_real = np.repeat(sor_means[0][None, :], num, axis=0)
        return pghost1_mean, pghost1_range, pghost1_mag, pghost1_mag_t, pghost1_num, pghost1_real, \
             pghost1pr_mean, pghost1pr_range, pghost1pr_mag, pghost1pr_mag_t, pghost1pr_num, pghost1pr_real, \
             pghost2pr_mean, pghost2pr_range, pghost2pr_mag, pghost2pr_mag_t, pghost2pr_num, pghost2pr_real
        
    def post_process_strict_level1(self, sor_mags, sor_means, sor_ranges, sor_thetas, sor_shapes, sor_nums, sor_labs):
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
            select = (simi_range[1:,0]) & (simi_theta[1:,0]==0)
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
        pghost1_index_pre = (sup_range[1:,0]) & (simi_theta[1:,0]) #& (np.sum(sup_range[1:,1:]*simi_theta[1:,1:],axis=1)==0) # (N-1,) #& (np.sum(sup_range[1:,1:]*simi_theta[1:,1:],axis=1)==0)
        pghost1_index_pre = np.where(pghost1_index_pre==1)[0]
        ''' ---------- pghost1pr (M4) 
        large than real range; different angle with real range; no larger than anything else '''
        pghost1pr_index = (simi_range[1:,1:][:,pghost1_index_pre]) & (simi_theta[1:,1:][:,pghost1_index_pre]==0) # (N-1, N-1)
        '''------------------------'''
        indices = np.where(pghost1pr_index==1) 
        pghost1pr_index = indices[0]; pghost1_index = pghost1_index_pre[indices[1]]
        check = [item not in pghost1_index for item in pghost1pr_index]
        pghost1pr_index = pghost1pr_index[check]; pghost1_index = pghost1_index[check]
        if len(pghost1pr_index) > 1:
            item_counts = Counter(pghost1_index); index_list = []
            for item, count in item_counts.items(): 
                if count>1: index_list.extend(np.where(pghost1_index == item)[0].tolist())
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
        pghost1_mean = sor_means[1:][pghost1_index]; pghost1_range = sor_ranges[1:][pghost1_index]
        pghost1_mag = sor_mags[1:][pghost1_index]; pghost1_mag_t = sor_mag_t[1:][pghost1_index]
        pghost1_num = sor_nums[1:][pghost1_index]
        num = len(pghost1_mean); pghost1_real = np.repeat(sor_means[0][None, :], num, axis=0)
        pghost1pr_mean = sor_means[1:][pghost1pr_index]; pghost1pr_range = sor_ranges[1:][pghost1pr_index]
        pghost1pr_mag = sor_mags[1:][pghost1pr_index]; pghost1pr_mag_t = sor_mag_t[1:][pghost1pr_index]
        pghost1pr_num = sor_nums[1:][pghost1pr_index]
        num = len(pghost1pr_mean); pghost1pr_real = np.repeat(sor_means[0][None, :], num, axis=0)
        if len(pghost1pr_index) > 0:
            pghost2pr_mean = sor_means[1:][pghost2pr_index]; pghost2pr_range = sor_ranges[1:][pghost2pr_index]
            pghost2pr_mag = sor_mags[1:][pghost2pr_index]; pghost2pr_mag_t = sor_mag_t[1:][pghost2pr_index]
            pghost2pr_num = sor_nums[1:][pghost2pr_index]
            num = len(pghost2pr_mean); pghost2pr_real = np.repeat(sor_means[0][None, :], num, axis=0)
        else:
            pghost2pr_mean, pghost2pr_range, pghost2pr_mag = None, None, None
            pghost2pr_mag_t, pghost2pr_num, pghost2pr_real = None, None, None
        return pghost1_mean, pghost1_range, pghost1_mag, pghost1_mag_t, pghost1_num, pghost1_real, \
             pghost1pr_mean, pghost1pr_range, pghost1pr_mag, pghost1pr_mag_t, pghost1pr_num, pghost1pr_real, \
             pghost2pr_mean, pghost2pr_range, pghost2pr_mag, pghost2pr_mag_t, pghost2pr_num, pghost2pr_real
    
    def normalize(self, vector):
        vector_sum = np.sum(vector, axis=1, keepdims=True)
        vector_nor = vector/(vector_sum+1e-8) # N, 16
        bin_value = np.array([i for i in range(-7, 9)])[None, :]
        vector_nor = vector_nor*bin_value
        dist_mean = np.mean(vector_nor, axis=1)
        dist_var = np.var(vector_nor, axis=1)
        dist_mean = np.mean(dist_mean)
        dist_var = np.mean(dist_var)
        return vector_nor, dist_mean, dist_var

    def compute_angle(self, u, v):
        dot_product = np.dot(u, v)
        magnitude_u = np.linalg.norm(u)
        magnitude_v = np.linalg.norm(v)
        cos_theta = dot_product / (magnitude_u * magnitude_v)
        angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Use clip to avoid potential numerical errors
        angle_degrees = np.degrees(angle_radians)
        return angle_radians