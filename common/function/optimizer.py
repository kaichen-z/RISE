import os
import pdb
import cv2
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture as _SkGaussianMixture

# sklearn's default reg_covar (1e-6) raises "not positive definite" on near-collinear clusters
# (scene11_set1, final_version_b9 bundle); 1e-4 is invisible on well-behaved scenes.
GMM_REG_COVAR = 1e-4


def GaussianMixture(*args, reg_covar=GMM_REG_COVAR, **kwargs):
    """sklearn GaussianMixture with a numerically safe default covariance regulariser."""
    return _SkGaussianMixture(*args, reg_covar=reg_covar, **kwargs)
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import RANSACRegressor

from scipy.spatial.distance import euclidean
from scipy.stats import gaussian_kde


class Optimizer_one_rel:
    def __init__(self, center, pghost1, pghost1pr, pghost2pr, \
            pghost1_real, pghost1pr_real, pghost2pr_real, \
            min_gmm, max_gmm, H, W):
        self.c = center[None, :]
        self.ref = np.array([1,0])[None,:]
        self.pg1 = pghost1
        self.pg1pr = pghost1pr
        self.pg2pr = pghost2pr
        self.pg1_r = pghost1_real
        self.pg1pr_r = pghost1pr_real
        self.pg2pr_r = pghost2pr_real
        self.compute_mirror()
        # ---------- Threshold for points
        self.threshold = 0.8
        self.reflector_point = np.zeros((5, 2))
        self.mirror_point = np.zeros((5, 2))
        self.RESOLUTION = 50
        # ---------- GMM Range
        self.min_gmm = min_gmm
        self.max_gmm = max_gmm
        # ---------- Drawing 
        self.H = H 
        self.W = W
        # ---------- TIME of GHOST_2pr
        self.threshold_density_neg = 5
        self.threshold_number_neg = 10
        self.threshold_density_pos = 0.5
        self.threshold_number_pos = 5
        # ---------- parameter for baseline
        self.dbscan_eps = 25
        self.dbscan_min_samples = 8
        # ---------- parameter for our
        self.dbscan_eps_ours = 15
        self.dbscan_min_samples_ours = 12
        # ---------- parameter for background
        self.dbscan_eps_bkg = 10
        self.dbscan_min_samples_bkg = 16
        # ---------- parameter for current method
        self.dbscan_eps_cl = 25
        self.dbscan_min_samples_cl = 8
        self.min_mae_improvement = 10
        # ---------- 如果通过则删除
        self.threshold_num = 10
        self.threshold_line = 25
        self.threshold_dis=6; self.threshold_ang=0.05
        # GMM splitting 最大组件数
        self.gmm_max_components = 2
        # 最小 MAE 改进幅度 (绝对值)
        self.mae_tol = 10
        # RANSAC 内点比例阈值
        self.prune_inliers_ratio = 0.5
        # BIC 罚项，用于判断分裂是否合理
        self.bic_penalty = 10.0

    def compute_mirror(self,):
        self.bg_start = np.array([210.5, 137.8125])
        self.bg_end = np.array([166, 83])
        x1, y1 = self.bg_start
        x2, y2 = self.bg_end
        xa, ya = self.c[0]
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        distance = (A * xa + B * ya + C) / np.sqrt(A**2 + B**2)
        Px = xa - (A / np.sqrt(A**2 + B**2)) * distance
        Py = ya - (B / np.sqrt(A**2 + B**2)) * distance
        P_i = (Px, Py)
        P_prime = (2 * P_i[0] - xa, 2 * P_i[1] - ya)
        self.real_mirror = P_prime

    def compute_reflector_1pr_(self,):
        # method 1
        center, real, ghost1, ghost2 = self.c, self.pg1pr_r, self.pg1, self.pg1pr
        # Generate samples
        R1 = np.linalg.norm((real-center), axis=1) # (n)
        R2 = np.linalg.norm((ghost1-center), axis=1) # (n)
        R3 = np.linalg.norm((ghost2-center), axis=1) # (n)
        R4 = R2 + R3 - R1   # Equation (6)
        original = np.array([0,0])[None,:]
        A1 = np.arccos(np.einsum('ij,ij->i', (real-center), (original-center)) \
                       / (np.linalg.norm(real-center, axis=1) * np.linalg.norm(original-center, axis=1)))
        A2 = np.arccos(np.einsum('ij,ij->i', (ghost1-center), (original-center)) \
                       / (np.linalg.norm(ghost1-center, axis=1) * np.linalg.norm(original-center, axis=1)))
        A3 = np.arccos(np.einsum('ij,ij->i', (ghost2-center), (original-center)) \
                       / (np.linalg.norm(ghost2-center, axis=1) * np.linalg.norm(original-center, axis=1)))
        A4 = A3 + np.arcsin(np.sin(A2-A1)*R1/R4) # Equation (11)
        R = (R4**2 - R1**2)/(2*R4 - 2*R1*np.cos(A4-A1)) # Equation (13)
        reflector_point = center + (ghost2 - center) * R[:, None] / R3[:, None]
        return reflector_point
    
    def compute_reflector_1pr(self,):
        # method 1
        center, real, ghost1pr = self.c, self.pg1pr_r, self.pg1pr
        # Generate samples
        l1 = np.linalg.norm((real-center), axis=1) # (n)
        g = np.linalg.norm((ghost1pr-center), axis=1) # (n)
        cos_theta = np.einsum('ij,ij->i', (real-center), (ghost1pr-center)) / (l1 * g)  # (n)
        l2 = (2*g*g-2*g*l1)/(2*g-l1*cos_theta-l1)
        reflector_point = center + (ghost1pr-center) * l2[:, None]/g[:, None] # n,2
        mirror_point = (reflector_point - real)/np.linalg.norm((reflector_point - real), axis=1, keepdims=True)*l2[:, None] + reflector_point  # n,2
        self.reflector_point_1pr = reflector_point
        self.mirror_point_1pr = mirror_point
        vector = self.mirror_point_1pr - self.c
        vector_norm = self.compute_norm(vector)
        distance_ratio = 2*g/l1
        return reflector_point, mirror_point, vector_norm, distance_ratio

    def compute_reflector_2pr(self,):
        # method 1
        center, real, ghost2pr = self.c, self.pg2pr_r, self.pg2pr
        # Generate samples
        l1 = np.linalg.norm((real-center), axis=1) # (n)
        h = np.linalg.norm((ghost2pr-center), axis=1) # (n)
        cos_theta = np.einsum('ij,ij->i', (real-center), (ghost2pr-center)) / (l1 * h)  # (n)
        l2 = (h*h-l1*l1)/(2*h-2*l1*cos_theta)
        reflector_point = center + (ghost2pr-center)*l2[:, None]/h[:, None] # n,2
        mirror_point = (reflector_point - real)/np.linalg.norm((reflector_point - real), axis=1, keepdims=True)*l2[:, None] + reflector_point  # n,2
        self.reflector_point_2pr = reflector_point
        self.mirror_point_pr = mirror_point
        vector = self.mirror_point_pr - self.c
        vector_norm = self.compute_norm(vector)
        distance_ratio = 2*h/l1
        return reflector_point, mirror_point, vector_norm, distance_ratio
    
    def compute_norm(self, vector):
        vector_norm = np.empty_like(vector)
        mask = vector[:, 0] > 0
        vector_norm[mask, 0] = vector[mask, 1]
        vector_norm[mask, 1] = vector[mask, 0]
        vector_norm[~mask, 0] = -vector[~mask, 1]
        vector_norm[~mask, 1] = vector[~mask, 0]
        magnitudes = np.linalg.norm(vector_norm, axis=1, keepdims=True)
        magnitudes[magnitudes == 0] = 1
        vector_norm = vector_norm / magnitudes
        return vector_norm

    def select_dbscan(self, data):
        # Step 1: Use DBSCAN to cluster the data and remove noise
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(data)
        labels = db.labels_
        mask = labels != -1
        return mask

    def gmm(self, data):
        n_components_range = range(self.min_gmm, self.max_gmm)
        bic_values = []
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(data)
            bic_values.append(gmm.bic(data))
        index = np.argmin(bic_values)
        n_components = n_components_range[index]
        print('---------Best Choice---------', n_components)
        gmm = GaussianMixture(reg_covar=GMM_REG_COVAR, n_components=n_components, random_state=42)
        gmm.fit(data)
        self.gaus_means = gmm.means_
        self.gaus_vars = gmm.covariances_
        self.gaus_weights = gmm.weights_
        self.prob = gmm.predict_proba(data) # Num_g1pr, Num_dist
    
    def fit_line(self, points):
        """Fit a 2D line to a cluster of points using linear regression."""
        x, y = points[:, 1].reshape(-1, 1), points[:, 0]
        model = LinearRegression().fit(x, y)
        y_pred = model.predict(x)
        mae = mean_absolute_error(y, y_pred)
        return model.coef_[0], model.intercept_, mae  

    def fit_line_list(self, points):
        line_point_lists = []
        for point in points:
            slope, intercept, _ = self.fit_line(point)
            lines_x = np.linspace(point[:,1].min(), point[:,1].max(), num=500)
            lines_y = slope * lines_x + intercept; lines_points = np.column_stack((lines_y, lines_x))
            line_point_lists.append(lines_points)
        return line_point_lists
    
    def gmm_baseline(self, points, MinMAE=10):
        """
        DBSCAN: distance: 20cm min-neighbors: 16.
        GMM: MaxK to 4, MinPts to 30, MinMAE-range 10 ~ 20cm.
        """
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(points)
        labels = db.labels_
        unique_labels = set(labels)
        line_segments = []  
        for cluster_id in unique_labels:
            if cluster_id == -1: continue
            cluster_points = points[labels == cluster_id]
            if len(cluster_points) < self.dbscan_min_samples: continue
            slope, intercept, mae = self.fit_line(cluster_points)
            improved = True
            while improved and len(cluster_points) > self.dbscan_min_samples:
                gmm = GaussianMixture(n_components=2, random_state=0).fit(cluster_points)
                split_labels = gmm.predict(cluster_points)
                points_cluster1 = cluster_points[split_labels == 0]
                points_cluster2 = cluster_points[split_labels == 1]
                if len(points_cluster1) < self.dbscan_min_samples or len(points_cluster2) < self.dbscan_min_samples: break
                _, _, mae1 = self.fit_line(points_cluster1)
                _, _, mae2 = self.fit_line(points_cluster2)
                combined_mae = mae1 + mae2
                if combined_mae >= mae or mae - combined_mae < MinMAE: improved = False
                else: 
                    cluster_points = points_cluster1 if mae1 < mae2 else points_cluster2
                    slope, intercept, mae = self.fit_line(cluster_points)
            line_segments.append((slope, intercept, cluster_points))
        return line_segments
    
    def gmm_ours(self, points, MinMAE=10):
        """
        DBSCAN: distance: 20cm min-neighbors: 16.
        GMM: MaxK to 4, MinPts to 30, MinMAE-range 10 ~ 20cm.
        """
        db = DBSCAN(eps=self.dbscan_eps_ours, min_samples=self.dbscan_min_samples_ours).fit(points)
        labels = db.labels_
        unique_labels = set(labels)
        line_segments = []  
        for cluster_id in unique_labels:
            if cluster_id == -1: continue
            cluster_points = points[labels == cluster_id]
            if len(cluster_points) < self.dbscan_min_samples_ours: continue
            slope, intercept, mae = self.fit_line(cluster_points)
            improved = True
            while improved and len(cluster_points) > self.dbscan_min_samples_ours:
                gmm = GaussianMixture(n_components=2, random_state=0).fit(cluster_points)
                split_labels = gmm.predict(cluster_points)
                points_cluster1 = cluster_points[split_labels == 0]
                points_cluster2 = cluster_points[split_labels == 1]
                if len(points_cluster1) < self.dbscan_min_samples_ours or len(points_cluster2) < self.dbscan_min_samples_ours: break
                _, _, mae1 = self.fit_line(points_cluster1)
                _, _, mae2 = self.fit_line(points_cluster2)
                combined_mae = mae1 + mae2
                if combined_mae >= mae or mae - combined_mae < MinMAE: improved = False
                else: 
                    cluster_points = points_cluster1 if mae1 < mae2 else points_cluster2
                    slope, intercept, mae = self.fit_line(cluster_points)
            line_segments.append((slope, intercept, cluster_points))
        return line_segments

    def gmm_clustering(self, points):
        db = DBSCAN(eps=self.dbscan_eps_cl, min_samples=self.dbscan_min_samples_cl).fit(points)
        labels = db.labels_; unique_labels = set(labels)
        # Prepare to store refined clusters and their labels
        final_labels = -1 * np.ones(points.shape[0], dtype=int)  # Initialize all labels as noise (-1)
        next_cluster_id = 0; clusters = []
        for cluster_id in unique_labels:
            if cluster_id == -1: continue
            cluster_points = points[labels == cluster_id]
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_points) < self.dbscan_min_samples_cl: continue
            improved = True
            while improved and len(cluster_points) > self.dbscan_min_samples_cl:
                gmm = GaussianMixture(reg_covar=GMM_REG_COVAR, n_components=2, random_state=0).fit(cluster_points)
                split_labels = gmm.predict(cluster_points)
                points_cluster1 = cluster_points[split_labels == 0]
                points_cluster2 = cluster_points[split_labels == 1]
                indices_cluster1 = cluster_indices[split_labels == 0]
                indices_cluster2 = cluster_indices[split_labels == 1]
                if len(points_cluster1) < self.dbscan_min_samples_cl or len(points_cluster2) < self.dbscan_min_samples_cl: break
                center1 = np.mean(points_cluster1, axis=0)
                center2 = np.mean(points_cluster2, axis=0)
                mae1 = np.mean(np.linalg.norm(points_cluster1 - center1, axis=1))
                mae2 = np.mean(np.linalg.norm(points_cluster2 - center2, axis=1))
                combined_mae = mae1 + mae2
                current_center = np.mean(cluster_points, axis=0)
                current_mae = np.mean(np.linalg.norm(cluster_points - current_center, axis=1))
                if combined_mae >= current_mae or (current_mae - combined_mae) < self.min_mae_improvement: improved = False
                else:                 
                    if mae1 < mae2: cluster_points, cluster_indices = points_cluster1, indices_cluster1
                    else: cluster_points, cluster_indices = points_cluster2, indices_cluster2
            clusters.append(cluster_points)
            final_labels[cluster_indices] = next_cluster_id
            next_cluster_id += 1
        return clusters, final_labels

    def p1_dist(self,):
        center = self.c
        pghost1 = self.pg1
        real = self.pg1_r
        radius = np.linalg.norm((pghost1-real), axis=1) # (n)
        distance = np.linalg.norm((real[:, None, :] - self.gaus_means[None, :, :]), axis=2) # (n, m)
        ratio = distance/radius[:, None] # (n, m)
        _ratio_ = np.mean(ratio, axis=0) # (m)
        return _ratio_

    def weight_norm(self, x):
        pi = np.array([np.pi])[:]
        mu = np.array([1.])[:]        
        variance = np.array([1.5])[:]
        sigma = np.sqrt(variance)  
        pdf_value = (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(-((x - mu) ** 2) / (2 * variance))
        pdf_value = pdf_value/np.sum(pdf_value)
        return pdf_value

    def compute_length(self, final_possibility):
        # only based on ghostn_pr
        choice = np.argmax(self.prob, axis=1)
        reals = []; ghosts = []; mirrors = []; reflectors = []
        reflectors_prob = []; reflectors_filter = []
        # iteration along different walls.
        for i in range(self.prob.shape[1]):
            reals.append(self.real_point[choice==i])
            ghosts.append(self.ghost_point[choice==i])
            mirrors.append(self.mirror_point[choice==i])
            reflectors.append(self.reflector_point[choice==i])
            reflectors_prob.append(self.prob[choice==i][:, i])
            reflectors_filter.append(reflectors[-1][reflectors_prob[-1]>self.threshold])
        return reals, ghosts, mirrors, reflectors, reflectors_prob, reflectors_filter

    def compute_perpendicular_line(self, A, B, Points):
        # 计算垂直于A，B的线
        x_a, y_a = A; x_b, y_b = B
        x_m = (x_a + x_b) / 2; y_m = (y_a + y_b) / 2
        M = np.array([[x_m, y_m]])
        if x_b - x_a == 0:  # vertical line
            m_perp = 0  # perpendicular line will be horizontal
            m_dir = np.array([[0, 1]])
        else:
            m_ab = (y_b - y_a) / (x_b - x_a); m_perp = -1 / m_ab
            m_dir = np.array([[1, m_perp]])
        m_dir = m_dir/np.linalg.norm(m_dir)
        proj_len = np.einsum('ij,ij->i', (Points-M), (m_dir))
        start_m = M + np.min(proj_len) * m_dir
        end_m = M + np.max(proj_len) * m_dir
        return start_m[0], end_m[0]
    
    def compute_perpendicular_line_withprob(self, A, B, Points):
        # A是雷达位置，B是高斯中心
        # 计算垂直于A，B的线
        x_a, y_a = A; x_b, y_b = B
        x_m = (x_a + x_b) / 2; y_m = (y_a + y_b) / 2
        M = np.array([[x_m, y_m]])
        if x_b - x_a == 0:  # vertical line
            m_perp = 0  # perpendicular line will be horizontal
            m_dir = np.array([[0, 1]])
        else:
            m_ab = (y_b - y_a) / (x_b - x_a); m_perp = -1 / m_ab
            m_dir = np.array([[1, m_perp]])
        m_dir = m_dir/np.linalg.norm(m_dir)
        proj_len = np.einsum('ij,ij->i', (Points-M), (m_dir))
        lower_bound, upper_bound = self.estimate_gaussian(proj_len)
        start_m = M + lower_bound * m_dir
        end_m = M + upper_bound * m_dir
        number = len(proj_len) > self.threshold_number_pos
        density = len(proj_len)/np.linalg.norm(end_m - start_m) > self.threshold_density_pos
        return start_m[0], end_m[0], number, density

    def estimate_gaussian(self, points, k=2):
        points = np.array(points)
        mean = np.mean(points)
        variance = np.var(points)  
        std_dev = np.sqrt(variance)  
        lower_bound = mean - k * std_dev
        upper_bound = mean + k * std_dev
        return lower_bound, upper_bound

    # Function to cluster points into lines using DBSCAN and fit lines
    def cluster_lines_dbscan(self, points): # For Negative point
        # Step 1: Cluster points using DBSCAN
        dbscan = DBSCAN(eps=self.dbscan_eps_bkg, min_samples=self.dbscan_eps_bkg)  # Adjust `eps` and `min_samples` for your data
        labels = dbscan.fit_predict(points)
        # Step 2: Fit a line to each cluster
        lines = []
        final_labels = -1 * np.ones(points.shape[0], dtype=int)  # Initialize all labels as noise (-1)
        next_cluster_id = 0; clusters = []
        for label in np.unique(labels):
            if label == -1:
                continue  # Skip noise points
            cluster_points = points[labels == label]
            cluster_indices = np.where(labels == label)[0]
            # Use PCA to find the direction of the cluster
            pca = PCA(n_components=1)
            pca.fit(cluster_points)
            direction = pca.components_[0]  # Principal direction
            # Project points onto the principal line
            cluster_mean = np.mean(cluster_points, axis=0)
            projected_points = np.dot(cluster_points - cluster_mean, direction[:, None]) * direction + cluster_mean
            # Find the start and end points of the cluster
            line_start = projected_points.min(axis=0)
            line_end = projected_points.max(axis=0)
            num_point = len(cluster_points)
            length = np.linalg.norm(line_end - line_start)
            density = num_point/length
            if density > self.threshold_density_neg or num_point > self.threshold_number_neg: 
                lines.append((line_start, line_end))
                clusters.append(cluster_points)
                final_labels[cluster_indices] = next_cluster_id
                next_cluster_id += 1
            else: continue
        return clusters, final_labels, lines

    def select_close_line(self, line_list, np_center, H):
        obstacle_col = 100
        world2_edge_obsline = np.zeros((self.H, self.W), dtype=np.uint8)
        for start, end in line_list:
            start_point = tuple(map(int, start))  # Convert to tuple (x, y)
            end_point = tuple(map(int, end))# Convert to tuple (x, y)
            cv2.line(world2_edge_obsline, start_point, end_point, color=obstacle_col, thickness=1)  # Draw the line
        world2_edge_obsline = cv2.dilate(world2_edge_obsline, np.ones((3, 3), dtype=np.uint8))
        world2_edge_obsline = world2_edge_obsline.T # mainly cause by cv2
        radius_vect= np.arange(H*10+1)
        radius_vect= radius_vect.reshape(1, radius_vect.shape[0])
        angles_vect = np.arange(0, np.pi, step=0.5/180*np.pi)
        angles = angles_vect.reshape(angles_vect.shape[0], 1)
        try:
            x_rangeCoordMat = np_center[0].numpy() + np.matmul(np.cos(angles), radius_vect)
            y_rangeCoordMat = np_center[1].numpy() + np.matmul(np.sin(angles), radius_vect)
        except:
            x_rangeCoordMat = np_center[0] + np.matmul(np.cos(angles), radius_vect)
            y_rangeCoordMat = np_center[1] + np.matmul(np.sin(angles), radius_vect)
        slamMap= np.zeros((self.H, self.W))
        dslamMap= np.zeros((self.H, self.W))
        slamMap, dslamMap = self._build_map_with_rangeCoordMat(x_rangeCoordMat, y_rangeCoordMat, world2_edge_obsline, slamMap, dslamMap, obstacle_col)
        return dslamMap
    
    def gauss_point_sampling(self, cluster_points, time_samples=100):
        sampled_maps = []; means = []; covariances = []; starters = []; enders = []; sampled_lines = []; density_list = []
        for cluster in cluster_points:
            mean = np.mean(cluster, axis=0) 
            covariance = np.cov(cluster, rowvar=False)  
            means.append(mean); covariances.append(covariance)
            num_samples = time_samples*len(cluster)
            sampled_points = np.random.multivariate_normal(mean, covariance, size=num_samples)
            sampled_maps.append(sampled_points)
            eigvals, eigvecs = np.linalg.eigh(covariance)
            largest_eigenvalue_index = np.argmax(eigvals)
            largest_eigenvector = eigvecs[:, largest_eigenvalue_index]
            largest_eigenvalue = eigvals[largest_eigenvalue_index]
            std_dev = np.sqrt(largest_eigenvalue)
            
            starters.append(mean + largest_eigenvector * 2 * std_dev)
            enders.append(mean - largest_eigenvector * 2 * std_dev)
            x_coords = np.linspace(starters[-1][0], enders[-1][0], num_samples)
            y_coords = np.linspace(starters[-1][1], enders[-1][1], num_samples)
            sampled_lines.append(np.column_stack((x_coords, y_coords)))
            density_list.append(len(cluster)/std_dev)
        return sampled_maps, sampled_lines, density_list
    
    def gauss_point_sampling_impr(self, cluster_points, time_samples=100):
        sampled_lines = []
        for cluster in cluster_points:
            points = cluster
            centroid = np.mean(points, axis=0)
            centered_points = points - centroid
            covariance_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
            best_fit_direction = eigenvectors[:, np.argmax(eigenvalues)]
            projections = []
            distances = []
            for p in centered_points:
                projection = centroid + np.dot(p, best_fit_direction) * best_fit_direction
                projections.append(projection)
                distances.append(np.linalg.norm(p - projection))  
            projections = np.array(projections)
            # Remove outliers based on distance
            distance_threshold = np.percentile(distances, 90)  # Use top 10% as outliers
            filtered_indices = np.where(distances <= distance_threshold)[0]
            filtered_projections = projections[filtered_indices]
            # Find starting and ending points
            start_point = filtered_projections[np.argmin(np.dot(centered_points[filtered_indices], best_fit_direction))]
            end_point = filtered_projections[np.argmax(np.dot(centered_points[filtered_indices], best_fit_direction))]
            num_samples = time_samples*len(cluster)
            x_coords = np.linspace(start_point[0], end_point[0], num_samples)
            y_coords = np.linspace(start_point[1], end_point[1], num_samples)
            sampled_lines.append(np.column_stack((x_coords, y_coords)))
        return sampled_lines

    def filter_points(self, points, min_val=0, max_val=600):
        max_val = self.H; points = np.array(points)
        valid_msk = (points[:, 0] >= min_val) * (points[:, 0] < max_val) * \
            (points[:, 1] >= min_val) * (points[:, 1] < max_val)
        return valid_msk    

    def dilate_line(self, line_points, scale=3):
        # convert line to dilate
        map_size = np.zeros((self.H, self.H))
        map_size[line_points[:,0], line_points[:,1]] = 1
        # Define the dilation kernel
        kernel_size = (scale, scale)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        dilated_image = cv2.dilate(map_size, kernel, iterations=1)
        new_points = np.argwhere(dilated_image > 0)  # Get coordinates of non-zero pixels
        return new_points
    
    def dilate_line_label(self, line_points, labels, scale=3):
        # convert line to dilate
        kernel_size = (scale, scale)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        points_list = []; labels_list = []
        for label in np.unique(labels):
            map_size = np.zeros((self.H, self.H))
            map_size[line_points[labels==label,0], line_points[labels==label,1]] = 1
            # Define the dilation kernel
            dilated_image = cv2.dilate(map_size, kernel, iterations=1)
            new_points = np.argwhere(dilated_image > 0)  # Get coordinates of non-zero pixels
            new_labels = label * np.ones((len(new_points)))
            points_list.append(new_points); labels_list.append(new_labels)
        final_points = np.vstack(points_list)
        final_labels = np.concatenate(labels_list)
        return final_points, final_labels
        
    def compute_kde(self, cluster_points, sampled_maps, bandwidth=10):
        kde_estimates = []
        for j, sampled_points in enumerate(sampled_maps):
            kde = gaussian_kde(cluster_points[j].T, bw_method=bandwidth)
            density_values = kde(sampled_points.T)
            kde_estimates.append(np.mean(density_values)*100)
        return kde_estimates
        
    def select_close_point(self, points_cluster, np_center):
        obstacle_col = 100
        world2_edge_obsline = np.zeros((self.H, self.W), dtype=np.uint8)
        condition = (points_cluster[:, 0] >= 0) & (points_cluster[:, 0] < self.H) & \
            (points_cluster[:, 1] >= 0) & (points_cluster[:, 1] < self.H)
        points_cluster = points_cluster[condition]
        world2_edge_obsline[points_cluster[:,1], points_cluster[:,0]] = obstacle_col
        world2_edge_obsline = world2_edge_obsline.T
        radius_vect = np.arange(self.H*0.75+1)
        radius_vect = radius_vect.reshape(1, radius_vect.shape[0])
        angles_vect = np.arange(0, np.pi, step=0.01/180*np.pi)
        #angles_vect = np.arange(np.pi*20/180, np.pi*160/180, step=0.5/180*np.pi) # only using point in this range
        angles = angles_vect.reshape(angles_vect.shape[0], 1)
        try: x_rangeCoordMat = np_center[0].numpy() + np.matmul(np.cos(angles), radius_vect); \
            y_rangeCoordMat = np_center[1].numpy() + np.matmul(np.sin(angles), radius_vect)
        except: x_rangeCoordMat = np_center[0] + np.matmul(np.cos(angles), radius_vect); \
            y_rangeCoordMat = np_center[1] + np.matmul(np.sin(angles), radius_vect)
        slamMap= np.zeros((self.H, self.W))
        dslamMap= np.zeros((self.H, self.W))
        slamMap, dslamMap = self._build_map_with_rangeCoordMat(x_rangeCoordMat, y_rangeCoordMat, world2_edge_obsline, slamMap, dslamMap, obstacle_col)
        return slamMap, dslamMap

    def _build_map_with_rangeCoordMat(self, y_rangeCoordMat, x_rangeCoordMat, world_map, slamMap, dslamMap, obstacle_col):
        # Round y and x coord into int
        y_rangeCoordMat = (np.round(y_rangeCoordMat)).astype(np.int64)
        x_rangeCoordMat = (np.round(x_rangeCoordMat)).astype(np.int64) # （角度，长度）
        """ Check for index of y_mat and x_mat that are within the world """
        inBound_ind = self.within_bound(np.array([y_rangeCoordMat, x_rangeCoordMat]), world_map.shape)
        """ delete coordinate that are not within the bound """
        outside_ind = np.argmax(~inBound_ind, axis=1) # 找到最近的不在范围点 （角度）
        ok_ind = np.where(outside_ind == 0)[0] # 第一个点就有问题 
        need_amend_ind = np.where(outside_ind != 0)[0] # 第一个点没有问题
        outside_ind = np.delete(outside_ind, ok_ind) # 删掉无用点 （剩余角度）
        inside_ind = np.copy(outside_ind)  
        inside_ind[inside_ind != 0] -= 1
        bound_ele_x = x_rangeCoordMat[need_amend_ind, inside_ind]
        bound_ele_y = y_rangeCoordMat[need_amend_ind, inside_ind]
        # 把boundary之外的点都换成boundary点
        for count, i in enumerate(need_amend_ind): 
            x_rangeCoordMat[i, ~inBound_ind[i,:]] = bound_ele_x[count] # [1,2,3,3,3,3,...], if 3 is the obstacle.
            y_rangeCoordMat[i, ~inBound_ind[i,:]] = bound_ele_y[count]
        """ find obstacle along the laser range """
        obstacle_ind = np.argmax(world_map[y_rangeCoordMat, x_rangeCoordMat] == obstacle_col, axis=1)
        obstacle_ind[obstacle_ind == 0] = x_rangeCoordMat.shape[1] # 直接设定成最远距离
        """ generate a matrix of [[1,2,3,...],[1,2,3...],[1,2,3,...],...] for comparing with the obstacle coord """
        bx = np.arange(x_rangeCoordMat.shape[1]).reshape(1, x_rangeCoordMat.shape[1])
        by = np.ones((x_rangeCoordMat.shape[0], 1))
        b = np.matmul(by, bx)
        """ get the coord that the robot can percieve (ignore pixel beyond obstacle) """
        b = b <= obstacle_ind.reshape(obstacle_ind.shape[0], 1) # [T,T,T,F,F,F,...], if 3 is the obstacle.
        y_coord = y_rangeCoordMat[b]
        x_coord = x_rangeCoordMat[b]
        """ laser noise + slam matching error """
        slamMap[y_coord, x_coord]= world_map[y_coord,x_coord]
        """ dilate/close to fill the holes """
        dslamMap = cv2.dilate(slamMap, np.ones((3,3)), iterations=1)
        return slamMap, dslamMap

    def within_bound(self, p, shape, r=0):
        return (p[0] >= r) & (p[0] < shape[0]-r) & (p[1] >= r) & (p[1] < shape[1]-r)

    def draw_line(self, start, end, address, number, density, reals=np.array([[0,0]]), ghosts=np.array([[0,0]]), reflectors_filter=np.array([[0,0]]), mirror=np.array([[0,0]]), possibility=np.array([1])):
        num_points = 1000
        bg_start = start - (end - start)*0.1
        bg_end = end + (end - start)*0.1
        x_coords = np.linspace(bg_start[0], bg_end[0], num_points)
        y_coords = np.linspace(bg_start[1], bg_end[1], num_points)
        points = np.column_stack((x_coords, y_coords))
        fixed_color = np.array([1, 1, 1])  # Mid gray (RGB)
        possibility_color = (possibility - np.min(possibility))/(np.max(possibility) - np.min(possibility)) * 0.5 + 0.25
        colors = np.abs(fixed_color * possibility_color[:, np.newaxis]  - 1)# Scale fixed color by possibility
        plt.figure(figsize=(8, 8))
        plt.scatter(reals[:, 1], reals[:, 0], label=f'Reals_{possibility[0]:.3f}', marker='x', alpha=0.9, color='yellow')
        plt.scatter(ghosts[:, 1], ghosts[:, 0], label=f'Ghost_{possibility[0]:.3f}', marker='o', alpha=0.9, color='green')
        plt.scatter(reflectors_filter[:, 1], reflectors_filter[:, 0], label=f'Points_{possibility[0]:.3f}', marker='x', alpha=0.9, color='pink')
        plt.scatter(mirror[:, 1], mirror[:, 0], label=f'Mirror_{possibility[0]:.3f}', marker='o', alpha=0.9, color='blue')
        plt.scatter(points[:, 1], points[:, 0], label=f'Reflector_{possibility[0]:.3f}', marker='.', alpha=0.5, color=colors)
        plt.legend()
        plt.title(f"_num{number}_den{density}_")
        plt.xlim(-self.H, self.H)
        plt.ylim(-self.H, self.H)
        plt.savefig(address, dpi=500, bbox_inches='tight')
        plt.close()
        return points

    def draw_scatter(self, points, labels, sizes, address):
        plt.figure(figsize=(8, 6))
        for i, point in enumerate(points):
            plt.scatter(points[i][:, 1], points[i][:, 0], label=f'{labels[i]}', marker='v', s=sizes[i])
        plt.legend()
        plt.xlim(0, self.H)
        plt.ylim(0, self.H)
        plt.savefig(address, dpi=500, bbox_inches='tight')
        plt.close()

    def compute_angle(self, u, v):
        if v.shape == (2,):
            v = v.reshape(1, 2)
        dot_product = np.sum(u * v, axis=1)  # Shape (n, 1), dot product of each row in u with v
        magnitude_u = np.linalg.norm(u, axis=1)  # Shape (n,)
        magnitude_v = np.linalg.norm(v, axis=1)  # Shape (1,); for a single vector, it will be a scalar
        cos_theta = dot_product / (magnitude_u * magnitude_v)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_radians = np.degrees(np.arccos(cos_theta))  # Shape (n,)
        return angle_radians
    
    def eliminate_angle(self, input_cluster):
        new_input_cluster = []
        for cluster in input_cluster:
            angle = self.compute_angle(cluster-self.c, np.zeros((2,))-self.c)
            keep_mask = (angle < 150) * (angle > 30)
            if len(cluster[keep_mask]) > self.threshold_num:
                new_input_cluster.append(cluster[keep_mask])
        return new_input_cluster

    def eliminate_short_line_iter(self, _map_, iteration_eli=1):
        for i in tqdm(range(iteration_eli)):
            _map_ = self.eliminate_short_line(_map_)
        return _map_
    
    def eliminate_short_line(self, _map_):
        center = self.c
        coordinates = np.argwhere(_map_ > 0)
        # Step 2: Compute angles (theta)
        angles = self.compute_angle(coordinates - center, self.ref)
        # Step 3: Sorting theta
        sorted_indices = np.argsort(angles)
        sorted_coors = coordinates[sorted_indices]
        sorted_angles = angles[sorted_indices]
        eliminates = self.finding_longest(sorted_coors, sorted_angles, self.threshold_dis, self.threshold_ang)
        if len(eliminates) > 0:
            elim_index = np.concatenate(eliminates)
            elim_coords = sorted_coors[elim_index]
            _map_[elim_coords[:,0], elim_coords[:,1]] = 0 
        return _map_
    
    def finding_longest(self, coordinates, thetas, threshold_dis=6, threshold_ang=0.05):
        distances = np.linalg.norm(np.diff(coordinates, axis=0), axis=1) # (n-1)
        current = [0]; eliminates = []
        for num_dis, dis in enumerate(distances):
            if dis < threshold_dis: current.append(num_dis+1)
            elif len(current) > 0:
                vector_ratio = np.linalg.norm(coordinates[current[-1]] - coordinates[current[0]])/np.linalg.norm((coordinates[current[-1]] + coordinates[current[0]])/2-self.c) 
                angle_start, angle_end = self.compute_angle(coordinates[current[-1]] - self.c, self.ref), self.compute_angle(coordinates[current[0]] - self.c, self.ref)
                if vector_ratio < threshold_ang or (max(angle_start, angle_end) < 30 or min(angle_start, angle_end) > 150): eliminates.append(current)
                current = [num_dis+1]
        if len(current) > 0: # 保证最后一个被删除
            vector_ratio = np.linalg.norm(coordinates[current[-1]] - coordinates[current[0]])/np.linalg.norm((coordinates[current[-1]] + coordinates[current[0]])/2) 
            angle_start, angle_end = self.compute_angle(coordinates[current[-1]] - self.c, self.ref), self.compute_angle(coordinates[current[0]] - self.c, self.ref)
            if vector_ratio < threshold_ang or (max(angle_start, angle_end) < 30 or min(angle_start, angle_end) > 150): eliminates.append(current)
        return eliminates
    
    def create_line(self, dialte_whole_cluster, dialte_whole_labels, whole_clusters_coor):
        label_mask = np.zeros((self.H, self.W)); line_map = np.zeros((self.H, self.W))
        label_mask[dialte_whole_cluster[:,0], dialte_whole_cluster[:,1]] = dialte_whole_labels
        line_map[whole_clusters_coor[:,0], whole_clusters_coor[:,1]] = 1
        label_map = label_mask * line_map
        points_list = []
        threshold = 5
        for label in np.unique(label_map): # 单独处理每一个label.
            if label != 0:
                mask = label_map == label
                coords = np.column_stack(np.nonzero(mask))
                angles = self.compute_angle(coords - self.c, self.ref)
                sorted_indices = np.argsort(angles); coords = coords[sorted_indices]
                distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
                checkpoint = np.sum(distances > threshold)
                indices = [index+1 for index, value in enumerate(distances > threshold) if value > 0]
                indices = [0] + indices + [len(coords)-1]
                for i in range(checkpoint + 1):
                    if np.linalg.norm(coords[indices[i]] - coords[indices[i+1]]) > threshold:
                        start_point, end_point = self.best_fit_line(coords[indices[i]:indices[i+1]])
                        if np.linalg.norm((start_point-end_point)) > threshold:
                            points_list.append([start_point.astype(int), end_point.astype(int)])
        return points_list
    
    def best_fit_line(self, coords):
        # Separate the coordinates into x and y
        x = coords[:, 1].reshape(-1, 1); y = coords[:, 0]  
        # Perform linear regression to find the best-fit line
        model = LinearRegression(); model.fit(x, y)
        # Get the slope and intercept of the line
        slope = model.coef_[0]; intercept = model.intercept_
        # Compute the start and end points based on min and max x-values
        x_min = np.min(x); x_max = np.max(x)
        # Compute corresponding y values for the start and end points
        y_min = slope * x_min + intercept
        y_max = slope * x_max + intercept
        # Return the start and end points of the line
        start_point = np.array([x_min, y_min])
        end_point = np.array([x_max, y_max])
        return start_point, end_point

    def fit_line_ransac(self, pts):
        """
        使用 RANSAC 进行鲁棒直线拟合，返回 slope, intercept, 噪音点 MAE, 内点掩码
        """
        X = pts[:, 0].reshape(-1, 1)
        y = pts[:, 1]
        ransac = RANSACRegressor(LinearRegression(),
                                min_samples=self.dbscan_min_samples_ours,
                                residual_threshold=0.05)
        ransac.fit(X, y)
        in_mask = ransac.inlier_mask_
        line_pred = ransac.predict(X[in_mask])
        mae = np.mean(np.abs(line_pred - y[in_mask]))
        k = ransac.estimator_.coef_[0]
        b = ransac.estimator_.intercept_
        return k, b, mae, in_mask

    def gmm_ours2(self, points):
        # 1. DBSCAN 聚类
        db = DBSCAN(eps=self.dbscan_eps_ours,
                    min_samples=self.dbscan_min_samples_ours).fit(points)
        labels = db.labels_
        segments = []
        for cid in set(labels):
            if cid == -1:
                continue
            cluster_pts = points[labels == cid]
            if len(cluster_pts) < self.dbscan_min_samples_ours:
                continue
            # 2. 初始 RANSAC 拟合
            k0, b0, mae0, in0 = self.fit_line_ransac(cluster_pts)
            bic0 = self.compute_bic(cluster_pts[in0], n_components=1)
            best = {'pts': cluster_pts[in0], 'k': k0, 'b': b0, 'mae': mae0}
            # 3. GMM 分裂判断
            if self.gmm_max_components >= 2:
                gmm = GaussianMixture(n_components=2, random_state=0)
                try:
                    labels2 = gmm.fit_predict(cluster_pts)
                    bic2 = (self.compute_bic(cluster_pts[labels2 == 0], n_components=1) +
                            self.compute_bic(cluster_pts[labels2 == 1], n_components=1))
                except Exception:
                    bic2 = np.inf
                # BIC 和 MAE 作为分裂判据
                if bic2 + self.bic_penalty < bic0:
                    for comp in [0,1]:
                        sub_pts = cluster_pts[labels2 == comp]
                        if len(sub_pts) < self.dbscan_min_samples_ours:
                            continue
                        k1, b1, mae1, in1 = self.fit_line_ransac(sub_pts)
                        if mae1 + self.mae_tol < best['mae']:
                            best = {'pts': sub_pts[in1], 'k': k1, 'b': b1, 'mae': mae1}
            # 剪枝：丢弃内点比例不够的
            if len(best['pts']) / len(cluster_pts) < self.prune_inliers_ratio:
                continue
            segments.append((best['k'], best['b'], best['pts']))
        return segments

    def compute_bic(self, pts, n_components):
        if len(pts) < self.dbscan_min_samples_ours:
            return np.inf
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(pts)
        log_lik = gmm.score(pts) * len(pts)
        p = n_components * (pts.shape[1] + pts.shape[1] * pts.shape[1])
        bic = -2 * log_lik + p * np.log(len(pts))
        return bic

def merge_similar_lines(lines, slope_threshold=0.1, distance_threshold=10):
    if lines is None or len(lines) == 0: return []
    merged_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)  # 避免除零错误
        merged = False
        for i, (mx1, my1, mx2, my2, mslope) in enumerate(merged_lines):
            # 计算斜率差异
            if abs(mslope - slope) < slope_threshold:
                # 计算端点之间的最小距离
                d1 = np.sqrt((x1 - mx1) ** 2 + (y1 - my1) ** 2)
                d2 = np.sqrt((x2 - mx2) ** 2 + (y2 - my2) ** 2)
                d3 = np.sqrt((x1 - mx2) ** 2 + (y1 - my2) ** 2)
                d4 = np.sqrt((x2 - mx1) ** 2 + (y2 - my1) ** 2)
                min_distance = min(d1, d2, d3, d4)
                if min_distance < distance_threshold:
                    # 合并直线，取更长的端点范围
                    merged_lines[i] = [
                        min(x1, x2, mx1, mx2), min(y1, y2, my1, my2),
                        max(x1, x2, mx1, mx2), max(y1, y2, my1, my2),
                        (mslope + slope) / 2]
                    merged = True
                    break
        if not merged:
            merged_lines.append([x1, y1, x2, y2, slope])
    return [line[:4] for line in merged_lines]

def count_merged_lines(map_image):
    # 确保输入是二值化的（0: 背景, 1: 线）
    binary_map = (map_image > 0).astype(np.uint8) * 255  
    # 使用 Canny 检测边缘
    edges = cv2.Canny(binary_map, 50, 150)
    # 使用概率霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
    if lines is None: return 0
    # 合并相似直线
    merged_lines = merge_similar_lines(lines)
    return merged_lines

def reflect_point_over_line(point, line_p0, line_p1):
    x0, y0 = point
    x1, y1 = line_p0
    x2, y2 = line_p1
    if x2 == x1:
        xr = 2*x1 - x0
        yr = y0
        return xr, yr
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a*x1
    denom = 1 + a*a
    xr = ((1 - a*a)*x0 + 2*a*(y0 - b)) / denom
    yr = (2*a*x0 + (a*a - 1)*y0 + 2*b) / denom
    return xr, yr

def mirror_of_P_over_lines(point_list, P):
    mirrors = []
    for (p0, p1) in point_list:
        xr, yr = reflect_point_over_line(P, p0, p1)
        print(p0, p1, "=========", xr, yr)
        mirrors.append((xr, yr))
    return mirrors

def fit_line_params(pts):
    X = pts[:, 0].reshape(-1, 1)
    y = pts[:, 1]
    model = LinearRegression().fit(X, y)
    return model.coef_[0], model.intercept_

def compute_segment_endpoints(m, c, pts):
    X0 = np.array([0.0, c])
    u = np.array([1.0, m])
    u = u / np.linalg.norm(u)
    s = (pts - X0).dot(u)
    s_min, s_max = s.min(), s.max()
    P_min = X0 + s_min * u
    P_max = X0 + s_max * u
    return tuple(P_min.astype(int)), tuple(P_max.astype(int))

def fuse_lines_from_mirrors(mirror_pts, sample_lists,
                            d_thresh=40.0,
                            mae_thresh=40.0):
    N = len(mirror_pts)
    pts = np.array(mirror_pts, float)
    # KD-tree for spatial neighbors
    tree = cKDTree(pts)
    # Build adjacency list: connect if within d_thresh
    adj = {i: set() for i in range(N)}
    for i in range(N):
        neighbors = tree.query_ball_point(pts[i], d_thresh)
        for j in neighbors:
            if j > i:
                adj[i].add(j)
                adj[j].add(i)
    # Find connected components via DFS
    visited = [False]*N
    line_segments = []
    fused_samples = []
    for i in range(N):
        if visited[i]:
            continue
        # collect component
        stack = [i]
        comp = []
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    stack.append(v)
        # Merge all samples in component
        combined = np.vstack([sample_lists[k] for k in comp])
        # Fit line and compute MAE
        m, c = fit_line_params(combined)
        y_pred = m * combined[:,0] + c
        mae = np.mean(np.abs(combined[:,1] - y_pred))
        # Accept fusion if MAE ok and more than one mirror in comp
        if mae <= mae_thresh and len(comp) > 1:
            start, end = compute_segment_endpoints(m, c, combined)
            line_segments.append((start, end))
            fused_samples.append(combined)
        else:
            # fallback to individual lines
            for k in comp:
                single = sample_lists[k]
                m_k, c_k = fit_line_params(single)
                start_k, end_k = compute_segment_endpoints(m_k, c_k, single)
                line_segments.append((start_k, end_k))
                fused_samples.append(single)
    return line_segments, fused_samples


if __name__ == '__main__':
    case = 2
    if case == 1:
        fix_list = [f'test{num}' for num in range(62,65)]
        #fix_list = [f'test{num}' for num in range(53,62)]
        fix_list = [f'test{num}' for num in range(53,65)]
        fix_list = [f'test{num}' for num in range(53,65)]
        min_gmm, max_gmm = 1, 20 
        H, W = 500, 500
    elif case == 2:
        fix_list = [f'{num}' for num in range(69,77)]
        min_gmm, max_gmm = 1, 8
        H, W = 600, 600
    elif case == 3:
        fix_list = [f'{num}' for num in range(82,87)]
        min_gmm, max_gmm = 1, 8
        H, W = 600, 600
    batch = 3 # Length for tracking
    pghost1_means_list, pghost1_reals_list = [], []
    pghost1pr_means_list, pghost1pr_reals_list = [], []
    pghost2pr_means_list, pghost2pr_reals_list = [], []
    for after_fix in fix_list:
        prefix = f'/mnt/c/kaichen/ghost/ti/mmwave_studio_03_00_00_14/mmWaveStudio/PostProc/Cascade_Capture_22xx_{after_fix}/preprocess/'
        prefix_save_cl_final = f'image_batch{batch}_range_ang_cluster_all_final'
        prefix_save_cl_final = os.path.join(prefix, prefix_save_cl_final)
        ADDRESS_DICT = os.path.join(prefix_save_cl_final, f'combine.npz')
        try:
            data = dict(np.load(ADDRESS_DICT, allow_pickle=True))
            data = data['arr_0'].item()
            center = data["REAL"]
            pghost1_means, pghost1_reals, pghost1_ranges, pghost1_mags, pghost1_nums =  \
                data["pghost1_means"], data["pghost1_reals"], data["pghost1_ranges"], data["pghost1_mags"], data["pghost1_nums"]
            pghost1pr_means, pghost1pr_reals, pghost1pr_ranges, pghost1pr_mags, pghost1pr_nums =  \
                data["pghost1pr_means"], data["pghost1pr_reals"], data["pghost1pr_ranges"], data["pghost1pr_mags"], data["pghost1pr_nums"]
            pghost2pr_means, pghost2pr_reals, pghost2pr_ranges, pghost2pr_mags, pghost2pr_nums =  \
                data["pghost2pr_means"], data["pghost2pr_reals"], data["pghost2pr_ranges"], data["pghost2pr_mags"], data["pghost2pr_nums"]
            pghost1_means_list.append(pghost1_means)
            pghost1_reals_list.append(pghost1_reals)
            pghost1pr_means_list.append(pghost1pr_means)
            pghost1pr_reals_list.append(pghost1pr_reals)
            pghost2pr_means_list.append(pghost2pr_means)
            pghost2pr_reals_list.append(pghost2pr_reals)
        except:
            print('-------------END--------------', after_fix)
    # ----------- Concatenating all necessary ghost 1 and ghost 1 pr.
    pghost1_means_list = np.concatenate(pghost1_means_list, axis=0)
    pghost1_reals_list = np.concatenate(pghost1_reals_list, axis=0)
    pghost1pr_means_list = np.concatenate(pghost1pr_means_list, axis=0)
    pghost1pr_reals_list = np.concatenate(pghost1pr_reals_list, axis=0)
    pghost2pr_means_list = np.concatenate(pghost2pr_means_list, axis=0)
    pghost2pr_reals_list = np.concatenate(pghost2pr_reals_list, axis=0)
    # ----------- Initializing optimization class.
    optimizer = Optimizer_one_rel(center, pghost1_means_list, pghost1pr_means_list, pghost2pr_means_list, \
        pghost1_reals_list, pghost1pr_reals_list, pghost2pr_reals_list, \
        min_gmm, max_gmm, H, W)
    # ----------- Initialize background points.
    if case == 1:
        bk_class = bk_case1(center, optimizer.RESOLUTION)
    elif case == 2:
        bk_class = bk_case2(center, optimizer.RESOLUTION)
    elif case == 3:
        bk_class = bk_case3(center, optimizer.RESOLUTION)
    gt_bg_points = bk_class.bg_points
    # ----------- Making use of ghost1pr
    reflector_point_1pr, mirror_point_1pr = optimizer.compute_reflector_1pr()
    reflector_point_2pr, mirror_point_2pr = optimizer.compute_reflector_2pr()
    print(len(reflector_point_1pr), '-------------Length of 1pr')
    print(len(reflector_point_2pr), '-------------Length of 2pr')
    choices = 1
    # choice:1 -> Both mirror_point_1pr and mirror_point_2pr
    # choice:2 -> Only mirror_point_2pr
    # choice:3 -> Only mirror_point_1pr
    if choices == 1:
        optimizer.mirror_point = np.concatenate([mirror_point_1pr, mirror_point_2pr], axis=0)  
        optimizer.reflector_point = np.concatenate([reflector_point_1pr, reflector_point_2pr], axis=0)  
    elif choices == 2:
        optimizer.mirror_point = mirror_point_2pr
        optimizer.reflector_point = reflector_point_2pr
    elif choices == 3:
        optimizer.mirror_point = mirror_point_1pr
        optimizer.reflector_point = reflector_point_1pr
    # ----------- Starting Optimization
    optimizer.gmm(optimizer.mirror_point)
    # ----------- Making use of ghost1
    ratio = optimizer.p1_dist() # Computing ratios for every pghost1 point
    pdf_value = optimizer.weight_norm(ratio) # Computing the probability of each pghost1 points based on distance
    print(optimizer.gaus_weights, '----------- Prob 1 -----------\n')
    print(pdf_value, '----------- Prob 2 -----------\n')
    # ----------- Combining probability of ghost1 and ghost1pr
    final_possibility = pdf_value*optimizer.gaus_weights
    final_possibility = final_possibility/np.sum(final_possibility)
    print(final_possibility, '----------- Prob 3 -----------')
    _, _, reflectors_filter = optimizer.compute_length(final_possibility)
    # ----------- Drawing line
    prefix_save = f'/mnt/c/kaichen/ghost/ti/mmwave_studio_03_00_00_14/mmWaveStudio/PostProc/Cascade_Capture_22xx_{fix_list[0]}_{fix_list[-1]}/preprocess/'
    if not os.path.exists(prefix_save):
        os.makedirs(prefix_save)
    rels = []
    for i in range(len(final_possibility)):
        start, end = optimizer.compute_perpendicular_line(optimizer.c[0], optimizer.gaus_means[i], reflectors_filter[i])
        address = os.path.join(prefix_save, f'large_predict_{final_possibility[i]:.3f}.png')
        est_bg_points = optimizer.draw_line(start, end, address, reflectors_filter[i], possibility=final_possibility[i:i+1])
        rels.append(est_bg_points)
    rels = np.concatenate(rels, axis=0)
    address = os.path.join(prefix_save, 'All.png')
    optimizer.draw_scatter([rels, gt_bg_points], ['EST', 'GT'], [10, 1], address)

    address = os.path.join(prefix_save, 'Cl_1.png')
    optimizer.draw_scatter([optimizer.pg1, optimizer.pg1_r], ['Gh', 'Obj'], [3, 3], address)
    address = os.path.join(prefix_save, 'Cl_1pr.png')
    optimizer.draw_scatter([optimizer.pg1pr, optimizer.pg1pr_r], ['Gh', 'Obj'], [3, 3], address)
    address = os.path.join(prefix_save, 'Cl_2pr.png')
    optimizer.draw_scatter([optimizer.pg2pr, optimizer.pg2pr_r], ['Gh', 'Obj'], [3, 3], address)


    """
    address = os.path.join(prefix_save, 'Cluster.png')
    optimizer.draw_scatter(address)
    address = os.path.join(prefix_save, 'Cluster_pr.png')
    optimizer.draw_scatter_pr(address)
    address = os.path.join(prefix_save, 'Cluster_2pr.png')
    optimizer.draw_scatter_2pr(address)
    """