import yaml
import random
import cv2, os
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import simulator.envs.simulator.util as util
import simulator.envs.simulator.jsonReader as jsonReader
map_color= {'uncertain':-101, 'free':0, 'obstacle':100}
move_choice= {'forward':np.array([1,0]), 'left': np.array([0,1]), 'right': np.array([0,-1])}

def random_point_on_edge(p1, p2):
    t = random.uniform(0, 1)  
    return (1 - t) * p1 + t * p2

def random_line_in_rectangle(rect, min_distance=0.90):
    edges = [(rect[i], rect[(i + 1) % 4]) for i in range(4)]  # Edges of the rectangle
    edges_length = [np.linalg.norm(rect[i] - rect[(i + 1) % 4]) for i in range(4)]
    random_edge1 = random.choice(edges)
    random_edge2 = random.choice(edges)
    while True:
        point1 = random_point_on_edge(np.array(random_edge1[0]), np.array(random_edge1[1]))
        point2 = random_point_on_edge(np.array(random_edge2[0]), np.array(random_edge2[1]))
        t1 = random.uniform(0, (1.-min_distance)/2.)  # Random proportion of the distance
        t2 = random.uniform(1.-(1.-min_distance)/2., 1.)
        start_point = point1 + t1*(point2-point1)
        end_point = point1 + t2*(point2-point1)
        if np.linalg.norm(end_point-start_point) > min_distance*np.min(edges_length): 
            return start_point, end_point

class pseudoSlam():
    def __init__(self, param_file, obstacle_config=None):
        """ define class variable """
        self.m2p= 0; self.robotRadius= 0
        self.stepLength_linear= 0; self.stepLength_angular= 0
        self.config_poseInit= np.array([0,0,0])
        self.robotPose_init= np.array([0.,0.,0.])
        self.robotResetRandomPose= 1; self.obs_num= 0
        self.obs_sizeRange= np.array([0,0])
        self.laser_range=0; self.laser_fov= 0
        self.laser_resol= 0; self.laser_noiseSigma= 0
        self.slamErr_linear= 0; self.slamErr_angular= 0
        self.state_size= np.array([0,0])
        self.world= np.zeros([1,1])
        self.obstacle_config = False
        self.visual_obs = False
        self.Flag = False
        with open(param_file) as stream:
            self.config = yaml.load(stream, Loader=yaml.FullLoader)
        """json reader init"""
        self.json_reader = jsonReader.jsonReader(self.config['json_dir'], self.config['meter2pixel'])
        """ set map_color """
        self.map_color = map_color
        self.map_id_set = np.loadtxt(os.path.join(os.path.dirname(__file__), "../", self.config['map_id_set']), str)
        """ Initialize user config param """
        self.initialize_param(self.config)
        """ pre calculate radius and angle vector that will be used in building map """
        radius_vect = np.arange(self.laser_range*10+1)
        radius_vect_gt = np.arange(self.laser_range*10+1)
        self.radius_vect= radius_vect.reshape(1, radius_vect.shape[0]) # generate radius vector of [0,1,2,...,laser_range]
        self.radius_vect= radius_vect.reshape(1, radius_vect.shape[0]) # generate radius vector of [0,1,2,...,laser_range]
        self.radius_vect_gt = radius_vect_gt.reshape(1, radius_vect_gt.shape[0]) # generate radius vector of [0,1,2,...,laser_range]
        self.radius_vect_gt = radius_vect_gt.reshape(1, radius_vect_gt.shape[0]) # generate radius vector of [0,1,2,...,laser_range]
        angles_vect = np.arange(-self.laser_fov*0.5, self.laser_fov*0.5, step=self.laser_resol*0.02)
        self.angles_vect = angles_vect.reshape(angles_vect.shape[0], 1) # generate angles vector from -laser_angle/2 to laser_angle
        self.robotPose = self.robotPose_init
        self.threshold_start_end = 10; num = 0
        while not self.reset(): num += 1
        return

    def initialize_param(self, config):
        """ world & robot param """
        self.m2p= config["meter2pixel"] # X pixel= 1 meter
        self.robotRadius= util.meter2pixel(config["robotRadius"], self.m2p) # robot radius in pixel
        self.stepLength_linear= util.meter2pixel(config["stepLength"]["linear"], self.m2p) # robot linear movement in each step in pixel
        self.stepLength_angular= util.deg2rad( config["stepLength"]["angular"] ) # robot angular movement in each step in rad
        """ robot starting pose """
        # robot starting pose in world coordinate and rad with the form of [y;x;theta]
        self.config_poseInit[0]= config["startPose"]["y"]
        self.config_poseInit[1]= config["startPose"]["x"]
        self.config_poseInit[2]= util.deg2rad( config["startPose"]["theta"] )
        # flag of robot randomly reset start pose in each reset
        self.robotResetRandomPose= config["resetRandomPose"]
        """ obstacle param """
        self.obs_num= config["obstacle"]["number"] # number of obstacle added to the world
        self.obs_num= np.random.choice([0,1,2,3])
        # size of obstacle added in the form of [min;max] in pixel
        self.obs_sizeRange= np.zeros((2,1))
        self.obs_sizeRange[0]= util.meter2pixel( config["obstacle"]["size"]["min"], self.m2p)
        self.obs_sizeRange[1]= util.meter2pixel( config["obstacle"]["size"]["max"], self.m2p)
        """ laser param """
        self.laser_range= util.meter2pixel(config["laser"]["range"], self.m2p) # laser range in pixel
        self.laser_fov= util.deg2rad( config["laser"]["fov"] ) # laser field of view in rad
        self.laser_resol= util.deg2rad( config["laser"]["resolution"])# laser rotation resolution in rad
        self.laser_noiseSigma= config["laser"]["noiseSigma"] # sigma of Gaussian distribution of laser noise
        """ slam error """
        self.slamErr_linear= config["slamError"]["linear"] # slam linear registration error in pixel?
        self.slamErr_angular= util.deg2rad( config["slamError"]["angular"] ) # slam rotational registration error in rad
        """ state size """
        self.state_size= ( config["stateSize"]["x"] * self.m2p, config["stateSize"]["y"] * self.m2p ) # state size in the form of [x;y]
        return

    def create_world(self, order=False, padding=10):
        if order: # Reading according to order
            map_id = self.map_id_set[0]
            self.map_id_set = np.delete(self.map_id_set, 0)
        else: map_id = np.random.choice(self.map_id_set)
        self.map_id = map_id
        #map_id = "3b33d5b6677a12f23e99ffbaac744a91"
        #map_id = "a99ab6fcd3771f073aa9b780d7192e86"
        input_world = self.json_reader.read_json(map_id)
        # ================= get map (indoor 0 outdoor 1) =================
        self.world = self._map_process(input_world, padding=padding)
        (h,w) = self.world.shape
        # ================= further processing wall (wall 1 non-wall 0) =================
        self.world1 = self.world.copy()
        _, self.world1_edge = edge_detection(self.world1.astype(np.uint8))
        self.world1_edge[self.world1_edge>1] = self.map_color['obstacle']
        # ================ change the position of camera to map position ================
        self.robotPose_init[0:2]= util.world2mapCoord(self.config_poseInit, (h*0.5,w*0.5), self.m2p)
        self.robotPose_init[2]= self.config_poseInit[2]

    def _map_process(self,input_world, padding=5):
        # =================  =================
        """ process input map into simulator compatible map """
        map_gt = np.zeros_like(input_world)
        """ input world obstacle= 0 & free space= 255 | convert into simulator config """
        map_gt[input_world == 0] = self.map_color['obstacle']
        map_gt[input_world == 255] = self.map_color['free']
        """ crop out redundent obstacle region in boundaries """
        index = np.where(map_gt == self.map_color['obstacle'])
        [index_row_max, index_row_min, index_col_max, index_col_min] = \
            [np.max(index[0]), np.min(index[0]), np.max(index[1]), np.min(index[1])]
        map_gt = map_gt[index_row_min:index_row_max+1, index_col_min:index_col_max+1]
        """ pad the map with obstacle on near boundary """
        map_gt = np.pad(map_gt, padding, mode='constant', constant_values=self.map_color['obstacle'])
        return map_gt

    def add_obstacle(self):
        # =================  =================
        """ Randomly add obstacle to world """
        # needing: self.transformed_world1_edge; self.transformed_world1
        prox_min = 20; closet_obs = 50 # prox_min Distance between Obstacle; closet_obs Distance between Obstacle and Camera.
        transformed_world1_obs = self.map_for_obstacle.copy()
        transformed_world1_obs[:,:closet_obs] = 1.
        self.world_pure_obs = np.zeros_like(transformed_world1_obs)
        self.world_pure_obsline = np.zeros_like(transformed_world1_obs)
        if self.obs_num == 0: print('==========NO OBS=========='); return # No obstacle added.
        for i in range(self.obs_num):
            attempts = 0
            while (1):
                if attempts > 25: print('==========FAIL=========='); break
                attempts += 1
                # randomly generate obstacle orientation & obstacle size that fall within obstacle_sizeRange
                if self.obs_sizeRange[0] != self.obs_sizeRange[1]:
                    obs_a = np.random.randint(min(self.obs_sizeRange[0], self.obs_sizeRange[1]), max(self.obs_sizeRange[0], self.obs_sizeRange[1]))*0.5
                    obs_b = np.random.randint(min(self.obs_sizeRange[0], self.obs_sizeRange[1]), max(self.obs_sizeRange[0], self.obs_sizeRange[1]))*0.5
                else: obs_a, obs_b = self.obs_sizeRange[0], self.obs_sizeRange[1]
                obs_len = np.ceil(np.sqrt(obs_a*obs_a + obs_b*obs_b) + prox_min*2).astype(np.uint32)[0]
                # Preventing Collison.
                cur_transformed_world1_obs = cv2.dilate(transformed_world1_obs, np.ones((obs_len,obs_len)), iterations=1)
                cur_transformed_world1_obs = (cur_transformed_world1_obs>0).astype(np.float32)
                zero_indices = np.argwhere(cur_transformed_world1_obs == 0)
                if len(zero_indices) > 0: selected_coordinate = zero_indices[np.random.choice(len(zero_indices))]
                else: continue
                obs_y, obs_x = selected_coordinate
                obs_theta = np.random.random()*360; cthe = np.cos(np.pi/180* obs_theta); sthe = np.sin(np.pi/180* obs_theta)
                rect = np.array([[obs_x + (-obs_a * cthe - -obs_b * sthe), obs_y + (-obs_a * sthe + -obs_b * cthe)],
                                    [obs_x + (-obs_a * cthe - obs_b * sthe), obs_y + (-obs_a * sthe + obs_b * cthe)],
                                    [obs_x + (obs_a * cthe - obs_b * sthe), obs_y + (obs_a * sthe + obs_b * cthe)],
                                    [obs_x + (obs_a * cthe - -obs_b * sthe), obs_y + (obs_a * sthe + -obs_b * cthe)]], np.int64)
                cv2.fillPoly(transformed_world1_obs, [rect], self.map_color["obstacle"])
                cv2.fillPoly(self.world_pure_obs, [rect], self.map_color["obstacle"])
                for _ in range(weighted_choice()): # 20% - 2; 80% - 1
                    start_point, end_point = random_line_in_rectangle(rect)
                    long_axis_x1, long_axis_y1 = int(start_point[0]), int(start_point[1])
                    long_axis_x2, long_axis_y2 = int(end_point[0]), int(end_point[1])
                    cv2.line(self.world_pure_obsline, (long_axis_x1, long_axis_y1), (long_axis_x2, long_axis_y2), self.map_color["obstacle"], thickness=1)
                break

    def initial_point_median(self,):
        foreground_coords = np.argwhere(self.world == 0)
        median_x = int(np.median(foreground_coords[:, 0]))
        median_y = int(np.median(foreground_coords[:, 1]))
        median_point = np.array([median_x, median_y])
        distances = cdist([median_point], foreground_coords)  # Calculate distances to all foreground points
        closest_index = np.argmin(distances)
        validated_point = tuple(foreground_coords[closest_index]) 
        self.robotPose[0] = validated_point[0]
        self.robotPose[1] = validated_point[1]
        self.robotPose[2] = float(np.random.rand() * np.pi * 2)
        self.robotPose[2] = np.random.choice([0, 1/4*np.pi, 1/2*np.pi, np.pi, float(np.random.rand() * np.pi * 2), float(np.random.rand() * np.pi * 2), float(np.random.rand() * np.pi * 2)])
        return self.robotPose

    def reset(self, order=False):
        #T1 =  time.time()
        self.create_world(order)
        self.initial_point_median()
        # ========= rotating world map
        self.transformed_world1 = transfer(self.robotPose, self.world1, borderValue=1) 
        self.transformed_world1 = (self.transformed_world1 >= 1).astype(float)
        self.transformed_world1_edge = transfer(self.robotPose, self.world1_edge, borderValue=0) 
        self.transformed_world1_edge = (self.transformed_world1_edge >= 1).astype(float)
        if self.check_wall(self.transformed_world1): print('--------Fail due to the wall--------'); return False
        # ===========1 Get ground truth based on robot position.
        transformed_world1_edge = (cv2.dilate(self.transformed_world1_edge, np.ones((4, 4)), iterations=1) > 0.5)*100.
        robotPose = [300,0,0]; self.slamMap_nobs, _ = self.build_map(robotPose, transformed_world1_edge)
        self.transformed_gt_wall = transfer(self.robotPose, self.slamMap_nobs+100) # wall ground truth
        self.new_gt_wall, self.new_gt_wall_nd = self.complete_wall(self.transformed_gt_wall); self.map_for_obstacle = (self.new_gt_wall != 100).astype(float)
        self.transformed_new_wall_clear = (self.new_gt_wall>=125).astype(float) # This is what we want GT + Wall        
        self.transformed_new_wall_clear_nd = (self.new_gt_wall_nd>=125).astype(float) # This is what we want GT + Wall        
        if self.visual_obs:
            plt.imshow(self.world1, cmap='viridis', aspect='auto'); plt.colorbar(); plt.title("Heatmap of NumPy Array")
            plt.savefig("1_original_world.png", dpi=300, bbox_inches='tight'); plt.close()
            plt.imshow(self.world1_edge, cmap='viridis', aspect='auto'); plt.colorbar(); plt.title("Heatmap of NumPy Array")
            plt.savefig("1_original_world_edge.png", dpi=300, bbox_inches='tight'); plt.close()
            plt.imshow(self.transformed_world1, cmap='viridis', aspect='auto'); plt.colorbar(); plt.title("Heatmap of NumPy Array")
            plt.savefig("1_original_world_edge_transfer.png", dpi=300, bbox_inches='tight'); plt.close()
            plt.imshow(self.new_gt_wall, cmap='viridis', aspect='auto'); plt.colorbar(); plt.title("Heatmap of NumPy Array")
            plt.savefig("1_GT_WALL_Field.png", dpi=300, bbox_inches='tight'); plt.close()
            plt.imshow(self.transformed_new_wall_clear_nd, cmap='viridis', aspect='auto'); plt.colorbar(); plt.title("Heatmap of NumPy Array")
            plt.savefig("1_GT_WALL_ND.png", dpi=300, bbox_inches='tight'); plt.close()
            plt.imshow(self.transformed_new_wall_clear, cmap='viridis', aspect='auto'); plt.colorbar(); plt.title("Heatmap of NumPy Array")
            plt.savefig("1_GT_WALL.png", dpi=300, bbox_inches='tight'); plt.close()
            plt.imshow(self.map_for_obstacle, cmap='viridis', aspect='auto'); plt.colorbar(); plt.title("Heatmap of NumPy Array")
            plt.savefig("1_GT_WALL_Field_new.png", dpi=300, bbox_inches='tight'); plt.close()
        self.add_obstacle()
        self.world_pure_obsline = (self.world_pure_obsline>=self.map_color["obstacle"]/2).astype(float)
        self.world_pure_obs = (self.world_pure_obs>=self.map_color["obstacle"]/2).astype(float) # This is what we want GT + Observation

        map_used_observation = self.transformed_new_wall_clear + self.world_pure_obsline
        map_used_observation = (cv2.dilate(map_used_observation.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0.5)*100. 
        self.slamMap_whole, _ = self.build_map(robotPose, map_used_observation) 
        obs_mask = cv2.dilate(self.world_pure_obs.astype(np.uint8), np.ones((8, 8), np.uint8)) > 0.0
        self.slamMap_whole = (self.slamMap_whole > 0).astype(float)
        self.slamMap_wall = self.slamMap_whole * (1 - obs_mask)
        self.slamMap_obs = self.slamMap_whole * obs_mask
        self.slamMap_obs = straighten_curves(cv2.dilate(self.slamMap_obs.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0.5)
        if self.visual_obs:
            plt.imshow(map_used_observation, cmap='viridis', aspect='auto'); plt.colorbar(); plt.title("Heatmap of NumPy Array")
            plt.savefig("3_GT_OBS_input.png", dpi=300, bbox_inches='tight'); plt.close()
            plt.imshow(self.world_pure_obsline, cmap='viridis', aspect='auto'); plt.colorbar(); plt.title("Heatmap of NumPy Array")
            plt.savefig("3_GT_OBS_line.png", dpi=300, bbox_inches='tight'); plt.close()
            plt.imshow(self.world_pure_obs, cmap='viridis', aspect='auto'); plt.colorbar(); plt.title("Heatmap of NumPy Array")
            plt.savefig("3_GT_OBS.png", dpi=300, bbox_inches='tight'); plt.close()
            plt.imshow(self.slamMap_wall, cmap='viridis', aspect='auto'); plt.colorbar(); plt.title("Heatmap of NumPy Array")
            plt.savefig("3_OBS_WALL.png", dpi=300, bbox_inches='tight'); plt.close()
            plt.imshow(self.slamMap_obs, cmap='viridis', aspect='auto'); plt.colorbar(); plt.title("Heatmap of NumPy Array")
            plt.savefig("3_OBS_OBS.png", dpi=300, bbox_inches='tight'); plt.close()
        return True
    
    def check_wall(self, wall_clear):
        if (np.sum(wall_clear[0:50,:]==0) + np.sum(wall_clear[-50:,:]==0)): return True
        else: return False

    def complete_wall(self, img):
        output_img = img.copy(); foreground_mask = (img > 0)
        output_img_ndilate = img.copy()
        background_mask = (img <= 0); kernel = np.ones((3, 3), np.uint8)
        dilated_foreground = cv2.dilate(foreground_mask.astype(np.uint8), kernel)  
        output_img[(dilated_foreground > 0.5) & background_mask] = 200
        output_img_ndilate[(foreground_mask > 0.5) & background_mask] = 200
        return output_img, output_img_ndilate

    def _build_map_with_rangeCoordMat(self, y_rangeCoordMat, x_rangeCoordMat, world_map, slamMap, dslamMap):
        # Round y and x coord into int
        y_rangeCoordMat = (np.round(y_rangeCoordMat)).astype(np.int64)
        x_rangeCoordMat = (np.round(x_rangeCoordMat)).astype(np.int64) # (Angle， Ranege)
        """ Check for index of y_mat and x_mat that are within the world """
        inBound_ind = util.within_bound(np.array([y_rangeCoordMat, x_rangeCoordMat]), world_map.shape)
        """ delete coordinate that are not within the bound """
        outside_ind = np.argmax(~inBound_ind, axis=1) # Finding the nearest point
        ok_ind = np.where(outside_ind == 0)[0] # The first point is problematic
        need_amend_ind = np.where(outside_ind != 0)[0] # The first point is not problematic and start detect
        outside_ind = np.delete(outside_ind, ok_ind) # Delete useless point
        inside_ind = np.copy(outside_ind); inside_ind[inside_ind != 0] -= 1
        bound_ele_x = x_rangeCoordMat[need_amend_ind, inside_ind]
        bound_ele_y = y_rangeCoordMat[need_amend_ind, inside_ind]
        for count, i in enumerate(need_amend_ind):
            x_rangeCoordMat[i, ~inBound_ind[i,:]] = bound_ele_x[count]
            y_rangeCoordMat[i, ~inBound_ind[i,:]] = bound_ele_y[count]
        """ find obstacle along the laser range """
        obstacle_ind = np.argmax(world_map[y_rangeCoordMat, x_rangeCoordMat] == self.map_color['obstacle'], axis=1)
        obstacle_ind[obstacle_ind == 0] = x_rangeCoordMat.shape[1]
        """ generate a matrix of [[1,2,3,...],[1,2,3...],[1,2,3,...],...] for comparing with the obstacle coord """
        bx = np.arange(x_rangeCoordMat.shape[1]).reshape(1, x_rangeCoordMat.shape[1])
        by = np.ones((x_rangeCoordMat.shape[0], 1)); b = np.matmul(by, bx)
        """ get the coord that the robot can percieve (ignore pixel beyond obstacle) """
        b = b <= obstacle_ind.reshape(obstacle_ind.shape[0], 1)
        y_coord = y_rangeCoordMat[b]; x_coord = x_rangeCoordMat[b]
        """ laser noise + slam matching error """
        slamMap[y_coord,x_coord] = world_map[y_coord,x_coord]
        """ dilate/close to fill the holes """
        dslamMap = cv2.dilate(slamMap, np.ones((3,3)), iterations=1)
        return slamMap, dslamMap

    def build_map(self, pose, world):
        """ build perceived map based on robot position and its simulated laser info
        pose: [y;x;theta] in pixel in img coord | robotPose= pose"""
        """ input pose can be in decimal place, it will be rounded off in _build_map_with_rangeCoordMat """
        self.robotPose = pose
        """ find the coord matrix that the laser cover """
        angles = pose[2] + self.angles_vect
        y_rangeCoordMat = pose[0] - np.matmul(np.sin(angles), self.radius_vect)
        x_rangeCoordMat = pose[1] + np.matmul(np.cos(angles), self.radius_vect)
        slam = np.ones(world.shape)*self.map_color["uncertain"]
        dslam= np.ones(world.shape)*self.map_color["uncertain"]
        slamMap, dslamMap = self._build_map_with_rangeCoordMat(y_rangeCoordMat, x_rangeCoordMat, world, slam, dslam)
        return slamMap, dslamMap

    def world2state(self):
        # state= cv2.resize(self.slamMap, self.state_size, interpolation=cv2.INTER_LINEAR)
        state= self.slamMap.copy()
        # draw robot position on state
        cv2.circle(state, (int(self.robotPose[1]), int(self.robotPose[0])), self.robotRadius, 50, thickness=-1)
        # draw robot orientation heading on state
        headRadius = np.ceil(self.robotRadius/3.).astype(np.int64)
        headLen = self.robotRadius + headRadius
        # orientPt = util.transform_coord(self.robotPose[0], self.robotPose[1], self.robotPose, np.array([0, headLen, 0]))
        # cv2.circle(state, (orientPt[1],orientPt[2]), headRadius, 50, thickness=-1)
        head_y = self.robotPose[0] - np.sin(self.robotPose[2]) * headLen
        head_x = self.robotPose[1] + np.cos(self.robotPose[2]) * headLen
        cv2.circle(state, (int(head_x), int(head_y)), headRadius, 50, thickness=-1)
        return state

    def robotCrashed(self, pose):
        if ~util.within_bound(pose, self.world.shape, self.robotRadius):
            return True
        py= np.round(pose[0]).astype(int)
        px= np.round(pose[1]).astype(int)
        r= self.robotRadius
        # make a circle patch around robot location and check if there is obstacle pixel inside the circle
        robotPatch, _ = util.make_circle(r,1)
        worldPatch= self.world[py-r:py+r+1, px-r:px+r+1]
        worldPatch= worldPatch*robotPatch
        return np.sum(worldPatch==self.map_color["obstacle"])!=0

    def get_state(self):
        return self.world2state().copy()

def weighted_choice():
    return 1 if random.random() < 0.8 else 2

def edge_detection(map_data):
    binary_map = np.zeros_like(map_data)
    binary_map[map_data == 100] = 255  # Wall (outside) as 255, free space as 0
    edges = cv2.Canny(binary_map, threshold1=50, threshold2=150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=5)
    return lines, edges

def point_position_on_line(line_points, point):
    #print(line_points.shape)
    p1, p2 = line_points[0], line_points[1]
    v_line = p2 - p1
    v_point = point - p1
    cross_product = v_line[0] * v_point[1] - v_line[1] * v_point[0]
    if cross_product > 0: return "left"
    elif cross_product < 0: return "right"
    else: return "on"
        
def left_of_line(line_points, grid_points, check):
    # Line vector
    p1, p2 = line_points[0], line_points[1]
    v_line = p2 - p1  # (2,)
    # Grid vector: subtract p1 from each grid point
    v_points = grid_points - p1  # (H, W, 2)
    # Compute cross product for each grid point
    cross_product = (v_line[0] * v_points[..., 1] - v_line[1] * v_points[..., 0])
    # Boolean matrix: True if cross_product > 0 (left), else False
    if check == "left": mask = cross_product > 0
    if check == "right": mask = cross_product < 0
    return mask

def random_remove_segments(num_segments, full_list, min_len):
    for _ in range(num_segments):
        total_range = len(full_list)
        start = random.randint(0, total_range - 1 - min_len)
        end = random.randint(start + min_len, total_range)  # end > start
        full_list = [x for x in full_list if x < start or x > end]
    return full_list

def transfer(robotPose, matrix, borderValue=1):
    X, Y = 600-robotPose[1], 600 - robotPose[0]; CANVAS = 1200
    translation_matrix = np.array([[1, 0, X], [0, 1, Y]], dtype=np.float32)
    translated_image = cv2.warpAffine(matrix, translation_matrix, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue)
    theta_rad = - robotPose[2]
    rotation_center = (600, 600) 
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center, np.degrees(theta_rad), 1)
    rotation_image = cv2.warpAffine(translated_image, rotation_matrix, (CANVAS, CANVAS), borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue)
    CANVAS = 600
    translation_matrix = np.array([[1, 0, -600], [0, 1, -300]], dtype=np.float32)
    transformed_image = cv2.warpAffine(rotation_image, translation_matrix, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue)
    return transformed_image

from skimage.measure import label, regionprops
from skimage.draw import line

def straighten_curves(map):
    # For making sure that obs is a line.
    # create a new map to save it
    straightened_map = np.zeros_like(map)
    # mark all link area
    labeled_map = label(map, connectivity=2)
    # iterate all areas
    for region in regionprops(labeled_map):
        # get the coordinates of all link area
        coords = region.coords
        # link the head and end of each area
        min_row, min_col = coords[np.argmin(coords[:, 1])]
        max_row, max_col = coords[np.argmax(coords[:, 1])]
        # using line to link them
        rr, cc = line(min_row, min_col, max_row, max_col)
        straightened_map[rr, cc] = 1  # drawing line
    return straightened_map

def validate_rect_in_zero_area(rect, binary_map):
    for point in rect:
        x, y = point
        if binary_map[y, x] != 0:
            print(f"Point {point} is in the 1 area.")
            return False
    return True