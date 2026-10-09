import os 
import re
import cv2
import pdb
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def concatenate_image_normal(combine_address, address):
    img_list = []
    width = 300
    height = 300
    for combine in combine_address:
        img_list.append(cv2.resize(cv2.imread(combine), (width, height)))
    fin_image = np.concatenate(img_list, axis=1)
    cv2.imwrite(address, fin_image)

def finding_longest(coordinates, threshold=9):
    distances = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    longest, current = [], []
    for i, distance in enumerate(distances):
        if distance < threshold: current.append(i + 1)
        else:
            if len(current) > len(longest): longest = current
            current = []
    if len(current) > len(longest): longest = current
    return longest

def low_resolution(image_clear, down_size=256, padding=50):
    cropped_image = image_clear[padding:-padding, :-padding*2]
    intermediate_size = (cropped_image.shape[1] // 2, cropped_image.shape[0] // 2)
    resized_image = cv2.resize(cropped_image, intermediate_size, interpolation=cv2.INTER_AREA)
    resized_image = cv2.resize(resized_image, (down_size, down_size), interpolation=cv2.INTER_AREA)
    return resized_image

def random_rotation_further(_map_, center=[64,0], theta_min=5, theta_max=30, dis_min=0.5, dis_max=2, min_angle=10, max_angle=30):
    map = _map_.copy()
    H, W = map.shape
    center = np.array(center)
    coordinates = np.argwhere(map > 0)
    # Step 1: Compute distances
    distances = np.linalg.norm(coordinates - center, axis=1)
    # Step 2: Compute angles (theta)
    reference_line = np.array([1, 0])
    vectors = coordinates - center
    unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_reference = reference_line / np.linalg.norm(reference_line)
    dot_products = np.dot(unit_vectors, unit_reference)
    angles = np.degrees(np.arccos(np.clip(dot_products, -1.0, 1.0)))
    # Step 3: Randomly select a point
    random_index = np.random.choice(len(coordinates))
    selected_theta = angles[random_index]
    # Step 4: Find all points with similar theta (theta ± 5 degrees)
    theta_range = np.random.uniform(theta_min, theta_max)
    similar_theta_indices = np.where((angles >= selected_theta - theta_range) &
                                    (angles <= selected_theta + theta_range))[0]
    # Gather results
    similar_coordinates = coordinates[similar_theta_indices]
    similar_thetas = angles[similar_theta_indices]
    similar_distances = distances[similar_theta_indices]
    if len(similar_thetas) < 10: return map
    sorted_indices = np.argsort(similar_thetas)
    sorted_coors = similar_coordinates[sorted_indices]
    sorted_theta = similar_thetas[sorted_indices]
    sorted_distance = similar_distances[sorted_indices]
    sorted_indices = finding_longest(sorted_coors)
    #sorted_indices = np.arange(0, len(sorted_coors))
    sorted_coors = sorted_coors[sorted_indices]
    sorted_angles = sorted_theta[sorted_indices]
    sorted_distances = sorted_distance[sorted_indices]
    #print('========', sorted_angles, '+++++++', sorted_distances, '+++++++++++++')
    theta_step = 0.1
    dense_theta = np.arange(sorted_angles.min(), sorted_angles.max() + theta_step, step=theta_step)  # Fill gaps in theta
    dense_distance = np.interp(dense_theta, sorted_angles, sorted_distances)  # Interpolate distances
    '''------------------ Further Part ------------------'''
    distance_time = np.random.uniform(dis_min, dis_max)
    dense_coords_x = distance_time*dense_distance*np.cos(np.radians(dense_theta)) + center[0]
    dense_coords_y = distance_time*dense_distance*np.sin(np.radians(dense_theta)) + center[1]
    new_coors = np.stack((dense_coords_x, dense_coords_y), axis=1) 
    '''------------------ Rotation Part ------------------'''
    random_angle = np.random.uniform(min_angle, max_angle)
    new_coors = rotate_points(new_coors, random_angle)
    '''------------------ Making sure all line ------------------'''
    start, end = new_coors[0], new_coors[-1]; num_points = len(new_coors)
    x1, y1 = start; x2, y2 = end
    x = np.linspace(x1, x2, num_points, dtype=int); y = np.linspace(y1, y2, num_points, dtype=int)
    new_coors = np.stack((x, y), axis=-1)
    # eliminate out range data
    valid_mask = (new_coors[:, 0] >= 0) & (new_coors[:, 0]+0.5 < H) & (new_coors[:, 1] >= 0) & (new_coors[:, 1]+0.5 < H)
    if np.sum(valid_mask) > 10:
        filtered_coors = new_coors[valid_mask]
        map[tuple(sorted_coors.astype(int).T)] = 0
        map[tuple(np.round(filtered_coors).astype(int).T)] = 1
        return map
    else: return map

def rotate_points(coords, angle_degrees):
    center = np.mean(coords, axis=0)
    translated_coords = coords - center
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians),  np.cos(angle_radians)]])
    rotated_coords = np.dot(translated_coords, rotation_matrix.T)
    result_coords = rotated_coords + center
    return result_coords

def random_missing(_map_, center=[64,0], theta_min=3, theta_max=20, num_max=3):
    map = _map_.copy()
    center = np.array(center)
    coordinates = np.argwhere(map > 0)
    # Step 2: Compute angles (theta)
    reference_line = np.array([1, 0])
    vectors = coordinates - center
    unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_reference = reference_line / np.linalg.norm(reference_line)
    dot_products = np.dot(unit_vectors, unit_reference)
    angles = np.degrees(np.arccos(np.clip(dot_products, -1.0, 1.0)))
    # Step 3: Randomly select a point
    num_missing = np.random.randint(1, num_max)
    select_indices = []
    for i in range(num_missing):
        random_index = np.random.choice(len(coordinates))
        selected_theta = angles[random_index]
        # Step 4: Find all points with similar theta (theta ± 5 degrees)
        theta_range = np.random.uniform(theta_min, theta_max)
        similar_theta_indices = np.where((angles >= selected_theta - theta_range) &
                                    (angles <= selected_theta + theta_range))[0]
        select_indices.append(similar_theta_indices)
    select_indices = np.concatenate(select_indices)
    similar_coors = coordinates[select_indices]
    map[tuple(similar_coors.astype(int).T)] = 0
    return map

def checking(image_aug):
    FLAG = False
    if not np.sum(image_aug[:, 50:]): FLAG = True
    # Make sure there is something far away from radar. (Horizontal)
    if not np.sum(image_aug[:250,]) or not np.sum(image_aug[350:,]): FLAG = True
    # Make sure the distance is enough far aways. (Vertical)
    if np.sum(image_aug) < 20: FLAG = True
    # Make sure that we observe enough information
    if np.sum(image_aug[:25,275:325]): FLAG = True
    # Deleting all information within a cube observation range of a radar.
    return FLAG

def remove_obs(image_a, image_b): # _a: gt; _b: obs.
    # Ensure both images are binary and uint8 (0, 255)
    image_a = np.uint8(image_a * 255)
    image_b = np.uint8(image_b * 255)
    # Step 1: Find connected components in image_a
    num_labels, labels = cv2.connectedComponents(image_a)
    # Step 2: Create an empty image to store the result
    result_image = np.zeros_like(image_a)
    # Step 3: Iterate over each connected component and check if it exists in image_b
    for label in range(1, num_labels):  # Start from 1 to ignore the background label (0)
        # Find the pixels belonging to the current component
        component_mask = (labels == label).astype(np.uint8)
        # Check if any of the pixels in the component are 1 in image_b
        if np.any(component_mask * image_b):  # If there's any overlap with foreground in B
            result_image = np.maximum(result_image, component_mask)  # Keep the component in the result
    return result_image

def remove_noise(image, min_size=50):
    # 去除掉多余单个墙壁点
    # Ensure the image is of type uint8 (CV_8U) with values 0 or 255
    image = np.uint8(image * 255)  # Convert binary 0/1 to 0/255
    # Step 1: Find connected components (0 = background, 255 = foreground)
    num_labels, labels = cv2.connectedComponents(image)
    # Step 2: Create an empty image to store the result
    cleaned_image = np.zeros_like(image)
    # Step 3: Iterate over each connected component and keep only the large enough ones
    for label in range(1, num_labels):  # Start from 1 to ignore the background label (0)
        # Find the pixels belonging to the current component
        component_mask = (labels == label).astype(np.uint8)
        # If the component size is above the threshold, keep it in the result
        if np.sum(component_mask) >= min_size:
            cleaned_image = np.maximum(cleaned_image, component_mask)  # Combine the valid components
    return cleaned_image

if __name__ == "__main__":
    # Configurable so run_augment.sh works unedited:
    #   RISE_SIM_OUT   simulated scenes produced by simulator.envs.main
    #   RISE_AUG_OUT   where the augmented training bundles are written
    #   RISE_AUG_IMG   where the debug renders are written
    #   RISE_AUG_N     how many augmented samples to produce
    #   RISE_AUG_SPLIT "train" (default) or "test"
    shuffle = True
    split = os.environ.get("RISE_AUG_SPLIT", "train")
    training = split != "test"
    pre_address = os.environ.get("RISE_SIM_OUT", "out/scenes")
    default_out = "out/augmented" if training else "out/augmented_test"
    save_np = os.environ.get("RISE_AUG_OUT", default_out)
    save_img = os.environ.get("RISE_AUG_IMG", save_np + "_img")
    if not os.path.exists(save_np): os.makedirs(save_np)
    if not os.path.exists(save_img): os.makedirs(save_img)
    save_freq = 500; efficient_number = 0
    total_numer = int(os.environ.get("RISE_AUG_N", 100000 if training else 5000))

    files = os.listdir(pre_address); phase1 = len(files)
    pattern = r'dict_(.*)_\.npy'; filtered_files = []
    for file in files:
        match = re.match(pattern, file)
        if match: filtered_files.append((file, match.group(1)))  
        # Save the filename and the part between 'dict_' and '_.npy'
    sorted_files = sorted(filtered_files, key=lambda x: int(x[1]))
    sorted_file_names = [file[0] for file in sorted_files]

    theta_min=0; theta_max=25; num_max=6 # This one for random missing.
    min_radius=5; max_radius=25; min_angle=10; max_angle=20 # 
    center=[300,0]; fur_theta_min=0; fur_theta_max=20; dis_min=0.9; dis_max=1.1

    while efficient_number < total_numer:
        if shuffle: sort_file = random.choice(sorted_file_names)
        else: sort_file = sorted_file_names
        address = os.path.join(pre_address, sort_file)
        dictionary = np.load(address, allow_pickle=True)
        dictionary = dictionary.item()
        image_gt_obs = np.unpackbits(dictionary["image_gt_obs"]).reshape(600,600).astype(bool) # Thickness Inif
        image_gt_wall = np.unpackbits(dictionary['image_gt_wall']).reshape(600,600).astype(bool) # Thickness 3
        image_ob_obs = np.unpackbits(dictionary['image_ob_obs']).reshape(600,600).astype(bool) # Thickness 1
        image_ob_wall = np.unpackbits(dictionary['image_ob_wall']).reshape(600,600).astype(bool) # Thickness 1
        # Make sure observed wall could found its corresponding walls in ground truth wall.
        image_ob_wall = image_ob_wall * (cv2.dilate(image_gt_wall.astype(float), np.ones((3,3)), iterations=1) > 0)

        image_gt_obs = (image_gt_obs.astype(float)>0).astype(float)
        image_gt_wall = (image_gt_wall.astype(float)>0).astype(float)
        image_ob_obs = (image_ob_obs.astype(float)>0).astype(float)
        image_ob_wall = (image_ob_wall.astype(float)>0).astype(float)
        # Make sure there is no obstacle unobserved. (Imagine an obstacle behind another one.)
        image_gt_obs = remove_obs(image_gt_obs, image_ob_obs)
        # Three types of data augmentation
        # 1. Random missing 2. Random rotation 3. Random further.
        FLAG_MIS = random.choice([1]) 
        FLAG_ROT = random.choice([0, 1])
        FLAG_FUR = random.choice([0, 1])
        if FLAG_MIS: image_aug1 = random_missing(image_ob_wall, center, theta_min, theta_max, num_max)
        else: image_aug1 = image_ob_wall
        if checking(image_aug1): continue
        # --------- Combining the further and the rotation mechanism.
        if FLAG_ROT: 
            FLAG_ROT = np.random.randint(1, 5)
            for i in range(FLAG_ROT):
                if i == 0:
                    try:
                        image_aug2 = random_rotation_further(image_aug1, center, theta_min=fur_theta_min, theta_max=fur_theta_max, \
                            dis_min=dis_min, dis_max=dis_max, min_angle=min_angle, max_angle=max_angle)
                    except: print('---problem--1-'); continue
                else: 
                    try:
                        image_aug2 = random_rotation_further(image_aug2, center, theta_min=fur_theta_min, theta_max=fur_theta_max, \
                            dis_min=dis_min, dis_max=dis_max, min_angle=min_angle, max_angle=max_angle)
                    except: print('---problem--2-'); continue
        else: image_aug2 = image_aug1
        if checking(image_aug2): continue
        image_aug1 = (image_aug1>0).astype(float)
        image_aug2 = (image_aug2>0).astype(float)
        image_aug2_line = remove_noise(image_aug2, min_size=8)
        # ---------- Finishing data augmentation ----------
        kernel = np.ones((2, 2), np.uint8)
        image_ob_obs_ = (cv2.dilate(image_ob_obs, kernel, iterations=1)>0).astype(float)
        image_ob_wall_ = (cv2.dilate(image_ob_wall, kernel, iterations=1)>0).astype(float)
        image_gt_obs_ = (cv2.dilate(image_gt_obs, kernel, iterations=1)>0).astype(float)
        image_gt_wall_ = (cv2.dilate(image_gt_wall, kernel, iterations=1)>0).astype(float)
        image_aug1_ = (cv2.dilate(image_aug1, kernel, iterations=1)>0).astype(float)
        image_aug2_ = (cv2.dilate(image_aug2, kernel, iterations=1)>0).astype(float)
        image_aug2_line_ = (cv2.dilate(image_aug2_line, kernel, iterations=1)>0).astype(float)

        image_ob_obs_ = (low_resolution(image_ob_obs_, down_size=128)>0).astype(float)
        image_ob_wall_ = (low_resolution(image_ob_wall_, down_size=128)>0).astype(float)
        image_gt_obs_ = (low_resolution(image_gt_obs_, down_size=128)>0).astype(float)
        image_gt_wall_ = (low_resolution(image_gt_wall_, down_size=128)>0).astype(float)
        image_aug1_ = (low_resolution(image_aug1_, down_size=128)>0).astype(float)
        image_aug2_ = (low_resolution(image_aug2_, down_size=128)>0).astype(float)
        image_aug2_line_ = (low_resolution(image_aug2_line_, down_size=128)>0).astype(float)

        new_dictionary = {'image_ob_obs': np.packbits(image_ob_obs_.astype(bool)), \
                            'image_ob_wall': np.packbits(image_ob_wall_.astype(bool)), \
                            'image_gt_obs': np.packbits(image_gt_obs_.astype(bool)), \
                            'image_gt_wall': np.packbits(image_gt_wall_.astype(bool)), \
                            'image_aug2_line_': np.packbits(image_aug2_line_.astype(bool))}
        np.save(os.path.join(save_np, f"dict_{efficient_number}_.npy"), new_dictionary)

        observation1 = True
        observation2 = True
        if observation1 and efficient_number%save_freq==0:
            print('==============Non Dilation One=============', efficient_number)
            add1 = os.path.join(save_img, f"ob_obs_{efficient_number}_.png")
            add2 = os.path.join(save_img, f"ob_wall_{efficient_number}_.png")
            add3 = os.path.join(save_img, f"gt_obs_{efficient_number}_.png")
            add4 = os.path.join(save_img, f"gt_wall_{efficient_number}_.png")
            add5 = os.path.join(save_img, f"aug1_{efficient_number}_.png")
            add6 = os.path.join(save_img, f"aug2_{efficient_number}_.png")
            add7 = os.path.join(save_img, f"aug2_line_{efficient_number}_.png")
            plt.imshow(image_ob_obs, cmap='viridis', aspect='auto'); plt.title("ob_obs")
            plt.savefig(add1, dpi=150, bbox_inches='tight'); plt.close()
            plt.imshow(image_ob_wall, cmap='viridis', aspect='auto'); plt.title("ob_wall")
            plt.savefig(add2, dpi=150, bbox_inches='tight'); plt.close()
            plt.imshow(image_gt_obs, cmap='viridis', aspect='auto'); plt.title("gt_obs")
            plt.savefig(add3, dpi=150, bbox_inches='tight'); plt.close()
            plt.imshow(image_gt_wall, cmap='viridis', aspect='auto'); plt.title("gt_wall")
            plt.savefig(add4, dpi=150, bbox_inches='tight'); plt.close()
            plt.imshow(image_aug1, cmap='viridis', aspect='auto'); plt.title("aug1")
            plt.savefig(add5, dpi=150, bbox_inches='tight'); plt.close()
            plt.imshow(image_aug2, cmap='viridis', aspect='auto'); plt.title(f"aug2_{FLAG_ROT}")
            plt.savefig(add6, dpi=150, bbox_inches='tight'); plt.close()
            plt.imshow(image_aug2_line, cmap='viridis', aspect='auto'); plt.title(f"aug3_{FLAG_ROT}")
            plt.savefig(add7, dpi=150, bbox_inches='tight'); plt.close()
            address = os.path.join(save_img, f"combine_{efficient_number}_nodilate.png")
            concatenate_image_normal([add1,add2,add3,add4,add5,add6,add7],address)
            os.remove(add1), os.remove(add2), os.remove(add3)
            os.remove(add4), os.remove(add5), os.remove(add6), os.remove(add7)  

        if observation2 and efficient_number%save_freq==0:
            print('=============Dilation One==============', efficient_number)
            add1 = os.path.join(save_img, f"ob_obs_{efficient_number}_.png")
            add2 = os.path.join(save_img, f"ob_wall_{efficient_number}_.png")
            add3 = os.path.join(save_img, f"gt_obs_{efficient_number}_.png")
            add4 = os.path.join(save_img, f"gt_wall_{efficient_number}_.png")
            add5 = os.path.join(save_img, f"aug1_{efficient_number}_.png")
            add6 = os.path.join(save_img, f"aug2_{efficient_number}_.png")
            add7 = os.path.join(save_img, f"aug2_line_{efficient_number}_.png")
            plt.imshow(image_ob_obs_, cmap='viridis', aspect='auto'); plt.title("Missing_Augmentation")
            plt.savefig(add1, dpi=150, bbox_inches='tight'); plt.close()
            plt.imshow(image_ob_wall_, cmap='viridis', aspect='auto'); plt.title("Missing_Augmentation")
            plt.savefig(add2, dpi=150, bbox_inches='tight'); plt.close()
            plt.imshow(image_gt_obs_, cmap='viridis', aspect='auto'); plt.title("Rotation_Augmentation")
            plt.savefig(add3, dpi=150, bbox_inches='tight'); plt.close()
            plt.imshow(image_gt_wall_, cmap='viridis', aspect='auto'); plt.title("Further_Augmentation")
            plt.savefig(add4, dpi=150, bbox_inches='tight'); plt.close()
            plt.imshow(image_aug1_, cmap='viridis', aspect='auto'); plt.title("Further_Augmentation")
            plt.savefig(add5, dpi=150, bbox_inches='tight'); plt.close()
            plt.imshow(image_aug2_, cmap='viridis', aspect='auto'); plt.title("Further_Augmentation")
            plt.savefig(add6, dpi=150, bbox_inches='tight'); plt.close() 
            plt.imshow(image_aug2_line_, cmap='viridis', aspect='auto'); plt.title("Further_Augmentation")
            plt.savefig(add7, dpi=150, bbox_inches='tight'); plt.close() 
            address = os.path.join(save_img, f"combine_{efficient_number}_.png")
            concatenate_image_normal([add1,add2,add3,add4,add5,add6,add7],address)
            os.remove(add1), os.remove(add2), os.remove(add3)
            os.remove(add4), os.remove(add5), os.remove(add6), os.remove(add7)  
        efficient_number += 1
    