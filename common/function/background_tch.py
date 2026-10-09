import os 
import cv2
import pdb
import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from common.utils.utils_tch import low_resolution

class bk_case_scene_bkg:
    def __init__(self, center, resolution, address):
        # radar facing -z，right +x，upper +y
        point_cloud = o3d.io.read_point_cloud(address)
        points = np.asarray(point_cloud.points)
        mask = (points[:,1]>-0.5)*(points[:,1]<1.)
        points_msk = points[mask]*resolution
        result = np.column_stack((-points_msk[:, 0], -points_msk[:, 2]))
        result = result+center[None,:].numpy()
        self.bg_points = result
        print(self.bg_points.shape, '===== Shape of point cluster')

    def draw_line(self,address = 'image/background.png'):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.bg_points[:, 1], self.bg_points[:, 0], label=f'Reflector GT', marker='.', alpha=0.9)
        xmin, xmax = plt.xlim(); ymin, ymax = plt.ylim()
        plt.xlim(0, 600); plt.ylim(0, 600); plt.axis('off')
        # Mark the x-axis start and end points
        plt.text(xmin, ymin, f'X Start: {xmin:.2f}', fontsize=10, color='blue', verticalalignment='top', horizontalalignment='right')
        plt.text(xmax, ymin, f'X End: {xmax:.2f}', fontsize=10, color='blue', verticalalignment='top', horizontalalignment='left')
        # Mark the y-axis start and end points
        plt.text(xmin, ymin, f'Y Start: {ymin:.2f}', fontsize=10, color='red', verticalalignment='bottom', horizontalalignment='left')
        plt.text(xmin, ymax, f'Y End: {ymax:.2f}', fontsize=10, color='red', verticalalignment='top', horizontalalignment='left')
        plt.legend(); plt.savefig(address, dpi=500, bbox_inches='tight'); plt.close()

class bk_case_scene_bkg_refine:
    def __init__(self, address):
        image = cv2.imread(address, cv2.IMREAD_GRAYSCALE)
        image = Image.open(address)
        image = image.resize((600, 600))
        image_array = np.array(image)
        image_wall = image_array < 200
        non_zero_coords = np.nonzero(image_wall)
        non_zero_coords = (600 - non_zero_coords[0], non_zero_coords[1]) 
        coordinates = np.array(list(zip(non_zero_coords[0], non_zero_coords[1])))
        self.bg_points = coordinates

