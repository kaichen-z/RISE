import numpy as np
import json, cv2, os
from matplotlib import pyplot as plt
import pdb

class jsonReader():
    def __init__(self, json_prefix, meter2pixel):
        self.json_prefix = os.path.join(os.path.dirname(__file__), "../", json_prefix)      # json data dir
        self.meter2pixel = meter2pixel      # meter to pixel. E.g. meter2pixel=100 represents 1 meter = 100 pixels
        self.border_pad = 20                 # border padding for each side
        self.json_data = {}                 # house info read from json file
        self.cnt_map = []                   # contour map, 0-obstacle, 255-free space
        self.tp_map = []                    # room type map
    def read_json(self, file_name):
        #file_name = "b813cfcaf67b61168957043336b9f459"
        with open(self.json_prefix + file_name.split('.')[0] + '.json') as json_file:
            self.json_data = json.load(json_file)
        # Draw the contour
        vertices = self.json_data['verts']
        verts = (np.array(vertices) * self.meter2pixel).astype(np.int32)
        x_max, x_min, y_max, y_min = np.max(verts[:,0]), np.min(verts[:,0]), np.max(verts[:, 1]), np.min(verts[:,1])
        self.cnt_map = np.zeros((y_max - y_min + self.border_pad * 2, x_max - x_min + self.border_pad * 2))
        verts[:, 0] = verts[:, 0] - x_min + self.border_pad
        verts[:, 1] = verts[:, 1] - y_min + self.border_pad
        cv2.drawContours(self.cnt_map, [verts], 0, 255, -1)
        observe = False
        if observe:
            plt.imshow(self.cnt_map, cmap='viridis', aspect='auto', vmin=0, vmax=255)
            plt.colorbar(); plt.title("Heatmap of NumPy Array")
            plt.savefig("0_check1_.png", dpi=300, bbox_inches='tight'); plt.close()
        cnt_map = np.zeros_like(self.cnt_map)
        cv2.fillPoly(cnt_map, [verts], 255)
        thickness = 10
        kernel = np.ones((thickness, thickness), np.uint8)
        cnt_map = cv2.dilate(cnt_map, kernel, iterations=2)
        if observe:
            plt.imshow(cnt_map, cmap='viridis', aspect='auto', vmin=0, vmax=255)
            plt.colorbar(); plt.title("Heatmap of NumPy Array")
            plt.savefig("0_check2_.png", dpi=300, bbox_inches='tight'); plt.close()
        self.cnt_map = self.cnt_map + cnt_map
        self.cnt_map[self.cnt_map > 0] = 255.
        H, W = self.cnt_map.shape 
        if np.sum(self.cnt_map==255.) == 0: self.cnt_map[:,int(W/10):-int(W/10)] = 255.
        if np.sum(self.cnt_map==0.) == 0: self.cnt_map[:int(H/10),:] = 0; self.cnt_map[-int(H/10):,:] = 0.;self.cnt_map[:,:int(W/10)] = 0.;self.cnt_map[:,-int(W/10):] = 0.
        if observe:
            plt.imshow(self.cnt_map, cmap='viridis', aspect='auto', vmin=0, vmax=255)
            plt.colorbar(); plt.title("Heatmap of NumPy Array"); 
            plt.savefig("0_check3_.png", dpi=300, bbox_inches='tight'); plt.close()
        return self.cnt_map.copy()


