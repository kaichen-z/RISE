import os                            
import cv2                           
import time                            
import numpy as np                   
from os import path                    
from tqdm import tqdm                
from matplotlib import pyplot as plt  
from simulator.envs.simulator.pseudoSlam import pseudoSlam
import pdb
                     
# python -m simulator.envs.main

def main(gen):                        
    config_file= path.join(path.dirname(__file__), "config", "config_exploration.yaml")
    T = time.time()                  
    sim= pseudoSlam(config_file)     
    T2 = time.time(); print('======= TIME Final=======', T2-T)
    dictionary = {}  
    dictionary["map_id"] = sim.map_id 
    dictionary["image_gt_obs"] = np.packbits(sim.world_pure_obs.astype(bool))
    dictionary["image_gt_wall"] = np.packbits(sim.transformed_new_wall_clear_nd.astype(bool))
    dictionary["image_gt_wall_ori"] = np.packbits(sim.transformed_new_wall_clear.astype(bool))
    dictionary["image_ob_obs"] = np.packbits(sim.slamMap_obs.astype(bool))
    dictionary["image_ob_wall"] = np.packbits(sim.slamMap_wall.astype(bool))
    dictionary["image_ob_whole"] = np.packbits(sim.slamMap_whole.astype(bool))
    visual_obs = False
    if visual_obs:
        slamMap= sim.get_state()
        slamMap_np = np.array(slamMap)
        plt.imshow(slamMap_np, cmap='viridis', aspect='auto')
        plt.colorbar(); plt.title("Heatmap of NumPy Array")
        plt.savefig("heatmap.png", dpi=300, bbox_inches='tight'); plt.close()
    return dictionary, sim.Flag

def concatenate_image_normal(combine_address, address):
    img_list = []
    width = 1200
    height = 1200
    for combine in combine_address:
        img_list.append(cv2.resize(cv2.imread(combine), (width, height)))
    fin_image = np.concatenate(img_list, axis=1)
    cv2.imwrite(address, fin_image)

if __name__=='__main__':
    # Output folders and scene count are configurable so run_sim.sh works unedited.
    #   RISE_SIM_OUT     where the packed .npy scenes are written
    #   RISE_SIM_IMG     where the periodic debug mosaics are written
    #   RISE_SIM_N       how many scenes to simulate
    npy_address = os.environ.get("RISE_SIM_OUT", "out/scenes")
    image_address = os.environ.get("RISE_SIM_IMG", "out/scenes_img")
    n_scenes = int(os.environ.get("RISE_SIM_N", 100000))
    if not os.path.exists(image_address): os.makedirs(image_address)
    if not os.path.exists(npy_address): os.makedirs(npy_address)
    generation = [i for i in range(0, n_scenes)] # 56037
    for gen in tqdm(generation):
        dict_main, Flag = main(gen)
        if Flag:
            continue
        np.save(f"{npy_address}/dict_{gen}_.npy", dict_main)
        observation = True; frequency = int(os.environ.get("RISE_SIM_IMG_FREQ", 200))
        if observation and gen%frequency==0:
            address1 = f"{image_address}/Observation_{gen}.png"
            address2 = f"{image_address}/Observation_GT_{gen}.png"
            address3 = f"{image_address}/Transparent_GT_{gen}.png"
            address4 = f"{image_address}/Transparent_GT_{gen}4.png"
            address5 = f"{image_address}/Observation_Whole_{gen}4.png"
            address6 = f"{image_address}/Transparent_Whole_{gen}4.png"
            address7 = f"{image_address}/Transparent_Whole_{gen}5.png"
            plt.imshow(np.unpackbits(dict_main["image_ob_obs"]).reshape(600,600).astype(bool), cmap='viridis', aspect='auto')
            plt.colorbar(); plt.title("observation_obstacle")
            plt.savefig(address1, dpi=300, bbox_inches='tight'); plt.close()
            
            plt.imshow(np.unpackbits(dict_main["image_ob_wall"]).reshape(600,600).astype(bool), cmap='viridis', aspect='auto')
            plt.colorbar(); plt.title("observation_wall")
            plt.savefig(address2, dpi=300, bbox_inches='tight'); plt.close()
            
            plt.imshow(np.unpackbits(dict_main["image_gt_obs"]).reshape(600,600).astype(bool), cmap='viridis', aspect='auto')
            plt.colorbar(); plt.title("gt_obstacle")
            plt.savefig(address3, dpi=300, bbox_inches='tight'); plt.close()

            plt.imshow(np.unpackbits(dict_main["image_gt_wall"]).reshape(600,600).astype(bool), cmap='viridis', aspect='auto')
            plt.colorbar(); plt.title("gt_wall")
            plt.savefig(address4, dpi=300, bbox_inches='tight'); plt.close()
             
            plt.imshow(np.unpackbits(dict_main["image_ob_whole"]).reshape(600,600).astype(bool), cmap='viridis', aspect='auto')
            plt.colorbar(); plt.title("gt_wall")
            plt.savefig(address7, dpi=300, bbox_inches='tight'); plt.close()
             
            plt.imshow(np.unpackbits(dict_main["image_ob_wall"] + dict_main["image_ob_obs"]).reshape(600,600).astype(bool), cmap='viridis', aspect='auto')
            plt.colorbar(); plt.title("gt_wall")
            plt.savefig(address5, dpi=300, bbox_inches='tight'); plt.close()
             
            plt.imshow(np.unpackbits(dict_main["image_gt_wall"] + dict_main["image_gt_obs"]).reshape(600,600).astype(bool), cmap='viridis', aspect='auto')
            plt.colorbar(); plt.title("gt_wall")
            plt.savefig(address6, dpi=300, bbox_inches='tight'); plt.close()
             
            address = f"{image_address}/4_combine_{gen}.png"
            concatenate_image_normal([address1, address2, address3, address4, address7, address5, address6], address)
            os.remove(address1), os.remove(address2), os.remove(address3), os.remove(address4), os.remove(address5), os.remove(address6), os.remove(address7)

