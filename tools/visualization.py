import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from matplotlib import cm

if __name__ == "__main__":
    """
    image_path_pre = "/mnt/c/kaichen/ghost/ti/mmwave_studio_03_00_00_14/mmWaveStudio/PostProc/visualization/"
    image_name = "pic1.png"
    image_path = os.path.join(image_path_pre, image_name)
    img = Image.open(image_path)
    img_array = np.array(img)
    if img_array.shape[-1] == 3: 
        img_array = np.mean(img_array, axis=-1)  
    elif img_array.shape[-1] == 4: 
        img_array = np.mean(img_array[..., :3], axis=-1) 
    normalized_img = img_array / np.max(img_array)
    print(np.max(normalized_img), np.min(normalized_img), np.mean(normalized_img))
    jet_colormap = plt.get_cmap('jet')
    colored_img = jet_colormap(normalized_img)
    colored_img_rgb = (colored_img * 255).astype(np.uint8)
    output_image_path = os.path.join(image_path_pre, image_name.split('.')[0]+'save'+'.png')
    Image.fromarray(colored_img_rgb).save(output_image_path)
    print(f"Image saved as {output_image_path}.")
    """

    image_path_pre = "/mnt/c/kaichen/ghost/ti/mmwave_studio_03_00_00_14/mmWaveStudio/PostProc/visualization/"
    image_name = "pic3.png"
    image_path = os.path.join(image_path_pre, image_name)
    output_image_path = os.path.join(image_path_pre, image_name.split('.')[0]+'save'+'.png')
    heatmap_image = Image.open(image_path)
    img_array = np.array(heatmap_image)
    rgb_values = img_array[..., :3] / 255.0  # Normalize to [0, 1] range
    viridis = cm.get_cmap('viridis')
    norm = plt.Normalize(vmin=0, vmax=1)  # Assuming the original values were normalized to [0, 1]
    value_map = np.zeros((img_array.shape[0], img_array.shape[1]))
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            rgb = rgb_values[i, j]
            color_diff = np.linalg.norm(viridis(norm(np.linspace(0, 1, 256)))[:, :3] - rgb, axis=1)
            value_map[i, j] = np.argmin(color_diff) / 255.0  # Get the corresponding scalar value
    print(np.max(value_map), np.min(value_map), np.mean(value_map))
    jet_colormap = plt.get_cmap('jet')
    colored_img = jet_colormap(value_map)

    colored_img_pil = Image.fromarray((colored_img * 255).astype(np.uint8))
    new_size = (colored_img.shape[1] * 4, colored_img.shape[0] * 4)  # (W * 4, H * 4)
    resized_img_pil = colored_img_pil.resize(new_size, Image.ANTIALIAS)

    resized_img_pil.save(output_image_path)