from dataset import sevenScenesDataset
import open3d as o3d
import skimage
import numpy as np
import imageio
import os
import pandas as pd
import matplotlib.pyplot as plt
cmap = plt.get_cmap("turbo")
import yaml
from configs.config_utils import parse_dataset_config, import_class_from_module

from utils.geometry_utils import get_surface_normal_np



if __name__ == "__main__":
    config_path = 'configs/spann3r_7scenes.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    dataset_cls = import_class_from_module("dataset", config["dataset"])
    dataset = dataset_cls(**parse_dataset_config(config))
    save_dir = 'debug_output'

    print('len(dataset):', len(dataset))
    os.makedirs(save_dir, exist_ok=True)

    ######################### vis data to check

    df_save_path = os.path.join(save_dir, 'data.csv')
    dataframe = pd.DataFrame(columns=['idx', 'seq_name', 'frame_name', 'caption'])

    sample_list = []
    for idx in list(range(len(dataset)))[::5]:  # sample every 5 frames
        data = dataset[idx]
        scene_name = data['scene_name']
        images_list, normals_list = [], []
        sample_indices = list(range(0,len(data['images']),4))
        for vid in sample_indices:
            image = data['images'][vid].transpose(1,2,0)
            images_list.append(image.astype(np.uint8))

            coord = data['world_coord'][vid].transpose(1,2,0)  # [H, W, 3]
            normal = get_surface_normal_np(coord)  # [H, W, 3]
            normal = (((normal + 1) / 2) * 255).astype(np.uint8)
            normals_list.append(normal)

        sample = np.concatenate(images_list + normals_list, axis=1)
        sample_list.append(sample)

        scene_name = data['scene_name']
        frame_name = data['image_names']
        caption = data.get('caption', None)
        dataframe.loc[idx] = [idx, scene_name, frame_name, caption]

        ##### save
        if idx > 0 and idx % 50 == 0:
            sample = np.concatenate(sample_list, axis=0)
            save_path = f'{save_dir}/sample_{idx:05d}_{scene_name}.png'
            print(f"saving to {save_path}")
            imageio.imwrite(save_path, (sample).astype(np.uint8))

            dataframe.to_csv(df_save_path, index=False)
            sample_list = []

    ### save the last part
    if len(sample_list) > 0:
        sample = np.concatenate(sample_list, axis=0)
        save_path = f'{save_dir}/sample_{idx:05d}_{scene_name}.webp'
        print(f"saving to {save_path}")
        imageio.imwrite(save_path, (sample).astype(np.uint8))

        dataframe.to_csv(df_save_path, index=False)
