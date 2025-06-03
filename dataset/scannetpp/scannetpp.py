import os
from glob import glob
import os.path as osp
import re
import pandas as pd
import h5py

from tqdm import tqdm
import numpy as np
from PIL import Image

from ..dataset_core.dataset import Dataset, Sample, _get_sample_list_path
from utils.geometry_utils import backproject_to_cv_position


class ScannetPPSequence:
    def __init__(self, root, scene_name, clip_length=30, clip_overlap=0):

        self.root = root
        self.scene_name = scene_name
        
        self.extrinsics, self.intrinsics, self.rgb_path_list, self.normal_path_list, self.depth_path_list = self.load_meta_data(self.root, self.scene_name)

        ### For iphone data, frames are too many
        gap = 3
        self.extrinsics = self.extrinsics[::gap]
        self.intrinsics = self.intrinsics[::gap]
        self.rgb_path_list = self.rgb_path_list[::gap]
        self.normal_path_list = self.normal_path_list[::gap]
        self.depth_path_list = self.depth_path_list[::gap]

        ### get clip for each sequence, directly split in time
        num_seq = len(self.rgb_path_list)
        print(f"sequence name: {scene_name}, num_seq: {num_seq}")

        assert num_seq == len(self.extrinsics), "num_seq should be the same as extrinsics"
        assert num_seq == len(self.intrinsics), "num_seq should be the same as intrinsics"
        assert num_seq == len(self.normal_path_list), "num_seq should be the same as normal_path_list"
        assert num_seq == len(self.depth_path_list), "num_seq should be the same as depth_path_list"

        ### split into clips. keyview_idx is the first frame index
        interval, overlap = clip_length, clip_overlap
        self.source_ids = {}
        for idx in range(0, num_seq, interval-overlap):
            group = list(range(idx, min(idx+interval, num_seq)))
            if len(group) < interval:
                group += [group[-1]] * (interval - len(group))
            self.source_ids[idx] = group


    def load_meta_data(self, root, scene_name):
        meta_data = np.load(osp.join(root, scene_name, 'scene_metadata.npz'))
        poses = meta_data['trajectories']  # camera to world, N x 4 x 4, in opencv coordinate
        ### change to opengl coordinate
        OPENGL_TO_OPENCV = np.float32([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
        poses = np.einsum('ij,njk,kl->nil', OPENGL_TO_OPENCV, poses, OPENGL_TO_OPENCV)

        ### transform to world to camera
        poses = np.linalg.inv(poses)  # N x 4 x 4
        intrinsics = meta_data['intrinsics']  # N x 3 x 3

        image_names = meta_data['images'].tolist()
        image_path_list = [osp.join('images', image_name + '.webp') for image_name in image_names]
        normal_path_list = [osp.join('normal', image_name + '.webp') for image_name in image_names]
        depth_path_list = [osp.join('depth', image_name + '.png') for image_name in image_names]

        return poses, intrinsics, image_path_list, normal_path_list, depth_path_list
    

class ScannetPPSample(Sample):
    def __init__(self, base, name):
        # base is folder path, not used here
        # name is scene name: 02455b3d20
        self.base = base
        self.name = name
        self.data = {}

    def load(self, root):
        out_dict = {'_base': root, 'scene_name': '_'.join(self.name.split('/'))}

        ### images, data['images']: list of ['images/DSC06130.jpg', ... ]
        img_paths = self.data['images']
        out_dict['images'] = [self.load_image(os.path.join(root, self.name), x) for x in img_paths]
        out_dict['image_names'] = [os.path.basename(x) for x in img_paths]

        ### poses
        out_dict['extrinsics'] = [x.astype(np.float32) for x in self.data['poses']]

        ### intrinsics
        out_dict['intrinsics'] = [x.astype(np.float32) for x in self.data['intrinsics']]

        out_dict['keyview_idx'] = self.data['keyview_idx']

        ### add normal
        out_dict['cam_normal'] = [self.load_normal(os.path.join(root, self.name), x) for x in self.data['normal']]

        out_dict['cam_coord'] = [self.load_position(os.path.join(root, self.name), x, out_dict['intrinsics'][0]) for x in self.data['depth']]

        ### add caption
        out_dict['caption'] = ""

        return self.postprocess(out_dict)
    

    def load_image(self, root, path):
        filename = osp.join(root, path)
        """Load a single image given the filename."""
        image = np.array(Image.open(filename)).astype(np.float32)
        return image.transpose(2, 0, 1)  # [3,H,W]


    def load_normal(self, root, path):
        filename = osp.join(root, path)
        """Load the surface normal given the filename."""
        normal = np.array(Image.open(filename)).astype(np.float32)
        mask = np.all(normal < 1e-3, axis=2)
        normal = normal / 255.0 * 2 - 1  # in opengl coordinate
        normal[mask] = 0
        return normal.astype(np.float32).transpose(2, 0, 1)  # [3,H,W]
    

    def load_position(self, root, path, intrinsic):
        filename = osp.join(root, path)
        depth = np.array(Image.open(filename)).astype(np.float32)
        depth = depth/ 1000
        position = backproject_to_cv_position(depth, intrinsic)  # opencv coordinate
        position[...,1:] *= -1  # change to opengl coordinate
        return position.astype(np.float32).transpose(2,0,1)  # [3,H,W]


    def update_depth(self):
        pass

    def postprocess(self, out_dict):
        ##### rotate normal to the first view
        cam_normal_list, world_normal_list = [], []
        cam_coord_list, world_coord_list = [], []
        mask_list = []

        keyview_idx = out_dict['keyview_idx']
        ref_pose = out_dict['extrinsics'][keyview_idx]  # [4, 4], w2c
        for idx in range(len(out_dict['cam_normal'])):
            src_pose = out_dict['extrinsics'][idx]
            trans_mat = ref_pose @ np.linalg.inv(src_pose)

            cam_normal = out_dict['cam_normal'][idx]  # [3,H,W], in camera coordinate
            ### transform position, all in opengl coordinate
            cam_coord = out_dict['cam_coord'][idx]  # [3,H,W], in camera coordinate

            # cam_normal = fix_normal(cam_normal, cam_coord)  # fix normal, not necessary for scannet

            world_normal = np.matmul(trans_mat[:3,:3], cam_normal.reshape(3, -1)).reshape(3, *cam_normal.shape[1:])

            world_coord = (np.matmul(trans_mat[:3,:3], cam_coord.reshape(3, -1)) + trans_mat[:3,3][:,None]).reshape(3, *cam_coord.shape[1:])

            ### mask
            invalid_mask = np.isnan(cam_normal).any(axis=0) | np.isnan(cam_coord).any(axis=0)
            depth = -1 * cam_coord[2]
            depth[np.isnan(depth)] = 0
            invalid_mask = invalid_mask | (depth < 1e-3) | (depth > 80)     

            ### process nan value
            cam_normal[:, invalid_mask] = 0
            cam_coord[:, invalid_mask] = 0
            world_normal[:, invalid_mask] = 0
            world_coord[:, invalid_mask] = 0

            cam_normal_list.append(cam_normal)
            cam_coord_list.append(cam_coord)
            world_normal_list.append(world_normal)
            world_coord_list.append(world_coord)
            mask_list.append((~invalid_mask).astype(np.float32))

            #####################

        out_dict['cam_normal'] = cam_normal_list  # list of [3,H,W]
        out_dict['cam_coord'] = cam_coord_list
        out_dict['world_normal'] = world_normal_list
        out_dict['world_coord'] = world_coord_list
        out_dict['mask'] = mask_list  # list of [H,W]

        ##### add extrinsic transformation
        out_dict['extrinsics'] = [x @ np.linalg.inv(ref_pose) for x in out_dict['extrinsics']]
        return out_dict


class ScannetPPDataset(Dataset):
    base_dataset = 'scannetpp'

    def __init__(self, root=None, layouts=None, split='test', clip_length=17, clip_overlap=0, **kwargs):
        root = root if root is not None else self._get_path("scannetpp", "root")
        self.split = split
        self.clip_length, self.clip_overlap = clip_length, clip_overlap

        super().__init__(root=root, layouts=layouts, **kwargs)

    def _init_samples(self,):
        sample_list_path = _get_sample_list_path(self.name, self.clip_length, self.clip_overlap)
        if sample_list_path is not None and osp.isfile(sample_list_path):
            super()._init_samples_from_list()
        else:
            self._init_samples_from_root_dir()
            self._write_samples_list()

    def _init_samples_from_root_dir(self,):
        # self.samples is defined in the base class, filled with Sample objects

        ##### first, get the scene names
        split = 'train' if self.split == 'train' else 'nvs_sem_val'

        ### get sequences
        self.split_file = os.path.join(os.path.dirname(__file__), "splits", "{}.txt".format(split))
        with open(self.split_file, "r") as f:
            self.split_list = f.read().splitlines()

        print("Loading the {} dataset".format(split))
        seqs = [ScannetPPSequence(self.root, scene_name, 
        clip_length=self.clip_length, clip_overlap=self.clip_overlap) for scene_name in self.split_list]

        for seq in (tqdm(seqs) if self.verbose else seqs):
            for key_id in seq.source_ids.keys():
                
                all_source_ids = seq.source_ids[key_id]
                all_ids = all_source_ids  # use the updated source_ids

                images = [seq.rgb_path_list[i] for i in all_ids]
                poses = [seq.extrinsics[i] for i in all_ids]
                intrinsics = [seq.intrinsics[i] for i in all_ids]
                depth = [seq.depth_path_list[i] for i in all_ids]
                normal = [seq.normal_path_list[i] for i in all_ids]

                sample = ScannetPPSample(base=self.root, name=seq.scene_name)

                sample.data['images'] = images
                sample.data['poses'] = poses
                sample.data['intrinsics'] = intrinsics
                sample.data['depth'] = depth
                sample.data['keyview_idx'] = 0
                sample.data['normal'] = normal

                self.samples.append(sample)