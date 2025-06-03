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


class boonSequence:
    def __init__(self, root, scene_name, clip_length=30, clip_overlap=0):

        self.root = root
        self.scene_name = scene_name
        
        self.extrinsics, self.intrinsics, self.rgb_path_list, self.depth_path_list = self.load_meta_data(self.root, self.scene_name)

        gap = 1
        self.extrinsics = self.extrinsics[::gap]
        self.intrinsics = self.intrinsics[::gap]
        self.rgb_path_list = self.rgb_path_list[::gap]
        self.depth_path_list = self.depth_path_list[::gap]

        ### get clip for each sequence, directly split in time
        num_seq = len(self.rgb_path_list)
        print(f"sequence name: {scene_name}, num_seq: {num_seq}")

        assert num_seq == len(self.extrinsics), "num_seq should be the same as extrinsics"
        assert num_seq == len(self.intrinsics), "num_seq should be the same as intrinsics"
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
        # root: /mnt/pfs/data/RGBD/bonn/rgbd_bonn_dataset
        # sceen_name: rgbd_bonn_balloon2/rgb_110
        rgb_files = sorted(glob(osp.join(root, scene_name, "rgb_110/*.png")))
        depth_files = sorted(glob(osp.join(root, scene_name, "depth_110/*.png")))
        poses_file = osp.join(root, scene_name, "groundtruth_110.txt")  # TUM-RGBD format

        # https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html
        fx = 542.822841
        fy = 542.576870
        cx = 315.593520
        cy = 237.756098
        intrinsics = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])

        from evo.tools import file_interface
        traj = file_interface.read_tum_trajectory_file(poses_file)
        poses = np.stack(traj.poses_se3)  # shape: (N, 4, 4)

        ### change to opengl coordinate
        OPENGL_TO_OPENCV = np.float32([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
        poses = np.einsum('ij,njk,kl->nil', OPENGL_TO_OPENCV, poses, OPENGL_TO_OPENCV)
        poses = np.linalg.inv(poses)  # [N, 4, 4]

        intrinsics_list = [intrinsics
                           for _ in range(len(poses))]

        return poses, intrinsics_list, rgb_files, depth_files
        


class bonnSample(Sample):
    def __init__(self, base, name):
        # base is folder path, not used here
        # name is scene name: rgbd_bonn_balloon2
        self.base = base
        self.name = name
        self.data = {}

    
    def load(self, root):
        out_dict = {'_base': root, 'scene_name': '_'.join(self.name.split('/'))}

        ### images, data['images']: list of ['rgb_110/1548266531.51903.png', ... ]
        img_paths = self.data['images']
        out_dict['images'] = [self.load_image(os.path.join(root, self.name), x) for x in img_paths]
        out_dict['image_names'] = [os.path.basename(x) for x in img_paths]

        ### poses
        out_dict['extrinsics'] = [x.astype(np.float32) for x in self.data['poses']]

        ### intrinsics
        out_dict['intrinsics'] = [x.astype(np.float32) for x in self.data['intrinsics']]

        out_dict['keyview_idx'] = self.data['keyview_idx']

        out_dict['cam_coord'] = [self.load_position(os.path.join(root, self.name), x, out_dict['intrinsics'][0]) for x in self.data['depth']]

        ### add caption
        out_dict['caption'] = ""

        return self.postprocess(out_dict)


    def load_image(self, root, path):
        filename = osp.join(root, path)
        """Load a single image given the filename."""
        image = np.array(Image.open(filename)).astype(np.float32)
        return image.transpose(2, 0, 1)  # [3,H,W]


    def load_position(self, root, path, intrinsic):
        filename = osp.join(root, path)
        depth_png = np.array(Image.open(filename)).astype(np.float32)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert np.max(depth_png) > 255
        depth = depth_png.astype(np.float64) / 5000.0

        depth[depth_png == 0] = np.nan

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
        for idx in range(len(out_dict['cam_coord'])):
            src_pose = out_dict['extrinsics'][idx]
            trans_mat = ref_pose @ np.linalg.inv(src_pose)


            ### transform position, all in opengl coordinate
            cam_coord = out_dict['cam_coord'][idx]  # [3,H,W], in camera coordinate
            world_coord = (np.matmul(trans_mat[:3,:3], cam_coord.reshape(3, -1)) + trans_mat[:3,3][:,None]).reshape(3, *cam_coord.shape[1:])

            ### mask
            invalid_mask = np.isnan(cam_coord).any(axis=0)
            depth = -1 * cam_coord[2]
            depth[np.isnan(depth)] = 0
            invalid_mask = invalid_mask | (depth < 1e-3) | (depth > 20)  # indoor scene

            ### process nan value
            # cam_normal[:, invalid_mask] = 0
            cam_coord[:, invalid_mask] = 0
            # world_normal[:, invalid_mask] = 0
            world_coord[:, invalid_mask] = 0

            # cam_normal_list.append(cam_normal)
            cam_coord_list.append(cam_coord)
            # world_normal_list.append(world_normal)
            world_coord_list.append(world_coord)
            mask_list.append((~invalid_mask).astype(np.float32))

        # out_dict['cam_normal'] = cam_normal_list  # list of [3,H,W]
        out_dict['cam_coord'] = cam_coord_list
        # out_dict['world_normal'] = world_normal_list
        out_dict['world_coord'] = world_coord_list
        out_dict['mask'] = mask_list  # list of [H,W]

        ##### add extrinsic transformation
        out_dict['extrinsics'] = [x @ np.linalg.inv(ref_pose) for x in out_dict['extrinsics']]
        return out_dict




def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array
    depth_png = np.asarray(Image.open(filename))
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255
    depth = depth_png.astype(np.float64) / 5000.0
    depth[depth_png == 0] = -1.0
    return depth


class bonnDataset(Dataset):
    base_dataset = 'bonn'

    def __init__(self, root=None, layouts=None, split='test', clip_length=17, clip_overlap=0, **kwargs):
        root = root if root is not None else self._get_path("bonn", "root")
        self.split = split
        self.clip_length, self.clip_overlap = clip_length, clip_overlap

        super().__init__(root=root, layouts=layouts, **kwargs)

    def _init_samples(self,):
        sample_list_path = _get_sample_list_path(self.name, self.clip_length, self.clip_overlap)
        if sample_list_path is not None and osp.isfile(sample_list_path):
            # always recompute the sample list
            super()._init_samples_from_list()
        else:
            self._init_samples_from_root_dir()
            self._write_samples_list()


    def _init_samples_from_root_dir(self,):
        # self.samples is defined in the base class, filled with Sample objects

        ##### first, get the scene names
        split = self.split

        ### get sequences
        self.split_file = os.path.join(os.path.dirname(__file__), "splits", "{}.txt".format(split))
        with open(self.split_file, "r") as f:
            self.split_list = f.read().splitlines()

        print("Loading the {} dataset".format(split))

        seqs = [boonSequence(self.root, scene_name, 
                clip_length=self.clip_length, clip_overlap=self.clip_overlap) for scene_name in self.split_list]

        for seq in (tqdm(seqs) if self.verbose else seqs):
            for key_id in seq.source_ids.keys():
                
                all_source_ids = seq.source_ids[key_id]
                all_ids = all_source_ids  # use the updated source_ids

                images = [seq.rgb_path_list[i] for i in all_ids]
                poses = [seq.extrinsics[i] for i in all_ids]
                intrinsics = [seq.intrinsics[i] for i in all_ids]
                depth = [seq.depth_path_list[i] for i in all_ids]

                sample = bonnSample(base=self.root, name=seq.scene_name)

                sample.data['images'] = images
                sample.data['poses'] = poses
                sample.data['intrinsics'] = intrinsics
                sample.data['depth'] = depth
                sample.data['keyview_idx'] = 0

                self.samples.append(sample)