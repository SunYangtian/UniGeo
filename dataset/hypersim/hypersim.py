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
from utils.geometry_utils import fix_normal
import skimage


class HyperSimSequence:
    def __init__(self, root, scene_name, camera_name='cam_00', num_source_views=1, compute_pair=True):

        self.root = root
        self.load_hdf5 = True
        self.scene_name = scene_name
        self.camera_name = camera_name
        self.num_source_views = num_source_views

        scene_dir = os.path.join(root, scene_name)

        ### Before loading, read meta info: camera parameters and image valid info
        # Read the camera parameters
        camera_parameters_path = os.path.join(root, "metadata_camera_parameters.csv")
        self.df_camera_parameters = pd.read_csv(camera_parameters_path, index_col="scene_name")
        images_info_path = os.path.join(root, "metadata_images_split_scene_v1.csv")
        self.df_images_info = pd.read_csv(images_info_path)

        ##### load all cameras
        # Load the RGB image and depth map
        cam_name = camera_name
        rgb_image_dir = os.path.join(scene_dir, "images", f"scene_{cam_name}_final_preview")
        normal_image_dir = os.path.join(scene_dir, "images", f"scene_{cam_name}_geometry_hdf5")

        rgb_path_list = sorted(glob(rgb_image_dir + "/*tonemap.jpg"))
        normal_path_list = sorted(glob(normal_image_dir + "/*normal_cam.hdf5"))
        position_path_list = sorted(glob(normal_image_dir + "/*position.hdf5"))

        ### save relative path to root
        self.rgb_path_list = [os.path.join("images", f"scene_{cam_name}_final_preview", os.path.basename(x)) for x in rgb_path_list]
        self.normal_path_list = [os.path.join("images", f"scene_{cam_name}_geometry_hdf5", os.path.basename(x)) for x in normal_path_list]
        self.position_path_list = [os.path.join("images", f"scene_{cam_name}_geometry_hdf5", os.path.basename(x)) for x in position_path_list]

        # self.intrinsics = self.get_intrinsic(scene_name)
        self.Mproj, self.ndc2screen = self.get_proj(scene_name)
        # stay in OpenGL coordinate
        self.extrinsics = []
        # for cam_name in camera_list:
        cam2world_list = self.load_extrinsic(scene_name, cam_name)
        ### some images are not invalid
        valid_mask = self.df_images_info[(self.df_images_info.scene_name==scene_name) & (self.df_images_info.camera_name==cam_name)]['included_in_public_release'].to_list()
        self.extrinsics += [np.linalg.inv(x) for idx, x in enumerate(cam2world_list) if valid_mask[idx]]
        

        num_seq = len(self.rgb_path_list)
        assert len(self.extrinsics) == num_seq, f"misaligned extrinsic {len(self.extrinsics)} != image number: {num_seq} in {scene_name}/{cam_name}"
        assert len(self.normal_path_list) == num_seq, f"misaligned normal {len(self.normal_path_list)} != image number: {num_seq} in {scene_name}/{cam_name}"
        assert len(self.position_path_list) == num_seq, f"misaligned position {len(self.position_path_list)} != image number: {num_seq} in {scene_name}/{cam_name}"

        print(f"load {num_seq} images in {scene_name}/{cam_name}")

        ##### some data is invalid filtered manully
        label_csv = os.path.join(root, scene_name, f"{camera_name}_label.csv")
        frame_labels = pd.read_csv(label_csv)['label'].to_numpy()
        assert len(frame_labels) == num_seq, f"misaligned label {len(frame_labels)} != image number: {num_seq} in {scene_name}/{cam_name}"

        if True:
            # don't open when calculate mask_score
            self.rgb_path_list = [x for idx, x in enumerate(self.rgb_path_list) if frame_labels[idx]]
            self.normal_path_list = [x for idx, x in enumerate(self.normal_path_list) if frame_labels[idx]]
            self.position_path_list = [x for idx, x in enumerate(self.position_path_list) if frame_labels[idx]]
            self.extrinsics = [x for idx, x in enumerate(self.extrinsics) if frame_labels[idx]]

            num_seq = len(self.rgb_path_list)
            print(f"After filtering, load {num_seq} valid images")
        
        if compute_pair:
            # don't open when calculate mask_score
            ### generate tuples based on mask_score
            mask_score_path = os.path.join(scene_dir, f"{scene_name}_{camera_name}_mask_score.csv")
            mask_score_frame = pd.read_csv(mask_score_path, index_col=0)
            mask_score_frame = mask_score_frame.loc[frame_labels, frame_labels]
            ### rename index and columes
            mask_score_frame.columns = range(mask_score_frame.shape[1])
            mask_score_frame = mask_score_frame.reset_index(drop=True)

            assert mask_score_frame.shape[0] == mask_score_frame.shape[1] == num_seq

            ### condiser both ref -> src and src -> ref
            ### mask_score_frame has the same index and columns
            mask_score_frame_merge = 0.5 * (mask_score_frame + mask_score_frame.T)

            self.source_ids = {}
            for idx in range(num_seq):
                row = mask_score_frame_merge.iloc[idx]
                selected_index = row.nlargest(self.num_source_views+1).index.tolist()
                selected_value = row.nlargest(self.num_source_views+1).values
                if selected_value.mean() < 0.7:
                    continue
                self.source_ids[idx] = selected_index
                # self.source_ids[idx] = [int(x) for x in row.nlargest(self.num_source_views+1).index.tolist()]
            print(f"After filtering, loading {len(self.source_ids)} sequences \n")
            


    def get_proj(self, scene_name: str) -> np.ndarray:
        df_ = self.df_camera_parameters.loc[scene_name]

        width_pixels  = int(df_["settings_output_img_width"])
        height_pixels = int(df_["settings_output_img_height"])
        M_proj = np.matrix([[ df_["M_proj_00"], df_["M_proj_01"], df_["M_proj_02"], df_["M_proj_03"] ],
                        [ df_["M_proj_10"], df_["M_proj_11"], df_["M_proj_12"], df_["M_proj_13"] ],
                        [ df_["M_proj_20"], df_["M_proj_21"], df_["M_proj_22"], df_["M_proj_23"] ],
                        [ df_["M_proj_30"], df_["M_proj_31"], df_["M_proj_32"], df_["M_proj_33"] ]])

        # matrix to map to integer screen coordinates from normalized device coordinates
        M_screen_from_ndc = np.matrix([[0.5*(width_pixels-1), 0,                      0,   0.5*(width_pixels-1)],
                                    [0,                    -0.5*(height_pixels-1), 0,   0.5*(height_pixels-1)],
                                    [0,                    0,                      0.5, 0.5],
                                    [0,                    0,                      0,   1.0]])

        return M_proj, M_screen_from_ndc
    
    def load_extrinsic(self, scene_name: str, cam_name: str) -> list:
        ### get camera extrinsic
        ### scene_name: ai_001_002, cam_name: cam_02
        camera_dir = os.path.join(self.root, scene_name, "_detail", cam_name)
        camera_positions_hdf5_file    = os.path.join(camera_dir, "camera_keyframe_positions.hdf5")
        camera_orientations_hdf5_file = os.path.join(camera_dir, "camera_keyframe_orientations.hdf5")

        with h5py.File(camera_positions_hdf5_file, "r") as f: camera_positions    = f["dataset"][:]
        with h5py.File(camera_orientations_hdf5_file, "r") as f: camera_orientations = f["dataset"][:]

        scene_metadata = pd.read_csv(os.path.join(os.path.dirname(camera_dir), "metadata_scene.csv"))

        meters_per_asset_unit = scene_metadata[
        scene_metadata.parameter_name == "meters_per_asset_unit"
        ]
        assert len(meters_per_asset_unit) == 1  # Should not be multiply defined
        meters_per_asset_unit = meters_per_asset_unit.parameter_value[0]
        scale_factor = float(meters_per_asset_unit)  # about 0.01. camera_positions is very large, maybe in centimeter

        cam2world_list = []
        for index in range(len(camera_positions)):
            camera_position = camera_positions[index]
            camera_rotation = camera_orientations[index]

            # Get camera pose and intrinsic
            camera_position_world = camera_position
            R_world_from_cam = camera_rotation
            t_world_from_cam = np.array(camera_position_world).T * scale_factor

            # Camera to world transformation matrix (OpenGL)
            cam2world = np.eye(4)
            cam2world[:3, :3] = R_world_from_cam
            cam2world[:3, 3] = t_world_from_cam

            cam2world_list.append(cam2world)
        return cam2world_list



class HyperSimSample(Sample):

    def __init__(self, base, name):
        self.base = base
        self.name = name
        self.data = {}

    @staticmethod
    def load_image(root, path):
        img_path = osp.join(root, path)
        img = np.array(Image.open(img_path))
        img = img.transpose(2, 0, 1).astype(np.float32)
        return img  # [3,H,W]
    
    @staticmethod
    def load_normal(root, path):
        normal_path = osp.join(root, path)
        if normal_path.endswith(".png"):
            normal = np.array(Image.open(normal_path))
        else:
            with h5py.File(normal_path, "r") as f:
                normal = f["dataset"][:].astype(np.float32)

        normal = normal / (np.linalg.norm(normal, axis=2, keepdims=True) + 1e-6)
        return normal.transpose(2,0,1)  # [1,H,W]
    
    @staticmethod
    def load_position(root, path):
        position_path = osp.join(root, path)
        with h5py.File(position_path, "r") as f:
            position = f["dataset"][:].astype(np.float32)

        if True:
            camera_dir = os.path.join(root, "_detail")
            scene_metadata = pd.read_csv(os.path.join(camera_dir, "metadata_scene.csv"))
            meters_per_asset_unit = scene_metadata[
            scene_metadata.parameter_name == "meters_per_asset_unit"
            ]
            assert len(meters_per_asset_unit) == 1  # Should not be multiply defined
            meters_per_asset_unit = meters_per_asset_unit.parameter_value[0]
            scale_factor = float(meters_per_asset_unit)
            position *= scale_factor

        ### change inf to nan
        position[np.isinf(position)] = np.nan
        return position.transpose(2,0,1)  # [3,H,W]


    def load(self, root):
        out_dict = {'_base': root, 'scene_name': self.name}
        # ai_001_001_cam_00 -> ai_001_001
        scene_name = '_'.join(self.name.split('_')[:3])

        ### images
        img_paths = self.data['images']
        out_dict['images'] = [self.load_image(os.path.join(root, scene_name), x) for x in img_paths]
        out_dict['image_names'] = [os.path.basename(x) for x in img_paths]

        ### poses
        out_dict['extrinsics'] = [x.astype(np.float32) for x in self.data['poses']]

        ### intrinsics
        # out_dict['intrinsics'] = [self.data['intrinsics']] * len(self.data['poses'])
        proj_mat = np.array(self.data['proj'])
        ndc2screen = np.array(self.data['ndc2screen'])

        def get_intrinsic(proj_mat, ndc2screen):
            fx = ndc2screen[0,0] * proj_mat[0,0]
            fy = -1 * ndc2screen[1,1] * proj_mat[1,1]
            cx = ndc2screen[0,3]
            cy = ndc2screen[1,3]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        out_dict['intrinsics'] = [get_intrinsic(proj_mat, ndc2screen) for _ in self.data['poses']]

        out_dict['keyview_idx'] = self.data['keyview_idx']

        ### add normal
        out_dict['cam_normal'] = [self.load_normal(os.path.join(root, scene_name), x) for x in self.data['normal']]

        ### add position
        out_dict['world_coord'] = [self.load_position(os.path.join(root, scene_name), x) for x in self.data['position']]

        ### add caption
        out_dict['caption'] = self.data.get('caption', "")

        return self.postprocess(out_dict)
    
    def postprocess(self, out_dict):
        ##### rotate attributes to the first view
        cam_normal_list, world_normal_list = [], []
        cam_coord_list, world_coord_list = [], []
        mask_list = []

        keyview_idx = out_dict['keyview_idx']
        ref_pose = out_dict['extrinsics'][keyview_idx]
        for idx in range(len(out_dict['cam_normal'])):
            src_pose = out_dict['extrinsics'][idx]
            cam_normal = out_dict['cam_normal'][idx]  # [3,H,W], camera coordinate
            global_world_coord = out_dict['world_coord'][idx]  # [3,H,W], world coordinate

            ### transform position
            cam_coord = (src_pose[:3,:3] @ global_world_coord.reshape(3,-1) + src_pose[:3,3:4]).reshape(3, *global_world_coord.shape[1:])

            ### fix inverse normal
            cam_normal = fix_normal(cam_normal, cam_coord)

            trans_mat = np.linalg.inv(src_pose)
            trans_mat = ref_pose @ trans_mat
            world_normal = np.matmul(trans_mat[:3,:3], cam_normal.reshape(3, -1)).reshape(3, *cam_normal.shape[1:])

            world_coord = (np.matmul(ref_pose[:3,:3], global_world_coord.reshape(3, -1)) + ref_pose[:3,3][:,None]).reshape(3, *global_world_coord.shape[1:])

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

        out_dict['cam_normal'] = cam_normal_list  # list of [3,H,W]
        out_dict['cam_coord'] = cam_coord_list
        out_dict['world_normal'] = world_normal_list
        out_dict['world_coord'] = world_coord_list
        out_dict['mask'] = mask_list  # list of [H,W]

        ##### add extrinsic transformation
        out_dict['extrinsics'] = [x @ np.linalg.inv(ref_pose) for x in out_dict['extrinsics']]
        return out_dict



class HyperSimDataset(Dataset):
    base_dataset = 'hypersim'

    def __init__(self, root=None, layouts=None, split='train', **kwargs):
        root = root if root is not None else self._get_path("hypersim", "root")
        self.split = split
        self.clip_length = self.clip_overlap = -1

        super().__init__(root=root, layouts=layouts, **kwargs)

    def _init_samples(self, **kwargs):
        sample_list_path = _get_sample_list_path(self.name, self.clip_length, self.clip_overlap)
        if sample_list_path is not None and osp.isfile(sample_list_path):
            super()._init_samples_from_list()
        else:
            self._init_samples_from_root_dir()
            self._write_samples_list(path=sample_list_path)

    def _init_samples_from_root_dir(self,):
        # self.samples is defined in the base class, filled with Sample objects

        ##### first, get the scene names
        split = self.split
        self.split_file = os.path.join(os.path.dirname(__file__), "splits", "train_filter.txt")
        with open(self.split_file, "r") as f:
            self.split_list = f.read().splitlines()

        print("Loading the {} dataset".format(split))
        # scene_name: ai_001_001/cam_00
        seqs = [HyperSimSequence(self.root, scene_name.split('/')[0], scene_name.split('/')[1]) for scene_name in self.split_list]

        for seq in (tqdm(seqs) if self.verbose else seqs):
            for key_id in seq.source_ids.keys():

                all_source_ids = seq.source_ids[key_id]
                # all_ids = [key_id] + all_source_ids
                all_ids = all_source_ids  # use the updated source_ids

                images = [seq.rgb_path_list[i] for i in all_ids]
                poses = [seq.extrinsics[i] for i in all_ids]
                # intrinsics = seq.intrinsics
                proj = seq.Mproj
                ndc2screen = seq.ndc2screen
                # depth = [seq.depth_path_list[i] for i in all_ids]
                normal = [seq.normal_path_list[i] for i in all_ids]
                position = [seq.position_path_list[i] for i in all_ids]

                sample = HyperSimSample(base=self.root, name=seq.scene_name+'_'+seq.camera_name)
                sample.data['images'] = images
                sample.data['poses'] = poses
                sample.data['proj'] = proj
                sample.data['ndc2screen'] = ndc2screen
                # sample.data['depth'] = depth
                sample.data['keyview_idx'] = 0
                sample.data['normal'] = normal
                sample.data['position'] = position

                self.samples.append(sample)
