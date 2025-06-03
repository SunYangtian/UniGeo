#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Script to pre-process the scannet++ dataset.
# Usage:
# python3 datasets_preprocess/preprocess_scannetpp.py --scannetpp_dir /path/to/scannetpp --precomputed_pairs /path/to/scannetpp_pairs --pyopengl-platform egl
# --------------------------------------------------------

# refer to: https://github1s.com/jiah-cloud/Align3R/blob/main/datasets_preprocess/preprocess_scannetpp.py

import os
import argparse
import os.path as osp
import re
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation
import pyrender
import trimesh
import trimesh.exchange.ply
import PIL.Image
import numpy as np
import cv2
import PIL.Image as Image
import matplotlib.cm as cm

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC

inv = np.linalg.inv
norm = np.linalg.norm
REGEXPR_DSLR = re.compile(r'^DSC(?P<frameid>\d+).JPG$')
REGEXPR_IPHONE = re.compile(r'frame_(?P<frameid>\d+).jpg$')

DEBUG_VIZ = None  # 'iou'
if DEBUG_VIZ is not None:
    import matplotlib.pyplot as plt  # noqa


OPENGL_TO_OPENCV = np.float32([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]])


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannetpp_dir', default='/mnt/pfs/data/ScanNetPP')
    parser.add_argument('--precomputed_pairs', default=None)
    parser.add_argument('--output_dir', default='data/scannetpp_processed_iphone')
    parser.add_argument('--target_resolution', default=920, type=int, help="images resolution")
    parser.add_argument('--pyopengl-platform', type=str, default='', help='PyOpenGL env variable')
    return parser


class ImageList:
    """ Convenience class to aply the same operation to a whole set of images.
    """

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch('resize', *args, **kwargs))

    def crop(self, *args, **kwargs):
        return ImageList(self._dispatch('crop', *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


def camera_matrix_of_crop(input_camera_matrix, input_resolution, output_resolution, scaling=1, offset_factor=0.5, offset=None):
    # Margins to offset the origin
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    # Generate new camera parameters
    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

    return output_camera_matrix


def rescale_image_depthmap(image, depthmap, pred_depth, camera_intrinsics, output_resolution, force=True):
    """ Jointly rescale a (image, depthmap) 
        so that (out_width, out_height) >= output_res
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        # can also use this with masks instead of depthmaps
        assert tuple(depthmap.shape[:2]) == image.size[::-1]
    if pred_depth is not None:
        # can also use this with masks instead of depthmaps
        assert tuple(pred_depth.shape[:2]) == image.size[::-1]
    # define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:  # image is already smaller than what is asked
        return (image.to_pil(), depthmap, pred_depth, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)
    output_resolution = list(output_resolution)
    # first rescale the image so that it contains the crop
    image = image.resize(output_resolution, resample=lanczos if scale_final < 1 else bicubic)
    if depthmap is not None:
        depthmap = cv2.resize(depthmap, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)
    if pred_depth is not None:
        pred_depth = cv2.resize(pred_depth, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)

    # no offset here; simple rescaling
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)

    return image.to_pil(), depthmap, pred_depth, camera_intrinsics


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def pose_from_qwxyz_txyz(elems):
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(pose)  # returns cam2world


def get_frame_number(name, cam_type='dslr'):
    if cam_type == 'dslr':
        regex_expr = REGEXPR_DSLR
    elif cam_type == 'iphone':
        regex_expr = REGEXPR_IPHONE
    else:
        raise NotImplementedError(f'wrong {cam_type=} for get_frame_number')
    matches = re.match(regex_expr, name)
    return matches['frameid']


def load_sfm(sfm_dir, cam_type='dslr'):
    # load cameras
    with open(osp.join(sfm_dir, 'cameras.txt'), 'r') as f:
        raw = f.read().splitlines()[3:]  # skip header

    intrinsics = {}
    for camera in tqdm(raw, position=1, leave=False):
        camera = camera.split(' ')
        intrinsics[int(camera[0])] = [camera[1]] + [float(cam) for cam in camera[2:]]

    # load images
    with open(os.path.join(sfm_dir, 'images.txt'), 'r') as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith('#')]  # skip header

    img_idx = {}
    img_infos = {}
    for image, points in tqdm(zip(raw[0::2], raw[1::2]), total=len(raw) // 2, position=1, leave=False):
        image = image.split(' ')
        points = points.split(' ')

        idx = image[0]
        img_name = image[-1]
        assert img_name not in img_idx, 'duplicate db image: ' + img_name
        img_idx[img_name] = idx  # register image name

        current_points2D = {int(i): (float(x), float(y))
                            for i, x, y in zip(points[2::3], points[0::3], points[1::3]) if i != '-1'}
        img_infos[idx] = dict(intrinsics=intrinsics[int(image[-2])],
                              path=img_name,
                              frame_id=get_frame_number(img_name, cam_type),
                              cam_to_world=pose_from_qwxyz_txyz(image[1: -2]),
                              sparse_pts2d=current_points2D)

    # load 3D points
    with open(os.path.join(sfm_dir, 'points3D.txt'), 'r') as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith('#')]  # skip header

    points3D = {}
    observations = {idx: [] for idx in img_infos.keys()}
    for point in tqdm(raw, position=1, leave=False):
        point = point.split()
        point_3d_idx = int(point[0])
        points3D[point_3d_idx] = tuple(map(float, point[1:4]))
        if len(point) > 8:
            for idx, point_2d_idx in zip(point[8::2], point[9::2]):
                observations[idx].append((point_3d_idx, int(point_2d_idx)))

    return img_idx, img_infos, points3D, observations

def load_imu(imu_path):
    with open(imu_path, 'r') as f:
        imu_data = json.load(f)

    img_names = sorted(imu_data.keys())
    poses = [imu_data[frame_name]['aligned_pose'] for frame_name in img_names]
    poses = np.array(poses)  # N x 4 x 4

    intrinsics = [imu_data[frame_name]['intrinsic'] for frame_name in img_names]
    intrinsics = np.array(intrinsics)  # N x 3 x 3

    img_infos = {}
    for idx, frame_name in enumerate(img_names):
        img_infos[frame_name] = dict(
            path=frame_name,
            # frame_id=get_frame_number(frame_name, cam_type='iphone'),
            cam_to_world=poses[idx],
            intrinsic = intrinsics[idx],
        )

    return img_names, img_infos


def subsample_img_infos(img_infos, num_images, allowed_name_subset=None):
    img_infos_val = [(idx, val) for idx, val in img_infos.items()]
    if allowed_name_subset is not None:
        img_infos_val = [(idx, val) for idx, val in img_infos_val if val['path'] in allowed_name_subset]

    if len(img_infos_val) > num_images:
        img_infos_val = sorted(img_infos_val, key=lambda x: x[1]['frame_id'])
        kept_idx = np.round(np.linspace(0, len(img_infos_val) - 1, num_images)).astype(int).tolist()
        img_infos_val = [img_infos_val[idx] for idx in kept_idx]
    return {idx: val for idx, val in img_infos_val}


def undistort_images(intrinsics, rgb, mask):
    camera_type = intrinsics[0]

    width = int(intrinsics[1])
    height = int(intrinsics[2])
    fx = intrinsics[3]
    fy = intrinsics[4]
    cx = intrinsics[5]
    cy = intrinsics[6]
    distortion = np.array(intrinsics[7:])

    K = np.zeros([3, 3])
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    K[2, 2] = 1

    K = colmap_to_opencv_intrinsics(K)
    if camera_type == "OPENCV_FISHEYE":
        assert len(distortion) == 4

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            distortion,
            (width, height),
            np.eye(3),
            balance=0.0,
        )
        # Make the cx and cy to be the center of the image
        new_K[0, 2] = width / 2.0
        new_K[1, 2] = height / 2.0

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion, (width, height), 1, (width, height), True)
        map1, map2 = cv2.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)

    undistorted_image = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    undistorted_mask = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    new_K = opencv_to_colmap_intrinsics(new_K)
    return width, height, new_K, undistorted_image, undistorted_mask


class CustomShaderCache():
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram("shaders/mesh.vert", "shaders/mesh.frag", defines=defines)
        return self.program


def process_scenes(root, pairsdir, output_dir, target_resolution):
    os.makedirs(output_dir, exist_ok=True)

    # default values from
    # https://github.com/scannetpp/scannetpp/blob/main/common/configs/render.yml
    znear = 0.05
    zfar = 20.0

    if pairsdir is None:
        scenes = sorted(os.listdir(os.path.join(root, 'data')))
    elif False:
        listfile = osp.join(pairsdir, 'scene_list.json')
        with open(listfile, 'r') as f:
            scenes = json.load(f)
    else:
        listfile = osp.join(pairsdir, 'nvs_sem_val.txt')
        with open(listfile, 'r') as f:
            scenes = f.read().splitlines()

    # for each of these, we will select some dslr images and some iphone images
    # we will undistort them and render their depth
    renderer = pyrender.OffscreenRenderer(0, 0)
    # for scene in tqdm(scenes, position=0, leave=True):
    for scene_id, scene in enumerate(scenes):
        print(f"Processing {scene} ({scene_id + 1}/{len(scenes)})")
        data_dir = os.path.join(root, 'data', scene)
        dir_dslr = os.path.join(data_dir, 'dslr')
        dir_iphone = os.path.join(data_dir, 'iphone')
        dir_scans = os.path.join(data_dir, 'scans')

        output_dir_scene = os.path.join(output_dir, scene)
        scene_metadata_path = osp.join(output_dir_scene, 'scene_metadata.npz')
        if osp.isfile(scene_metadata_path):
            continue

        # set up the output paths
        output_dir_scene_rgb = os.path.join(output_dir_scene, 'images')
        output_dir_scene_depth = os.path.join(output_dir_scene, 'depth')
        output_dir_scene_depth_vis = os.path.join(output_dir_scene, 'depth_vis')
        output_dir_scene_normal = os.path.join(output_dir_scene, 'normal')
        # output_dir_scene_rgb_all = os.path.join(output_dir_scene, 'images_all')
        os.makedirs(output_dir_scene_rgb, exist_ok=True)
        os.makedirs(output_dir_scene_depth, exist_ok=True)
        os.makedirs(output_dir_scene_depth_vis, exist_ok=True)
        os.makedirs(output_dir_scene_normal, exist_ok=True)
        # os.makedirs(output_dir_scene_rgb_all, exist_ok=True)

        ply_path = os.path.join(dir_scans, 'mesh_aligned_0.05.ply')

        sfm_dir_dslr = os.path.join(dir_dslr, 'colmap')
        rgb_dir_dslr = os.path.join(dir_dslr, 'resized_images')
        mask_dir_dslr = os.path.join(dir_dslr, 'resized_anon_masks')

        sfm_dir_iphone = os.path.join(dir_iphone, 'colmap')
        rgb_dir_iphone = os.path.join(dir_iphone, 'rgb')
        mask_dir_iphone = os.path.join(dir_iphone, 'rgb_masks')

        if not os.path.exists(ply_path):
            print(f"Skipping {scene} as {ply_path} does not exist")
            continue

        # load the mesh
        with open(ply_path, 'rb') as f:
            mesh_kwargs = trimesh.exchange.ply.load_ply(f)
        mesh_scene = trimesh.Trimesh(**mesh_kwargs)

        if not mesh_scene.is_winding_consistent:
            # # 修复反向面片
            # print(f"Fixing normals for {scene}")
            # mesh_scene.fix_normals()
            pass

        # read colmap reconstruction, we will only use the intrinsics and pose here
        img_idx_dslr, img_infos_dslr, points3D_dslr, observations_dslr = load_sfm(sfm_dir_dslr, cam_type='dslr')
        dslr_paths = {
            "in_colmap": sfm_dir_dslr,
            "in_rgb": rgb_dir_dslr,
            "in_mask": mask_dir_dslr,
        }

        # img_idx_iphone, img_infos_iphone, points3D_iphone, observations_iphone = load_sfm(
        #     sfm_dir_iphone, cam_type='iphone')
        img_idx_iphone, img_infos_iphone = load_imu(os.path.join(dir_iphone, 'pose_intrinsic_imu.json'))
        iphone_paths = {
            "in_colmap": sfm_dir_iphone,
            "in_rgb": rgb_dir_iphone,
            "in_mask": mask_dir_iphone,
        }

        mesh = pyrender.Mesh.from_trimesh(mesh_scene, smooth=False, material=pyrender.MetallicRoughnessMaterial(doubleSided=True))
        # mesh = pyrender.Mesh.from_trimesh(mesh_scene, smooth=False)
        pyrender_scene = pyrender.Scene()
        pyrender_scene.add(mesh)

        selection_dslr   = sorted([imgname for imgname in os.listdir(rgb_dir_dslr) if imgname.startswith('DSC')])
        selection_iphone = sorted([imgname[:-4] for imgname in os.listdir(rgb_dir_iphone) if imgname.startswith('frame_')]) if os.path.exists(rgb_dir_iphone) else []
        selection = selection_dslr + selection_iphone
        selection = [imgname[:-4] for imgname in selection]

        # resize the image to a more manageable size and render depth
        # for selection_cam, img_idx, img_infos, paths_data in [(selection_dslr, img_idx_dslr, img_infos_dslr, dslr_paths)]:  #,
        #                                                     #   (selection_iphone, img_idx_iphone, img_infos_iphone, iphone_paths)]:
        
        valid_imgname_list = []
        for selection_cam, img_idx, img_infos, paths_data in [(selection_iphone, img_idx_iphone, img_infos_iphone, iphone_paths)]:
            rgb_dir = paths_data['in_rgb']
            mask_dir = paths_data['in_mask']
            for imgname in tqdm(selection_cam, position=1, leave=False):

                if imgname not in img_idx:
                    continue
                valid_imgname_list.append(imgname)

                # imgidx = img_idx[imgname]
                img_infos_idx = img_infos[imgname]
                rgb = np.array(Image.open(os.path.join(rgb_dir, img_infos_idx['path'] + '.jpg')))
                mask = np.array(Image.open(os.path.join(mask_dir, img_infos_idx['path'] + '.png')))


                # _, _, K, rgb, mask = undistort_images(img_infos_idx['intrinsic'], rgb, mask)

                # # rescale_image_depthmap assumes opencv intrinsics
                intrinsics = img_infos_idx['intrinsic']
                image, mask, _, intrinsics = rescale_image_depthmap(
                    rgb, mask, None, intrinsics, (target_resolution, target_resolution * 3.0 / 4))

                W, H = image.size
                intrinsics = opencv_to_colmap_intrinsics(intrinsics)

                # update inpace img_infos_idx
                img_infos_idx['intrinsics'] = intrinsics
                rgb_outpath = os.path.join(output_dir_scene_rgb, img_infos_idx['path'] + '.webp')
                image.save(rgb_outpath)

                depth_outpath = os.path.join(output_dir_scene_depth, img_infos_idx['path'] + '.png')
                # render depth image
                renderer.viewport_width, renderer.viewport_height = W, H
                fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
                camera = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy, znear=znear, zfar=zfar)
                camera_node = pyrender_scene.add(camera, pose=img_infos_idx['cam_to_world'] @ OPENGL_TO_OPENCV)

                # depth = renderer.render(pyrender_scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
                old_cache = renderer._renderer._program_cache
                renderer._renderer._program_cache = CustomShaderCache()
                normal, depth = renderer.render(pyrender_scene, flags=pyrender.RenderFlags.VERTEX_NORMALS)


                normal = (normal / 255) * 2 - 1
                normal = normal @ (img_infos_idx['cam_to_world'] @ OPENGL_TO_OPENCV)[:3, :3]
                normal = (normal + 1) / 2 * 255
                normal = (normal * (depth > 0)[:, :, None]).astype('uint8')

                renderer._renderer._program_cache = old_cache
                pyrender_scene.remove_node(camera_node)  # dont forget to remove camera

                depth = (depth * 1000).astype('uint16')
                # invalidate depth from mask before saving
                depth_mask = (mask < 255)
                depth[depth_mask] = 0
                Image.fromarray(depth).save(depth_outpath)

                normal_outpath = os.path.join(output_dir_scene_normal, img_infos_idx['path'] + '.webp')
                Image.fromarray(normal).save(normal_outpath)

                # Normalize depth image to range [0, 1]
                depth_normalized = depth / np.max(depth)
                colormap = cm.get_cmap('turbo')
                depth_colored = colormap(depth_normalized)
                depth_colored = (depth_colored[:, :, :3] * 255).astype('uint8')
                depth_colored_image = Image.fromarray(depth_colored)
                depth_outpath_vis = os.path.join(output_dir_scene_depth_vis, img_infos_idx['path'] + '.webp')
                depth_colored_image.save(depth_outpath_vis)


        if len(valid_imgname_list) == 0:
            print(f"Skipping {scene} as valid_imgname_list is empty")
            continue

        trajectories = []
        intrinsics = []
        for imgname in valid_imgname_list:
            if imgname.startswith('DSC'):
                # imgidx = img_idx_dslr[imgname + '.JPG']
                imgidx = img_idx_dslr[imgname]
                img_infos_idx = img_infos_dslr[imgidx]
            elif imgname.startswith('frame_'):
                # imgidx = img_idx_iphone[imgname + '.jpg']
                # imgidx = img_idx_iphone[imgname]
                img_infos_idx = img_infos_iphone[imgname]
            else:
                raise ValueError('invalid image name')

            intrinsics.append(img_infos_idx['intrinsics'])
            trajectories.append(img_infos_idx['cam_to_world'])

        intrinsics = np.stack(intrinsics, axis=0)
        trajectories = np.stack(trajectories, axis=0)
        # save metadata for this scene
        np.savez(scene_metadata_path,
                 trajectories=trajectories,
                 intrinsics=intrinsics,
                 images=valid_imgname_list)

        del img_infos
        del pyrender_scene

    if False:
        # concat all scene_metadata.npz into a single file
        scene_data = {}
        for scene_subdir in scenes:
            scene_metadata_path = osp.join(output_dir, scene_subdir, 'scene_metadata.npz')
            with np.load(scene_metadata_path) as data:
                trajectories = data['trajectories']
                intrinsics = data['intrinsics']
                images = data['images']
            scene_data[scene_subdir] = {'trajectories': trajectories,
                                        'intrinsics': intrinsics,
                                        'images': images}

        offset = 0
        counts = []
        scenes = []
        sceneids = []
        images = []
        intrinsics = []
        trajectories = []
        for scene_idx, (scene_subdir, data) in enumerate(scene_data.items()):
            num_imgs = data['images'].shape[0]

            scenes.append(scene_subdir)
            sceneids.extend([scene_idx] * num_imgs)

            images.append(data['images'])

            intrinsics.append(data['intrinsics'])
            trajectories.append(data['trajectories'])
            counts.append(offset)

            offset += num_imgs

        images = np.concatenate(images, axis=0)
        intrinsics = np.concatenate(intrinsics, axis=0)
        trajectories = np.concatenate(trajectories, axis=0)
        np.savez(osp.join(output_dir, 'all_metadata.npz'),
                counts=counts,
                scenes=scenes,
                sceneids=sceneids,
                images=images,
                intrinsics=intrinsics,
                trajectories=trajectories)
        print('all done')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.pyopengl_platform.strip():
        os.environ['PYOPENGL_PLATFORM'] = args.pyopengl_platform
    process_scenes(args.scannetpp_dir, args.precomputed_pairs, args.output_dir, args.target_resolution)
