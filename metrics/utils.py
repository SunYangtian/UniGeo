import numpy as np
from scipy.spatial import cKDTree as KDTree
import torch
import cv2


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)

    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)

        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)

        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median


def compute_iou(pred_vox, target_vox):
    # Get voxel indices
    v_pred_indices = [voxel.grid_index for voxel in pred_vox.get_voxels()]
    v_target_indices = [voxel.grid_index for voxel in target_vox.get_voxels()]

    # Convert to sets for set operations
    v_pred_filled = set(tuple(np.round(x, 4)) for x in v_pred_indices)
    v_target_filled = set(tuple(np.round(x, 4)) for x in v_target_indices)

    # Compute intersection and union
    intersection = v_pred_filled & v_target_filled
    union = v_pred_filled | v_target_filled

    # Compute IoU
    iou = len(intersection) / len(union)
    return iou



##### solve depth and camera from 3D points

from metrics.geometry import xy_grid, geotrf

def estimate_focal_knowing_depth(pts3d, pp, focal_mode='median', min_focal=0., max_focal=np.inf):
    """ Reprojection method, for when the absolute depth is known:
        1) estimate the camera focal using a robust estimator
        2) reproject points onto true rays, minimizing a certain error
    """
    # pts3d: [B, H, W, 3]
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    # centered pixel grid
    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)

    if focal_mode == 'median':
        with torch.no_grad():
            # direct estimation of focal
            u, v = pixels.unbind(dim=-1)
            x, y, z = pts3d.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-1).values

    elif focal_mode == 'weiszfeld':
        # init focal with l2 closed form
        # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|
        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(posinf=0, neginf=0)  # homogeneous (x,y,1)

        dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)

        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip(min=1e-8).reciprocal()
            # update the scaling with the new weights
            focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)
    else:
        raise ValueError(f'bad {focal_mode=}')

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal*focal_base, max=max_focal*focal_base)
    # print(focal)
    return focal


def solve_depth_and_camera_from_3d_points(pts3d_list,):
    ### pts: list of [Bs, H,W,3], torch.tensor, global coordinate
    ### output: 
    ### cam_coord_list: list of [H,W,3], numpy, camera coordinate
    ### extrinsic_list: list of [4,4], numpy, world to camera
    ### intrinsic_list: list of [3,3], numpy, camera intrinsic
    H, W = pts3d_list[0].shape[1:3]
    pp = torch.tensor((W/2, H/2))

    ### assert first frame coord is world coord
    focal = estimate_focal_knowing_depth(pts3d_list[0].cpu(), pp, focal_mode='weiszfeld')
    intrinsic = np.eye(3)
    intrinsic[0, 0] = focal
    intrinsic[1, 1] = focal
    intrinsic[:2, 2] = pp

    cam_coord_list = []
    extrinsic_list, intrinsic_list = [], []
    for j, pts in enumerate(pts3d_list):  
        pts = pts.cpu().numpy()  # [1, H, W, 3]     
        ##### Solve PnP-RANSAC
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        points_2d = np.stack((u, v), axis=-1)
        dist_coeffs = np.zeros(4).astype(np.float32)
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            pts.reshape(-1, 3).astype(np.float32), 
            points_2d.reshape(-1, 2).astype(np.float32), 
            intrinsic.astype(np.float32), 
            dist_coeffs)
    
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Extrinsic parameters (4x4 matrix)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
        extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))
        cam_pts3d = geotrf(extrinsic_matrix , pts)
        cam_coord_list.append(cam_pts3d)
        extrinsic_list.append(extrinsic_matrix)
        intrinsic_list.append(intrinsic)

    return cam_coord_list, extrinsic_list, intrinsic_list




############ camera pose utils ###################
from scipy.spatial.transform import Rotation
from metrics.device import to_numpy

def c2w_to_tumpose(c2w):
    """
    Convert a camera-to-world matrix to a tuple of translation and rotation

    input: c2w: 4x4 matrix
    output: tuple of translation and rotation (x y z qw qx qy qz)
    """
    # convert input to numpy
    c2w = to_numpy(c2w)
    xyz = c2w[:3, -1]
    rot = Rotation.from_matrix(c2w[:3, :3])
    qx, qy, qz, qw = rot.as_quat()
    tum_pose = np.concatenate([xyz, [qw, qx, qy, qz]])
    return tum_pose


def get_tum_poses(poses):
    """
    poses: list of 4x4 arrays
    """
    tt = np.arange(len(poses)).astype(float)
    tum_poses = [c2w_to_tumpose(p) for p in poses]
    tum_poses = np.stack(tum_poses, 0)
    return [tum_poses, tt]