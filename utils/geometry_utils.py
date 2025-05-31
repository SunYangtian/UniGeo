import torch
from torch import nn

import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
import cv2

def get_surface_normal(xyz, patch_size=5):
    # xyz: [1, h, w, 3]
    x, y, z = torch.unbind(xyz, dim=3)
    x = torch.unsqueeze(x, 0)
    y = torch.unsqueeze(y, 0)
    z = torch.unsqueeze(z, 0)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    patch_weight = torch.ones((1, 1, patch_size, patch_size), requires_grad=False)
    xx_patch = nn.functional.conv2d(xx, weight=patch_weight, padding=int(patch_size / 2))
    yy_patch = nn.functional.conv2d(yy, weight=patch_weight, padding=int(patch_size / 2))
    zz_patch = nn.functional.conv2d(zz, weight=patch_weight, padding=int(patch_size / 2))
    xy_patch = nn.functional.conv2d(xy, weight=patch_weight, padding=int(patch_size / 2))
    xz_patch = nn.functional.conv2d(xz, weight=patch_weight, padding=int(patch_size / 2))
    yz_patch = nn.functional.conv2d(yz, weight=patch_weight, padding=int(patch_size / 2))
    ATA = torch.stack([xx_patch, xy_patch, xz_patch, xy_patch, yy_patch, yz_patch, xz_patch, yz_patch, zz_patch],
                      dim=4)
    ATA = torch.squeeze(ATA)
    ATA = torch.reshape(ATA, (ATA.size(0), ATA.size(1), 3, 3))
    eps_identity = 1e-6 * torch.eye(3, device=ATA.device, dtype=ATA.dtype)[None, None, :, :].repeat([ATA.size(0), ATA.size(1), 1, 1])
    ATA = ATA + eps_identity
    x_patch = nn.functional.conv2d(x, weight=patch_weight, padding=int(patch_size / 2))
    y_patch = nn.functional.conv2d(y, weight=patch_weight, padding=int(patch_size / 2))
    z_patch = nn.functional.conv2d(z, weight=patch_weight, padding=int(patch_size / 2))
    AT1 = torch.stack([x_patch, y_patch, z_patch], dim=4)
    AT1 = torch.squeeze(AT1)
    AT1 = torch.unsqueeze(AT1, 3)

    patch_num = 4
    patch_x = int(AT1.size(1) / patch_num)
    patch_y = int(AT1.size(0) / patch_num)
    n_img = torch.randn(AT1.shape)
    overlap = patch_size // 2 + 1
    for x in range(int(patch_num)):
        for y in range(int(patch_num)):
            left_flg = 0 if x == 0 else 1
            right_flg = 0 if x == patch_num -1 else 1
            top_flg = 0 if y == 0 else 1
            btm_flg = 0 if y == patch_num - 1 else 1
            at1 = AT1[y * patch_y - top_flg * overlap:(y + 1) * patch_y + btm_flg * overlap,
                  x * patch_x - left_flg * overlap:(x + 1) * patch_x + right_flg * overlap]
            ata = ATA[y * patch_y - top_flg * overlap:(y + 1) * patch_y + btm_flg * overlap,
                  x * patch_x - left_flg * overlap:(x + 1) * patch_x + right_flg * overlap]
            # n_img_tmp, _ = torch.solve(at1, ata)
            # n_img_tmp = torch.linalg.solve(ata, at1)
            n_img_tmp = torch.linalg.lstsq(ata, at1).solution

            n_img_tmp_select = n_img_tmp[top_flg * overlap:patch_y + top_flg * overlap, left_flg * overlap:patch_x + left_flg * overlap, :, :]
            n_img[y * patch_y:y * patch_y + patch_y, x * patch_x:x * patch_x + patch_x, :, :] = n_img_tmp_select

    n_img_L2 = torch.sqrt(torch.sum(n_img ** 2, dim=2, keepdim=True))
    n_img_norm = n_img / n_img_L2

    # re-orient normals consistently
    orient_mask = torch.sum(torch.squeeze(n_img_norm) * torch.squeeze(xyz), dim=2) > 0
    n_img_norm[orient_mask] *= -1
    return n_img_norm


def get_surface_normal_np(xyz, patch_size=5):
    # xyz: [h, w, 3]
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    patch_weight = np.ones((patch_size, patch_size)) / (patch_size ** 2)

    xx_patch = convolve2d(xx, patch_weight, mode='same')
    yy_patch = convolve2d(yy, patch_weight, mode='same')
    zz_patch = convolve2d(zz, patch_weight, mode='same')
    xy_patch = convolve2d(xy, patch_weight, mode='same')
    xz_patch = convolve2d(xz, patch_weight, mode='same')
    yz_patch = convolve2d(yz, patch_weight, mode='same')

    ATA = np.stack([xx_patch, xy_patch, xz_patch, xy_patch, yy_patch, yz_patch, xz_patch, yz_patch, zz_patch], axis=-1)
    ATA = ATA.reshape(*ATA.shape[:-1], 3, 3)
    eps_identity = 1e-6 * np.eye(3)[None, None, :, :]
    ATA += eps_identity

    x_patch = convolve2d(x, patch_weight, mode='same')
    y_patch = convolve2d(y, patch_weight, mode='same')
    z_patch = convolve2d(z, patch_weight, mode='same')
    AT1 = np.stack([x_patch, y_patch, z_patch], axis=-1).reshape(*x_patch.shape, 3, 1)

    patch_num = 4
    patch_x = int(AT1.shape[1] / patch_num)
    patch_y = int(AT1.shape[0] / patch_num)
    n_img = np.random.randn(*AT1.shape)
    overlap = patch_size // 2 + 1

    for x in range(patch_num):
        for y in range(patch_num):
            left_flg = 0 if x == 0 else 1
            right_flg = 0 if x == patch_num - 1 else 1
            top_flg = 0 if y == 0 else 1
            btm_flg = 0 if y == patch_num - 1 else 1

            at1 = AT1[y * patch_y - top_flg * overlap:(y + 1) * patch_y + btm_flg * overlap,
                      x * patch_x - left_flg * overlap:(x + 1) * patch_x + right_flg * overlap]
            ata = ATA[y * patch_y - top_flg * overlap:(y + 1) * patch_y + btm_flg * overlap,
                      x * patch_x - left_flg * overlap:(x + 1) * patch_x + right_flg * overlap]

            n_img_tmp = np.linalg.solve(ata, at1)

            n_img_tmp_select = n_img_tmp[top_flg * overlap:patch_y + top_flg * overlap, left_flg * overlap:patch_x + left_flg * overlap, :, :]
            n_img[y * patch_y:y * patch_y + patch_y, x * patch_x:x * patch_x + patch_x, :, :] = n_img_tmp_select

    n_img_L2 = np.sqrt(np.sum(n_img ** 2, axis=2, keepdims=True))
    n_img_norm = n_img / (n_img_L2 + 1e-6)

    # re-orient normals consistently
    orient_mask = np.sum(np.squeeze(n_img_norm) * np.squeeze(xyz), axis=2) > 0
    n_img_norm[orient_mask] *= -1

    return n_img_norm.squeeze().astype(np.float32)

def get_surface_normal_v2(depth_refine,fx=-1,fy=-1,cx=-1,cy=-1,bbox=np.array([0]),refine=True):
    # Copied from https://github.com/kirumang/Pix2Pose/blob/master/pix2pose_util/common_util.py
    '''
    fast normal computation. 
    depth_refine: [H, W]
    '''
    res_y = depth_refine.shape[0]
    res_x = depth_refine.shape[1]
    centerX=cx
    centerY=cy
    constant_x = 1/fx
    constant_y = 1/fy

    if(refine):
        depth_refine = np.nan_to_num(depth_refine)
        mask = np.zeros_like(depth_refine).astype(np.uint8)
        mask[depth_refine==0]=1
        depth_refine = depth_refine.astype(np.float32)
        depth_refine = cv2.inpaint(depth_refine,mask,2,cv2.INPAINT_NS)
        depth_refine = depth_refine.astype(np.float)
        depth_refine = ndimage.gaussian_filter(depth_refine,2)

    uv_table = np.zeros((res_y,res_x,2),dtype=np.int16)
    column = np.arange(0,res_y)
    uv_table[:,:,1] = np.arange(0,res_x) - centerX #x-c_x (u)
    uv_table[:,:,0] = column[:,np.newaxis] - centerY #y-c_y (v)

    if(bbox.shape[0]==4):
        uv_table = uv_table[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        v_x = np.zeros((bbox[2]-bbox[0],bbox[3]-bbox[1],3))
        v_y = np.zeros((bbox[2]-bbox[0],bbox[3]-bbox[1],3))
        normals = np.zeros((bbox[2]-bbox[0],bbox[3]-bbox[1],3))
        depth_refine=depth_refine[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    else:
        v_x = np.zeros((res_y,res_x,3))
        v_y = np.zeros((res_y,res_x,3))
        normals = np.zeros((res_y,res_x,3))
    
    uv_table_sign= np.copy(uv_table)
    uv_table=np.abs(np.copy(uv_table))

    
    dig=np.gradient(depth_refine,2,edge_order=2)
    v_y[:,:,0]=uv_table_sign[:,:,1]*constant_x*dig[0]
    v_y[:,:,1]=depth_refine*constant_y+(uv_table_sign[:,:,0]*constant_y)*dig[0]
    v_y[:,:,2]=dig[0]

    v_x[:,:,0]=depth_refine*constant_x+uv_table_sign[:,:,1]*constant_x*dig[1]
    v_x[:,:,1]=uv_table_sign[:,:,0]*constant_y*dig[1]
    v_x[:,:,2]=dig[1]

    cross = np.cross(v_x.reshape(-1,3),v_y.reshape(-1,3))
    norm = np.expand_dims(np.linalg.norm(cross,axis=1),axis=1)
    norm[norm==0]=1
    cross = cross/norm
    if(bbox.shape[0]==4):
        cross =cross.reshape((bbox[2]-bbox[0],bbox[3]-bbox[1],3))
    else:
        cross =cross.reshape(res_y,res_x,3)
    cross= np.nan_to_num(cross)
    return cross



def pose_distance(reference_pose, measurement_pose):
    """
    :param reference_pose: 4x4 numpy array, reference frame camera-to-world pose
        (not extrinsic matrix!)
    :param measurement_pose: 4x4 numpy array, measurement frame camera-to-world
        pose (not extrinsic matrix!)
    :return combined_measure: float, combined pose distance measure
    :return R_measure: float, rotation distance measure
    :return t_measure: float, translation distance measure
    """
    rel_pose = np.dot(np.linalg.inv(reference_pose), measurement_pose)
    R = rel_pose[:3, :3]
    t = rel_pose[:3, 3]
    R_measure = np.sqrt(2 * (1 - min(3.0, np.matrix.trace(R)) / 3))
    t_measure = np.linalg.norm(t)
    combined_measure = np.sqrt(t_measure**2 + R_measure**2)
    return combined_measure, R_measure, t_measure


def backproject(depth, intrinsic, opengl_coord=True):
    ### depth: [H, W]
    ### intrinsic: [3, 3]
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    z = depth
    x = (i - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (j - intrinsic[1, 2]) * z / intrinsic[1, 1]
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    if opengl_coord:
        points[:, 1:] *= -1
    return points.reshape(h, w, 3)




def fix_normal(normal, position):
    # normal: [3,H,W], float32, [-1,1]
    # position: [3,H,W], float32, from origin to point
    # return: [3,H,W], float32, [-1,1]
    # cos<normal, position> should <= 0
    position_direction = position / (np.linalg.norm(position, axis=0, keepdims=True) + 1e-6)
    mask = (normal * position_direction).sum(axis=0) > 0.01
    normal[:, mask] *= -1
    return normal


def backproject_to_cv_position(depth, intrinsic):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    z = depth
    x = (i - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (j - intrinsic[1, 2]) * z / intrinsic[1, 1]
    points = np.stack((x, y, z), axis=-1).reshape(h,w,3)
    return points



def crop_image_and_adjust_intrinsics(K, input_h, input_w, aspect_ratio):
    """
    计算满足宽高比的裁剪坐标，并调整内参矩阵。
    
    :param K: 3x3 内参矩阵 (numpy array)
    :param input_h: 输入图像高度
    :param input_w: 输入图像宽度
    :param aspect_ratio: 目标宽高比 (w/h)
    :return: (crop_x1, crop_y1, crop_x2, crop_y2), new_K
    """
    input_ratio = input_w / input_h

    if input_ratio > aspect_ratio:
        # 需要裁剪宽度
        new_w = int(input_h * aspect_ratio)
        new_h = input_h
        crop_x1 = (input_w - new_w) // 2
        crop_x2 = crop_x1 + new_w
        crop_y1 = 0
        crop_y2 = input_h
    else:
        # 需要裁剪高度
        new_h = int(input_w / aspect_ratio)
        new_w = input_w
        crop_y1 = (input_h - new_h) // 2
        crop_y2 = crop_y1 + new_h
        crop_x1 = 0
        crop_x2 = input_w

    # 计算新的内参矩阵
    new_K = K.copy()
    new_K[0, 2] -= crop_x1  # 调整 cx
    new_K[1, 2] -= crop_y1  # 调整 cy

    return (crop_x1, crop_y1, crop_x2, crop_y2), new_K