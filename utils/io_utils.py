import numpy as np
import torch

def prepare_gt_label(data):

    ### opengl to opencv
    OPENGL_TO_OPENCV = np.float32([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])

    num_views = len(data['images'])

    gt_world_pts, gt_masks, gt_poses, gt_depths = [], [], [], []
    gt_rgbs = []

    for i in range(num_views):
        extrinsic = data["extrinsics"][i].astype(np.float32)
        camera_pose = np.linalg.inv(extrinsic)  # [4,4], camera2world
        camera_pose = np.einsum('ij,jk,kl->il', OPENGL_TO_OPENCV, camera_pose, OPENGL_TO_OPENCV)

        pts3d = data["world_coord"][i].astype(np.float32)  # [3,h,w]
        pts3d[1:] *= -1  # opengl to opencv

        cam_pts3d = data["cam_coord"][i].astype(np.float32)  # [3,h,w]
        cam_pts3d[1:] *= -1  # opengl to opencv

        # -----
        gt_world_pts.append(torch.from_numpy(pts3d).permute(1,2,0).unsqueeze(0))  # [1, H, W, 3]
        gt_masks.append(torch.from_numpy(data["mask"][i]).unsqueeze(0).bool())  # [1, H, W]
        gt_poses.append(torch.from_numpy(camera_pose).unsqueeze(0))  # [1, 4, 4]
        gt_depths.append(torch.from_numpy(cam_pts3d).permute(1,2,0).unsqueeze(0)[..., -1])  # [1, H, W]
        gt_rgbs.append(torch.from_numpy(data['images'][i]).permute(2, 0, 1).unsqueeze(0))  # [1, 3, H, W]
        # -----

    gt_label = {
        "gt_world_pts": gt_world_pts,  # list of [1, H, W, 3]
        "gt_masks": gt_masks,  # list of [1, H, W]
        "gt_poses": torch.cat(gt_poses, 0),  # [B, 4, 4]
        "gt_depths": torch.cat(gt_depths, 0),  # [B, H, W]
        "gt_rgbs": torch.cat(gt_rgbs, 0),  # [B, 3, H, W]
    }

    return gt_label