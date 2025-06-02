import numpy as np
import torch
from metrics.pcd_alignment import Regr3D_t_ScaleShiftInv
import open3d as o3d
from metrics.utils import accuracy, completion
from copy import deepcopy
# from pytorch3d.ops import sample_farthest_points

### https://github.com/CUT3R/CUT3R/blob/main/eval/mv_recon/launch.py
def pcd_evaluation(
    predicted_pcd_original,
    ground_truth_pcd_original,
    masks,
    rgbs=None,
    threshold = 0.1,
    downsample_num = -1,
):
    ### torch.tensor, global coordinate
    ### predicted_pcd_original: [Nf,H,W,3]
    ### ground_truth_pcd_original: [Nf,H,W,3]
    ### masks: [Nf, H, W], bool
    ### rgbs: [Nf, H, W, 3], float, 0~1
    criterion = Regr3D_t_ScaleShiftInv(norm_mode=False, gt_scale=True)

    #### modify input here
    predicted_pcd_original = [x[None] for x in predicted_pcd_original]  # -> list of [Bs, H, W, 3]
    ground_truth_pcd_original = [x[None] for x in ground_truth_pcd_original]  # -> list of [Bs, H, W, 3]
    masks = [x[None] for x in masks]  # -> list of [Bs, H, W]
    rgbs = [x[None] for x in rgbs] # -> list of [Bs, H, W, 3]


    gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = criterion.get_all_pts3d_t(gt_pts3d=ground_truth_pcd_original, 
                                   pred_pts3d=predicted_pcd_original,
                                   gt_masks=masks)

    ### extract align parameters
    pred_scale, gt_scale, pred_shift_z, gt_shift_z = (
        monitoring["pred_scale"],
        monitoring["gt_scale"],
        monitoring["pred_shift_z"],
        monitoring["gt_shift_z"],
    )

    ### ？？？
    pts_all = []
    pts_gt_all = []
    masks_all = []

    for j, _ in enumerate(predicted_pcd_original):

        mask = masks[j].cpu().numpy()[0]

        pts = predicted_pcd_original[j].cpu().numpy()[0]
        pts_gt = ground_truth_pcd_original[j].detach().cpu().numpy()[0]

        # H, W = image.shape[:2]
        # cx = W // 2
        # cy = H // 2
        # l, t = cx - 112, cy - 112
        # r, b = cx + 112, cy + 112
        # image = image[t:b, l:r]
        # mask = mask[t:b, l:r]
        # pts = pts[t:b, l:r]
        # pts_gt = pts_gt[t:b, l:r]

        #### Align predicted 3D points to the ground truth
        pts[..., -1] += gt_shift_z.cpu().numpy().item()

        pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()

        pts_all.append(pts[None, ...])
        pts_gt_all.append(pts_gt[None, ...])
        masks_all.append(mask[None, ...])

    pts_all = np.concatenate(pts_all, axis=0)
    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
    masks_all = np.concatenate(masks_all, axis=0)
    images_all = torch.cat(rgbs, dim=0).cpu().numpy()

    pts_all_masked = pts_all[masks_all > 0]
    pts_gt_all_masked = pts_gt_all[masks_all > 0]
    images_all_masked = images_all[masks_all > 0]

    ##### farthest point sampling to accelerate
    if downsample_num > 0:
        # pts_all_masked = fps_pytorch3d(pts_all_masked, num_samples=downsample_num)
        # pts_gt_all_masked = fps_pytorch3d(pts_gt_all_masked, num_samples=downsample_num)

        if False:
            images_all_masked_bak = images_all_masked.copy()

            pts_all_masked, images_all_masked_pred = fps_pytorch3d_with_color(pts_all_masked, images_all_masked_bak, num_samples=downsample_num)
            pts_gt_all_masked, images_all_masked_gt = fps_pytorch3d_with_color(pts_gt_all_masked, images_all_masked_bak, num_samples=downsample_num)
        else:
            mask_indices = np.random.choice(
                pts_all_masked.shape[0], downsample_num, replace=False
            )
            pts_all_masked = pts_all_masked[mask_indices]
            pts_gt_all_masked = pts_gt_all_masked[mask_indices]
            images_all_masked_pred = images_all_masked_gt = images_all_masked[mask_indices]


    ##### do icp
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        pts_all_masked.reshape(-1, 3)
    )
    pcd.colors = o3d.utility.Vector3dVector(
        images_all_masked_pred.reshape(-1, 3)
    )

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(
        pts_gt_all_masked.reshape(-1, 3)
    )
    pcd_gt.colors = o3d.utility.Vector3dVector(
        images_all_masked_gt.reshape(-1, 3)
    )


    ##### record the point cloud before icp
    result = {
        "pred_pcd": deepcopy(pcd),
        "gt_pcd": deepcopy(pcd_gt),
    }


    trans_init = np.eye(4)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd,
        pcd_gt,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    transformation = reg_p2p.transformation

    pcd = pcd.transform(transformation)
    pcd.estimate_normals()
    pcd_gt.estimate_normals()

    gt_normal = np.asarray(pcd_gt.normals)
    pred_normal = np.asarray(pcd.normals)

    acc, acc_med, nc1, nc1_med = accuracy(
        pcd_gt.points, pcd.points, gt_normal, pred_normal
    )
    comp, comp_med, nc2, nc2_med = completion(
        pcd_gt.points, pcd.points, gt_normal, pred_normal
    )
    # print(
    #     f"Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
    # )

    result.update({
                    "acc": acc,
                    "comp": comp,
                    "nc1": nc1,
                    "nc2": nc2,
                    "acc_med": acc_med,
                    "comp_med": comp_med,
                    "nc1_med": nc1_med,
                    "nc2_med": nc2_med,
                })

    return result



def fps_pytorch3d(pcd, num_samples):
    # pcd: [N,3], np.array
    # num_samples: int

    points = torch.from_numpy(pcd).float().cuda()
    points = points.unsqueeze(0)  # [1,N,3]
    sampled_points, sampled_indices = sample_farthest_points(points, K=num_samples)
    return sampled_points.squeeze(0).cpu().numpy()

def fps_pytorch3d_with_color(pcd, color, num_samples):
    # pcd: [N,3], np.array
    # color: [N,3], np.array
    # num_samples: int

    points = torch.from_numpy(pcd).float().cuda()
    color = torch.from_numpy(color).float().cuda()
    points = points.unsqueeze(0)  # [1,N,3]
    color = color.unsqueeze(0)  # [1,N,3]
    sampled_points, sampled_indices = sample_farthest_points(points, K=num_samples)
    sampled_color = color.gather(1, sampled_indices.unsqueeze(-1).expand(-1, -1, 3))  # [1, num_samples, 3]
    # sampled_color = color[:, sampled_indices]  # [1, 1, num_samples, 3]
    return sampled_points.squeeze(0).cpu().numpy(), sampled_color.squeeze(0).cpu().numpy()
    
    



if __name__ == "__main__":
    gt_pcd = [torch.randn(1, 480, 640, 3)]
    pred_pcd = [torch.randn(1, 480, 640, 3)]
    masks = [torch.ones(1, 480, 640).bool()]
    import time
    t1 = time.time()
    res = pcd_evaluation(predicted_pcd_original=pred_pcd, ground_truth_pcd_original=gt_pcd, masks=masks)
    t2 = time.time()
    print(t2 - t1)
    print(res)

    # gt_pcd2 = [x * 10 for x in gt_pcd]
    # pred_pcd2 = [x * 10 for x in pred_pcd]
    res = pcd_evaluation(predicted_pcd_original=pred_pcd, ground_truth_pcd_original=gt_pcd, masks=masks, downsample_num=-1)
    t3 = time.time()
    print(t3 - t2)
    print(res)
