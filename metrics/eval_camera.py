import numpy as np
import torch

from .alignment import *
from metrics.utils import get_tum_poses
from metrics.evo_utils import eval_metrics
from metrics.evo_utils import plot_trajectory

def camera_pose_evaluation(
    pred_pose,
    gt_pose,
):
    # pred_pose: [B, 4, 4], camera to world
    # gt_pose: [B, 4, 4], camera to world

    pred_traj = get_tum_poses(pred_pose)
    gt_traj = get_tum_poses(gt_pose)

    ate, rpe_trans, rpe_rot = eval_metrics(
        pred_traj,
        gt_traj,
    )

    # plot_trajectory(pred_traj, gt_traj, title=seq, filename=f"{save_dir}/{seq}.png")

    return ate, rpe_trans, rpe_rot



if __name__ == "__main__":
    pred_pose = torch.randn(10, 4, 4)
    gt_pose = torch.randn(10, 4, 4)
    ate, rpe_trans, rpe_rot = camera_pose_evaluation(pred_pose, gt_pose)

    print(f"ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n")