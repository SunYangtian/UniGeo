import open3d as o3d
import numpy as np


def align_pcd(points_pred, points_gt, threshold = 0.1):
    # align pcd1 to pcd2
    # points_pred: [N,3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_pred)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(points_gt)

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

    return np.asarray(pcd.points), transformation