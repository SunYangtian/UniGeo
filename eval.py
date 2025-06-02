import os
import yaml
from utils.io_utils import prepare_gt_label
from configs.config_utils import parse_dataset_config, import_class_from_module, parse_metric_config
from metrics import MetricsManager, depth_evaluation, pcd_evaluation, camera_pose_evaluation
from utils.vis_utils import save_point_cloud, save_depth_normal_maps
import torch

if __name__ == "__main__":
    config_path = "configs/spann3r_7scenes.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ### dataset
    dataset_cls = import_class_from_module("dataset", config["dataset"])
    dataset = dataset_cls(**parse_dataset_config(config))

    ### model
    model_cls = import_class_from_module("model", config["model_name"])
    model = model_cls(**config["model_params"])

    ### metrics
    metric_names = parse_metric_config(config)
    metrics_manager = MetricsManager(metric_names=metric_names)

    save_dir = "./debug_output"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"metrics.csv")


    for data_idx, data in enumerate(dataset):

        seq = f"{data_idx:03d}_{data['scene_name']}"
        print("processing seq:", seq)

        # Forward pass through the model
        output = model.forward(data)

        gt_label = prepare_gt_label(data)  # in opencv coordinates

        ### metric
        metric = {"seq_name" : seq}

        breakpoint()
        ### depth
        alignment = config["eval_depth"]["depth_alignment"]
        res = depth_evaluation(predicted_depth_original=output["pred_depths"], ground_truth_depth_original=gt_label["gt_depths"], custom_mask=gt_label["gt_masks"], align_with_lstsq=True)
        print(res[0])
        metric.update(res[0])

        ### point cloud
        pcd_eval_res = pcd_evaluation(
            predicted_pcd_original=output["pred_world_pts"],
            ground_truth_pcd_original=gt_label["gt_world_pts"],
            masks=gt_label["gt_masks"],
            rgbs=gt_label["gt_rgbs"],
            downsample_num=config["eval_pcd"]["pcd_downsample_num"],
        )
        print(pcd_eval_res)
        metric.update(pcd_eval_res)

        ### camera pose
        ate, rpe_trans, rpe_rot = camera_pose_evaluation(pred_pose=output["pred_poses"], gt_pose=gt_label["gt_poses"])


        print(f"| ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n")
        pose_eval_res = {
            "ATE": ate,
            "RPE trans" : rpe_trans,
            "RPE rot" : rpe_rot
        }
        metric.update(pose_eval_res)

        ######## update metric
        metrics_manager.update_metrics(metric)
        metrics_manager.export_to_csv(save_path)


        ### visualization
        depth_save_dir = os.path.join(save_dir, "depth_maps")
        os.makedirs(depth_save_dir, exist_ok=True)
        pcd_save_dir = os.path.join(save_dir, "point_clouds")
        os.makedirs(pcd_save_dir, exist_ok=True)

        save_depth_normal_maps(output["pred_depths"], output["pred_normals"], depth_save_dir, rgbs=gt_label["gt_rgbs"])

        pred_pcd, gt_pcd = pcd_eval_res['pred_pcd'], pcd_eval_res['gt_pcd']
        save_point_cloud(pred_pcd, os.path.join(pcd_save_dir, f"pred.ply"))
        save_point_cloud(gt_pcd, os.path.join(pcd_save_dir, f"gt.ply"))

        breakpoint()

