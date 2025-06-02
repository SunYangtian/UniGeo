import numpy as np
import torch

from .alignment import *

def depth_evaluation(
    predicted_depth_original,
    ground_truth_depth_original,
    max_depth=80,
    custom_mask=None,
    post_clip_min=None,
    post_clip_max=None,
    pre_clip_min=None,
    pre_clip_max=None,
    align_with_lstsq=False,
    align_with_lad=False,
    align_with_lad2=False,
    metric_scale=False,
    lr=1e-4,
    max_iters=1000,
    use_gpu=False,
    align_with_scale=False,
    disp_input=False,
):
    """
    Evaluate the depth map.

    Args:
        predicted_depth (numpy.ndarray or torch.Tensor): The predicted depth map. [H,w] or [Nf,H,W]
        ground_truth_depth (numpy.ndarray or torch.Tensor): The ground truth depth map. [H,w] or [Nf,H,W]
        custom_mask: [H,w] or [Nf,H,W]
        max_depth (float): The maximum depth value to consider. Default is 80 meters.
        align_with_lstsq (bool): If True, perform least squares alignment of the predicted depth with ground truth.

    Returns:
        dict: A dictionary containing the evaluation metrics.
        torch.Tensor: The depth error parity map.
    """
    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original)
    if custom_mask is not None and isinstance(custom_mask, np.ndarray):
        custom_mask = torch.from_numpy(custom_mask)

    # if the dimension is 3, flatten to 2d along the batch dimension
    if predicted_depth_original.dim() == 3:
        _, h, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.view(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.view(-1, w)
        if custom_mask is not None:
            custom_mask = custom_mask.view(-1, w)

    # put to device
    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()

    # Filter out depths greater than max_depth
    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (
            ground_truth_depth_original < max_depth
        )
    else:
        mask = ground_truth_depth_original > 0
    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]

    # Clip the depth values
    if pre_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=pre_clip_min)
    if pre_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=pre_clip_max)

    if disp_input:  # align the pred to gt in the disparity space
        real_gt = ground_truth_depth.clone()
        ground_truth_depth = 1 / (ground_truth_depth + 1e-8)

    # various alignment methods
    if metric_scale:
        predicted_depth = predicted_depth
    elif align_with_lstsq:
        s, t = align_with_lstsq_torch(predicted_depth=predicted_depth, ground_truth_depth=ground_truth_depth)

        # Apply scale and shift
        predicted_depth = s * predicted_depth + t
    elif align_with_lad:
        s, t = absolute_value_scaling_torch(
            predicted_depth,
            ground_truth_depth,
            s=torch.median(ground_truth_depth) / torch.median(predicted_depth),
        )
        predicted_depth = s * predicted_depth + t
    elif align_with_lad2:
        s_init = (
            torch.median(ground_truth_depth) / torch.median(predicted_depth)
        ).item()
        s, t = absolute_value_scaling2_torch(
            predicted_depth,
            ground_truth_depth,
            s_init=s_init,
            lr=lr,
            max_iters=max_iters,
        )
        predicted_depth = s * predicted_depth + t
    elif align_with_scale:
        s = align_with_scale_torch(predicted_depth=predicted_depth, ground_truth_depth=ground_truth_depth)

        # Optionally clip 's' to prevent extreme scaling
        s = s.clamp(min=1e-3)

        # Detach 's' if you want to stop gradients from flowing through it
        s = s.detach()

        # Apply the scale factor to the predicted depth
        predicted_depth = s * predicted_depth

    else:
        # Align the predicted depth with the ground truth using median scaling
        scale_factor = torch.median(ground_truth_depth) / torch.median(predicted_depth)
        predicted_depth *= scale_factor

    if disp_input:
        # convert back to depth
        ground_truth_depth = real_gt
        predicted_depth = depth2disparity(predicted_depth)

    # Clip the predicted depth values
    if post_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=post_clip_min)
    if post_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=post_clip_max)

    if custom_mask is not None:
        assert custom_mask.shape == ground_truth_depth_original.shape
        mask_within_mask = custom_mask.cpu()[mask]
        predicted_depth = predicted_depth[mask_within_mask]
        ground_truth_depth = ground_truth_depth[mask_within_mask]

    # Calculate the metrics
    abs_rel = torch.mean(
        torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth
    ).item()
    sq_rel = torch.mean(
        ((predicted_depth - ground_truth_depth) ** 2) / ground_truth_depth
    ).item()

    # Correct RMSE calculation
    rmse = torch.sqrt(torch.mean((predicted_depth - ground_truth_depth) ** 2)).item()

    # Clip the depth values to avoid log(0)
    predicted_depth = torch.clamp(predicted_depth, min=1e-5)
    log_rmse = torch.sqrt(
        torch.mean((torch.log(predicted_depth) - torch.log(ground_truth_depth)) ** 2)
    ).item()

    # Calculate the accuracy thresholds
    max_ratio = torch.maximum(
        predicted_depth / ground_truth_depth, ground_truth_depth / predicted_depth
    )
    threshold_0 = torch.mean((max_ratio < 1.0).float()).item()
    threshold_1 = torch.mean((max_ratio < 1.25).float()).item()
    threshold_2 = torch.mean((max_ratio < 1.25**2).float()).item()
    threshold_3 = torch.mean((max_ratio < 1.25**3).float()).item()

    # Compute the depth error parity map
    if metric_scale:
        predicted_depth_original = predicted_depth_original
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / ground_truth_depth_original
        )
    elif align_with_lstsq or align_with_lad or align_with_lad2:
        predicted_depth_original = predicted_depth_original * s + t
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / ground_truth_depth_original
        )
    elif align_with_scale:
        predicted_depth_original = predicted_depth_original * s
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / ground_truth_depth_original
        )
    else:
        predicted_depth_original = predicted_depth_original * scale_factor
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / ground_truth_depth_original
        )

    # Reshape the depth_error_parity_map back to the original image size
    depth_error_parity_map_full = torch.zeros_like(ground_truth_depth_original)
    depth_error_parity_map_full = torch.where(
        mask, depth_error_parity_map, depth_error_parity_map_full
    )

    predict_depth_map_full = predicted_depth_original
    gt_depth_map_full = torch.zeros_like(ground_truth_depth_original)
    gt_depth_map_full = torch.where(
        mask, ground_truth_depth_original, gt_depth_map_full
    )

    num_valid_pixels = (
        torch.sum(mask).item()
        if custom_mask is None
        else torch.sum(mask_within_mask).item()
    )
    if num_valid_pixels == 0:
        (
            abs_rel,
            sq_rel,
            rmse,
            log_rmse,
            threshold_0,
            threshold_1,
            threshold_2,
            threshold_3,
        ) = (0, 0, 0, 0, 0, 0, 0, 0)

    results = {
        "Abs Rel": abs_rel,
        "Sq Rel": sq_rel,
        "RMSE": rmse,
        "Log RMSE": log_rmse,
        "delta < 1.": threshold_0,
        "delta < 1.25": threshold_1,
        "delta < 1.25^2": threshold_2,
        "delta < 1.25^3": threshold_3,
        "valid_pixels": num_valid_pixels,
    }

    return (
        results,
        depth_error_parity_map_full,
        predict_depth_map_full,
        gt_depth_map_full,
    )



def depth_evaluation_in_global_coord(
    predicted_depth_original,
    ground_truth_depth_original,
    ground_truth_radius,
    cam2world,
    intrinsics,
    max_depth=80,
    custom_mask=None,
    post_clip_min=None,
    post_clip_max=None,
    pre_clip_min=None,
    pre_clip_max=None,
    align_with_lstsq=False,
    align_with_lad=False,
    align_with_lad2=False,
    metric_scale=False,
    lr=1e-4,
    max_iters=1000,
    use_gpu=False,
    align_with_scale=False,
    disp_input=False,
):
    """
    Evaluate the depth map using various metrics and return a depth error parity map, with an option for least squares alignment.

    Args:
        predicted_depth (numpy.ndarray or torch.Tensor): The predicted depth map. [H,w] or [Nf,H,W]
        ground_truth_depth (numpy.ndarray or torch.Tensor): The ground truth depth map. [H,w] or [Nf,H,W]
        ground_truth_radius (numpy.ndarray or torch.Tensor): The radius of the ground truth points. [H,w] or [Nf,H,W]
        cam2world: np.array c2v, opencv coord. [Nf,4,4]
        intrinsics: np.array, [Nf,3,3]
        custom_mask: [H,w] or [Nf,H,W]
        max_depth (float): The maximum depth value to consider. Default is 80 meters.
        align_with_lstsq (bool): If True, perform least squares alignment of the predicted depth with ground truth.

    Returns:
        dict: A dictionary containing the evaluation metrics.
        torch.Tensor: The depth error parity map.
    """
    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original)
    if custom_mask is not None and isinstance(custom_mask, np.ndarray):
        custom_mask = torch.from_numpy(custom_mask)

    if isinstance(ground_truth_radius, np.ndarray):
        ground_truth_radius = torch.from_numpy(ground_truth_radius)

    predicted_depth_original_nhw = predicted_depth_original.clone()
    # if the dimension is 3, flatten to 2d along the batch dimension
    if predicted_depth_original.dim() == 3:
        _, h, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.view(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.view(-1, w)
        if custom_mask is not None:
            custom_mask = custom_mask.view(-1, w)

    # put to device
    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()

    # Filter out depths greater than max_depth
    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (
            ground_truth_depth_original < max_depth
        )
    else:
        mask = ground_truth_depth_original > 0
    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]

    # Clip the depth values
    if pre_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=pre_clip_min)
    if pre_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=pre_clip_max)

    if disp_input:  # align the pred to gt in the disparity space
        real_gt = ground_truth_depth.clone()
        ground_truth_depth = 1 / (ground_truth_depth + 1e-8)

    # various alignment methods
    assert align_with_lstsq
    s, t = align_with_lstsq_torch(predicted_depth=predicted_depth, ground_truth_depth=ground_truth_depth)

    # Apply scale and shift
    predicted_depth_original_nhw = s * predicted_depth_original_nhw + t

    # Clip the predicted depth values
    if post_clip_min is not None:
        predicted_depth_original_nhw = torch.clamp(predicted_depth_original_nhw, min=post_clip_min)
    if post_clip_max is not None:
        predicted_depth_original_nhw = torch.clamp(predicted_depth_original_nhw, max=post_clip_max)

    ############## coordinate transform ##############
    from geometry_utils import backproject_to_cv_position
    cam_pts3d = [backproject_to_cv_position(x.numpy(), intrinsics[i]) for i,x in enumerate(predicted_depth_original_nhw)]  # list of [H, W, 3]
    world_pts3d = [
        (cam_pts3d[i].reshape(-1,3) @ cam2world[i][:3,:3].T + cam2world[i][:3,3][:,None].T).reshape(cam_pts3d[i].shape)
        for i in range(len(cam_pts3d))]  # list of [H, W, 3]

    predicted_radius = np.linalg.norm(np.stack(world_pts3d, axis=0), axis=-1)  # [Nf, H, W]
    predicted_radius = torch.from_numpy(predicted_radius).float()  # [Nf, H, W]
    predicted_radius_original_nhw = predicted_radius.clone()
    ground_truth_radius_original_nhw = ground_truth_radius.clone()


    assert predicted_radius.dim() == 3
    predicted_radius = predicted_radius.view(-1, w)
    ground_truth_radius = ground_truth_radius.view(-1, w)

    predicted_radius = predicted_radius[mask]
    ground_truth_radius = ground_truth_radius[mask]

    assert align_with_lstsq
    s, t = align_with_lstsq_torch(predicted_depth=predicted_radius, ground_truth_depth=    ground_truth_radius
)
    # Apply scale and shift
    predicted_radius_aligned_nhw = s * predicted_radius_original_nhw + t
    predicted_radius_nhw_output = predicted_radius_aligned_nhw.clone().view(predicted_depth_original_nhw.shape)

    # ### replace
    predicted_depth = predicted_radius_aligned_nhw.view(-1, w)[mask]
    ground_truth_depth = ground_truth_radius_original_nhw.view(-1, w)[mask]
    #################################################

    if custom_mask is not None:
        assert custom_mask.shape == ground_truth_depth_original.shape
        mask_within_mask = custom_mask.cpu()[mask]
        predicted_depth = predicted_depth[mask_within_mask]
        ground_truth_depth = ground_truth_depth[mask_within_mask]

    # Calculate the metrics
    abs_rel = torch.mean(
        torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth
    ).item()
    sq_rel = torch.mean(
        ((predicted_depth - ground_truth_depth) ** 2) / ground_truth_depth
    ).item()

    # Correct RMSE calculation
    rmse = torch.sqrt(torch.mean((predicted_depth - ground_truth_depth) ** 2)).item()

    # Clip the depth values to avoid log(0)
    predicted_depth = torch.clamp(predicted_depth, min=1e-5)
    log_rmse = torch.sqrt(
        torch.mean((torch.log(predicted_depth) - torch.log(ground_truth_depth)) ** 2)
    ).item()

    # Calculate the accuracy thresholds
    max_ratio = torch.maximum(
        predicted_depth / ground_truth_depth, ground_truth_depth / predicted_depth
    )
    threshold_0 = torch.mean((max_ratio < 1.0).float()).item()
    threshold_1 = torch.mean((max_ratio < 1.25).float()).item()
    threshold_2 = torch.mean((max_ratio < 1.25**2).float()).item()
    threshold_3 = torch.mean((max_ratio < 1.25**3).float()).item()

    num_valid_pixels = (
        torch.sum(mask).item()
        if custom_mask is None
        else torch.sum(mask_within_mask).item()
    )
    if num_valid_pixels == 0:
        (
            abs_rel,
            sq_rel,
            rmse,
            log_rmse,
            threshold_0,
            threshold_1,
            threshold_2,
            threshold_3,
        ) = (0, 0, 0, 0, 0, 0, 0, 0)

    results = {
        "Abs Rel": abs_rel,
        "Sq Rel": sq_rel,
        "RMSE": rmse,
        "Log RMSE": log_rmse,
        "delta < 1.": threshold_0,
        "delta < 1.25": threshold_1,
        "delta < 1.25^2": threshold_2,
        "delta < 1.25^3": threshold_3,
        "valid_pixels": num_valid_pixels,
    }

    return (
        results, predicted_radius_nhw_output
    )


if __name__ == "__main__":
    gt_depth = np.random.randn(480, 640)
    pred_depth = np.random.randn(480, 640)

    res = depth_evaluation(predicted_depth_original=pred_depth, ground_truth_depth_original=gt_depth, align_with_scale=True)

    # default: median scaling
    # possible options: metric_scale / align_with_lstsq / align_with_lad / align_with_lad2 / align_with_scale

    print(res[0])