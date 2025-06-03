import numpy as np
import torch

def compute_normal_metrics(pred_norm, gt_norm, mask=None):
    """ compute surface normal metrics (used for benchmarking)
        NOTE: total_normal_errors should be a 1D torch tensor of errors in degrees

        pred_norm: [Nf, 3, h, w]
        gt_norm: [Nf, 3, h, w]
        mask: [Nf, h, w]
    """
    dot_product = (pred_norm * gt_norm).sum(dim=1)  # [Nf, h, w]
    norm_A = torch.norm(pred_norm, dim=1)  # [Nf, h, w]
    norm_B = torch.norm(gt_norm, dim=1)  # [Nf, h, w]
    pred_error = dot_product / (norm_A  * norm_B + 1e-6)

    pred_error = torch.clamp(pred_error, -1.0, 1.0)
    pred_error = torch.arccos(pred_error) * 180.0 / np.pi

    total_normal_errors = pred_error[mask]  # [num_valid_pixels,]

    num_pixels = total_normal_errors.shape[0]

    metrics = {
        'normal mean': torch.mean(total_normal_errors),
        'normal median': torch.median(total_normal_errors),
        'normal rmse': torch.sqrt(torch.sum(total_normal_errors * total_normal_errors) / num_pixels),
        'angle < 5': 100.0 * (torch.sum(total_normal_errors < 5) / num_pixels),
        'angle < 7.5': 100.0 * (torch.sum(total_normal_errors < 7.5) / num_pixels),
        'angle < 11.25': 100.0 * (torch.sum(total_normal_errors < 11.25) / num_pixels),
        'angle < 22.5': 100.0 * (torch.sum(total_normal_errors < 22.5) / num_pixels),
        'angle < 30': 100.0 * (torch.sum(total_normal_errors < 30) / num_pixels)
    }
    return metrics



def normal_evaluation(
    predicted_normal_original,
    ground_truth_normal_original,
    custom_mask=None,
):
    """
    Evaluate the normal map.

    Args:
        predicted_normal (numpy.ndarray or torch.Tensor): The predicted depth map. [Nf,H,W,3]
        ground_truth_normal (numpy.ndarray or torch.Tensor): The ground truth depth map. [Nf,H,W,3]
        custom_mask: [Nf,H,W]

    Returns:
        dict: A dictionary containing the evaluation metrics.
        torch.Tensor: The depth error parity map.
    """
    if isinstance(predicted_normal_original, np.ndarray):
        predicted_normal_original = torch.from_numpy(predicted_normal_original)
    if isinstance(ground_truth_normal_original, np.ndarray):
        ground_truth_normal_original = torch.from_numpy(ground_truth_normal_original)
    if custom_mask is not None and isinstance(custom_mask, np.ndarray):
        custom_mask = torch.from_numpy(custom_mask)

    
    predicted_normal_original = predicted_normal_original.permute(0, 3, 1, 2)  # [Nf, 3, H, W]
    ground_truth_normal_original = ground_truth_normal_original.permute(0, 3, 1, 2)  # [Nf, 3, H, W]
    
    output =  compute_normal_metrics(
        pred_norm=predicted_normal_original,
        gt_norm=ground_truth_normal_original,
        mask=custom_mask
    )

    return {k: v.item() for k, v in output.items()}