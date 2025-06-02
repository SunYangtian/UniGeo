import numpy as np
import torch
import cv2
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from scipy.optimize import minimize
import os
from collections import defaultdict

# https://github.com/prs-eth/Marigold/blob/main/src/util/alignment.py
def align_depth_least_square_np(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):

    # gt_arr: [H, W]
    # pred_arr: [H, W]
    # valid_mask_arr: [H, W]
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred

# https://github.com/CUT3R/CUT3R/blob/main/eval/video_depth/tools.py


def absolute_error_loss(params, predicted_depth, ground_truth_depth):
    s, t = params

    predicted_aligned = s * predicted_depth + t

    abs_error = np.abs(predicted_aligned - ground_truth_depth)
    return np.sum(abs_error)


def absolute_value_scaling_torch(predicted_depth, ground_truth_depth, s=1, t=0):
    # predicted_depth: [H, W]
    # ground_truth_depth: [H, W]
    predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1)
    ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1)

    initial_params = [s, t]  # s = 1, t = 0

    result = minimize(
        absolute_error_loss,
        initial_params,
        args=(predicted_depth_np, ground_truth_depth_np),
    )

    s, t = result.x
    return s, t

def absolute_value_scaling2_torch(
    predicted_depth,
    ground_truth_depth,
    s_init=1.0,
    t_init=0.0,
    lr=1e-4,
    max_iters=1000,
    tol=1e-6,
):
    # Initialize s and t as torch tensors with requires_grad=True
    s = torch.tensor(
        [s_init],
        requires_grad=True,
        device=predicted_depth.device,
        dtype=predicted_depth.dtype,
    )
    t = torch.tensor(
        [t_init],
        requires_grad=True,
        device=predicted_depth.device,
        dtype=predicted_depth.dtype,
    )

    optimizer = torch.optim.Adam([s, t], lr=lr)

    prev_loss = None

    for i in range(max_iters):
        optimizer.zero_grad()

        # Compute predicted aligned depth
        predicted_aligned = s * predicted_depth + t

        # Compute absolute error
        abs_error = torch.abs(predicted_aligned - ground_truth_depth)

        # Compute loss
        loss = torch.sum(abs_error)

        # Backpropagate
        loss.backward()

        # Update parameters
        optimizer.step()

        # Check convergence
        if prev_loss is not None and torch.abs(prev_loss - loss) < tol:
            break

        prev_loss = loss.item()

    return s.detach().item(), t.detach().item()


def align_with_lstsq_torch(predicted_depth, ground_truth_depth):
    # predicted_depth: [H, W]
    # ground_truth_depth: [H, W]
    # Convert to numpy for lstsq
    predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1, 1)  # [N,1]
    ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1, 1)  # [N,1]

    # Add a column of ones for the shift term
    A = np.hstack([predicted_depth_np, np.ones_like(predicted_depth_np)])

    # Solve for scale (s) and shift (t) using least squares
    result = np.linalg.lstsq(A, ground_truth_depth_np, rcond=None)
    s, t = result[0][0], result[0][1]

    # convert to torch tensor
    s = torch.tensor(s, device=predicted_depth.device)
    t = torch.tensor(t, device=ground_truth_depth.device)
    return s, t


def align_with_scale_torch(predicted_depth, ground_truth_depth):
    # predicted_depth: [H, W]
    # ground_truth_depth: [H, W]
    # Compute initial scale factor 's' using the closed-form solution (L2 norm)
    dot_pred_gt = torch.nanmean(ground_truth_depth)
    dot_pred_pred = torch.nanmean(predicted_depth)
    s = dot_pred_gt / dot_pred_pred

    # Iterative reweighted least squares using the Weiszfeld method
    for _ in range(10):
        # Compute residuals between scaled predictions and ground truth
        residuals = s * predicted_depth - ground_truth_depth
        abs_residuals = (
            residuals.abs() + 1e-8
        )  # Add small constant to avoid division by zero

        # Compute weights inversely proportional to the residuals
        weights = 1.0 / abs_residuals

        # Update 's' using weighted sums
        weighted_dot_pred_gt = torch.sum(
            weights * predicted_depth * ground_truth_depth
        )
        weighted_dot_pred_pred = torch.sum(weights * predicted_depth**2)
        s = weighted_dot_pred_gt / weighted_dot_pred_pred

    return s