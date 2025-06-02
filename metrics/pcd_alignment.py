
### https://github.com/CUT3R/CUT3R/blob/main/eval/mv_recon/criterion.py
import torch
import torch.nn as nn
from copy import copy, deepcopy
from metrics.misc import invalid_to_nans, invalid_to_zeros


def Sum(losses, masks, conf=None):
    loss, mask = losses[0], masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        if conf is not None:
            return losses, masks, conf
        return losses, masks
    else:
        # we are returning the global loss
        for loss2 in losses[1:]:
            loss = loss + loss2
        return loss


def get_norm_factor(pts, norm_mode="avg_dis", valids=None, fix_first=True):
    assert pts[0].ndim >= 3 and pts[0].shape[-1] == 3
    assert pts[1] is None or (pts[1].ndim >= 3 and pts[1].shape[-1] == 3)
    norm_mode, dis_mode = norm_mode.split("_")

    nan_pts = []
    nnzs = []

    if norm_mode == "avg":
        # gather all points together (joint normalization)

        for i, pt in enumerate(pts):
            nan_pt, nnz = invalid_to_zeros(pt, valids[i], ndim=3)
            nan_pts.append(nan_pt)
            nnzs.append(nnz)

            if fix_first:
                break
        all_pts = torch.cat(nan_pts, dim=1)

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)
        if dis_mode == "dis":
            pass  # do nothing
        elif dis_mode == "log1p":
            all_dis = torch.log1p(all_dis)
        else:
            raise ValueError(f"bad {dis_mode=}")

        norm_factor = all_dis.sum(dim=1) / (torch.cat(nnzs).sum() + 1e-8)
    else:
        raise ValueError(f"Not implemented {norm_mode=}")

    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts[0].ndim:
        norm_factor.unsqueeze_(-1)

    return norm_factor


def normalize_pointcloud_t(
    pts, norm_mode="avg_dis", valids=None, fix_first=True, gt=False
):
    if gt:
        norm_factor = get_norm_factor(pts, norm_mode, valids, fix_first)
        res = []

        for i, pt in enumerate(pts):
            res.append(pt / norm_factor)

    else:
        # pts_l, pts_r = pts
        # use pts_l and pts_r[-1] as pts to normalize
        norm_factor = get_norm_factor(pts, norm_mode, valids, fix_first)

        res = []

        for i in range(len(pts)):
            res.append(pts[i] / norm_factor)
            # res_r.append(pts_r[i] / norm_factor)

        # res = [res_l, res_r]

    return res, norm_factor


@torch.no_grad()
def get_joint_pointcloud_depth(zs, valid_masks=None, quantile=0.5):
    # set invalid points to NaN
    _zs = []
    for i in range(len(zs)):
        valid_mask = valid_masks[i] if valid_masks is not None else None
        _z = invalid_to_nans(zs[i], valid_mask).reshape(len(zs[i]), -1)
        _zs.append(_z)

    _zs = torch.cat(_zs, dim=-1)

    # compute median depth overall (ignoring nans)
    if quantile == 0.5:
        shift_z = torch.nanmedian(_zs, dim=-1).values
    else:
        shift_z = torch.nanquantile(_zs, quantile, dim=-1)
    return shift_z  # (B,)


@torch.no_grad()
def get_joint_pointcloud_center_scale(pts, valid_masks=None, z_only=False, center=True):
    # set invalid points to NaN

    _pts = []
    for i in range(len(pts)):
        valid_mask = valid_masks[i] if valid_masks is not None else None
        _pt = invalid_to_nans(pts[i], valid_mask).reshape(len(pts[i]), -1, 3)
        _pts.append(_pt)

    _pts = torch.cat(_pts, dim=1)

    # compute median center
    _center = torch.nanmedian(_pts, dim=1, keepdim=True).values  # (B,1,3)
    if z_only:
        _center[..., :2] = 0  # do not center X and Y

    # compute median norm
    _norm = ((_pts - _center) if center else _pts).norm(dim=-1)
    scale = torch.nanmedian(_norm, dim=1).values
    return _center[:, None, :, :], scale[:, None, None, None]


class Regr3D_t:
    def __init__(self, norm_mode="avg_dis", gt_scale=False, fix_first=True):
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        self.fix_first = fix_first

    def get_all_pts3d_t(self, gt_pts3d, pred_pts3d, gt_masks, dist_clip=None):
        ### gt_pts3d: Bs, H, W, 3, world space
        ### pred_pts3d: Bs, H, W, 3, world space
        ### gt_masks: Bs, H, W, valid mask

        gt_pts = gt_pts3d
        valids = gt_masks
        pr_pts = pred_pts3d

        # pr_pts = (pr_pts_l, pr_pts_r)

        if self.norm_mode:
            pr_pts, pr_factor = normalize_pointcloud_t(
                pr_pts, self.norm_mode, valids, fix_first=self.fix_first, gt=False
            )
        else:
            pr_factor = None

        if self.norm_mode and not self.gt_scale:
            gt_pts, gt_factor = normalize_pointcloud_t(
                gt_pts, self.norm_mode, valids, fix_first=self.fix_first, gt=True
            )
        else:
            gt_factor = None

        return gt_pts, pr_pts, gt_factor, pr_factor, valids, {}



class Regr3D_t_ShiftInv(Regr3D_t):
    """Same than Regr3D but invariant to depth shift."""

    def get_all_pts3d_t(self, gt_pts3d, pred_pts3d, gt_masks):
        # compute unnormalized points
        gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
            super().get_all_pts3d_t(gt_pts3d, pred_pts3d, gt_masks)
        )

        # pred_pts_l, pred_pts_r = pred_pts
        gt_zs = [gt_pt[..., 2] for gt_pt in gt_pts]

        pred_zs = [pred_pt[..., 2] for pred_pt in pred_pts]
        # pred_zs.append(pred_pts_r[-1][..., 2])

        # compute median depth
        gt_shift_z = get_joint_pointcloud_depth(gt_zs, masks)[:, None, None]
        pred_shift_z = get_joint_pointcloud_depth(pred_zs, masks)[:, None, None]

        # subtract the median depth
        for i in range(len(gt_pts)):
            gt_pts[i][..., 2] -= gt_shift_z

        for i in range(len(pred_pts)):
            # for j in range(len(pred_pts[i])):
            pred_pts[i][..., 2] -= pred_shift_z

        monitoring = dict(
            monitoring,
            gt_shift_z=gt_shift_z.mean().detach(),
            pred_shift_z=pred_shift_z.mean().detach(),
        )
        return gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring


class Regr3D_t_ScaleInv(Regr3D_t):
    """Same than Regr3D but invariant to depth shift.
    if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d_t(self, gt_pts3d, pred_pts3d, gt_masks):
        # compute depth-normalized points
        gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
            super().get_all_pts3d_t(gt_pts3d, pred_pts3d, gt_masks)
        )

        # measure scene scale

        # pred_pts_l, pred_pts_r = pred_pts

        pred_pts_all = [
            x.clone() for x in pred_pts
        ]  # [pred_pt for pred_pt in pred_pts_l]
        # pred_pts_all.append(pred_pts_r[-1])

        _, gt_scale = get_joint_pointcloud_center_scale(gt_pts, masks)
        _, pred_scale = get_joint_pointcloud_center_scale(pred_pts_all, masks)

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            for i in range(len(pred_pts)):
                # for j in range(len(pred_pts[i])):
                pred_pts[i] *= gt_scale / pred_scale

        else:
            for i in range(len(pred_pts)):
                # for j in range(len(pred_pts[i])):
                pred_pts[i] *= pred_scale / gt_scale

            for i in range(len(gt_pts)):
                gt_pts[i] *= gt_scale / pred_scale

        monitoring = dict(
            monitoring, gt_scale=gt_scale.mean(), pred_scale=pred_scale.mean().detach()
        )

        return gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring


class Regr3D_t_ScaleShiftInv(Regr3D_t_ScaleInv, Regr3D_t_ShiftInv):
    # calls Regr3D_ShiftInv first, then Regr3D_ScaleInv
    pass
