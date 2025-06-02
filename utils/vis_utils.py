import cv2

import numpy as np
import torch
import torch.nn as nn
import roma
from copy import deepcopy
import tqdm
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.spatial.transform import Rotation
from PIL import Image
import imageio.v2 as iio
from matplotlib.figure import Figure
import open3d as o3d


def save_point_cloud(point_cloud, save_path):
    points = np.asarray(point_cloud.points)  # shape: (N, 3)

    # Y 和 Z 乘以 -1
    points[:, 1] *= -1  # Y 坐标取反
    points[:, 2] *= -1  # Z 坐标取反

    # 更新点云坐标
    point_cloud.points = o3d.utility.Vector3dVector(points)  

    o3d.io.write_point_cloud(
        save_path,
        point_cloud,
    )
    print(f"Saved point cloud to {save_path}")



def save_depth_normal_maps(depth_maps, normal_maps, path, conf_self=None, rgbs=None):
    ## depth_maps: [Nf,H,W], torch.tensor
    ## normal_maps: [Nf,H,W,3], torch.tensor
    ## rgbs: list of [H,W,3], torch.tensor

    min_depth = depth_maps.min()  # float(torch.quantile(out, 0.01))
    max_depth = depth_maps.max()  # float(torch.quantile(out, 0.99))
    colored_depth = colorize(
        depth_maps,
        cmap_name="Spectral_r",
        range=(min_depth, max_depth),
        append_cbar=True,
    )
    images = []

    ##### add normal
    normals = [x*0.5+0.5 for x in normal_maps]


    if conf_self is not None:
        conf_selfs = torch.concat(conf_self, 0)
        min_conf = torch.log(conf_selfs.min())  # float(torch.quantile(out, 0.01))
        max_conf = torch.log(conf_selfs.max())  # float(torch.quantile(out, 0.99))
        colored_conf = colorize(
            torch.log(conf_selfs),
            cmap_name="jet",
            range=(min_conf, max_conf),
            append_cbar=True,
        )

    for i, depth_map in enumerate(colored_depth):
        # Apply color map to depth map
        img_path = f"{path}/frame_{(i):04d}.webp"
        if rgbs is None:
            to_save = torch.cat([normals[i], depth_map], dim=1)
            to_save = (to_save * 255).detach().cpu().numpy().astype(np.uint8)
        else:
            to_save = torch.cat([rgbs[i], normals[i], depth_map], dim=1)
            to_save = (to_save * 255).detach().cpu().numpy().astype(np.uint8)
        iio.imwrite(img_path, to_save)
        images.append(Image.open(img_path))
        # np.save(f"{path}/frame_{(i):04d}.npy", depth_maps[i].detach().cpu().numpy())

    # comment this as it may fail sometimes
    # images[0].save(f'{path}/_depth_maps.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

    return depth_maps



def get_vertical_colorbar(h, vmin, vmax, cmap_name="jet", label=None, cbar_precision=2):
    """
    :param w: pixels
    :param h: pixels
    :param vmin: min value
    :param vmax: max value
    :param cmap_name:
    :param label
    :return:
    """
    fig = Figure(figsize=(2, 8), dpi=100)
    # fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, ticks=tick_loc, orientation="vertical"
    )

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)
    if label is not None:
        cb1.set_label(label)

    # fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.0
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im



def colorize_np(
    x,
    cmap_name="jet",
    mask=None,
    range=None,
    append_cbar=False,
    cbar_in_image=False,
    cbar_precision=2,
):
    """
    turn a grayscale image into a color image
    :param x: input grayscale, [H, W]
    :param cmap_name: the colorization method
    :param mask: the mask image, [H, W]
    :param range: the range for scaling, automatic if None, [min, max]
    :param append_cbar: if append the color bar
    :param cbar_in_image: put the color bar inside the image to keep the output image the same size as the input image
    :return: colorized image, [H, W]
    """
    if range is not None:
        vmin, vmax = range
    elif mask is not None:
        # vmin, vmax = np.percentile(x[mask], (2, 100))
        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])
        # vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        # print(vmin, vmax)
    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += 1e-6

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.ones_like(x_new) * (1.0 - mask)

    cbar = get_vertical_colorbar(
        h=x.shape[0],
        vmin=vmin,
        vmax=vmax,
        cmap_name=cmap_name,
        cbar_precision=cbar_precision,
    )

    if append_cbar:
        if cbar_in_image:
            x_new[:, -cbar.shape[1] :, :] = cbar
        else:
            x_new = np.concatenate(
                (x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1
            )
        return x_new
    else:
        return x_new


# tensor
def colorize(
    x, cmap_name="jet", mask=None, range=None, append_cbar=False, cbar_in_image=False
):
    """
    turn a grayscale image into a color image
    :param x: torch.Tensor, grayscale image, [H, W] or [B, H, W]
    :param mask: torch.Tensor or None, mask image, [H, W] or [B, H, W] or None
    """

    device = x.device
    x = x.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy() > 0.99
        kernel = np.ones((3, 3), np.uint8)

    if x.ndim == 2:
        x = x[None]
        if mask is not None:
            mask = mask[None]

    out = []
    for x_ in x:
        if mask is not None:
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        x_ = colorize_np(x_, cmap_name, mask, range, append_cbar, cbar_in_image)
        out.append(torch.from_numpy(x_).to(device).float())
    out = torch.stack(out).squeeze(0)
    return out