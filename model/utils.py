from PIL import Image
import numpy as np
import torch
import torchvision.transforms as tvf

def prepare_input_Dust3R(data,):
    ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    views = []
    images_list = [ImgNorm(Image.fromarray(x.transpose(1,2,0).astype(np.uint8)))[None] for x in data['images']]  # list of [1, 3, H, W]

    num_views = len(images_list)

    for i in range(num_views):

        extrinsic = data["extrinsics"][i].astype(np.float32)
        camera_pose = np.linalg.inv(extrinsic)  # [4,4], camera2world
        ### opengl to opencv
        OPENGL_TO_OPENCV = np.float32([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
        camera_pose = np.einsum('ij,jk,kl->il', OPENGL_TO_OPENCV, camera_pose, OPENGL_TO_OPENCV)

        pts3d = data["world_coord"][i].astype(np.float32)  # [3,h,w]
        pts3d[1:] *= -1  # opengl to opencv

        cam_pts3d = data["cam_coord"][i].astype(np.float32)  # [3,h,w]
        cam_pts3d[1:] *= -1  # opengl to opencv

        view = {
            "img": images_list[i],
            "ray_map": torch.full(
                (
                    images_list[i].shape[0],
                    6,
                    images_list[i].shape[-2],
                    images_list[i].shape[-1],
                ),
                torch.nan,
            ),
            "true_shape": torch.tensor([data['images'][i].shape[1:]]),
            "idx": i,
            "instance": str(i),
            "camera_intrinsics": torch.from_numpy(
                data["intrinsics"][i].astype(np.float32)
            ).unsqueeze(0),  # [1, 3, 3]

            "camera_pose": torch.from_numpy(
                camera_pose
            ).unsqueeze(0),  # c2w, opencv coord

            "pts3d": torch.from_numpy(
                pts3d
            ).permute(1,2,0).unsqueeze(0),  # [1, H, W, 3]
            
            "cam_pts3d": torch.from_numpy(
                cam_pts3d
            ).permute(1,2,0).unsqueeze(0),  # [1, H, W, 3]

            "valid_mask": torch.from_numpy(
                data["mask"][i]
            ).unsqueeze(0).bool(),  # [1, H, W]

            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)
    return views



def prepare_input_Dust3R_simple(data,):
    """
        Used for data with only images and no camera poses.
    """

    ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    views = []
    images_list = [ImgNorm(Image.fromarray(x.transpose(1,2,0).astype(np.uint8)))[None] for x in data['images']]  # list of [1, 3, H, W]

    num_views = len(images_list)

    for i in range(num_views):
        view = {
            "img": images_list[i],
            "ray_map": torch.full(
                (
                    images_list[i].shape[0],
                    6,
                    images_list[i].shape[-2],
                    images_list[i].shape[-1],
                ),
                torch.nan,
            ),
            "true_shape": torch.tensor([data['images'][i].shape[1:]]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                0
            ),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)
    return views