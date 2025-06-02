import sys
import os
import numpy as np
import torch
from .utils import prepare_input_Dust3R
from metrics.utils import solve_depth_and_camera_from_3d_points
from utils.geometry_utils import get_surface_normal

class Spann3R:
    def __init__(self, model_dir, dust3r_ckpt, ckpt_path, **kwargs):
        torch.cuda.init()  # 
        torch.cuda.synchronize()  # 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        sys.path.append(model_dir)
        from spann3r.model import Spann3R as Spann3R_model
        self.model = Spann3R_model(dus3r_name=dust3r_ckpt, use_feat=False).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device)['model'])
        self.model.eval()
        print(f"Model loaded from {ckpt_path}")


    def forward(self, data):
        """
        Forward pass through the SPANN3R model.
        """
        ###### Prepare input data
        batch_data = prepare_input_Dust3R(data)
        for view in batch_data:
            for name in 'img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres'.split():  # pseudo_focal
                if name not in view:
                    continue
                view[name] = view[name].to(self.device, non_blocking=True)

        ###### Forward pass
        with torch.no_grad():
            preds, preds_all = self.model.forward(batch_data)

        ###### Prepare output
        pred_world_pts = [pred['pts3d' if j==0 else 'pts3d_in_other_view'].detach().cpu() for (j, pred) in enumerate(preds)] # list of [1, H, W, 3]

        cam_coord_list, extrinsic_list, intrinsic_list = solve_depth_and_camera_from_3d_points(pred_world_pts)
        # cam_coord_list: list of [1, H, W, 3], np.array

        pr_poses = [np.linalg.inv(x) for x in extrinsic_list]
        pred_poses = torch.from_numpy(np.stack(pr_poses)).float()  # [B, 4, 4]

        pts3ds_self = [torch.from_numpy(x[0]).float() for x in cam_coord_list]  # list of [H, W, 3]
        pred_depths = torch.stack([pts3d_self[..., -1] for pts3d_self in pts3ds_self], 0)  # [B, H, W]

        pred_normals = [get_surface_normal(x[None]).squeeze() for x in pts3ds_self]  # list of [H, W, 3]
        ### change to opengl coord
        # Flip y and z axes for OpenGL coordinates
        for i in range(len(pred_normals)):
            pred_normals[i][:, :, 1:] = - pred_normals[i][:, :, 1:]  # Flip yz axes


        output = {
            'pred_world_pts': torch.cat(pred_world_pts, 0),  # [Nf, H, W, 3]
            'pred_depths' : pred_depths,  # [Nf, H, W]
            'pred_normals' : torch.stack(pred_normals, 0),  # [Nf, H, W, 3]
            'pred_poses' : pred_poses,  # [Nf, 4, 4]
        }
        return output