import sys
import os
import numpy as np
import torch
from .utils import prepare_input_Dust3R
from metrics.utils import solve_depth_and_camera_from_3d_points
from utils.geometry_utils import get_surface_normal
from metrics.camera import pose_encoding_to_camera
from metrics.utils import estimate_focal_knowing_depth

class Cut3R:
    def __init__(self, model_dir, ckpt_path, **kwargs):
        torch.cuda.init()  # 
        torch.cuda.synchronize()  # 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        sys.path.append(model_dir)
        from dust3r.model import ARCroco3DStereo
        from dust3r.inference import inference
        self.inference = inference

        model = ARCroco3DStereo.from_pretrained(ckpt_path)
        self.model = model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {ckpt_path}")


    def prepare_input(self, data):
        return prepare_input_Dust3R(data)

    def prepare_output(self, outputs, revisit=1, solve_pose=False):
        valid_length = len(outputs["pred"]) // revisit
        outputs["pred"] = outputs["pred"][-valid_length:]
        outputs["views"] = outputs["views"][-valid_length:]

        if solve_pose:
            pts3ds_self = [
                output["pts3d_in_self_view"].cpu() for output in outputs["pred"]
            ]
            pts3ds_other = [
                output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
            ]
            conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
            conf_other = [output["conf"].cpu() for output in outputs["pred"]]
            pr_poses, focal, pp = recover_cam_params(
                torch.cat(pts3ds_self, 0),
                torch.cat(pts3ds_other, 0),
                torch.cat(conf_self, 0),
                torch.cat(conf_other, 0),
            )
            pts3ds_self = torch.cat(pts3ds_self, 0)
        else:

            pts3ds_self = [
                output["pts3d_in_self_view"].cpu() for output in outputs["pred"]
            ]
            pts3ds_other = [
                output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
            ]
            conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
            conf_other = [output["conf"].cpu() for output in outputs["pred"]]
            pts3ds_self = torch.cat(pts3ds_self, 0)
            pr_poses = [
                pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
                for pred in outputs["pred"]
            ]
            pr_poses = torch.cat(pr_poses, 0)

            B, H, W, _ = pts3ds_self.shape
            pp = (
                torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
                .float()
                .repeat(B, 1)
                .reshape(B, 2)
            )
            focal = estimate_focal_knowing_depth(
                pts3ds_self, pp, focal_mode="weiszfeld"
            )

        colors = [0.5 * (output["rgb"][0] + 1.0) for output in outputs["pred"]]
        cam_dict = {
            "focal": focal.cpu().numpy(),
            "pp": pp.cpu().numpy(),
        }

        pred_normals = [get_surface_normal(x[None]).squeeze() for x in pts3ds_self]  # list of [H, W, 3]
        ### change to opengl coord
        # Flip y and z axes for OpenGL coordinates
        for i in range(len(pred_normals)):
            pred_normals[i][:, :, 1:] = - pred_normals[i][:, :, 1:]  # Flip yz axes

        output = {
            'pred_world_pts': pts3ds_self,  # [Nf, H, W, 3]
            'pred_depths' : torch.stack([pts3d_self[..., -1] for pts3d_self in pts3ds_self], 0),  # [B, H, W]
            'pred_normals' : torch.stack(pred_normals, 0),  # [Nf, H, W, 3]
            'pred_poses' : pr_poses,  # [Nf, 4, 4]
        }
        return output


    def forward(self, data):
        """
        Forward pass through the CUT3R model.
        """
        ###### Prepare input data
        batch_data = self.prepare_input(data)

        ###### Forward pass
        with torch.no_grad():
            outputs, _ = self.inference(batch_data, self.model, self.device)

        ###### Prepare output
        return self.prepare_output(outputs)