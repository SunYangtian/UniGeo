import sys
import os
import numpy as np
import torch
from utils.geometry_utils import get_surface_normal, backproject_to_cv_position

class DepthCrafter:
    def __init__(self, model_dir, unet_path, pre_train_path, **kwargs):
        torch.cuda.init()  # 
        torch.cuda.synchronize()  # 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        sys.path.append(model_dir)
        from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
        from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )

        self.pipeline = DepthCrafterPipeline.from_pretrained(
                pre_train_path,
                unet=unet,
                torch_dtype=torch.float16,
                variant="fp16",
            )

        self.pipeline.to(self.device)

        self.pipeline.enable_xformers_memory_efficient_attention()
        self.pipeline.enable_attention_slicing()

        print(f"Model loaded from {unet_path}")


    def prepare_input(self, data):
        """
        Prepare input data for the DepthCrafter model.
        """
        frames = [x.transpose(1,2,0).astype(np.uint8) for x in data['images']]  # list of [H, W, 3], np.uint8
        frames = np.stack(frames, axis=0).astype(np.float32) / 255.  # [Nf, H, W, 3]
        return frames


    def prepare_output(self, depthcrafter_depths, data):
        gt_intrinsics = data['intrinsics']  # list of [H,W,3,3], np.array

        pts3ds_self = [backproject_to_cv_position(depthcrafter_depths[i], gt_intrinsics[i]) for i in range(len(depthcrafter_depths))]
        pts3ds_self = [torch.from_numpy(x).float() for x in pts3ds_self]  # list of [H, W, 3]

        pred_normals = [get_surface_normal(x[None]).squeeze() for x in pts3ds_self]  # list of [H, W, 3]

        ### change to opengl coord
        # Flip y and z axes for OpenGL coordinates
        for i in range(len(pred_normals)):
            pred_normals[i][:, :, 1:] = - pred_normals[i][:, :, 1:]  # Flip yz axes


        pred_depths = [torch.from_numpy(x).float() for x in depthcrafter_depths]  # list of [H, W]
        pred_depths = torch.stack(pred_depths, dim=0)  # [B, H, W]

        output = {
            'pred_depths' : pred_depths,  # [Nf, H, W]
            'pred_normals' : torch.stack(pred_normals, 0),  # [Nf, H, W, 3]
        }
        return output



    def forward(self, data):
        """
        Forward pass through the DepthCrafter model.
        """
        frames = self.prepare_input(data)
        # inference the depth map using the DepthCrafter pipeline
        with torch.inference_mode():
            res = self.pipeline(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=1.0,
                num_inference_steps=5,
                window_size=len(frames),
                overlap=25,
                track_time=False,
            ).frames[0]
        
        # convert the three-channel output to a single channel depth map
        res = res.sum(-1) / res.shape[-1]
        # normalize the depth map to [0, 1] across the whole video
        res = (res - res.min()) / (res.max() - res.min())  # [Nf,H,W]
        # https://github.com/Tencent/DepthCrafter/blob/ee2c6e8c3a7ccdc221adaf749b67bd21db712ce4/visualization/visualization_pcd.py#L118
        depthcrafter_depths = [1 / (x + 0.1) for x in res]  # list of [H,W], np.array

        return self.prepare_output(depthcrafter_depths, data)