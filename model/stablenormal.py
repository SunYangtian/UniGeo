import sys
import os
import numpy as np
import torch
from utils.geometry_utils import get_surface_normal, backproject_to_cv_position
from PIL import Image

class StableNormal:
    def __init__(self, **kwargs):
        torch.cuda.init()  # 
        torch.cuda.synchronize()  # 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create predictor instance
        self.predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)

        print(f"Model loaded")


    def prepare_input(self, data):
        """
        Prepare input data for the DepthCrafter model.
        """
        frames = [x.transpose(1,2,0).astype(np.uint8) for x in data['images']]  # list of [H, W, 3], np.uint8
        frames = np.stack(frames, axis=0).astype(np.float32) / 255.  # [Nf, H, W, 3]
        return frames


    def forward(self, data):
        """
        Forward pass through the DepthCrafter model.
        """
        frames = self.prepare_input(data)
        # inference the depth map using the DepthCrafter pipeline
        with torch.inference_mode():
            images = [Image.fromarray(x.transpose(1,2,0).astype(np.uint8)) for x in data['images']]

        pred_normals = [self.predictor(image) for image in images]
        pred_normals = [np.array(normal) for normal in pred_normals]
        ### flip x coordinate
        for i in range(len(pred_normals)):
            pred_normals[i][:,:,0] = -pred_normals[i][:,:,0]

        pred_normals = [normal / 255. * 2 - 1 for normal in pred_normals]

        output = {
            'pred_normals': torch.stack([torch.from_numpy(x).float() for x in pred_normals], dim=0),  # [Nf, H, W, 3]
            'pred_depths': torch.zeros_like(torch.stack([torch.from_numpy(x[...,0]).float() for x in pred_normals], dim=0)),  # [Nf, H, W]
        }

        return output