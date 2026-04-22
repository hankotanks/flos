import numpy as np

import forest_segmentor_bindings

import torch

from .model import PointTransformerV3Lightning

class InferenceHelper:
    def __init__(self):
        ckpt_path = "/root/ros_ws/src/forest_segmentation/models/last.ckpt"
        print("Loading checkpoint from", ckpt_path, "for traversability inference.")
        self.model = PointTransformerV3Lightning.load_from_checkpoint(ckpt_path)
        self.model.to("cuda")
        self.model.eval()

    def inference(self, points, intensity):
        np_points = np.asarray(points, dtype=np.float32)
        np_intensity = np.asarray(intensity, dtype=np.float32).reshape(-1, 1)
        if not np_points.size:
            raise Exception("points is empty")
        if np_points.shape[0] != np_intensity.shape[0]:
            raise Exception("points and intensity dont match shapes")
        if np_points.shape[1] != 3:
            raise Exception("points needs to be Nx3")
        # points needs an additional batch column at the start
        batched_points = np.zeros((np_points.shape[0], 4), dtype=np.float32)
        batched_points[:, 1:] = np_points
        with torch.no_grad():
            logits = self.model.infer_logits(
                torch.from_numpy(batched_points).to("cuda"),
                torch.from_numpy(np_intensity).to("cuda"),
            )
        # note that the vector type should match the pybind side, otherwise might end up with copies 
        np_logits = logits.cpu().numpy().reshape(-1).astype(np.float32)
        # although the return is very likely still a copy
        return forest_segmentor_bindings.FloatVector(np_logits)
