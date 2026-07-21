import numpy as np
import torch

from segmentation_model.segmentation_model import PointTransformerV3Lightning

import cloud_bindings

class InferenceHelper:
    def __init__(self, path_to_model):
        self.model = PointTransformerV3Lightning.load_from_checkpoint(path_to_model)
        self.model.to("cuda")
        self.model.eval()

    def infer(self, points):
        np_points = np.asarray(points, dtype = np.float32)
        if not np_points.size:
            raise Exception("points is empty")
        if np_points.shape[1] != 3:
            raise Exception("points needs to be Nx3")
        # points needs an additional batch column at the start
        np_points_batched = np.zeros((np_points.shape[0], 4), dtype = np.float32)
        np_points_batched[:, 1:] = np_points
        # construct dict
        batch = {"pos": torch.from_numpy(np_points_batched).to("cuda")}
        with torch.no_grad(): pred, conf = self.model.infer(batch)
        # note that the vector type should match the pybind side, otherwise might end up with copies 
        np_pred = pred.cpu().numpy().reshape(-1).astype(np.uint8)
        np_conf = conf.cpu().numpy().astype(np.float32)
        # although the return is very likely still a copy
        return cloud_bindings.FieldU1(np_pred), [cloud_bindings.FieldF4(np_conf[:, i]) for i in range(np_conf.shape[1])]