import numpy as np

import forest_segmentor_bindings

class InferenceHelper:
    def __init__(self):
        pass

    def infer(self, points):
        np_points = np.asarray(points, dtype=np.float32)
        print(np_points.shape)
        np_out = np.sum(np_points, axis=1)
        return forest_segmentor_bindings.FloatVector(np_out)