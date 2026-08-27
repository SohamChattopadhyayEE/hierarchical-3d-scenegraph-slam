"""Back-projects a 2D instance mask + depth image into per-object 3D points."""

import numpy as np


class BackProjector:

    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def project(self, label_mask: np.ndarray, depth_m: np.ndarray) -> dict:
        """Returns {instance_id: (points_Nx3, centroid_3)}, camera-frame, metres."""
        v, u = np.mgrid[0:depth_m.shape[0], 0:depth_m.shape[1]]
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy

        objects = {}
        for instance_id in np.unique(label_mask):
            if instance_id == 0:
                continue
            valid = (label_mask == instance_id) & (depth_m > 0)
            if not np.any(valid):
                continue
            points = np.stack([x[valid], y[valid], depth_m[valid]], axis=1)
            objects[int(instance_id)] = (points, points.mean(axis=0))
        return objects
