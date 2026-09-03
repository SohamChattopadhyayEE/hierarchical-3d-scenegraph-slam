"""Back-projects a 2D instance mask + depth image into per-object 3D points."""

import cv2
import numpy as np


class BackProjector:

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 erosion_kernel: int = 3, depth_discontinuity_thresh: float = 0.05,
                 min_area: int = 3000):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        # Cheap, at-the-source cleanup, applied before any point ever reaches
        # back-projection/tracking: erosion drops mixed-pixel mask boundary
        # pixels (part object, part background); discontinuity rejection
        # drops depth-sensor "flying pixels" straddling an edge. Both are
        # exactly the pixels that otherwise pollute a frame's centroid and
        # destabilize the EKF that tracks it.
        self.erosion_kernel = (
            np.ones((erosion_kernel, erosion_kernel), np.uint8) if erosion_kernel > 1 else None)
        self.depth_discontinuity_thresh = depth_discontinuity_thresh
        # Object selection: a mask this small (px) is more likely noise/a
        # sliver than a real object worth putting in the map.
        self.min_area = min_area

    def project(self, label_mask: np.ndarray, depth_m: np.ndarray) -> dict:
        """Returns {instance_id: (points_Nx3, centroid_3)}, camera-frame, metres."""
        v, u = np.mgrid[0:depth_m.shape[0], 0:depth_m.shape[1]]
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy

        # Computed once for the whole image (cheap), reused for every
        # instance below: a pixel next to a sharp depth jump -- a real
        # object edge, or a valid/invalid boundary -- is unreliable either
        # way, so it's rejected regardless of which instance it falls in.
        dx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
        stable_depth = np.hypot(dx, dy) < self.depth_discontinuity_thresh

        objects = {}
        for instance_id in np.unique(label_mask):
            if instance_id == 0:
                continue
            mask = (label_mask == instance_id)
            if mask.sum() < self.min_area:
                continue
            if self.erosion_kernel is not None:
                mask = cv2.erode(mask.astype(np.uint8), self.erosion_kernel,
                                  iterations=1).astype(bool)

            valid = mask & (depth_m > 0) & stable_depth
            if not np.any(valid):
                continue
            points = np.stack([x[valid], y[valid], depth_m[valid]], axis=1)
            objects[int(instance_id)] = (points, points.mean(axis=0))
        return objects
