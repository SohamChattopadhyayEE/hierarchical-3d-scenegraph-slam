"""Tracks back-projected objects in 3D world space with a per-object EKF."""

import os

import cv2
import numpy as np


class TrackedObject:
    """Constant-velocity EKF: state = [x, y, z, vx, vy, vz] in world frame.

    The measurement (a world-frame position, already transformed by the known
    camera pose) is linear in the state, so H is just a constant matrix --
    still an EKF, just one with no nonlinearity left to linearize here.
    """

    def __init__(self, obj_id: int, pos_world: np.ndarray):
        self.id = obj_id
        self.x = np.concatenate([pos_world, np.zeros(3)])
        self.P = np.eye(6) * 1.0
        self.observations = 1  # this construction counts as the first observation

    def predict(self, dt: float):
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        Q = np.eye(6) * 0.01
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z_world: np.ndarray):
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        R = np.eye(3) * 0.05
        y = z_world - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.observations += 1

    @property
    def position(self) -> np.ndarray:
        return self.x[:3]


class Tracker:

    def __init__(self, output_dir: str, gate: float = 0.5, min_track_length: int = 5):
        self.objects = {}
        self.next_id = 1
        self.output_dir = output_dir
        self.gate = gate
        # Object selection: a track this short is more likely segmentation
        # flicker than a real, persistent object worth putting in the map.
        # Its first (min_track_length - 1) observations are never saved --
        # not retroactively backfilled once it does qualify, so short-lived
        # noise tracks leave nothing on disk at all.
        self.min_track_length = min_track_length
        self.last_t = None

    def update(self, detections: dict, R_cam2world: np.ndarray, t_cam2world: np.ndarray,
               timestamp: float, rgb: np.ndarray, label_mask: np.ndarray):
        dt = 0.0 if self.last_t is None else timestamp - self.last_t
        self.last_t = timestamp
        for obj in self.objects.values():
            obj.predict(dt)

        for instance_id, (points_cam, centroid_cam) in detections.items():
            centroid_world = R_cam2world @ centroid_cam + t_cam2world
            points_world = points_cam @ R_cam2world.T + t_cam2world

            match, best_dist = None, self.gate
            for obj in self.objects.values():
                dist = np.linalg.norm(obj.position - centroid_world)
                if dist < best_dist:
                    match, best_dist = obj, dist

            if match is None:
                match = TrackedObject(self.next_id, centroid_world)
                self.objects[self.next_id] = match
                self.next_id += 1
            else:
                match.update(centroid_world)

            if match.observations >= self.min_track_length:
                self._save(match.id, timestamp, match.position, points_world,
                           rgb, label_mask == instance_id, R_cam2world, t_cam2world)

    def _save(self, obj_id: int, timestamp: float, centroid: np.ndarray,
              points_world: np.ndarray, rgb: np.ndarray, obj_mask: np.ndarray,
              R_cam2world: np.ndarray, t_cam2world: np.ndarray):
        folder = os.path.join(self.output_dir, str(obj_id))
        os.makedirs(folder, exist_ok=True)
        stamp = f'{timestamp:.6f}'

        with open(os.path.join(folder, 'trajectory.txt'), 'a') as f:
            f.write(f'{stamp} {centroid[0]:.4f} {centroid[1]:.4f} {centroid[2]:.4f}\n')

        np.save(os.path.join(folder, f'{stamp}_points.npy'), points_world)

        ys, xs = np.where(obj_mask)
        x_min, y_min, x_max, y_max = xs.min(), ys.min(), xs.max(), ys.max()
        crop = rgb[y_min:y_max + 1, x_min:x_max + 1]
        cv2.imwrite(os.path.join(folder, f'{stamp}.png'), crop)

        # Camera pose + this crop's bounding box in the full frame -- lets a
        # later stage (e.g. GaussianSplatter) reconstruct the posed View
        # (T_world_cam, K adjusted for the crop offset, image) this frame's
        # crop corresponds to, without needing anything beyond what's
        # already saved here plus the global camera intrinsics.
        np.savez(os.path.join(folder, f'{stamp}_pose.npz'),
                 R=R_cam2world, t=t_cam2world,
                 bbox=np.array([x_min, y_min, x_max, y_max]))
