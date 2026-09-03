#!/usr/bin/env python3
"""Builds one combined Gaussian-Splat map straight from a TUM RGB-D dataset
-- same technique as test_modules/combined_complete_map.py (pool multi-view
points, clean with reconstruct_object, fit ONE joint GaussianSplatter), but
directly from rgb.txt/depth.txt/groundtruth.txt instead of tracked
per-object outputs, using each whole frame instead of segmented crops.

Usage: python3 gs_map_from_rgbd.py datasets/rgbd_dataset_freiburg1_desk
       python3 gs_map_from_rgbd.py datasets/rgbd_dataset_freiburg1_desk --stride 10 --iters 4000
"""

import argparse
import os

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree

from mapping_from_file import nearest, quat_to_matrix, read_tum_file
from reconstruction_3d.gaussian_splatter import GaussianSplatter, View
from reconstruction_3d.reconstruct import reconstruct_object

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def backproject_frame(bgr, depth_m, fx, fy, cx, cy, discontinuity_thresh=0.05):
    """Whole-frame back-projection, camera frame, with exact per-pixel color
    (no crop, so no need for the approximate texture lookup the per-object
    viewers use) plus the same depth-discontinuity rejection
    reconstruction_3d/back_projector.py applies."""
    h, w = depth_m.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy

    dx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
    stable = np.hypot(dx, dy) < discontinuity_thresh
    valid = (depth_m > 0) & stable

    points = np.stack([x[valid], y[valid], depth_m[valid]], axis=1)
    colors = bgr[valid][:, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, [0,1]
    return points, colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')
    parser.add_argument('--config', default=os.path.join(REPO_ROOT, 'config', 'params.yaml'))
    parser.add_argument('--stride', type=int, default=5, help='use every Nth rgb frame')
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default=None, help='defaults to <dataset_path>/gs_map.ply')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    cam = config['mapping_node']['ros__parameters']
    depth_scale = config['dataset_streamer']['ros__parameters']['depth_scale']
    fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    rgb_entries = read_tum_file(os.path.join(args.dataset_path, 'rgb.txt'))[::args.stride]
    depth_entries = read_tum_file(os.path.join(args.dataset_path, 'depth.txt'))
    gt_entries = read_tum_file(os.path.join(args.dataset_path, 'groundtruth.txt'))
    print(f"{len(rgb_entries)} frames (stride {args.stride})")

    clouds, colors_all, views = [], [], []
    for ts, fields in rgb_entries:
        bgr = cv2.imread(os.path.join(args.dataset_path, fields[0]), cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        _, depth_fields = nearest(depth_entries, ts)
        raw = cv2.imread(os.path.join(args.dataset_path, depth_fields[0]), cv2.IMREAD_UNCHANGED)
        depth_m = raw.astype(np.float32) / depth_scale

        _, gt_fields = nearest(gt_entries, ts)
        tx, ty, tz, qx, qy, qz, qw = (float(v) for v in gt_fields)
        R, t = quat_to_matrix(qx, qy, qz, qw), np.array([tx, ty, tz])

        points_cam, colors = backproject_frame(bgr, depth_m, fx, fy, cx, cy)
        clouds.append(points_cam @ R.T + t)   # world frame, same as Tracker.update()
        colors_all.append(colors)

        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = R
        T_world_cam[:3, 3] = t
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        views.append(View(T_world_cam=T_world_cam, K=K, image=image))

    print(f"{sum(c.shape[0] for c in clouds)} raw points across {len(clouds)} views")

    print("cleaning/densifying...")
    reconstructed = reconstruct_object(clouds)
    raw_points = np.concatenate(clouds, axis=0)
    raw_colors = np.concatenate(colors_all, axis=0)
    _, nn_idx = cKDTree(raw_points).query(reconstructed, k=1)
    colors = raw_colors[nn_idx]
    print(f"{reconstructed.shape[0]} points going into the joint GS fit")

    print("fitting ONE Gaussian Splat map (needs CUDA + gsplat)...")
    gs = GaussianSplatter(iters=args.iters, device=args.device).reconstruct(
        reconstructed, colors, views)

    output = args.output or os.path.join(args.dataset_path, 'gs_map.ply')
    gs.export_ply(output)
    print(f"saved {output}")

    import open3d as o3d
    o3d.visualization.draw_geometries(
        [o3d.io.read_point_cloud(output)], window_name=f"GS map -- {args.dataset_path}")


if __name__ == '__main__':
    main()
