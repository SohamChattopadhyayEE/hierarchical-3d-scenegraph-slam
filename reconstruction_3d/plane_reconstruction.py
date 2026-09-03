#!/usr/bin/env python3
"""Extracts large planes (walls/floor) from a mapped scene: iterates the
saved segmentation masks, takes each mask's BLACK (background/non-object)
pixels, back-projects the corresponding depth into world-frame points, then
RANSAC-fits planes across all of them.

By default it stops there and shows the extracted planes -- one color per
plane -- so extraction can be checked by eye. --texture instead paints them
with their real image colors (carried through the voxel downsample as a
per-voxel average of the pixels that produced each point, so it is the
actual observed color, not an approximate lookup). --gs additionally fits
Gaussian Splatting to them (same technique as gs_map_from_rgbd.py /
combined_complete_map.py, needs CUDA + gsplat).

Only physically LARGE planes are kept: both in-plane dimensions must reach
--min_extent metres. Point count alone doesn't distinguish a wall from a
slab through desk clutter.

Usage: python3 reconstruction_3d/plane_reconstruction.py \
           datasets/rgbd_dataset_freiburg1_desk \
           datasets/segmented_masks/fastsam/mask
       ... --texture                 # planes painted with real image colors
       ... --min_extent 1.5          # stricter: only really big surfaces
       ... --gs                      # also run Gaussian Splatting
"""

import argparse
import colorsys
import glob
import os
import sys

import cv2
import numpy as np
import open3d as o3d
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from mapping_from_file import nearest, quat_to_matrix, read_tum_file  # noqa: E402
from reconstruction_3d.gaussian_splatter import GaussianSplatter, View  # noqa: E402
from reconstruction_3d.reconstruct import oriented_box  # noqa: E402


def backproject_background(bgr, depth_m, background, fx, fy, cx, cy,
                            discontinuity_thresh=0.05):
    """Back-projects only background==True pixels, camera frame, with exact
    per-pixel color and the same depth-discontinuity rejection
    reconstruction_3d/back_projector.py applies."""
    h, w = depth_m.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy

    dx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
    stable = np.hypot(dx, dy) < discontinuity_thresh
    valid = background & (depth_m > 0) & stable

    points = np.stack([x[valid], y[valid], depth_m[valid]], axis=1)
    colors = bgr[valid][:, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, [0,1]
    return points, colors


def voxel_resample_colored(points, colors, voxel):
    """voxel_resample() from reconstruct.py, but averaging colors per voxel
    too -- so a downsampled point keeps the real image color of the pixels
    that produced it, instead of needing a lookup back into the full cloud."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n = int(inv.max()) + 1
    p_sums, c_sums, counts = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n)
    np.add.at(p_sums, inv, points)
    np.add.at(c_sums, inv, colors)
    np.add.at(counts, inv, 1.0)
    return p_sums / counts[:, None], c_sums / counts[:, None]


def extract_planes(points, distance_threshold=0.02, ransac_n=3, num_iterations=1000,
                    min_plane_points=2000, min_extent=1.0, max_planes=6):
    """Iterative RANSAC: repeatedly fits the biggest remaining plane and
    removes its inliers, so multiple planes come out instead of just the
    single dominant one.

    A candidate only counts as a wall/floor if it is physically LARGE: both
    of its in-plane dimensions (PCA extents, ignoring thickness) must reach
    min_extent metres. Point count alone is a useless test -- on a dense
    cloud any 2cm slab through desk clutter holds thousands of points, which
    is exactly how a cluttered scene ends up "86% planes". Rejected
    candidates still get their inliers removed, otherwise the next round
    refits the identical slab forever.

    Returns a list of INDEX arrays into `points`, one per plane, so a
    caller can slice matching per-point colors with the same indices.
    """
    remaining_idx = np.arange(points.shape[0])
    planes = []
    for _ in range(max_planes * 4):          # attempts, not accepted planes
        if len(planes) >= max_planes:
            break
        if remaining_idx.shape[0] < max(ransac_n, min_plane_points):
            break

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[remaining_idx])
        _, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
        if len(inliers) < min_plane_points:
            break                            # nothing big enough left at all

        inlier_mask = np.zeros(remaining_idx.shape[0], dtype=bool)
        inlier_mask[inliers] = True
        candidate_idx = remaining_idx[inlier_mask]
        remaining_idx = remaining_idx[~inlier_mask]

        _, _, extent = oriented_box(points[candidate_idx])
        in_plane = np.sort(extent)[::-1][:2]  # two largest = the plane's own dims
        if in_plane[0] >= min_extent and in_plane[1] >= min_extent:
            planes.append(candidate_idx)
            print(f"  plane {len(planes)}: {candidate_idx.shape[0]} pts, "
                  f"{in_plane[0]:.2f}m x {in_plane[1]:.2f}m")
        else:
            print(f"  rejected: {candidate_idx.shape[0]} pts but only "
                  f"{in_plane[0]:.2f}m x {in_plane[1]:.2f}m (< {min_extent}m)")
    return planes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='e.g. datasets/rgbd_dataset_freiburg1_desk')
    parser.add_argument('mask_dir', help='e.g. datasets/segmented_masks/fastsam/mask')
    parser.add_argument('--config', default=os.path.join(REPO_ROOT, 'config', 'params.yaml'))
    parser.add_argument('--stride', type=int, default=5, help='use every Nth mask frame')
    parser.add_argument('--distance_threshold', type=float, default=0.02,
                         help='RANSAC plane-fit tolerance, metres')
    parser.add_argument('--max_planes', type=int, default=6)
    parser.add_argument('--min_extent', type=float, default=1.0,
                         help='metres; both in-plane dimensions must reach this, so only '
                              'long/large walls and floors qualify -- not desk tops or '
                              'clutter slabs')
    parser.add_argument('--min_plane_points', type=int, default=2000)
    parser.add_argument('--voxel', type=float, default=0.03,
                         help='metres; voxel-downsample the background cloud BEFORE '
                              'RANSAC. 0 disables.')
    parser.add_argument('--texture', action='store_true',
                         help='color the planes with their real image colors instead of '
                              'one flat color per plane')
    parser.add_argument('--gs', action='store_true',
                         help='also fit Gaussian Splatting to the extracted planes '
                              '(needs CUDA + gsplat); off by default, extraction only')
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default=None, help='defaults to <mask_dir>/../planes.ply')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    cam = config['mapping_node']['ros__parameters']
    depth_scale = config['dataset_streamer']['ros__parameters']['depth_scale']
    fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    depth_entries = read_tum_file(os.path.join(args.dataset_path, 'depth.txt'))
    gt_entries = read_tum_file(os.path.join(args.dataset_path, 'groundtruth.txt'))
    mask_paths = sorted(glob.glob(os.path.join(args.mask_dir, '*.png')))[::args.stride]
    print(f"{len(mask_paths)} mask frames (stride {args.stride})")

    points_all, colors_all, views = [], [], []
    for mask_path in mask_paths:
        ts = float(os.path.splitext(os.path.basename(mask_path))[0])
        mask_img = cv2.imread(mask_path)
        background = np.all(mask_img == 0, axis=-1)
        if not np.any(background):
            continue

        rgb_path = os.path.join(args.dataset_path, 'rgb', os.path.basename(mask_path))
        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        _, depth_fields = nearest(depth_entries, ts)
        raw = cv2.imread(os.path.join(args.dataset_path, depth_fields[0]), cv2.IMREAD_UNCHANGED)
        depth_m = raw.astype(np.float32) / depth_scale

        _, gt_fields = nearest(gt_entries, ts)
        tx, ty, tz, qx, qy, qz, qw = (float(v) for v in gt_fields)
        R, t = quat_to_matrix(qx, qy, qz, qw), np.array([tx, ty, tz])

        points_cam, colors = backproject_background(bgr, depth_m, background, fx, fy, cx, cy)
        if points_cam.shape[0] == 0:
            continue
        points_all.append(points_cam @ R.T + t)  # world frame, same as Tracker.update()
        colors_all.append(colors)

        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = R
        T_world_cam[:3, 3] = t
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        views.append(View(T_world_cam=T_world_cam, K=K, image=image))

    if not points_all:
        print("No background points found -- nothing to reconstruct.")
        sys.exit(1)

    raw_points = np.concatenate(points_all, axis=0)
    raw_colors = np.concatenate(colors_all, axis=0)
    print(f"{raw_points.shape[0]} raw background points across {len(views)} views")

    # Downsample BEFORE RANSAC: on a 20M-point cloud the thresholds below are
    # meaningless (any thin slab holds thousands of points) and each fit is
    # needlessly slow. Colors are averaged per voxel alongside the positions,
    # so every search point keeps the real image color of the pixels behind it.
    search_points, search_colors = raw_points, raw_colors
    if args.voxel > 0:
        search_points, search_colors = voxel_resample_colored(
            raw_points, raw_colors, args.voxel)
        print(f"voxel-downsampled @ {args.voxel * 1000:.0f}mm -> "
              f"{search_points.shape[0]} points for plane search")

    print("fitting planes with RANSAC...")
    plane_idx = extract_planes(search_points, distance_threshold=args.distance_threshold,
                                min_plane_points=args.min_plane_points,
                                min_extent=args.min_extent, max_planes=args.max_planes)
    if not plane_idx:
        print("No planes found -- loosen --min_extent / --min_plane_points, or raise "
              "--distance_threshold.")
        sys.exit(1)

    plane_points = np.concatenate([search_points[i] for i in plane_idx], axis=0)
    # Same indices -> the exact per-voxel image colors of those points.
    image_colors = np.concatenate([search_colors[i] for i in plane_idx], axis=0)
    print(f"\n{len(plane_idx)} large planes kept, {plane_points.shape[0]} points "
          f"({100 * plane_points.shape[0] / search_points.shape[0]:.0f}% of the background)")

    if args.texture:
        plane_colors = image_colors
        label = f"{len(plane_idx)} planes -- original image colors"
    else:
        # Distinct color per plane so extraction is verifiable by eye, same
        # golden-ratio hue spacing colorize_label_mask uses.
        plane_colors = np.concatenate([
            np.tile(colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.85, 0.95),
                    (idx.shape[0], 1))
            for i, idx in enumerate(plane_idx)
        ])
        label = f"{len(plane_idx)} planes -- one color each"

    output = args.output or os.path.join(os.path.dirname(args.mask_dir), 'planes.ply')
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(plane_points)
    cloud.colors = o3d.utility.Vector3dVector(plane_colors)
    o3d.io.write_point_cloud(output, cloud)
    print(f"saved {output}")

    o3d.visualization.draw_geometries([cloud], window_name=label)

    if not args.gs:
        return

    # --gs: refit the planes as Gaussians, seeded with the real image colors
    # (image_colors, already carried through the voxel downsample -- no
    # nearest-neighbour lookup into the full 20M-point cloud needed).
    print("fitting ONE Gaussian Splat model for the walls/floor (needs CUDA + gsplat)...")
    gs = GaussianSplatter(iters=args.iters, device=args.device).reconstruct(
        plane_points, image_colors, views)

    gs_output = os.path.splitext(output)[0] + '_gs.ply'
    gs.export_ply(gs_output)
    print(f"saved {gs_output}")

    o3d.visualization.draw_geometries(
        [o3d.io.read_point_cloud(gs_output)], window_name=f"planes (GS) -- {args.mask_dir}")


if __name__ == '__main__':
    main()
