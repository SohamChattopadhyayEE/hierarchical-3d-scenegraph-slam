#!/usr/bin/env python3
"""Reconstructs a combined multi-object map with ONE joint Gaussian
Splatting fit, not one object at a time (that's view_objects_map.py).

Every requested object's raw per-frame clouds and posed views (from what
Tracker._save writes: <ts>_points.npy, <ts>.png, <ts>_pose.npz) are pooled
together BEFORE reconstruction -- reconstruct_object() cleans/densifies the
whole pooled point cloud as one scene (it doesn't need per-object identity,
just geometry), and GaussianSplatter then fits ONE set of Gaussians against
the combined view pool. More views supervising one shared parameter set and
one shared densification pass is what makes this smoother/less seamy at
object boundaries than reconstructing each object in isolation.

Usage: python3 test_modules/combined_complete_map.py datasets/objects 3 7 12
       python3 test_modules/combined_complete_map.py datasets/objects 3 7 12 \
           --iters 4000 --output datasets/objects/combined_map.ply
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from view_multiview_pointcloud import texture_colors  # noqa: E402
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from reconstruction_3d.reconstruct import reconstruct_object  # noqa: E402
from reconstruction_3d.gaussian_splatter import GaussianSplatter, load_views  # noqa: E402


def load_object(folder: str):
    """Raw per-frame clouds + texture colors + posed views for one object."""
    clouds, colors_all = [], []
    for path in sorted(glob.glob(os.path.join(folder, '*_points.npy'))):
        points = np.load(path)
        crop_path = path[:-len('_points.npy')] + '.png'
        crop = cv2.imread(crop_path) if os.path.isfile(crop_path) else None
        if crop is not None:
            colors = texture_colors(points, crop)
        else:
            colors = np.full((points.shape[0], 3), 0.5, dtype=np.float32)
        clouds.append(points)
        colors_all.append(colors)
    return clouds, colors_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('objects_root', help='e.g. datasets/objects')
    parser.add_argument('object_ids', nargs='+')
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default=None,
                         help='defaults to <objects_root>/combined_map.ply')
    args = parser.parse_args()

    with open(os.path.join(REPO_ROOT, 'config', 'params.yaml')) as f:
        cam = yaml.safe_load(f)['mapping_node']['ros__parameters']

    all_clouds, all_colors, all_views = [], [], []
    for obj_id in args.object_ids:
        folder = os.path.join(args.objects_root, str(obj_id))
        clouds, colors = load_object(folder)
        if not clouds:
            print(f"object {obj_id}: no *_points.npy found in {folder}, skipping")
            continue
        views = load_views(folder, cam['fx'], cam['fy'], cam['cx'], cam['cy'])
        print(f"object {obj_id}: {sum(c.shape[0] for c in clouds)} raw points, "
              f"{len(clouds)} frames, {len(views)} posed views")
        all_clouds += clouds
        all_colors += colors
        all_views += views

    if not all_clouds:
        print("Nothing to reconstruct.")
        sys.exit(1)
    if not all_views:
        print("No *_pose.npz files found across the requested objects -- "
              "re-run the mapping pipeline to produce them (see Tracker._save).")
        sys.exit(1)

    print(f"\npooling {len(args.object_ids)} objects: "
          f"{sum(c.shape[0] for c in all_clouds)} raw points, {len(all_views)} views")

    raw = np.concatenate(all_clouds, axis=0)
    raw_colors = np.concatenate(all_colors, axis=0)

    print("cleaning/densifying the combined scene...")
    reconstructed = reconstruct_object(all_clouds)
    _, nn_idx = cKDTree(raw).query(reconstructed, k=1)
    colors = raw_colors[nn_idx]
    print(f"combined scene: {reconstructed.shape[0]} points going into the joint GS fit")

    print("fitting ONE Gaussian Splat model across the whole combined scene "
          "(needs CUDA + gsplat)...")
    gs = GaussianSplatter(iters=args.iters, device=args.device).reconstruct(
        reconstructed, colors, all_views)

    output = args.output or os.path.join(args.objects_root, 'combined_map.ply')
    gs.export_ply(output)
    print(f"saved {output}")

    import open3d as o3d
    o3d.visualization.draw_geometries(
        [o3d.io.read_point_cloud(output)],
        window_name=f"combined map -- objects {', '.join(args.object_ids)}")


if __name__ == '__main__':
    main()
