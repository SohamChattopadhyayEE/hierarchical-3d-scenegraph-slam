#!/usr/bin/env python3
"""Reconstructs one tracked object's shape from its accumulated per-frame
point clouds (reconstruction_3d/reconstruct.py's multi-view fusion +
consistency filtering) and shows raw-fused vs. reconstructed side by side,
then its dense reconstruction two ways: reconstruction_3d/poisson_mesher.py's
PoissonMesher, and reconstruction_3d/gaussian_splatter.py's GaussianSplatter
-- real, trained 3D Gaussian Splatting (needs CUDA + gsplat; posed views are
built by that module's load_views() from what Tracker._save writes per
frame: <ts>.png crops + <ts>_pose.npz camera pose/bbox).

Each raw point is colored by its frame's original image texture (same
lookup as view_multiview_pointcloud.py's texture_colors). reconstruct_object
only outputs geometry (voxel-averaged points, not the original ones), so
color isn't carried through it -- each reconstructed point instead borrows
the color of its nearest raw point (scipy cKDTree), same as
view_objects_map.py.

Usage: python3 test_modules/view_reconstructed_object.py datasets/objects/3
"""

import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  -- registers the '3d' projection
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from view_multiview_pointcloud import texture_colors  # noqa: E402
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from reconstruction_3d.reconstruct import reconstruct_object  # noqa: E402
from reconstruction_3d.poisson_mesher import PoissonMesher  # noqa: E402
from reconstruction_3d.gaussian_splatter import GaussianSplatter, load_views  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <object_folder>")
        sys.exit(1)

    paths = sorted(glob.glob(os.path.join(sys.argv[1], '*_points.npy')))
    if not paths:
        print(f"No *_points.npy files found in {sys.argv[1]}")
        sys.exit(1)

    clouds, colors_all = [], []
    for path in paths:
        points = np.load(path)
        crop_path = path[:-len('_points.npy')] + '.png'
        crop = cv2.imread(crop_path) if os.path.isfile(crop_path) else None
        if crop is not None:
            colors = texture_colors(points, crop)
        else:
            colors = np.full((points.shape[0], 3), 0.5, dtype=np.float32)  # gray fallback
        clouds.append(points)
        colors_all.append(colors)

    raw = np.concatenate(clouds, axis=0)
    raw_colors = np.concatenate(colors_all, axis=0)
    print(f"{len(clouds)} views, {raw.shape[0]} raw points total")

    clean = reconstruct_object(clouds)
    _, nn_idx = cKDTree(raw).query(clean, k=1)
    clean_colors = raw_colors[nn_idx]

    fig = plt.figure(figsize=(10, 5))

    ax_raw = fig.add_subplot(1, 2, 1, projection='3d')
    ax_raw.scatter(raw[:, 0], raw[:, 1], raw[:, 2], s=1, c=raw_colors)
    ax_raw.set_title(f'raw fused ({raw.shape[0]} pts)')

    ax_clean = fig.add_subplot(1, 2, 2, projection='3d')
    ax_clean.scatter(clean[:, 0], clean[:, 1], clean[:, 2], s=3, c=clean_colors)
    ax_clean.set_title(f'reconstructed ({clean.shape[0]} pts)')

    for ax in (ax_raw, ax_clean):
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    fig.suptitle(sys.argv[1])
    plt.show()

    print("Running Poisson reconstruction for a dense mesh...")
    mesh = PoissonMesher().reconstruct(clean, clean_colors)
    print(f"mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    import open3d as o3d
    o3d.visualization.draw_geometries([mesh], window_name=f"{sys.argv[1]} -- Poisson mesh")

    print("Fitting Gaussian Splatting (alternative to Poisson; needs CUDA + gsplat)...")
    with open(os.path.join(REPO_ROOT, 'config', 'params.yaml')) as f:
        cam = yaml.safe_load(f)['mapping_node']['ros__parameters']
    views = load_views(sys.argv[1], cam['fx'], cam['fy'], cam['cx'], cam['cy'])
    if not views:
        print(f"no *_pose.npz files in {sys.argv[1]} -- run the pipeline again "
              "after this change to produce them, skipping Gaussian Splatting")
    else:
        gs = GaussianSplatter().reconstruct(clean, clean_colors, views)
        ply_path = os.path.join(sys.argv[1], 'gaussian_splat.ply')
        gs.export_ply(ply_path)
        print(f"saved {ply_path}")
        o3d.visualization.draw_geometries(
            [o3d.io.read_point_cloud(ply_path)],
            window_name=f"{sys.argv[1]} -- Gaussian splats")


if __name__ == '__main__':
    main()
