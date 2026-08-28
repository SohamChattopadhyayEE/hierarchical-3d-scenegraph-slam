#!/usr/bin/env python3
"""Combines several tracked objects' RECONSTRUCTED point clouds -- all
already in the shared map/world frame (reconstruction_3d/object_tracker.py
saves points transformed by each frame's own pose) -- into one 3D scene.

Each object's raw per-frame clouds are cleaned/densified by
reconstruction_3d/reconstruct.py's reconstruct_object() (multi-view fusion +
consistency filtering). That step only outputs geometry (voxel-averaged
points, not the original ones), so color isn't carried through it -- instead,
each raw point is first colored by its original image texture (same lookup
as view_multiview_pointcloud.py's texture_colors), and each reconstructed
point then borrows the color of its nearest raw point (scipy cKDTree).

Usage: python3 test_modules/view_objects_map.py datasets/objects 3 7 12
"""

import argparse
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  -- registers the '3d' projection
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from view_multiview_pointcloud import texture_colors  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reconstruction_3d.reconstruct import reconstruct_object  # noqa: E402


def load_object(folder: str):
    """Loads every frame's raw points + texture colors for one object folder.

    Returns (clouds, raw_points, raw_colors): clouds is the list of
    per-frame arrays (reconstruct_object's expected input), raw_points/
    raw_colors are those same points concatenated, 1:1 with their colors.
    """
    clouds, colors_all = [], []
    for path in sorted(glob.glob(os.path.join(folder, '*_points.npy'))):
        points = np.load(path)
        crop_path = path[:-len('_points.npy')] + '.png'
        crop = cv2.imread(crop_path) if os.path.isfile(crop_path) else None
        if crop is not None:
            colors = texture_colors(points, crop)
        else:
            colors = np.full((points.shape[0], 3), 0.5, dtype=np.float32)  # gray fallback
        clouds.append(points)
        colors_all.append(colors)

    if not clouds:
        return None, None, None
    return clouds, np.concatenate(clouds), np.concatenate(colors_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('objects_root', help='e.g. datasets/objects')
    parser.add_argument('object_ids', nargs='+')
    args = parser.parse_args()

    all_points, all_colors = [], []
    for obj_id in args.object_ids:
        folder = os.path.join(args.objects_root, str(obj_id))
        clouds, raw_points, raw_colors = load_object(folder)
        if clouds is None:
            print(f"object {obj_id}: no *_points.npy found in {folder}, skipping")
            continue

        print(f"object {obj_id}: reconstructing from {raw_points.shape[0]} raw points "
              f"across {len(clouds)} views")
        reconstructed = reconstruct_object(clouds, verbose=False)

        # nearest-neighbor color transfer: reconstructed point -> closest raw point's color
        _, nn_idx = cKDTree(raw_points).query(reconstructed, k=1)
        colors = raw_colors[nn_idx]

        print(f"object {obj_id}: {reconstructed.shape[0]} reconstructed points")
        all_points.append(reconstructed)
        all_colors.append(colors)

    if not all_points:
        print("Nothing to show.")
        sys.exit(1)

    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=3, c=colors)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f"objects {', '.join(args.object_ids)}  ({points.shape[0]} points, reconstructed)")
    plt.show()


if __name__ == '__main__':
    main()
