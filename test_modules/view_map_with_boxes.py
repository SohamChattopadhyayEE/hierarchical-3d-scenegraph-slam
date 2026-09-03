#!/usr/bin/env python3
"""Overlays a 3D oriented bounding box per tracked object onto a
reconstructed map (combined_map.ply for now; gs_map.ply later -- both are
just point clouds in the same map/world frame, so this works for either),
in one open3d scene.

Each object's box is fit to its RECONSTRUCTED points (reconstruct_object's
multi-view fusion + consistency filtering), not the raw accumulated ones, so
noise/fliers don't inflate the box.

Usage: python3 test_modules/view_map_with_boxes.py datasets/objects/combined_map.ply datasets/objects 3 7 12
"""

import argparse
import colorsys
import glob
import os
import sys

import numpy as np
import open3d as o3d

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from reconstruction_3d.reconstruct import reconstruct_object  # noqa: E402


def object_bbox(folder: str, color) -> o3d.geometry.OrientedBoundingBox:
    clouds = [np.load(p) for p in sorted(glob.glob(os.path.join(folder, '*_points.npy')))]
    points = reconstruct_object(clouds, verbose=False)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    box = pcd.get_oriented_bounding_box()
    box.color = color
    return box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('map_path', help='e.g. datasets/objects/combined_map.ply')
    parser.add_argument('objects_root', help='e.g. datasets/objects')
    parser.add_argument('object_ids', nargs='+')
    args = parser.parse_args()

    scene = o3d.io.read_point_cloud(args.map_path)
    print(f"{args.map_path}: {len(scene.points)} points")

    boxes = []
    for i, obj_id in enumerate(args.object_ids):
        folder = os.path.join(args.objects_root, str(obj_id))
        if not glob.glob(os.path.join(folder, '*_points.npy')):
            print(f"object {obj_id}: no *_points.npy found in {folder}, skipping")
            continue

        hue = (i * 0.618033988749895) % 1.0  # same golden-ratio spacing as colorize_label_mask
        color = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        box = object_bbox(folder, color)
        print(f"object {obj_id}: bbox center={box.center} extent={box.extent}")
        boxes.append(box)

    o3d.visualization.draw_geometries(
        [scene] + boxes, window_name=f"{args.map_path} + object boxes")


if __name__ == '__main__':
    main()
