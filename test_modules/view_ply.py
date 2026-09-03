#!/usr/bin/env python3
"""Visualizes one or more .ply files together in one open3d window, each
keeping its own colors.

Usage: python3 test_modules/view_ply.py datasets/objects/combined_map.ply
       python3 test_modules/view_ply.py planes.ply combined_map.ply gs_map.ply
"""

import os
import sys

import open3d as o3d


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to.ply> [more.ply ...]")
        sys.exit(1)

    clouds = []
    for path in sys.argv[1:]:
        if not os.path.isfile(path):
            print(f"{path}: not found, skipping")
            continue
        cloud = o3d.io.read_point_cloud(path)
        if len(cloud.points) == 0:
            print(f"{path}: no points, skipping")
            continue
        print(f"{path}: {len(cloud.points)} points")
        clouds.append(cloud)

    if not clouds:
        print("Nothing to show.")
        sys.exit(1)

    total = sum(len(c.points) for c in clouds)
    print(f"showing {len(clouds)} file(s), {total} points total")
    o3d.visualization.draw_geometries(
        clouds, window_name=' + '.join(os.path.basename(p) for p in sys.argv[1:]))


if __name__ == '__main__':
    main()
