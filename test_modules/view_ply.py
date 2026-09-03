#!/usr/bin/env python3
"""Visualizes a .ply file (e.g. combined_map.ply / gs_map.ply) in open3d.

Usage: python3 test_modules/view_ply.py datasets/objects/combined_map.ply
"""

import sys

import open3d as o3d


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to.ply>")
        sys.exit(1)

    cloud = o3d.io.read_point_cloud(sys.argv[1])
    print(f"{sys.argv[1]}: {len(cloud.points)} points")
    o3d.visualization.draw_geometries([cloud], window_name=sys.argv[1])


if __name__ == '__main__':
    main()
