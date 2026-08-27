#!/usr/bin/env python3
"""Visualizes one <timestamp>_points.npy file (Nx3 world-frame points) saved
by reconstruction_3d/object_tracker.py, as a 3D scatter plot.

Usage: python3 test_modules/view_pointcloud.py datasets/objects/3/1305031452.9_points.npy
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  -- registers the '3d' projection


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_points.npy>")
        sys.exit(1)

    points = np.load(sys.argv[1])
    print(f"{points.shape[0]} points, x/y/z range: "
          f"{points.min(axis=0)} to {points.max(axis=0)}")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(sys.argv[1])
    plt.show()


if __name__ == '__main__':
    main()
