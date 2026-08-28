#!/usr/bin/env python3
"""Interactive viewer for *_points.npy files in a folder: press Right/Left
to step through frames. One window, two side-by-side panels that update
together -- the 3D point cloud on the left, the original <ts>.png crop for
that same frame on the right.

Each point in the 3D window is colored by looking up its position in that
frame's saved <ts>.png crop (both files are already written by
reconstruction_3d/object_tracker.py's Tracker._save for every frame).
There's no exact per-pixel correspondence saved between the two, so this
approximates one: the crop is the object's bounding-box rectangle, and the
points span that same rectangle in camera-plane X/Y, so each point's X/Y is
min-max normalized against this frame's own point cloud and used to index
into the crop's pixel grid.

Usage: python3 test_modules/view_multiview_pointcloud.py datasets/objects/3
"""

import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  -- registers the '3d' projection


def texture_colors(points: np.ndarray, crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    x, y = points[:, 0], points[:, 1]
    nx = (x - x.min()) / (x.max() - x.min() + 1e-9)
    ny = (y - y.min()) / (y.max() - y.min() + 1e-9)
    cols = np.clip((nx * (w - 1)).astype(int), 0, w - 1)
    rows = np.clip((ny * (h - 1)).astype(int), 0, h - 1)
    bgr = crop_bgr[rows, cols]
    return bgr[:, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, [0,1] for matplotlib


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <folder_of_points.npy>")
        sys.exit(1)

    paths = sorted(glob.glob(os.path.join(sys.argv[1], '*_points.npy')))
    if not paths:
        print(f"No *_points.npy files found in {sys.argv[1]}")
        sys.exit(1)

    fig = plt.figure(figsize=(10, 5))
    ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax_img = fig.add_subplot(1, 2, 2)

    state = {'i': 0}

    def draw():
        path = paths[state['i']]
        points = np.load(path)
        crop_path = path[:-len('_points.npy')] + '.png'
        crop = cv2.imread(crop_path) if os.path.isfile(crop_path) else None
        label = f"{os.path.basename(path)}  ({state['i'] + 1}/{len(paths)})  -- Right/Left to step"

        ax_3d.cla()
        if crop is not None:
            ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], s=2,
                          c=texture_colors(points, crop))
        else:
            print(f"no matching crop for {os.path.basename(path)}, plotting uncolored")
            ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('z')
        ax_3d.set_title('point cloud')

        ax_img.cla()
        if crop is not None:
            ax_img.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        else:
            ax_img.text(0.5, 0.5, 'no crop image', ha='center', va='center')
        ax_img.set_title('original image')
        ax_img.axis('off')

        fig.suptitle(label)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right':
            state['i'] = min(state['i'] + 1, len(paths) - 1)
            draw()
        elif event.key == 'left':
            state['i'] = max(state['i'] - 1, 0)
            draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    draw()
    plt.show()


if __name__ == '__main__':
    main()
