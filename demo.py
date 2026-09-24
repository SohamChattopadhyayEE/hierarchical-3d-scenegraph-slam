#!/usr/bin/env python3
"""Side-by-side demo: the RGB stream plays on the left while the 3D map
builds up on the right.

Objects appear one at a time in the order given by --objects, the camera
trajectory (from the dataset's groundtruth poses) grows as the stream
plays, and --planes is dropped in near the end. When the stream finishes, a
plain open3d window shows the finished maps (--combined_map + --planes)
with the robot's trajectory, and nothing else.

Usage:
  python3 demo.py --objects 22 21 23 \
      --images datasets/rgbd_dataset_freiburg1_desk/rgb \
      --planes datasets/segmented_masks_size_gated/fastsam/planes.ply \
      --combined_map datasets/objects_size_gated/combined_map.ply
"""

import argparse
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  -- registers the '3d' projection
from scipy.spatial import cKDTree

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'test_modules'))
from mapping_from_file import read_tum_file  # noqa: E402
from reconstruction_3d.reconstruct import reconstruct_object  # noqa: E402
from view_multiview_pointcloud import texture_colors  # noqa: E402


def load_object(folder, max_points):
    """Cleaned, image-colored points for one object, subsampled for the
    animated view (matplotlib redraws get sluggish well before open3d does)."""
    raw, colors = [], []
    for path in sorted(glob.glob(os.path.join(folder, '*_points.npy'))):
        points = np.load(path)
        crop_path = path[:-len('_points.npy')] + '.png'
        crop = cv2.imread(crop_path) if os.path.isfile(crop_path) else None
        colors.append(texture_colors(points, crop) if crop is not None
                      else np.full((points.shape[0], 3), 0.5, np.float32))
        raw.append(points)
    if not raw:
        return None, None

    clouds, raw_points = raw, np.concatenate(raw)
    raw_colors = np.concatenate(colors)
    points = reconstruct_object(clouds, verbose=False)
    _, nn = cKDTree(raw_points).query(points, k=1)
    points, colors = points, raw_colors[nn]

    if points.shape[0] > max_points:                 # keep the animation fluid
        keep = np.random.default_rng(0).choice(points.shape[0], max_points, replace=False)
        points, colors = points[keep], colors[keep]
    return points, colors


def grab_frame(fig):
    """The figure as a BGR array, ready for cv2.VideoWriter."""
    fig.canvas.draw()
    return cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)


def load_ply(path, max_points):
    cloud = o3d.io.read_point_cloud(path)
    points = np.asarray(cloud.points)
    colors = (np.asarray(cloud.colors) if cloud.has_colors()
              else np.full((points.shape[0], 3), 0.6))
    if points.shape[0] > max_points:
        keep = np.random.default_rng(0).choice(points.shape[0], max_points, replace=False)
        points, colors = points[keep], colors[keep]
    return points, colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', nargs='+', required=True,
                         help='object ids, revealed in this order')
    parser.add_argument('--images', required=True, help='folder of RGB frames')
    parser.add_argument('--planes', default=None, help='planes .ply, added near the end')
    parser.add_argument('--combined_map', default=None,
                         help='combined map .ply, shown in the final window')
    parser.add_argument('--objects_root', default='datasets/objects')
    parser.add_argument('--dataset', default=None,
                         help='dataset root holding groundtruth.txt '
                              '(default: parent of --images)')
    parser.add_argument('--stride', type=int, default=2, help='play every Nth frame')
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--max_points', type=int, default=4000,
                         help='points per object in the animated pane')
    parser.add_argument('--save_video', default=None,
                         help='also record the run to this .mp4')
    args = parser.parse_args()

    dataset = args.dataset or os.path.dirname(os.path.abspath(args.images))
    gt = read_tum_file(os.path.join(dataset, 'groundtruth.txt'))
    gt_ts = np.array([t for t, _ in gt])
    gt_xyz = np.array([[float(f[0]), float(f[1]), float(f[2])] for _, f in gt])

    frames = sorted(glob.glob(os.path.join(args.images, '*.png')))[::args.stride]
    if not frames:
        print(f"No images found in {args.images}")
        sys.exit(1)

    print(f"loading {len(args.objects)} objects...")
    objects = []
    for obj_id in args.objects:
        points, colors = load_object(
            os.path.join(args.objects_root, str(obj_id)), args.max_points)
        if points is None:
            print(f"  object {obj_id}: no *_points.npy, skipping")
            continue
        print(f"  object {obj_id}: {points.shape[0]} points")
        objects.append((obj_id, points, colors))

    planes = load_ply(args.planes, args.max_points * 8) if args.planes else None
    if planes:
        print(f"planes: {planes[0].shape[0]} points")

    # Objects spread over the first 80% of the stream, planes at 90%, so the
    # map is visibly still being assembled while the video plays.
    reveal_at = [int(len(frames) * 0.8 * (k + 1) / max(len(objects), 1))
                 for k in range(len(objects))]
    planes_at = int(len(frames) * 0.9)

    # Fixed limits computed up front, so the view never jumps mid-animation.
    everything = [gt_xyz] + [p for _, p, _ in objects] + ([planes[0]] if planes else [])
    allpts = np.concatenate(everything)
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    pad = 0.05 * (hi - lo).max()

    plt.ion()
    fig = plt.figure(figsize=(13, 6))
    ax_img = fig.add_subplot(1, 2, 1)
    ax_img.axis('off')
    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
    ax_3d.set_xlim(lo[0] - pad, hi[0] + pad)
    ax_3d.set_ylim(lo[1] - pad, hi[1] + pad)
    ax_3d.set_zlim(lo[2] - pad, hi[2] + pad)
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z')

    im = ax_img.imshow(cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB))
    traj, = ax_3d.plot([], [], [], color='k', linewidth=2)

    next_obj, planes_shown, writer = 0, False, None
    for i, path in enumerate(frames):
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        im.set_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # Odometry grows with the stream: trajectory up to this frame's time.
        ts = float(os.path.splitext(os.path.basename(path))[0])
        seen = gt_ts <= ts
        traj.set_data_3d(gt_xyz[seen, 0], gt_xyz[seen, 1], gt_xyz[seen, 2])

        while next_obj < len(objects) and i >= reveal_at[next_obj]:
            obj_id, points, colors = objects[next_obj]
            ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c=colors)
            print(f"  [frame {i}] object {obj_id} added")
            next_obj += 1

        if planes and not planes_shown and i >= planes_at:
            ax_3d.scatter(planes[0][:, 0], planes[0][:, 1], planes[0][:, 2],
                          s=1, c=planes[1])
            print(f"  [frame {i}] planes added")
            planes_shown = True

        ax_img.set_title(f"frame {i + 1}/{len(frames)}")
        ax_3d.set_title(f"{next_obj}/{len(objects)} objects"
                        + (" + planes" if planes_shown else ""))
        plt.pause(1.0 / args.fps)

        if args.save_video:
            shot = grab_frame(fig)
            if writer is None:                     # size comes from the canvas
                h, w = shot.shape[:2]
                writer = cv2.VideoWriter(args.save_video,
                                          cv2.VideoWriter_fourcc(*'mp4v'),
                                          args.fps, (w, h))
                if not writer.isOpened():
                    print(f"could not open {args.save_video} for writing -- "
                          f"continuing without recording")
                    writer, args.save_video = None, None
            if writer is not None:
                writer.write(shot)

    if writer is not None:
        for _ in range(int(args.fps * 2)):         # hold the finished map ~2s
            writer.write(shot)
        writer.release()
        print(f"saved {args.save_video}")

    plt.ioff()
    print("\nstream finished -- close the plot window for the full 3D map")
    plt.show()

    # Final window: just the finished maps plus the robot's trajectory --
    # the per-object clouds have served their purpose in the animation.
    geoms = []
    for path in (args.combined_map, args.planes):
        if not path:
            continue
        if not os.path.isfile(path):
            print(f"{path}: not found, skipping")
            continue
        cloud = o3d.io.read_point_cloud(path)
        print(f"{path}: {len(cloud.points)} points")
        geoms.append(cloud)

    path_lines = o3d.geometry.LineSet()
    path_lines.points = o3d.utility.Vector3dVector(gt_xyz)
    path_lines.lines = o3d.utility.Vector2iVector(
        [[i, i + 1] for i in range(len(gt_xyz) - 1)])
    path_lines.paint_uniform_color([0.0, 0.0, 0.0])
    geoms.append(path_lines)

    o3d.visualization.draw_geometries(geoms, window_name="combined map + trajectory")


if __name__ == '__main__':
    main()
