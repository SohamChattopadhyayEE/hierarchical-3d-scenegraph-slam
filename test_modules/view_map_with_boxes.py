#!/usr/bin/env python3
"""Overlays a 3D bounding box per tracked object on top of one or more
reconstructed maps (planes.ply, combined_map.ply, gs_map.ply, ...), all in
one open3d scene. Every one of these lives in the same map/world frame, so
they line up when shown together; each .ply keeps its own colors.

Boxes stay upright -- Z is world-vertical -- while X/Y follow the PCA of
each object's footprint, so a rotated object gets a tight box without the
box itself tilting.

Each object's box is fit to its RECONSTRUCTED points (reconstruct_object's
multi-view fusion + consistency filtering), not the raw accumulated ones, so
noise/fliers don't inflate the box.

Usage: python3 test_modules/view_map_with_boxes.py datasets/objects 3 7 12 \
           --ply datasets/objects/combined_map.ply
       python3 test_modules/view_map_with_boxes.py datasets/objects --all \
           --ply datasets/objects/combined_map.ply planes.ply gs_map.ply
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


def discover_object_ids(objects_root: str) -> list:
    """Every subfolder holding tracked-object output (skips stray files like
    a combined_map.ply sitting in that same directory)."""
    ids = [d for d in os.listdir(objects_root)
           if os.path.isdir(os.path.join(objects_root, d))
           and glob.glob(os.path.join(objects_root, d, '*_points.npy'))]
    return sorted(ids, key=lambda d: int(d) if d.isdigit() else d)


def zaligned_pca_box_params(points, percentile: float = 1.0):
    """(center, R, extent) for a box kept upright in Z while X/Y follow the
    PCA of the footprint -- so it hugs a rotated object without tilting.

    Bounds come from percentiles, not min/max: the extremes are exactly
    where the stray points that survived reconstruct_object() live, and a
    single one of them would stretch the whole box. percentile=1 clips the
    outer 1% at each end per axis; 0.5-2 is the useful range (higher =
    tighter, but starts eating real extent).

    Built directly rather than hull-fitted, so unlike get_oriented_bounding_box()
    it can't fail on objects that reduce to a few coplanar/near-coincident points.
    """
    xy = points[:, :2]
    mean_xy = xy.mean(axis=0)
    centered = xy - mean_xy

    # Principal directions of the footprint, major axis first.
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
    R2 = eigvecs[:, np.argsort(eigvals)[::-1]]
    if np.linalg.det(R2) < 0:            # keep it a rotation, not a reflection
        R2[:, 1] *= -1

    # Coordinates in the box's own frame: PCA axes for x/y, world z for z.
    local = np.column_stack([centered @ R2, points[:, 2]])
    lo = np.percentile(local, percentile, axis=0)
    hi = np.percentile(local, 100 - percentile, axis=0)
    mid = (lo + hi) / 2

    center_xy = mean_xy + mid[:2] @ R2.T
    R = np.eye(3)
    R[:2, :2] = R2                        # Z column stays [0, 0, 1]
    center = np.array([center_xy[0], center_xy[1], mid[2]])
    return center, R, hi - lo


def object_bbox(folder: str, min_points: int = 10, percentile: float = 1.0):
    """((center, R, extent), n_points) for a Z-aligned, PCA-oriented box
    around this object's reconstructed points -- or (None, n_points) if
    there aren't enough of them to mean anything."""
    clouds = [np.load(p) for p in sorted(glob.glob(os.path.join(folder, '*_points.npy')))]
    points = reconstruct_object(clouds, verbose=False)
    if points.shape[0] < min_points:
        return None, points.shape[0]
    return zaligned_pca_box_params(points, percentile), points.shape[0]


def _align_z_to(direction):
    """Rotation taking +Z onto `direction` (Rodrigues)."""
    d = direction / np.linalg.norm(direction)
    v = np.cross([0.0, 0.0, 1.0], d)
    s, c = np.linalg.norm(v), float(d[2])
    if s < 1e-9:                                  # already parallel
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def box_edge_mesh(center, R, extent, color, radius):
    """The box's 12 edges as thin cylinders.

    draw_geometries() draws a LineSet's edges through glLineWidth, which
    modern OpenGL core profiles clamp to 1px -- so its line width setting
    does nothing. Real geometry has real thickness, and renders in the
    normal viewer with no change to it.
    """
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = center + (signs * (extent / 2.0)) @ R.T

    mesh = o3d.geometry.TriangleMesh()
    for i in range(8):
        for k in range(3):
            j = i ^ (1 << k)          # neighbour differing in exactly one axis
            if j < i:                 # each edge once
                continue
            p0, p1 = corners[i], corners[j]
            d = p1 - p0
            length = float(np.linalg.norm(d))
            if length < 1e-9:         # zero-thickness object along this axis
                continue
            cyl = o3d.geometry.TriangleMesh.create_cylinder(
                radius=radius, height=length, resolution=8)
            cyl.rotate(_align_z_to(d), center=(0.0, 0.0, 0.0))
            cyl.translate((p0 + p1) / 2.0)
            mesh += cyl

    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('objects_root', help='e.g. datasets/objects')
    parser.add_argument('object_ids', nargs='*',
                         help='objects to draw boxes for (ignored if --all is given)')
    parser.add_argument('--all', action='store_true',
                         help='box every object folder under objects_root')
    parser.add_argument('--ply', nargs='*', default=[],
                         help='map .ply file(s) to show underneath the boxes')
    parser.add_argument('--line_width', type=float, default=0.005,
                         help='box edge thickness in METRES (default 0.005 = 5mm). Edges '
                              'are drawn as cylinders because a wireframe is stuck at 1px '
                              'in this viewer regardless of any width setting.')
    parser.add_argument('--percentile', type=float, default=1.0,
                         help='clip this %% off each end of every axis when sizing a box, '
                              'so stray points cannot inflate it (0.5-2 is the useful '
                              'range; higher = tighter). 0 = plain min/max.')
    args = parser.parse_args()

    if args.all:
        object_ids = discover_object_ids(args.objects_root)
        print(f"--all: found {len(object_ids)} object folders under {args.objects_root}")
    elif args.object_ids:
        object_ids = args.object_ids
    else:
        parser.error('provide object_ids or use --all')

    clouds = []
    for path in args.ply:
        if not os.path.isfile(path):
            print(f"{path}: not found, skipping")
            continue
        cloud = o3d.io.read_point_cloud(path)
        if len(cloud.points) == 0:
            print(f"{path}: no points, skipping")
            continue
        print(f"{path}: {len(cloud.points)} points")
        clouds.append(cloud)

    boxes = []
    for i, obj_id in enumerate(object_ids):
        folder = os.path.join(args.objects_root, str(obj_id))
        if not glob.glob(os.path.join(folder, '*_points.npy')):
            print(f"object {obj_id}: no *_points.npy found in {folder}, skipping")
            continue

        hue = (i * 0.618033988749895) % 1.0  # same golden-ratio spacing as colorize_label_mask
        color = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        params, n_points = object_bbox(folder, percentile=args.percentile)
        if params is None:
            print(f"object {obj_id}: only {n_points} points after reconstruction, skipping")
            continue

        center, R, extent = params
        print(f"object {obj_id}: {n_points} pts, center={np.round(center, 3)} "
              f"size={np.round(extent, 3)}")
        boxes.append(box_edge_mesh(center, R, extent, color, args.line_width))

    if not clouds and not boxes:
        print("Nothing to show.")
        sys.exit(1)

    title = f"{len(clouds)} map(s) + {len(boxes)} object boxes"
    print(f"showing {title}")
    o3d.visualization.draw_geometries(clouds + boxes, window_name=title)


if __name__ == '__main__':
    main()
