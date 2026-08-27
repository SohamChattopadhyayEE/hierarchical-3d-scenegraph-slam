#!/usr/bin/env python3
"""Offline, non-ROS counterpart to mapping_node.py.

Iterates a TUM-format dataset directly (rgb.txt, depth.txt, groundtruth.txt)
and runs the exact same segmentation -> back-projection -> tracking -> save
pipeline as mapping_node.py, without ROS -- dataset in, dataset out.
"""

import argparse
import bisect
import os

import cv2
import numpy as np
import yaml

from object_segmentor.fastsam_segmentor import FastSAMSegmentor
from object_segmentor.mobilesam_segmentor import MobileSAMSegmentor
from object_segmentor.visualization import colorize_label_mask
from reconstruction_3d.back_projector import BackProjector
from reconstruction_3d.object_tracker import Tracker

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, 'config', 'params.yaml')


def read_tum_file(path):
    entries = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            entries.append((float(fields[0]), fields[1:]))
    return entries


def nearest(entries, ts):
    """entries: sorted list of (timestamp, ...). Returns the closest one to ts."""
    times = [e[0] for e in entries]
    i = bisect.bisect_left(times, ts)
    candidates = [e for e in (entries[i - 1] if i > 0 else None,
                               entries[i] if i < len(entries) else None) if e]
    return min(candidates, key=lambda e: abs(e[0] - ts))


def quat_to_matrix(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def build_segmentor(params):
    segmentor_type = params['segmentor_type']
    if segmentor_type == 'fastsam':
        return segmentor_type, FastSAMSegmentor(
            model_path=params['fastsam_model_path'],
            conf_threshold=params['conf_threshold'],
            iou_threshold=params['iou_threshold'],
            device=params['device'],
        )
    if segmentor_type == 'mobilesam':
        return segmentor_type, MobileSAMSegmentor(
            checkpoint_path=params['mobilesam_checkpoint_path'],
            model_type=params['mobilesam_model_type'],
            points_per_side=params['points_per_side'],
            pred_iou_thresh=params['pred_iou_thresh'],
            stability_score_thresh=params['stability_score_thresh'],
            device=params['device'],
        )
    raise ValueError(f"Unknown segmentor_type '{segmentor_type}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    params = config['mapping_node']['ros__parameters']
    depth_scale = config.get('dataset_streamer', {}).get('ros__parameters', {}).get(
        'depth_scale', 5000.0)

    segmentor_type, segmentor = build_segmentor(params)

    output_dir = os.path.join(params['output_dataset_dir'], segmentor_type)
    os.makedirs(os.path.join(output_dir, 'mask'), exist_ok=True)
    index_file = open(os.path.join(output_dir, 'mask.txt'), 'w')
    index_file.write('# segmentation masks\n# timestamp filename\n')

    back_projector = BackProjector(params['fx'], params['fy'], params['cx'], params['cy'])
    tracker = Tracker(output_dir=params['objects_output_dir'], gate=params['track_gate_m'])

    rgb_entries = read_tum_file(os.path.join(args.dataset_path, 'rgb.txt'))
    depth_entries = read_tum_file(os.path.join(args.dataset_path, 'depth.txt'))
    gt_entries = read_tum_file(os.path.join(args.dataset_path, 'groundtruth.txt'))

    for ts, fields in rgb_entries:
        bgr_image = cv2.imread(os.path.join(args.dataset_path, fields[0]), cv2.IMREAD_COLOR)
        if bgr_image is None:
            continue

        label_mask = segmentor.segment(bgr_image)

        stamp_str = f'{ts:.6f}'
        filename = os.path.join('mask', f'{stamp_str}.png')
        cv2.imwrite(os.path.join(output_dir, filename), colorize_label_mask(label_mask))
        index_file.write(f'{stamp_str} {filename}\n')
        index_file.flush()

        _, depth_fields = nearest(depth_entries, ts)
        raw = cv2.imread(os.path.join(args.dataset_path, depth_fields[0]), cv2.IMREAD_UNCHANGED)
        depth_m = raw.astype(np.float32) / depth_scale

        _, gt_fields = nearest(gt_entries, ts)
        tx, ty, tz, qx, qy, qz, qw = (float(x) for x in gt_fields)
        R, t = quat_to_matrix(qx, qy, qz, qw), np.array([tx, ty, tz])

        detections = back_projector.project(label_mask, depth_m)
        tracker.update(detections, R, t, ts, bgr_image, label_mask)

        print(f"{stamp_str}: {len(detections)} objects")

    index_file.close()


if __name__ == '__main__':
    main()
