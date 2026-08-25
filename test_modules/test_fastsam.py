#!/usr/bin/env python3
"""Standalone test for FastSAMSegmentor, outside of ROS/threading.

Loads one dataset image as a numpy array (the same way mapping_node.py
feeds frames in via cv_bridge) and runs it through FastSAMSegmentor.segment()
-- the exact class/method the ROS node uses -- to isolate whether a bug is
in that class itself versus something specific to running inside the node's
worker thread.
"""

import os
import sys

import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from object_segmentor.fastsam_segmentor import FastSAMSegmentor  # noqa: E402
from object_segmentor.visualization import colorize_label_mask  # noqa: E402

IMAGE_PATH = os.path.join(
    REPO_ROOT, 'datasets', 'rgbd_dataset_freiburg1_desk', 'rgb',
    '1305031452.791720.png')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    bgr_image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise FileNotFoundError(f"Could not read {IMAGE_PATH}")
    print(f"input image: {IMAGE_PATH} shape={bgr_image.shape} dtype={bgr_image.dtype}")

    segmentor = FastSAMSegmentor(
        model_path='FastSAM-s.pt',
        conf_threshold=0.1,
        iou_threshold=0.1,
        device='cuda:0',
    )

    label_mask = segmentor.segment(bgr_image)
    print(f"label_mask shape={label_mask.shape} dtype={label_mask.dtype} "
          f"max_id={label_mask.max()} nonzero_px={(label_mask > 0).sum()}")

    out_path = os.path.join(OUTPUT_DIR, 'test_fastsam_mask.png')
    cv2.imwrite(out_path, label_mask)
    print(f"saved raw label mask to {out_path}")

    # Raw instance IDs (1, 2, 3, ...) are visually indistinguishable from 0
    # on a 16-bit scale, so also save a distinct-color RGB rendering to
    # actually eyeball.
    color = colorize_label_mask(label_mask)
    visual_path = os.path.join(OUTPUT_DIR, 'test_fastsam_mask_visual.png')
    cv2.imwrite(visual_path, color)
    print(f"saved color visualization to {visual_path}")


if __name__ == '__main__':
    main()
