#!/usr/bin/env python3
"""Standalone test for MobileSAMSegmentor, outside of ROS/threading.

Mirrors test_fastsam.py: loads one dataset image as a numpy array (the same
way mapping_node.py feeds frames in via cv_bridge) and runs it through
MobileSAMSegmentor.segment() -- the exact class/method the ROS node uses --
so both segmentors go through the same test pipeline.
"""

import os
import sys

import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from object_segmentor.mobilesam_segmentor import MobileSAMSegmentor  # noqa: E402
from object_segmentor.visualization import colorize_label_mask  # noqa: E402

IMAGE_PATH = os.path.join(
    REPO_ROOT, 'datasets', 'rgbd_dataset_freiburg1_desk', 'rgb',
    '1305031452.791720.png')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Unlike FastSAM's auto-downloadable asset names, mobile_sam.sam_model_registry
# needs a real local checkpoint file. Download it from the official MobileSAM
# repo (https://github.com/ChaoningZhang/MobileSAM, weights/mobile_sam.pt) and
# place it at this path, or edit CHECKPOINT_PATH below.
CHECKPOINT_PATH = os.path.join(REPO_ROOT, 'mobile_sam.pt')


def main():
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"MobileSAM checkpoint not found at {CHECKPOINT_PATH}. Download "
            "it from https://github.com/ChaoningZhang/MobileSAM "
            "(weights/mobile_sam.pt) and place it there, or edit "
            "CHECKPOINT_PATH in this script.")

    bgr_image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise FileNotFoundError(f"Could not read {IMAGE_PATH}")
    print(f"input image: {IMAGE_PATH} shape={bgr_image.shape} dtype={bgr_image.dtype}")

    segmentor = MobileSAMSegmentor(
        checkpoint_path=CHECKPOINT_PATH,
        model_type='vit_t',
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        device='cuda:0',
    )

    label_mask = segmentor.segment(bgr_image)
    print(f"label_mask shape={label_mask.shape} dtype={label_mask.dtype} "
          f"max_id={label_mask.max()} nonzero_px={(label_mask > 0).sum()}")

    out_path = os.path.join(OUTPUT_DIR, 'test_mobilesam_mask.png')
    cv2.imwrite(out_path, label_mask)
    print(f"saved raw label mask to {out_path}")

    # Raw instance IDs (1, 2, 3, ...) are visually indistinguishable from 0
    # on a 16-bit scale, so also save a distinct-color RGB rendering to
    # actually eyeball.
    color = colorize_label_mask(label_mask)
    visual_path = os.path.join(OUTPUT_DIR, 'test_mobilesam_mask_visual.png')
    cv2.imwrite(visual_path, color)
    print(f"saved color visualization to {visual_path}")


if __name__ == '__main__':
    main()
