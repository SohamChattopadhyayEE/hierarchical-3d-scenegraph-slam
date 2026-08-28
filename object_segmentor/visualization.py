"""Turns segmentation/depth data into viewable color images.

Shared by mapping_node.py / mapping_from_file.py (saved visualizations) and
test_modules/*.py (local inspection).
"""

import colorsys

import cv2
import numpy as np


def colorize_label_mask(label_mask: np.ndarray) -> np.ndarray:
    """Maps each instance id to a distinct, well-separated color.

    Background (id 0) stays black. Hue steps by the golden ratio per id so
    consecutive ids land far apart on the color wheel regardless of how many
    instances there are -- unlike a plain intensity/colormap scaling, which
    compresses many instances into similar-looking values.

    Returns a BGR uint8 image (OpenCV/cv2.imwrite convention, matching the
    rest of this codebase), not RGB.
    """
    height, width = label_mask.shape
    color = np.zeros((height, width, 3), dtype=np.uint8)

    for instance_id in np.unique(label_mask):
        if instance_id == 0:
            continue
        hue = (instance_id * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        color[label_mask == instance_id] = (int(b * 255), int(g * 255), int(r * 255))

    return color


def colorize_depth(depth_m: np.ndarray, max_depth_m: float = 5.0) -> np.ndarray:
    """Maps a float depth image (metres) to a viewable BGR jet-colormap image.

    Near = red, far = blue (standard depth-visualization convention); invalid
    pixels (depth <= 0) clip to 0 and show as the colormap's low end.
    """
    clipped = np.clip(depth_m, 0, max_depth_m)
    normalized = (clipped / max_depth_m * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
