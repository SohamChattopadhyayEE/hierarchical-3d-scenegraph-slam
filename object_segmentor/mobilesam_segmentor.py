"""MobileSAM-backed unsupervised instance segmentor (via the mobile_sam package).

Uses SamAutomaticMaskGenerator's "segment everything" mode: a grid of point
prompts is sampled internally and MobileSAM proposes + filters masks for
whatever objects fall under them, with no manual prompting required.
"""

import cv2
import numpy as np

from .object_segmentor_base import ObjectSegmentorBase


class MobileSAMSegmentor(ObjectSegmentorBase):

    def __init__(self, checkpoint_path: str, model_type: str = 'vit_t',
                 points_per_side: int = 16, pred_iou_thresh: float = 0.88,
                 stability_score_thresh: float = 0.95, device: str = 'cuda:0'):
        # deferred: only import if this segmentor is used
        from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        sam.eval()

        self._generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )

    def segment(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        masks = self._generator.generate(rgb_image)  # list of dicts w/ 'segmentation', 'area'

        height, width = bgr_image.shape[:2]
        label_mask = np.zeros((height, width), dtype=np.uint16)

        # Largest objects first, so smaller ones stamped later aren't hidden
        # behind them in the shared label mask.
        for instance_id, mask in enumerate(
                sorted(masks, key=lambda m: m['area'], reverse=True), start=1):
            write = mask['segmentation'] & (label_mask == 0)
            label_mask[write] = instance_id

        return label_mask
