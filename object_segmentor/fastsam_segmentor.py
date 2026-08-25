"""FastSAM-backed unsupervised instance segmentor (via the ultralytics package)."""

import numpy as np

from .object_segmentor_base import ObjectSegmentorBase


class FastSAMSegmentor(ObjectSegmentorBase):

    def __init__(self, model_path: str, conf_threshold: float = 0.4,
                 iou_threshold: float = 0.7, device: str = 'cuda:0'):
        from ultralytics import FastSAM  # deferred: only import if this segmentor is used

        self._model = FastSAM(model_path)
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._device = device

    def segment(self, bgr_image: np.ndarray) -> np.ndarray:
        results = self._model.predict(
            bgr_image,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            device=self._device,
            retina_masks=True,  # masks at the original image resolution
            verbose=False,
        )
        result = results[0]

        height, width = bgr_image.shape[:2]
        label_mask = np.zeros((height, width), dtype=np.uint16)
        if result.masks is None:
            return label_mask

        masks = result.masks.data.cpu().numpy()  # (N, H, W), already binary
        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Largest objects first, so smaller ones stamped later aren't hidden
        # behind them in the shared label mask.
        for instance_id, idx in enumerate(np.argsort(-areas), start=1):
            binary = masks[idx] > 0.5
            write = binary & (label_mask == 0)
            label_mask[write] = instance_id

        return label_mask
