"""Common interface for unsupervised instance segmentors (FastSAM, MobileSAM, ...)."""

from abc import ABC, abstractmethod

import numpy as np


class ObjectSegmentorBase(ABC):

    @abstractmethod
    def segment(self, bgr_image: np.ndarray) -> np.ndarray:
        """Segments bgr_image (passed by reference, not copied).

        Returns a uint16 label mask the same height/width as bgr_image:
        0 is background, 1..N identifies each detected instance.
        """
        raise NotImplementedError
