
from typing import Tuple, Optional
# from cam_cal import CameraCalibrator
import cv2
import numpy as np

class ImageUndistorter:
    """Pre-builds remap arrays for fast per-frame undistortion."""

    def __init__(self, K: np.ndarray, dist: np.ndarray,
                 cal_wh: Tuple[int, int], alpha: float = 0.0):
        self._K_cal  = K.copy()
        self._dist   = dist.copy()
        self._cal_wh = cal_wh
        self._alpha  = alpha
        self._map1   = None
        self._map2   = None
        self._new_K  = None
        self._last_wh: Optional[Tuple[int, int]] = None

    def _scale_K(self, target_wh: Tuple[int, int]) -> np.ndarray:
        sx = target_wh[0] / self._cal_wh[0]
        sy = target_wh[1] / self._cal_wh[1]
        K  = self._K_cal.copy().astype(np.float64)
        K[0, 0] *= sx;  K[0, 2] *= sx
        K[1, 1] *= sy;  K[1, 2] *= sy
        return K

    def prepare(self, wh: Tuple[int, int]):
        if self._last_wh == wh:
            return
        self._last_wh = wh
        K_scaled = self._scale_K(wh)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K_scaled, self._dist, wh, self._alpha, wh)
        self._new_K = new_K
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            K_scaled, self._dist, None, new_K, wh, cv2.CV_32FC1)

    def undistort(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        self.prepare((w, h))
        return cv2.remap(image, self._map1, self._map2, cv2.INTER_LINEAR)

    @property
    def current_K(self) -> Optional[np.ndarray]:
        return self._new_K

