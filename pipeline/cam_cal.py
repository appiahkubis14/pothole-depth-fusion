


import csv
import cv2
import numpy as np
from typing import Tuple, List, Optional
import os

class CameraCalibrator:
    """Chessboard camera calibration → intrinsic K + distortion d."""

    def __init__(self,
                 chessboard_size: Tuple[int, int] = (8, 5),
                 square_size_m: float = 0.030):
        self.chessboard_size = chessboard_size
        self.square_size     = square_size_m
        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

        tpl = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        tpl[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self._tpl = tpl * square_size_m

        self._obj_pts:  List[np.ndarray] = []
        self._img_pts:  List[np.ndarray] = []
        self.image_shape: Optional[Tuple[int, int]] = None
        self.K:                   Optional[np.ndarray] = None
        self.dist:                Optional[np.ndarray] = None
        self.reprojection_error:  Optional[float]      = None

    def find_corners(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        size    = self.chessboard_size

        def _refine(g, c):
            return cv2.cornerSubPix(g, c, (11, 11), (-1, -1), self._criteria)

        flags_list = [
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        ]
        for g in (gray_eq, gray):
            for flags in flags_list:
                found, c = cv2.findChessboardCorners(g, size, flags)
                if found:
                    return True, _refine(g, c)
        for g in (gray_eq, gray):
            found, c = cv2.findChessboardCornersSB(g, size)
            if found:
                return True, c
        return False, None

    def add_image(self, image: np.ndarray) -> bool:
        if self.image_shape is None:
            h, w = image.shape[:2]
            self.image_shape = (w, h)
        found, corners = self.find_corners(image)
        if found:
            self._obj_pts.append(self._tpl.copy())
            self._img_pts.append(corners)
            print(f"  [CAL] ✓  Corners found  ({len(self._obj_pts)} images total)")
        else:
            print("  [CAL] ✗  No corners found")
        return found

    def calibrate(self) -> bool:
        n = len(self._img_pts)
        if n < 5:
            print(f"  [CAL] Need ≥ 5 images, have {n}")
            return False
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self._obj_pts, self._img_pts, self.image_shape, None, None)
        if not ret:
            return False
        self.K = K; self.dist = dist
        self._rvecs = rvecs; self._tvecs = tvecs
        self.reprojection_error = self._mean_reproj_error(rvecs, tvecs)
        print(f"  [CAL] fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  "
              f"cx={K[0,2]:.2f}  cy={K[1,2]:.2f}  err={self.reprojection_error:.4f}px")
        return True

    def _mean_reproj_error(self, rvecs, tvecs) -> float:
        
        total = 0.0
        for i in range(len(self._obj_pts)):
            proj, _ = cv2.projectPoints(self._obj_pts[i], rvecs[i], tvecs[i], self.K, self.dist)
            total += cv2.norm(self._img_pts[i], proj, cv2.NORM_L2) / len(proj)
        return total / len(self._obj_pts)

    def save(self, path: str = "outputs/cal/calibration.npz"):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, K=self.K, dist=self.dist,
                 image_shape=np.array(self.image_shape),
                 reprojection_error=np.array([self.reprojection_error]))
        print(f"  [CAL] Saved → {path}")

    def load(self, path: str = "outputs/cal/calibration.npz"):
        data = np.load(path, allow_pickle=True)
        self.K    = data["K"]
        self.dist = data["dist"]
        self.image_shape = tuple(data["image_shape"].tolist())
        if "reprojection_error" in data:
            self.reprojection_error = float(data["reprojection_error"][0])
        print(f"  [CAL] Loaded from {path}  "
              f"fx={self.K[0,0]:.2f}  fy={self.K[1,1]:.2f}  "
              f"err={self.reprojection_error:.4f}px")
