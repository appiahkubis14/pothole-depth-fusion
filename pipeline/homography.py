from typing import Tuple, Dict
import numpy as np

class HomographyMapper:
    """Maps image pixels → ground-plane world coordinates via IMU-corrected pinhole."""

    def __init__(self, K: np.ndarray, camera_height_m: float):
        self.K = K.copy().astype(np.float64)
        self.h = camera_height_m

    def update_K(self, K: np.ndarray):
        self.K = K.copy().astype(np.float64)

    def update_height(self, h: float):
        self.h = h

    def compute_H(self, pitch: float, roll: float, yaw: float = 0.0) -> np.ndarray:
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll),  np.sin(roll)
        cy, sy = np.cos(yaw),   np.sin(yaw)
        Rx = np.array([[1,  0,   0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0,  sp], [0,  1,   0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy,  0], [0,   0,  1]])
        R  = Rz @ Ry @ Rx
        K_inv  = np.linalg.inv(self.K)
        r1, r2 = R[:, 0], R[:, 1]
        t_cam  = np.array([0.0, 0.0, self.h])
        return np.column_stack([r1, r2, t_cam]) @ K_inv

    def pixel_to_world(self, px: float, py: float,
                       pitch: float, roll: float, yaw: float = 0.0
                       ) -> Tuple[float, float]:
        H     = self.compute_H(pitch, roll, yaw)
        w_hom = H @ np.array([px, py, 1.0])
        if abs(w_hom[2]) < 1e-9:
            return 0.0, 0.0
        return w_hom[0] / w_hom[2], w_hom[1] / w_hom[2]

    def bbox_world_dims(self, bbox: Tuple[int, int, int, int],
                        pitch: float, roll: float, yaw: float = 0.0) -> Dict:
        x1, y1, x2, y2 = bbox
        corners_w = [self.pixel_to_world(px, py, pitch, roll, yaw)
                     for px, py in [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]]
        Xs = [c[0] for c in corners_w]
        Ys = [c[1] for c in corners_w]
        return {
            "width_m":  round(abs(max(Xs) - min(Xs)), 4),
            "length_m": round(abs(max(Ys) - min(Ys)), 4),
            "area_m2":  round(abs(max(Xs)-min(Xs)) * abs(max(Ys)-min(Ys)), 6),
        }

    def confidence(self, pitch: float, roll: float) -> float:
        tilt = np.sqrt(pitch**2 + roll**2)
        return float(np.clip(1.0 - tilt / np.radians(30), 0.0, 1.0))

