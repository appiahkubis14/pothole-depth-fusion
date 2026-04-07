
from typing import Tuple, List
import numpy as np

from pipeline.homography import HomographyMapper

class GeometryAnchoredDepthScaler:
    """
    Converts MiDaS *inverse* depth to absolute *metric* depth.

    D_rel    = 1 / D_inv           (true relative depth)
    D_metric ≈ scale * D_rel + shift   (RANSAC-fit to ground anchors)
    EMA over last frames for temporal stability.
    """

    def __init__(self, K: np.ndarray,
                 min_anchors: int = 8,
                 ema_alpha: float = 0.6):
        self.K            = K.copy().astype(np.float64)
        self.min_anchors  = min_anchors
        self.ema_alpha    = ema_alpha
        self._scale_ema   = 1.0
        self._shift_ema   = 0.0
        self._initialized = False

    def update_K(self, K: np.ndarray):
        self.K = K.copy().astype(np.float64)

    def _ground_anchors(self,
                        inv_depth: np.ndarray,
                        homography_mapper: "HomographyMapper",
                        pitch: float, roll: float, yaw: float,
                        camera_height_m: float,
                        grid_h: int = 12, grid_w: int = 16
                        ) -> Tuple[np.ndarray, np.ndarray]:
        H, W = inv_depth.shape
        y0 = H // 2
        pairs_rel: List[float] = []
        pairs_met: List[float] = []

        for iy in range(grid_h):
            y = y0 + int(iy * (H - y0) / grid_h)
            for ix in range(grid_w):
                x = int(ix * W / grid_w)
                wx, wy = homography_mapper.pixel_to_world(x, y, pitch, roll, yaw)
                d_geom = float(np.sqrt(wx**2 + wy**2 + camera_height_m**2))
                if not (0.3 < d_geom < 60.0):
                    continue
                d_inv = float(inv_depth[y, x])
                if d_inv <= 0 or not np.isfinite(d_inv):
                    continue
                pairs_rel.append(1.0 / d_inv)
                pairs_met.append(d_geom)

        # Convert to numpy arrays and ensure they are 1D
        rel_arr = np.array(pairs_rel, dtype=np.float64).ravel()
        met_arr = np.array(pairs_met, dtype=np.float64).ravel()
        
        return rel_arr, met_arr
    @staticmethod
    def _ransac_linear(x: np.ndarray, y: np.ndarray,
                    n_iter: int = 30,
                    inlier_thresh_frac: float = 0.12
                    ) -> Tuple[float, float, float]:
        # Ensure 1D arrays and same length
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        
        n = len(x)
        if n < 4:
            return 1.0, 0.0, 0.0
        
        # Make sure x and y have the same length
        if len(x) != len(y):
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]
            n = min_len
        
        best_a, best_b, best_in = 1.0, 0.0, 0
        rng = np.random.default_rng(0)
        
        for _ in range(n_iter):
            i, j = rng.choice(n, 2, replace=False)
            xi = float(x[i])
            xj = float(x[j])
            yi = float(y[i])
            yj = float(y[j])
            
            dx = xj - xi
            if abs(dx) < 1e-9:
                continue
            
            a = (yj - yi) / dx
            b = yi - a * xi
            
            thresh = inlier_thresh_frac * np.median(y)
            inliers = int(np.sum(np.abs(a * x + b - y) < thresh))
            
            if inliers > best_in:
                best_in = inliers
                best_a, best_b = a, b
        
        # Refit with inliers if we have enough
        thresh = inlier_thresh_frac * np.median(y)
        mask = np.abs(best_a * x + best_b - y) < thresh
        if mask.sum() >= 2:
            A = np.vstack([x[mask], np.ones(mask.sum())]).T
            res = np.linalg.lstsq(A, y[mask], rcond=None)[0]
            best_a, best_b = float(res[0]), float(res[1])
        
        return best_a, best_b, best_in / n

    def scale_frame(self,
                    inv_depth: np.ndarray,
                    homography_mapper: "HomographyMapper",
                    pitch: float, roll: float, yaw: float,
                    camera_height_m: float
                    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Returns (abs_depth_m [float32], viz_u8 [uint8], confidence [0-1]).
        """
        D_rel, D_met = self._ground_anchors(
            inv_depth, homography_mapper, pitch, roll, yaw, camera_height_m)

        # DON'T modify D_rel here - it breaks the array
        # Remove this line or fix it:
        # corr = np.corrcoef(D_rel, D_met)[0, 1]
        # if corr < 0:
        #     D_rel = 1.0 / (inv_depth + 1e-6)  # This is wrong - inv_depth vs D_rel
        
        # Instead, just use D_rel as is
        if len(D_rel) >= self.min_anchors:
            scale, shift, conf = self._ransac_linear(D_rel, D_met)
        else:
            h, w = inv_depth.shape
            strip = inv_depth[int(h * 0.85):, w//4: 3*w//4]
            med_inv = float(np.median(strip[strip > 0])) if strip.size else 1.0
            d_rel_anchor = 1.0 / max(med_inv, 1e-6)
            d_met_anchor = camera_height_m / max(np.cos(pitch), 0.1)
            scale = d_met_anchor / max(d_rel_anchor, 1e-9)
            shift = 0.0
            conf = 0.0

        if self._initialized:
            self._scale_ema = self.ema_alpha * scale + (1 - self.ema_alpha) * self._scale_ema
            self._shift_ema = self.ema_alpha * shift + (1 - self.ema_alpha) * self._shift_ema
        else:
            self._scale_ema   = scale
            self._shift_ema   = shift
            self._initialized = True

        with np.errstate(divide="ignore", invalid="ignore"):
            d_rel_map = np.where(inv_depth > 0,
                                 1.0 / inv_depth.astype(np.float64),
                                 0.0).astype(np.float32)
        abs_depth = (self._scale_ema * d_rel_map + self._shift_ema).astype(np.float32)
        abs_depth = np.clip(abs_depth, 0.1, 200.0)

        near_clip = float(np.percentile(abs_depth, 2))
        far_clip  = float(np.percentile(abs_depth, 98))
        viz_norm  = np.clip((abs_depth - near_clip) / max(far_clip - near_clip, 0.1), 0, 1)
        viz_u8    = (255 * (viz_norm)).astype(np.uint8)  # near = bright

        return abs_depth, viz_u8, conf

