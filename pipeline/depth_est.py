"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        DEPTH & DIMENSION ESTIMATION PIPELINE                                ║
║        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                          ║
║  Data flow:                                                                 ║
║    RGB Frame ──► Undistort                                                  ║
║    IMU       ──► Pitch / Roll / Camera-height                               ║
║    Calib     ──► K, dist ──► Undistort ──► Homography metric anchors        ║
║    RGB Frame ──► MiDaS (relative inverse-depth)                             ║
║    Anchors + Relative-depth ──► GeometryAnchoredDepthScaler                 ║
║                           ──► Absolute Dense Depth Map                      ║
║    Absolute depth ──► Pothole detection / dimension fusion                  ║
║    Output: dual-view annotated video (RGB + depth) + dimensions CSV         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from math import radians, cos, sin, sqrt, atan2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed — YOLO detection disabled; using mock potholes")

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    print("[WARN] pandas not installed — CSV sensor loading disabled")

try:
    from scipy.ndimage import median_filter as _scipy_median_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    def _scipy_median_filter(arr, size):
        return arr


# ══════════════════════════════════════════════════════════════════════════════
# RUN MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class RunManager:
    """
    Creates a timestamped project directory for each pipeline run.

    Layout
    ──────
    <base_output_dir>/
      run_YYYYMMDD_HHMMSS/
        annotated/          ← per-frame JPEGs (written async)
        data/
          dimensions.csv    ← all pothole/object dimension records
        manifest.json
    """

    def __init__(self, base_dir: str = "output"):
        ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id  = f"run_{ts}"
        self.run_dir = os.path.join(base_dir, self.run_id)

        self.annotated_dir = os.path.join(self.run_dir, "annotated")
        self.data_dir      = os.path.join(self.run_dir, "data")

        for d in (self.annotated_dir, self.data_dir):
            os.makedirs(d, exist_ok=True)

        self._settings:   Dict  = {}
        self._start_time: float = time.time()
        print(f"  [RUN] Project → {self.run_dir}")

    def set_settings(self, **kwargs):
        self.settings = {**self._settings, **kwargs,
                         "run_id":     self.run_id,
                         "started_at": datetime.now().isoformat()}

    def save_manifest(self, summary: Dict, video_path: Optional[str] = None):
        inventory = []
        for root, dirs, files in os.walk(self.run_dir):
            dirs.sort()
            for fname in sorted(files):
                full = os.path.join(root, fname)
                rel  = os.path.relpath(full, self.run_dir)
                sz   = os.path.getsize(full)
                inventory.append({"path": rel, "bytes": sz})

        manifest = {
            "run_id":       self.run_id,
            "started_at":   getattr(self, "settings", {}).get("started_at", ""),
            "finished_at":  datetime.now().isoformat(),
            "elapsed_s":    round(time.time() - self._start_time, 1),
            "settings":     getattr(self, "settings", {}),
            "input_video":  video_path or "",
            "summary":      summary,
            "output_files": inventory,
        }
        path = os.path.join(self.run_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"  [RUN] Manifest → {path}")
        print(f"\n  ── Run {self.run_id} output files {'─'*30}")
        for item in inventory:
            sz = item["bytes"]
            if   sz > 1_000_000: label = f"{sz/1_000_000:6.1f} MB"
            elif sz > 1_000:     label = f"{sz/1_000:6.1f} KB"
            else:                label = f"{sz:6d}  B"
            print(f"    {item['path']:<50} {label}")
        print(f"  {'─'*68}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – CAMERA CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – IMAGE UNDISTORTION
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – MIDAS DEPTH ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

class MiDaSDepthEstimator:
    """
    Monocular depth via MiDaS.

    run(frame, frame_id) → (inv_depth_f32, viz_u8)
        inv_depth_f32 : raw MiDaS output; larger value = closer pixel.
        viz_u8        : uint8 single-channel normalised map (for colormap).
    """

    def __init__(self, model_type: str = "MiDaS_small",
             device: Optional[torch.device] = None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.available = False
        self._cache: Tuple[int, Optional[np.ndarray], Optional[np.ndarray]] = (-1, None, None)

        print(f"  [MIDAS] Loading {model_type} on {self.device} …")
        try:
            # Use direct import to avoid local file conflicts
            import sys
            # Ensure we're not picking up a local midas.py
            if 'midas' in sys.modules:
                del sys.modules['midas']
            
            self._model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True, trust_repo=True)
            self._model.to(self.device).eval()
            
            # Get transforms separately
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if "DPT" in model_type:
                self._transform = midas_transforms.dpt_transform
            else:
                self._transform = midas_transforms.small_transform
                
            self.available = True
            print("  [MIDAS] ✓ Loaded")
        except Exception as exc:
            print(f"  [MIDAS] ✗ Failed: {exc}")
            print("  [MIDAS] Attempting alternative load method...")
            try:
                # Alternative: load from local if cached
                import os
                cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints/")
                self._model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True, 
                                            force_reload=False, trust_repo=True)
                self._model.to(self.device).eval()
                self.available = True
                print("  [MIDAS] ✓ Loaded from cache")
            except Exception as exc2:
                print(f"  [MIDAS] ✗ Still failed: {exc2}")

    # ── single public entry point ─────────────────────────────────────────────

    def run(self, bgr_image: np.ndarray,
            frame_id: int = -1
            ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run MiDaS inference and return (inv_depth_f32, viz_u8).

        frame_id: pass the current frame counter so results are cached
                  per frame — call once, reuse as many times as needed.
        """
        if frame_id >= 0 and frame_id == self._cache[0]:
            return self._cache[1], self._cache[2]     # ← cached hit

        if not self.available:
            return None, None

        try:
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            inp = self._transform(rgb).to(self.device)
            with torch.no_grad():
                pred = self._model(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=rgb.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            inv = pred.cpu().numpy().astype(np.float32)
            viz = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            self._cache = (frame_id, inv, viz)
            return inv, viz
        except Exception as exc:
            print(f"  [MIDAS] inference error: {exc}")
            return None, None


    # def run(self, bgr_image: np.ndarray,
    #         frame_id: int = -1
    #         ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    #     if frame_id >= 0 and frame_id == self._cache[0]:
    #         return self._cache[1], self._cache[2]
    #     if not self.available:
    #         return None, None
    #     try:
    #         rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    #         inp = self._transform(rgb).to(self.device)
    #         with torch.no_grad():
    #             pred = self._model(inp)
    #             pred = torch.nn.functional.interpolate(
    #                 pred.unsqueeze(1),
    #                 size=rgb.shape[:2],
    #                 mode="bilinear",
    #                 align_corners=False,
    #             ).squeeze()
    #         inv = pred.cpu().numpy().astype(np.float32)
    #         viz = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #         self._cache = (frame_id, inv, viz)
    #         return inv, viz
    #     except Exception as exc:
    #         print(f"  [MIDAS] inference error: {exc}")
    #         return None, None

    @staticmethod
    def bbox_metric_depth(abs_depth: np.ndarray,
                          bbox: Tuple[int, int, int, int],
                          percentile: float = 20.0) -> float:
        """20th-percentile metric depth inside bbox (near portion)."""
        x1, y1, x2, y2 = bbox
        hh, ww = abs_depth.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(ww - 1, x2), min(hh - 1, y2)
        if x1 >= x2 or y1 >= y2:
            return float(np.nanmedian(abs_depth))
        roi    = abs_depth[y1:y2, x1:x2]
        finite = roi[np.isfinite(roi) & (roi > 0)]
        return float(np.percentile(finite, percentile)) if finite.size else float(np.nanmedian(abs_depth))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – GEOMETRY-ANCHORED DEPTH SCALER
# ══════════════════════════════════════════════════════════════════════════════

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
        H, W  = inv_depth.shape
        y0    = H // 2
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

        return np.array(pairs_rel, dtype=np.float64), np.array(pairs_met, dtype=np.float64)

    @staticmethod
    def _ransac_linear(x: np.ndarray, y: np.ndarray,
                       n_iter: int = 30,
                       inlier_thresh_frac: float = 0.12
                       ) -> Tuple[float, float, float]:
        n = len(x)
        if n < 4:
            return 1.0, 0.0, 0.0
        best_a, best_b, best_in = 1.0, 0.0, 0
        rng = np.random.default_rng(0)
        for _ in range(n_iter):
            i, j = rng.choice(n, 2, replace=False)
            dx = x[j] - x[i]
            if abs(dx) < 1e-9:
                continue
            a = (y[j] - y[i]) / dx
            b = y[i] - a * x[i]
            thresh  = inlier_thresh_frac * np.median(y)
            inliers = int(np.sum(np.abs(a * x + b - y) < thresh))
            if inliers > best_in:
                best_in = inliers
                best_a, best_b = a, b
        thresh = inlier_thresh_frac * np.median(y)
        mask   = np.abs(best_a * x + best_b - y) < thresh
        if mask.sum() >= 2:
            A   = np.vstack([x[mask], np.ones(mask.sum())]).T
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

        corr = np.corrcoef(D_rel, D_met)[0, 1]
        if corr < 0:  # Negative correlation means inverse relationship
            # Use direct inverse instead
            D_rel = 1.0 / (inv_depth + 1e-6)

        if len(D_rel) >= self.min_anchors:
            scale, shift, conf = self._ransac_linear(D_rel, D_met)
        else:
            h, w   = inv_depth.shape
            strip  = inv_depth[int(h * 0.85):, w//4: 3*w//4]
            med_inv = float(np.median(strip[strip > 0])) if strip.size else 1.0
            d_rel_anchor = 1.0 / max(med_inv, 1e-6)
            d_met_anchor = camera_height_m / max(np.cos(pitch), 0.1)
            scale = d_met_anchor / max(d_rel_anchor, 1e-9)
            shift = 0.0
            conf  = 0.0

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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – GROUND-PLANE HOMOGRAPHY
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – IMU PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class IMUProcessor:
    """Processes IMU data from Android phone CSV format."""
    
    def __init__(self, initial_height_m: float = 1.2):
        self.initial_height = initial_height_m
        self.static_height = initial_height_m
        self.dynamic_height = initial_height_m
        
        # Hybrid mode state
        self._integrating = False
        self._integration_start_time = None
        self._last_vel_z = 0.0
        self._last_height = initial_height_m
        self._frames_without_pothole = 0
        
        # Data storage
        self._t: List[float] = []
        self._pitch: List[float] = []
        self._roll: List[float] = []
        self._yaw: List[float] = []
        self._lin_accel_z: List[float] = []  # Linear acceleration (gravity removed)
        self._speed: List[float] = []
        self._is_stationary: List[bool] = []
        
        self.orientation = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}
        
    def load_from_csv(self, csv_path: str):
        """Load IMU data from Android phone CSV format."""
        if not _PANDAS_AVAILABLE:
            print("  [IMU] pandas not available")
            return
            
        if not os.path.exists(csv_path):
            print(f"  [IMU] File not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        
        print(f"  [IMU] CSV columns: {list(df.columns)}")
        
        # ── Parse timestamp ───────────────────────────────────────────────
        # Your format: "Sun Nov 30 08:42:19 GMT 2025"
        def parse_timestamp(raw):
            try:
                # Try the GMT format first
                return datetime.strptime(str(raw), "%a %b %d %H:%M:%S GMT %Y").timestamp()
            except:
                try:
                    return datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S").timestamp()
                except:
                    try:
                        return float(raw)
                    except:
                        return 0.0
        
        timestamps = df['Timestamp'].apply(parse_timestamp).to_numpy(float)
        t_sec = timestamps - timestamps[0]  # Normalize to start at 0
        
        # ── Extract sensor data ──────────────────────────────────────────
        # Linear acceleration (gravity already removed!) - perfect for integration
        lin_accel_z = df['LinAccelZ'].to_numpy(float)
        
        # Gyroscope for orientation (more accurate than accelerometer for pitch/roll)
        gyro_x = df['GyroX'].to_numpy(float)
        gyro_y = df['GyroY'].to_numpy(float)
        gyro_z = df['GyroZ'].to_numpy(float)
        
        # Speed for detecting stationary periods (ZUPT)
        speed = df['SpeedKmh'].to_numpy(float)
        
        # Detect stationary periods (speed < 0.5 km/h for > 1 second)
        is_stationary = speed < 0.5
        
        # ── Compute pitch and roll from gyroscope integration ────────────
        # Simple integration of gyro for short-term orientation
        dt = np.diff(t_sec, prepend=t_sec[0])
        pitch = np.zeros(len(t_sec))
        roll = np.zeros(len(t_sec))
        yaw = np.zeros(len(t_sec))
        
        for i in range(1, len(t_sec)):
            # Gyro integration (more accurate for fast motions)
            pitch[i] = pitch[i-1] + gyro_y[i] * dt[i]  # Pitch from Y-axis gyro
            roll[i] = roll[i-1] + gyro_x[i] * dt[i]   # Roll from X-axis gyro
            yaw[i] = yaw[i-1] + gyro_z[i] * dt[i]
        
        # Apply complementary filter with accelerometer to remove drift
        # For stationary periods, reset to accelerometer-based orientation
        window = 10
        for i in range(len(t_sec)):
            if is_stationary[i]:
                # When stationary, use accelerometer for absolute orientation
                ax = df['AccelX'].iloc[i]
                ay = df['AccelY'].iloc[i]
                az = df['AccelZ'].iloc[i]
                
                # Compute pitch from accelerometer
                accel_pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
                accel_roll = np.arctan2(ay, az)
                
                # Blend: trust accelerometer more when stationary
                alpha = 0.95
                pitch[i] = alpha * accel_pitch + (1 - alpha) * pitch[i]
                roll[i] = alpha * accel_roll + (1 - alpha) * roll[i]
        
        # Clip to reasonable range (±20 degrees for road driving)
        MAX_ANGLE = np.radians(20)
        pitch = np.clip(pitch, -MAX_ANGLE, MAX_ANGLE)
        roll = np.clip(roll, -MAX_ANGLE, MAX_ANGLE)
        
        # ── Store data ───────────────────────────────────────────────────
        self._t = t_sec.tolist()
        self._pitch = pitch.tolist()
        self._roll = roll.tolist()
        self._yaw = yaw.tolist()
        self._lin_accel_z = lin_accel_z.tolist()
        self._speed = speed.tolist()
        self._is_stationary = is_stationary.tolist()
        
        # Set initial orientation (median of first few seconds)
        initial_samples = min(30, len(pitch))
        self.orientation = {
            "pitch": float(np.median(pitch[:initial_samples])),
            "roll": float(np.median(roll[:initial_samples])),
            "yaw": float(np.median(yaw[:initial_samples])),
        }
        
        # Print statistics
        pitch_deg = np.degrees(pitch)
        roll_deg = np.degrees(roll)
        stationary_pct = sum(is_stationary) / len(is_stationary) * 100
        
        print(f"  [IMU] Loaded {len(t_sec)} rows, duration={t_sec[-1]:.1f}s")
        print(f"  [IMU] Stationary: {stationary_pct:.1f}% of time")
        print(f"  [IMU] Pitch: mean={np.mean(pitch_deg):.1f}°, min={np.min(pitch_deg):.1f}°, max={np.max(pitch_deg):.1f}°")
        print(f"  [IMU] Roll:  mean={np.mean(roll_deg):.1f}°, min={np.min(roll_deg):.1f}°, max={np.max(roll_deg):.1f}°")
        print(f"  [IMU] LinAccelZ: min={np.min(lin_accel_z):.2f}, max={np.max(lin_accel_z):.2f}")
        
        # Perform ZUPT calibration on stationary periods
        self._calibrate_zupt()
    
    def _calibrate_zupt(self):
        """Calibrate zero-velocity updates using stationary periods."""
        stationary_accels = [self._lin_accel_z[i] for i in range(len(self._t)) if self._is_stationary[i]]
        if stationary_accels:
            bias = np.mean(stationary_accels)
            print(f"  [IMU] ZUPT calibration: accelerometer bias = {bias:.4f} m/s²")
            # Remove bias from all linear acceleration values
            self._lin_accel_z = [a - bias for a in self._lin_accel_z]
    
    def _interp(self, arr: List[float], t_query: float) -> float:
        """Interpolate value at given timestamp."""
        if not self._t:
            return arr[0] if arr else 0.0
        if t_query <= self._t[0]:
            return arr[0]
        if t_query >= self._t[-1]:
            return arr[-1]
        return float(np.interp(t_query, self._t, arr))
    
    def orientation_at(self, t: float) -> Dict[str, float]:
        """Get interpolated orientation at timestamp."""
        if not self._t:
            return dict(self.orientation)
        return {
            "pitch": self._interp(self._pitch, t),
            "roll": self._interp(self._roll, t),
            "yaw": self._interp(self._yaw, t),
        }
    
    def get_linear_accel_z(self, timestamp: float) -> float:
        """Get interpolated vertical linear acceleration (gravity already removed)."""
        if not self._lin_accel_z:
            return 0.0
        return self._interp(self._lin_accel_z, timestamp)
    
    def is_stationary_at(self, timestamp: float) -> bool:
        """Check if vehicle was stationary at given timestamp."""
        if not self._is_stationary:
            return False
        # Find closest index
        idx = np.argmin(np.abs(np.array(self._t) - timestamp))
        return self._is_stationary[idx]
    
    # ── Hybrid Height Methods ─────────────────────────────────────────────
    
    def reset_integration(self):
        """Reset dynamic integration back to static baseline."""
        self._integrating = False
        self._integration_start_time = None
        self._last_vel_z = 0.0
        self.dynamic_height = self.static_height
        self._frames_without_pothole = 0
    
    def start_integration(self, timestamp: float, accel_z: float, dt: float):
        """Start IMU integration from current static height."""
        self._integrating = True
        self._integration_start_time = timestamp
        self._last_vel_z = 0.0
        self.dynamic_height = self.static_height
        self._last_height = self.static_height
    
    def update_dynamic_height(self, accel_z: float, dt: float) -> float:
        """
        Update dynamic height using double integration of linear acceleration.
        
        Since LinAccelZ already has gravity removed, we can integrate directly:
            velocity = ∫ acceleration dt
            height = ∫ velocity dt
        
        Args:
            accel_z: Linear vertical acceleration (m/s², gravity already removed)
            dt: Time since last update (seconds)
        """
        if not self._integrating:
            return self.static_height
        
        # Integrate to get velocity change
        vel_z = self._last_vel_z + accel_z * dt
        
        # Integrate to get position change (height)
        # Using trapezoidal integration for better accuracy
        delta_height = (self._last_vel_z + vel_z) * 0.5 * dt
        
        self.dynamic_height += delta_height
        self._last_vel_z = vel_z
        
        # Safety bounds (suspension travel limited to ±0.3m)
        self.dynamic_height = np.clip(self.dynamic_height, 0.7, 1.8)
        
        return self.dynamic_height
    
    def get_height(self, timestamp: float, pothole_detected: bool, dt: float = 0.033) -> float:
        """
        Main interface: returns best height estimate.
        
        Strategy:
        - If pothole detected: use dynamic integration (short burst, accurate)
        - If no pothole: use static height and reset integration
        - Additionally, reset if vehicle is stationary (ZUPT)
        """
        # ZUPT: Reset if vehicle is stationary (drift can't accumulate)
        if self.is_stationary_at(timestamp):
            self.reset_integration()
            return self.static_height
        
        if pothole_detected:
            # Pothole present: use dynamic mode
            self._frames_without_pothole = 0
            
            if not self._integrating:
                # First frame of pothole: start integration from static baseline
                accel_z = self.get_linear_accel_z(timestamp)
                self.start_integration(timestamp, accel_z, dt)
                return self.dynamic_height
            else:
                # Continuing pothole: update integration
                accel_z = self.get_linear_accel_z(timestamp)
                return self.update_dynamic_height(accel_z, dt)
        else:
            # No pothole: count frames without detection
            self._frames_without_pothole += 1
            
            # Reset after 0.5 seconds (about 15 frames at 30fps) of no detection
            if self._integrating and self._frames_without_pothole > 15:
                self.reset_integration()
            
            return self.static_height


# class IMUProcessor:
#     """Correctly computes pitch and roll from accelerometer data"""
    
#     def __init__(self, initial_height_m: float = 1.2):
#         self.initial_height = initial_height_m
#         self.current_height = initial_height_m
#         self._t: List[float] = []
#         self._pitch: List[float] = []
#         self._roll: List[float] = []
#         self._yaw: List[float] = []
#         self._height: List[float] = []
#         self.orientation = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}

#     def load_from_csv(self, csv_path: str):
#         if not _PANDAS_AVAILABLE:
#             print("  [IMU] pandas not available"); return
#         if not os.path.exists(csv_path):
#             print(f"  [IMU] File not found: {csv_path}"); return

#         df = pd.read_csv(csv_path)
#         df.columns = [c.strip() for c in df.columns]
        
#         print(f"  [IMU] CSV columns: {list(df.columns)}")
        
#         # Look for timestamp column
#         ts_col = None
#         for col in df.columns:
#             if 'timestamp' in col.lower() or 'time' in col.lower():
#                 ts_col = col
#                 break
        
#         # Look for accelerometer columns
#         ax_col = None
#         ay_col = None
#         az_col = None
#         for col in df.columns:
#             col_lower = col.lower()
#             if 'accelx' in col_lower or 'accel_x' in col_lower:
#                 ax_col = col
#             elif 'accely' in col_lower or 'accel_y' in col_lower:
#                 ay_col = col
#             elif 'accelz' in col_lower or 'accel_z' in col_lower:
#                 az_col = col
        
#         if ax_col is None or ay_col is None or az_col is None:
#             print(f"  [IMU] Could not find accelerometer columns")
#             print(f"  [IMU] Available: {df.columns.tolist()}")
#             return
        
#         print(f"  [IMU] Using Accel: {ax_col}, {ay_col}, {az_col}")
        
#         # Parse timestamps
#         def parse_timestamp(raw):
#             try:
#                 # Try the GMT format first
#                 return datetime.strptime(str(raw), "%a %b %d %H:%M:%S GMT %Y").timestamp()
#             except:
#                 try:
#                     # Try ISO format
#                     return datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S").timestamp()
#                 except:
#                     try:
#                         return float(raw)
#                     except:
#                         return 0.0
        
#         t_sec = df[ts_col].apply(parse_timestamp).to_numpy(float)
#         t_sec = t_sec - t_sec[0]  # Normalize to start at 0
        
#         # Get accelerometer data
#         ax = df[ax_col].to_numpy(float)
#         ay = df[ay_col].to_numpy(float)
#         az = df[az_col].to_numpy(float)
        
#         # Compute pitch and roll from accelerometer
#         # Pitch: angle around Y-axis (forward/back tilt)
#         # Roll: angle around X-axis (left/right tilt)
        
#         # Smooth accelerometer data
#         window_size = 5
#         ax_smooth = np.convolve(ax, np.ones(window_size)/window_size, mode='same')
#         ay_smooth = np.convolve(ay, np.ones(window_size)/window_size, mode='same')
#         az_smooth = np.convolve(az, np.ones(window_size)/window_size, mode='same')
        
#         # Calculate pitch and roll (in radians)
#         # pitch = arctan2(ax, sqrt(ay^2 + az^2))
#         # roll = arctan2(ay, sqrt(ax^2 + az^2))
#         pitches = np.arctan2(ax_smooth, np.sqrt(ay_smooth**2 + az_smooth**2))
#         rolls = np.arctan2(ay_smooth, np.sqrt(ax_smooth**2 + az_smooth**2))
        
#         # Limit to reasonable angles (±15 degrees for road driving)
#         MAX_ANGLE = np.radians(15)
#         pitches = np.clip(pitches, -MAX_ANGLE, MAX_ANGLE)
#         rolls = np.clip(rolls, -MAX_ANGLE, MAX_ANGLE)
        
#         # Estimate camera height from acceleration variance
#         az_std = np.std(az)
#         if az_std < 0.5:
#             h_est = self.initial_height
#         elif az_std < 1.0:
#             h_est = self.initial_height - 0.1
#         else:
#             h_est = self.initial_height - 0.2
        
#         self._t = t_sec.tolist()
#         self._pitch = pitches.tolist()
#         self._roll = rolls.tolist()
#         self._yaw = np.zeros(len(t_sec)).tolist()
#         self._height = np.full(len(t_sec), h_est).tolist()
#         self.current_height = h_est
        
#         # Print statistics
#         pitch_deg = np.degrees(pitches)
#         roll_deg = np.degrees(rolls)
#         print(f"  [IMU] Loaded {len(t_sec)} rows, duration={t_sec[-1]:.1f}s")
#         print(f"  [IMU] Height: {h_est:.2f}m")
#         print(f"  [IMU] Pitch: mean={np.mean(pitch_deg):.2f}°, min={np.min(pitch_deg):.2f}°, max={np.max(pitch_deg):.2f}°")
#         print(f"  [IMU] Roll:  mean={np.mean(roll_deg):.2f}°, min={np.min(roll_deg):.2f}°, max={np.max(roll_deg):.2f}°")
        
#         self.orientation = {
#             "pitch": float(np.median(pitches)),
#             "roll": float(np.median(rolls)),
#             "yaw": 0.0,
#         }

#     def _interp(self, arr: List[float], t_query: float) -> float:
#         if not self._t or len(arr) == 0:
#             return arr[0] if arr else 0.0
#         # Handle time queries outside range
#         if t_query <= self._t[0]:
#             return arr[0]
#         if t_query >= self._t[-1]:
#             return arr[-1]
#         return float(np.interp(t_query, self._t, arr))

#     def height_at(self, t: float) -> float:
#         return self._interp(self._height, t) if self._height else self.current_height

#     def orientation_at(self, t: float) -> Dict[str, float]:
#         if not self._t:
#             return dict(self.orientation)
#         return {
#             "pitch": self._interp(self._pitch, t),
#             "roll": self._interp(self._roll, t),
#             "yaw": self._interp(self._yaw, t),
#         }



# class IMUProcessor:
#     """Gravity-vector tilt for pitch/roll (no gyro integration drift)."""

#     def __init__(self, initial_height_m: float = 1.2):
#         self.initial_height = initial_height_m
#         self.current_height = initial_height_m
#         self._t:     List[float] = []
#         self._pitch: List[float] = []
#         self._roll:  List[float] = []
#         self._yaw:   List[float] = []
#         self._height: List[float] = []
#         self.orientation = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}

#     def load_from_csv(self, csv_path: str):
#         if not _PANDAS_AVAILABLE:
#             print("  [IMU] pandas not available"); return
#         if not os.path.exists(csv_path):
#             print(f"  [IMU] File not found: {csv_path}"); return

#         df = pd.read_csv(csv_path)
#         df.columns = [c.strip() for c in df.columns]

#         def col(*names):
#             return next((n for n in names if n in df.columns), None)

#         ts_col = col("Timestamp","timestamp","Time","time")
#         ax_col = col("AccelX","accel_x","ax")
#         ay_col = col("AccelY","accel_y","ay")
#         az_col = col("AccelZ","accel_z","az")
#         gx_col = col("GyroX","gyro_x","gx")
#         gy_col = col("GyroY","gyro_y","gy")
#         gz_col = col("GyroZ","gyro_z","gz")

#         FORMATS = ["%a %b %d %H:%M:%S GMT %Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]

#         def _parse(raw) -> float:
#             for fmt in FORMATS:
#                 try:
#                     return datetime.strptime(str(raw).strip(), fmt).timestamp()
#                 except ValueError:
#                     pass
#             try:
#                 return float(raw)
#             except Exception:
#                 return 0.0

#         n     = len(df)
#         epoch = df[ts_col].apply(_parse).to_numpy(float) if ts_col else np.arange(n, dtype=float)
#         t_sec = epoch - epoch[0]
#         ax    = df[ax_col].to_numpy(float) if ax_col else np.zeros(n)
#         ay    = df[ay_col].to_numpy(float) if ay_col else np.zeros(n)
#         az    = df[az_col].to_numpy(float) if az_col else np.full(n, 9.81)
#         gx    = df[gx_col].to_numpy(float) if gx_col else np.zeros(n)
#         gy    = df[gy_col].to_numpy(float) if gy_col else np.zeros(n)
#         gz    = df[gz_col].to_numpy(float) if gz_col else np.zeros(n)
#         self._process_arrays(t_sec, ax, ay, az, gx, gy, gz)
#         print(f"  [IMU] {n} rows  duration={t_sec[-1]:.1f}s")

#     def _process_arrays(self, t, ax, ay, az, gx, gy, gz):
#         n        = len(t)
#         gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
#         static   = gyro_mag < 0.08
#         if static.sum() < 5:
#             static = np.ones(n, bool)

#         g_med  = np.array([np.median(ax[static]),
#                            np.median(ay[static]),
#                            np.median(az[static])])
#         g_norm = float(np.linalg.norm(g_med))
#         if g_norm < 7.0:
#             g_norm = 9.81

#         fracs     = np.abs(g_med) / g_norm
#         grav_axis = int(np.argmax(fracs))
#         grav_sign = 1.0 if g_med[grav_axis] > 0 else -1.0

#         ax_s = _scipy_median_filter(ax, 5)
#         ay_s = _scipy_median_filter(ay, 5)
#         az_s = _scipy_median_filter(az, 5)

#         if grav_axis == 0:
#             pitches = np.arctan2(az_s, ax_s * grav_sign)
#             rolls   = np.arctan2(ay_s, ax_s * grav_sign)
#         elif grav_axis == 2:
#             pitches = np.arctan2(ax_s, az_s * grav_sign)
#             rolls   = np.arctan2(ay_s, az_s * grav_sign)
#         else:
#             pitches = np.arctan2(ax_s, ay_s * grav_sign)
#             rolls   = np.zeros(n)

#         MAX_TILT = np.radians(20)
#         pitches  = np.clip(pitches, -MAX_TILT, MAX_TILT)
#         rolls    = np.clip(rolls,   -MAX_TILT, MAX_TILT)

#         vert     = [ax, ay, az][grav_axis]
#         vert_lin = vert - grav_sign * g_norm
#         vert_std = float(np.std(vert_lin))
#         if   vert_std < 1.5: h_est = 1.7
#         elif vert_std < 2.5: h_est = 1.3
#         elif vert_std < 3.5: h_est = 1.0
#         else:                h_est = 0.8

#         deviation = abs(h_est - self.initial_height) / max(self.initial_height, 0.1)
#         h_final   = h_est if deviation > 0.25 else self.initial_height

#         self._t      = t.tolist()
#         self._pitch  = pitches.tolist()
#         self._roll   = rolls.tolist()
#         self._yaw    = np.zeros(n).tolist()
#         self._height = np.full(n, h_final).tolist()
#         self.current_height = h_final
#         self.orientation = {
#             "pitch": float(np.median(pitches)),
#             "roll":  float(np.median(rolls)),
#             "yaw":   0.0,
#         }
#         print(f"  [IMU] Height={h_final:.3f}m  "
#               f"pitch [{np.degrees(pitches.min()):.1f}°, {np.degrees(pitches.max()):.1f}°]  "
#               f"roll  [{np.degrees(rolls.min()):.1f}°, {np.degrees(rolls.max()):.1f}°]")

#     def _interp(self, arr: List[float], t_query: float) -> float:
#         if not self._t:
#             return arr[0] if arr else 0.0
#         return float(np.interp(t_query, self._t, arr))

#     def height_at(self, t: float) -> float:
#         return self._interp(self._height, t) if self._height else self.current_height

#     def orientation_at(self, t: float) -> Dict[str, float]:
#         if not self._t:
#             return dict(self.orientation)
#         return {
#             "pitch": self._interp(self._pitch, t),
#             "roll":  self._interp(self._roll,  t),
#             "yaw":   self._interp(self._yaw,   t),
#         }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – DIMENSION FUSION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DimensionEstimate:
    width_m:        float
    length_m:       float
    area_m2:        float
    volume_m3:      float
    depth_m:        float          # nearest metric depth to detection [m]
    confidence:     float
    method_weights: Dict[str, float] = field(default_factory=dict)


class DimensionFusionEngine:
    """
    Fuses three dimension estimation strategies:
      A) Homography  – projects bbox corners to ground plane.
      B) Depth-anchor – uses absolute MiDaS depth + pinhole projection.
      C) Known-height – focal length + camera height + pixel bbox size.
    Confidence-weighted mean. Volume from absolute depth-map ROI.
    """

    def __init__(self, K: np.ndarray):
        self.K  = K.copy().astype(np.float64)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])

    def update_K(self, K: np.ndarray):
        
        self.K  = K.copy().astype(np.float64)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])

    def _depth_dims(self, bbox: Tuple[int,int,int,int],
                    depth_m: float) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (max(1, x2-x1) * depth_m / self.fx,
                max(1, y2-y1) * depth_m / self.fy)

    def _known_height_dims(self, bbox: Tuple[int,int,int,int],
                           camera_height_m: float,
                           pitch: float) -> Tuple[float, float]:
        z_approx = camera_height_m / max(np.cos(pitch), 0.05)
        return self._depth_dims(bbox, z_approx)

    def _compute_volume(self,
                        bbox: Tuple[int,int,int,int],
                        abs_depth: Optional[np.ndarray]) -> float:
        """
        Pothole volume from absolute metric depth map.
        Ground reference = median depth of border ring around bbox.
        Volume = Σ max(0, D_pixel − D_ground) × pixel_footprint_area.
        """
        if abs_depth is None:
            return 0.0
        x1, y1, x2, y2 = bbox
        H, W = abs_depth.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W-1, x2), min(H-1, y2)
        if x1 >= x2 or y1 >= y2:
            return 0.0

        ring = 10
        rx1, ry1 = max(0, x1-ring), max(0, y1-ring)
        rx2, ry2 = min(W-1, x2+ring), min(H-1, y2+ring)
        border = np.zeros(abs_depth.shape, bool)
        border[ry1:ry2, rx1:rx2] = True
        border[y1:y2, x1:x2]    = False

        ground_d   = abs_depth[border]
        if ground_d.size == 0:
            return 0.0
        ground_ref = float(np.nanmedian(ground_d))

        roi      = abs_depth[y1:y2, x1:x2].astype(np.float32)
        extra    = np.clip(roi - ground_ref, 0.0, None)
        pix_area = (ground_ref / self.fx) * (ground_ref / self.fy)
        return round(float(np.sum(extra) * pix_area), 6)

    @staticmethod
    def _depth_confidence(depth_m: float, cam_h: float) -> float:
        ratio = depth_m / max(cam_h, 0.1)
        if   ratio < 2:  return 0.95
        elif ratio < 5:  return 0.85
        elif ratio < 15: return 0.70
        elif ratio < 30: return 0.50
        else:            return 0.30

    def fuse(self,
             bbox: Tuple[int,int,int,int],
             homography_mapper: HomographyMapper,
             pitch: float, roll: float, yaw: float,
             camera_height_m: float,
             abs_depth: Optional[np.ndarray]) -> DimensionEstimate:

        results: Dict[str, Tuple[float, float, float]] = {}
        depth_val: float = camera_height_m / max(np.cos(pitch), 0.1)

        # Method A: Homography
        try:
            hd = homography_mapper.bbox_world_dims(bbox, pitch, roll, yaw)
            hc = homography_mapper.confidence(pitch, roll)
            if hd["width_m"] > 0 and hd["length_m"] > 0:
                results["homography"] = (hd["width_m"], hd["length_m"], hc)
        except Exception as exc:
            print(f"  [FUSE] Homography error: {exc}")

        # Method B: Absolute depth from depth map
        if abs_depth is not None:
            d_m = MiDaSDepthEstimator.bbox_metric_depth(abs_depth, bbox)
            depth_val = d_m
            dc  = self._depth_confidence(d_m, camera_height_m)
            dw, dl = self._depth_dims(bbox, d_m)
            if dw > 0 and dl > 0:
                results["depth"] = (dw, dl, dc)

        # Method C: Known camera height
        kw, kl = self._known_height_dims(bbox, camera_height_m, pitch)
        if kw > 0 and kl > 0:
            results["known_height"] = (kw, kl, 0.5)

        if not results:
            return DimensionEstimate(0.0, 0.0, 0.0, 0.0, depth_val, 0.0)

        total_conf = sum(v[2] for v in results.values()) or 1.0
        width_m    = sum(v[0] * v[2] for v in results.values()) / total_conf
        length_m   = sum(v[1] * v[2] for v in results.values()) / total_conf
        conf       = min(1.0, total_conf / max(1, len(results)))
        volume_m3  = self._compute_volume(bbox, abs_depth)
        weights    = {k: round(v[2]/total_conf, 3) for k, v in results.items()}

        return DimensionEstimate(
            width_m        = round(width_m,  4),
            length_m       = round(length_m, 4),
            area_m2        = round(width_m * length_m, 6),
            volume_m3      = volume_m3,
            depth_m        = round(depth_val, 3),
            confidence     = round(conf, 3),
            method_weights = weights,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – SEVERITY CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def classify_pothole_severity(area_m2: float, volume_m3: float) -> str:
    al = 0 if area_m2   < 0.05  else 1 if area_m2   < 0.20  else 2 if area_m2   < 0.50  else 3
    vl = 0 if volume_m3 < 0.001 else 1 if volume_m3 < 0.005 else 2 if volume_m3 < 0.020 else 3
    return ["Minimal", "Low", "Medium", "High"][max(al, vl)]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – POTHOLE DETECTOR (YOLO or mock)
# ══════════════════════════════════════════════════════════════════════════════

class PotholeDetector:
    """Wraps pothole-YOLO; falls back to a centred mock bbox if model absent."""

    def __init__(self, model_path: str, conf: float = 0.35,
                 device: Optional[torch.device] = None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.conf   = conf
        self._model = None
        self._mode  = "mock"

        if _YOLO_AVAILABLE and os.path.exists(model_path):
            try:
                self._model = YOLO(model_path)
                self._model.to(str(self.device))
                self._mode  = "yolo"
                print(f"  [DET] Pothole YOLO ← {model_path}")
            except Exception as exc:
                print(f"  [DET] YOLO load failed: {exc} → mock")
        else:
            print(f"  [DET] Model not found ({model_path}) → mock")

    # def detect(self, image: np.ndarray) -> List[Dict]:
    #     if self._mode == "yolo" and self._model is not None:
    #         results = self._model(image, conf=self.conf, verbose=False)
    #         out = []
    #         for r in results:
    #             for box in r.boxes:
    #                 x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    #                 out.append({
    #                     "bbox":       (x1, y1, x2, y2),
    #                     "confidence": float(box.conf[0]),
    #                     "class_name": self._model.names[int(box.cls[0])],
    #                 })
    #         return out
    #     # Mock: single centred bbox
    #     h, w  = image.shape[:2]
    #     cx, cy = w // 2, int(h * 0.70)
    #     hw, hh = w // 12, h // 14
    #     return [{"bbox": (cx-hw, cy-hh, cx+hw, cy+hh),
    #              "confidence": 0.72, "class_name": "pothole"}]

    def detect(self, image: np.ndarray, lower_conf: float = 0.15) -> List[Dict]:
        if self._mode == "yolo" and self._model is not None:
            # Run with lower threshold first
            results = self._model(image, conf=lower_conf, verbose=False, iou=0.45)
            
            # Apply additional confidence filtering and NMS
            detections = []
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    # Use adaptive threshold: lower for edge regions
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    h, w = image.shape[:2]
                    
                    # Lower threshold near image edges (where potholes may appear truncated)
                    edge_factor = 1.0
                    if x1 < w*0.1 or x2 > w*0.9 or y1 < h*0.1:
                        edge_factor = 0.7
                    
                    adaptive_thresh = self.conf * edge_factor
                    
                    if conf > adaptive_thresh:
                        detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "confidence": conf,
                            "class_name": self._model.names[int(box.cls[0])],
                        })
            
            # Apply Non-Maximum Suppression
            return self._nms(detections, iou_threshold=0.3)
        
        # Mock fallback when YOLO not available
        h, w = image.shape[:2]
        cx, cy = w // 2, int(h * 0.70)
        hw, hh = w // 12, h // 14
        return [{"bbox": (cx-hw, cy-hh, cx+hw, cy+hh),
                "confidence": 0.72, 
                "class_name": "pothole"}]

    def _nms(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping detections
            detections = [d for d in detections 
                        if self._compute_iou(best['bbox'], d['bbox']) < iou_threshold]
        return keep

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter / (area1 + area2 - inter + 1e-6)






class PotholeTracker:
    """Simple IoU-based tracker for temporal consistency"""
    
    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 5):
        self.next_id = 0
        self.tracks = {}  # track_id -> {'bbox': tuple, 'lost': int, 'history': list}
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost_frames
    
    def update(self, detections: List[Dict], frame_idx: int) -> List[Dict]:
        """Update tracks with new detections"""
        if not detections:
            # Increment lost counter for all tracks
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    del self.tracks[tid]
            return []
        
        # Compute IoU between existing tracks and new detections
        matched = []
        for det in detections:
            det_bbox = det['bbox']
            best_iou = 0
            best_tid = None
            
            for tid, track in self.tracks.items():
                iou = self._compute_iou(det_bbox, track['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_tid = tid
            
            if best_tid is not None:
                # Update existing track
                self.tracks[best_tid]['bbox'] = det_bbox
                self.tracks[best_tid]['lost'] = 0
                self.tracks[best_tid]['history'].append((frame_idx, det_bbox))
                det['track_id'] = best_tid
                matched.append(best_tid)
            else:
                # New track
                self.tracks[self.next_id] = {
                    'bbox': det_bbox,
                    'lost': 0,
                    'history': [(frame_idx, det_bbox)]
                }
                det['track_id'] = self.next_id
                self.next_id += 1
                matched.append(self.next_id - 1)
        
        # Remove unmatched tracks
        for tid in list(self.tracks.keys()):
            if tid not in matched:
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    del self.tracks[tid]
        
        return detections
    
    def get_interpolated(self, frame_idx: int) -> List[Dict]:
        """Get interpolated detections for frames without detection"""
        results = []
        for tid, track in self.tracks.items():
            history = track['history']
            if len(history) >= 2:
                # Linear interpolation between last two known positions
                (f1, b1), (f2, b2) = history[-2], history[-1]
                if f1 < frame_idx < f2:
                    ratio = (frame_idx - f1) / (f2 - f1)
                    interp_bbox = tuple(
                        int(b1[i] + ratio * (b2[i] - b1[i]))
                        for i in range(4)
                    )
                    results.append({
                        'bbox': interp_bbox,
                        'track_id': tid,
                        'interpolated': True,
                        'confidence': 0.5  # Lower confidence for interpolated
                    })
        return results
    
    @staticmethod
    def _compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter / (area1 + area2 - inter + 1e-6)




# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – DIMENSIONS CSV LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class DimensionsLogger:
    """Writes one row per detection to dimensions.csv; flushes incrementally."""

    FIELDS = [
        "timestamp", "frame", "source_file",
        "detection_id",           # index within frame
        "class_name", "detector_confidence",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        # Absolute depth
        "depth_m",
        # Dimensions
        "width_m", "length_m", "area_m2", "volume_m3",
        # Severity
        "severity",
        # Fusion metadata
        "fuse_confidence", "method_weights",
        # IMU state
        "camera_height_m", "pitch_deg", "roll_deg",
        # Depth map stats at bbox
        "depth_map_min_m", "depth_map_max_m", "depth_map_mean_m",
        # RANSAC depth confidence
        "depth_scaler_confidence",
    ]

    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self._path    = os.path.join(output_dir, "dimensions.csv")
        self._records: List[Dict] = []
        with open(self._path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()
        print(f"  [LOG] Dimensions CSV → {self._path}")

    def add(self, record: Dict):
        self._records.append(record)
        with open(self._path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS,
                           extrasaction="ignore").writerow(record)

    @property
    def records(self) -> List[Dict]:
        return self._records

    @property
    def csv_path(self) -> str:
        return self._path


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 – DEPTH PANEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_depth_panel(viz_u8: Optional[np.ndarray],
                       abs_depth: Optional[np.ndarray],
                       w: int, h: int,
                       detections: Optional[List[Dict]] = None) -> np.ndarray:
    """
    Build an annotated JET colourmap depth panel.
    Overlays per-detection depth labels and a colour scale bar.
    """
    if viz_u8 is None:
        panel = np.zeros((h, w, 3), np.uint8)
        cv2.putText(panel, "Depth initializing…",
                    (w//2 - 110, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
        return panel

    gray  = viz_u8 if len(viz_u8.shape) == 2 else cv2.cvtColor(viz_u8, cv2.COLOR_BGR2GRAY)
    panel = cv2.applyColorMap(cv2.resize(gray, (w, h)), cv2.COLORMAP_TURBO)

    # Scale factor for bbox coords (viz may be smaller than display)
    if abs_depth is not None:
        dh, dw = abs_depth.shape
    else:
        dh, dw = h, w
    sx = w / dw
    sy = h / dh

    # Overlay detection bboxes on depth panel
    if detections:
        for det in detections:
            bx1, by1, bx2, by2 = [int(v * s) for v, s in
                                   zip(det["bbox"], [sx, sy, sx, sy])]
            col = (255, 255, 255)
            cv2.rectangle(panel, (bx1, by1), (bx2, by2), col, 1)
            d_m = det.get("depth_m", 0.0)
            cv2.putText(panel, f"{d_m:.1f}m",
                        (bx1+2, max(by1-4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    # Colour scale bar (right edge)
    bar_w, bar_h = 22, h - 100
    bx, by = w - bar_w - 15, 50
    # for i in range(bar_h):
    #     val = int(255 * (i / bar_h))  # Remove the inversion (was 1 - i/bar_h)
    #     col = cv2.applyColorMap(np.uint8([[val]]), cv2.COLORMAP_TURBO)[0][0]
    #     cv2.line(panel, (bx, by+i), (bx+bar_w, by+i),
    #             (int(col[0]), int(col[1]), int(col[2])), 1)
    # cv2.putText(panel, "Near", (bx - 28, by + bar_h - 12),  # Swapped positions
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
    # cv2.putText(panel, "Far",  (bx - 24, by + 12),  # Swapped positions
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
    
    # for i in range(bar_h):
    #     val = int(255 * (1 - i / bar_h))
    #     col = cv2.applyColorMap(np.uint8([[val]]), cv2.COLORMAP_TURBO)[0][0]
    #     cv2.line(panel, (bx, by+i), (bx+bar_w, by+i),
    #              (int(col[0]), int(col[1]), int(col[2])), 1)
    # cv2.putText(panel, "Near", (bx - 28, by + 12),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
    # cv2.putText(panel, "Far",  (bx - 24, by + bar_h - 5),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)

    # Title
    cv2.rectangle(panel, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.putText(panel, "ABSOLUTE DEPTH MAP (metric)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    return panel


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 – MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class DepthDimensionPipeline:
    """
    Focused pipeline: absolute dense depth + dimension estimation only.

      RGB Frame ──► Undistort
      IMU       ──► pitch / roll / camera height
      MiDaS     ──► inv_depth
      GeometryAnchoredDepthScaler ──► abs_depth_m (metric, dense)
      Pothole detection ──► DimensionFusionEngine ──► DimensionsLogger
      Output: dual-view video (RGB + TURBO depth) + dimensions.csv
    """

    def __init__(self,
                 calibration_path:   str,
                 pothole_model_path: str,
                 base_output_dir:    str   = "output",
                 midas_model_type:   str   = "MiDaS_small",
                 initial_height_m:   float = 1.2,
                 pothole_conf:       float = 0.35,
                 focal_length_mm:    float = 4.0,
                 sensor_width_mm:    float = 5.6,
                 depth_every_n:      int   = 1):
        print("=" * 72)
        print("  DEPTH & DIMENSION ESTIMATION PIPELINE")
        print("=" * 72)

        self.run          = RunManager(base_output_dir)
        self.depth_every  = max(1, depth_every_n)
        self._depth_ctr   = 0

        # self.detector = PotholeDetector(pothole_model_path, conf=pothole_conf)
        # self.tracker = PotholeTracker(iou_threshold=0.3, max_lost_frames=3)  # Add this line
        # self.logger = DimensionsLogger(self.run.data_dir)

        print("\n[1/7] Camera calibration …")
        self.calibrator = CameraCalibrator()
        self.calibrator.load(calibration_path)

        print("[2/7] Undistorter …")
        self.undistorter = ImageUndistorter(
            self.calibrator.K, self.calibrator.dist,
            self.calibrator.image_shape, alpha=0.0)

        print("[3/7] IMU processor …")
        self.imu = IMUProcessor(initial_height_m=initial_height_m)

        print("[4/7] MiDaS depth estimator …")
        self.depth_est = MiDaSDepthEstimator(model_type=midas_model_type)

        print("[5/7] Homography mapper …")
        self.homography = HomographyMapper(self.calibrator.K, initial_height_m)

        print("[6/7] Geometry-anchored depth scaler …")
        self.depth_scaler = GeometryAnchoredDepthScaler(self.calibrator.K)

        print("[7/7] Dimension fusion engine + detector + logger …")
        self.fuser    = DimensionFusionEngine(self.calibrator.K)
        self.detector = PotholeDetector(pothole_model_path, conf=pothole_conf)
        self.tracker = PotholeTracker(iou_threshold=0.3, max_lost_frames=3)  # Add this line
        self.logger   = DimensionsLogger(self.run.data_dir)

        self._focal_mm = focal_length_mm
        self._sens_mm  = sensor_width_mm

        # Async JPEG writes
        self._jpeg_pool = ThreadPoolExecutor(max_workers=2)

        # Depth cache
        self._cached_abs_depth: Optional[np.ndarray] = None
        self._cached_viz_u8:    Optional[np.ndarray] = None
        self._cached_conf:      float                = 0.0

        self.run.set_settings(
            calibration_path  = calibration_path,
            pothole_model     = pothole_model_path,
            midas_model       = midas_model_type,
            initial_height_m  = initial_height_m,
            pothole_conf      = pothole_conf,
            focal_length_mm   = focal_length_mm,
            sensor_width_mm   = sensor_width_mm,
            depth_every_n     = depth_every_n,
        )
        print("\n  ✓ Pipeline ready\n" + "=" * 72)

    # ── helpers ───────────────────────────────────────────────────────────────

    # def load_sensor_csv(self, path: str):
    #     self.imu.load_from_csv(path)

    def load_sensor_csv(self, path: str):
        self.imu.load_from_csv(path)
        # Test interpolation at different times
        print(f"  [IMU] Test at t=0: pitch={np.degrees(self.imu.orientation_at(0)['pitch']):.2f}°")
        if len(self.imu._t) > 10:
            mid_t = self.imu._t[len(self.imu._t)//2]
            print(f"  [IMU] Test at t={mid_t:.1f}: pitch={np.degrees(self.imu.orientation_at(mid_t)['pitch']):.2f}°")

    def _sync_K(self, K: np.ndarray):
        self.homography.update_K(K)
        self.depth_scaler.update_K(K)
        self.fuser.update_K(K)

    def _async_imwrite(self, path: str, img: np.ndarray, quality: int = 90):
        self._jpeg_pool.submit(
            cv2.imwrite, path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])

    # ── core per-frame processing ─────────────────────────────────────────────

    def _process_frame(self,
                    raw_frame:   np.ndarray,
                    timestamp:   float,
                    source_name: str,
                    frame_idx:   int
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full pipeline for one frame.
        Returns (rgb_annotated_BGR, depth_panel_BGR).
        """
        # STEP 1: Undistort → effective K
        frame = self.undistorter.undistort(raw_frame)
        H, W  = frame.shape[:2]

        effective_K = self.undistorter.current_K
        if effective_K is not None:
            self._sync_K(effective_K)

        # STEP 2: Pothole detection (needed BEFORE height selection)
        detections = self.detector.detect(frame)
        detections = self.tracker.update(detections, frame_idx)
        pothole_detected = len(detections) > 0
        
        # STEP 3: Get IMU orientation (always from IMU)
        ori = self.imu.orientation_at(timestamp)
        pitch = ori["pitch"]
        roll = ori["roll"]
        yaw = ori["yaw"]
        
        # STEP 4: Get hybrid height (static normally, dynamic during potholes)
        dt = 1.0 / 30.0  # Default for 30fps, you can compute from actual timestamps if needed
        cam_h = self.imu.get_height(timestamp, pothole_detected, dt)
        
        # Update homography with current height
        self.homography.update_height(cam_h)
        
        # Debug output (optional)
        if pothole_detected and self.imu._integrating:
            print(f"  [HEIGHT] Frame {frame_idx}: Dynamic mode, height={cam_h:.3f}m")
        elif not pothole_detected:
            print(f"  [HEIGHT] Frame {frame_idx}: Static mode, height={cam_h:.3f}m")

        # STEP 5: MiDaS + GeometryAnchoredDepthScaler (throttled)
        self._depth_ctr += 1
        if self._depth_ctr % self.depth_every == 0 or self._cached_abs_depth is None:
            inv_depth, _ = self.depth_est.run(frame, frame_id=frame_idx)
            if inv_depth is not None:
                abs_depth, viz_u8, conf = self.depth_scaler.scale_frame(
                    inv_depth, self.homography, pitch, roll, yaw, cam_h)
                self._cached_abs_depth = abs_depth
                self._cached_viz_u8    = viz_u8
                self._cached_conf      = conf

        abs_depth: Optional[np.ndarray] = self._cached_abs_depth
        viz_u8:    Optional[np.ndarray] = self._cached_viz_u8
        depth_conf: float               = self._cached_conf

        # STEP 6: Add interpolated detections for missed frames
        interpolated = self.tracker.get_interpolated(frame_idx)
        if interpolated:
            detections.extend(interpolated)

        rgb_ann = frame.copy()

        SEV_COLORS = {
            "Minimal": (0, 230, 80),
            "Low":     (0, 220, 150),
            "Medium":  (0, 160, 255),
            "High":    (0, 0, 255),
        }

        det_meta: List[Dict] = []   # for depth panel overlay

        for i, det in enumerate(detections):
            bbox = det["bbox"]
            dims = self.fuser.fuse(
                bbox=bbox,
                rgb_image=frame,  # Pass the RGB frame
                homography_mapper=self.homography,
                pitch=pitch,
                roll=roll,
                yaw=yaw,
                camera_height_m=cam_h,
                abs_depth=abs_depth,
            )
            # dims = self.fuser.fuse(
            #     bbox              = bbox,
            #     homography_mapper = self.homography,
            #     pitch             = pitch,
            #     roll              = roll,
            #     yaw               = yaw,
            #     camera_height_m   = cam_h,
            #     abs_depth         = abs_depth,
            # )
            severity = classify_pothole_severity(dims.area_m2, dims.volume_m3)
            col      = SEV_COLORS.get(severity, (0, 200, 255))
            x1, y1, x2, y2 = bbox

            # Depth map stats inside bbox (for CSV)
            if abs_depth is not None:
                ax1 = max(0, x1); ay1 = max(0, y1)
                ax2 = min(W-1, x2); ay2 = min(H-1, y2)
                if ax1 < ax2 and ay1 < ay2:
                    roi = abs_depth[ay1:ay2, ax1:ax2]
                    dm_min  = round(float(np.nanmin(roi)), 3)
                    dm_max  = round(float(np.nanmax(roi)), 3)
                    dm_mean = round(float(np.nanmean(roi)), 3)
                else:
                    dm_min = dm_max = dm_mean = dims.depth_m
            else:
                dm_min = dm_max = dm_mean = dims.depth_m

            # ── Draw bbox + measurements on RGB frame ─────────────────────────
            # Thick coloured border
            cv2.rectangle(rgb_ann, (x1, y1), (x2, y2), col, 2)
            # Corner accents
            corner_len = min(18, (x2-x1)//4, (y2-y1)//4)
            for cx_off, cy_off in [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]:
                dx = corner_len if cx_off == x1 else -corner_len
                dy = corner_len if cy_off == y1 else -corner_len
                cv2.line(rgb_ann, (cx_off, cy_off), (cx_off+dx, cy_off), col, 3)
                cv2.line(rgb_ann, (cx_off, cy_off), (cx_off, cy_off+dy), col, 3)

            # Label box
            lines = [
                f"[{severity}]  conf:{det['confidence']:.2f}",
                f"W:{dims.width_m:.3f}m  L:{dims.length_m:.3f}m",
                f"A:{dims.area_m2:.4f}m²  V:{dims.volume_m3:.5f}m³",
                f"D:{dims.depth_m:.2f}m  fuse:{dims.confidence:.2f}",
            ]
            fs   = 0.38
            lh   = 14
            tw   = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0] for l in lines)
            lbx1 = x1
            lby1 = y2 + 4
            lby2 = lby1 + lh * len(lines) + 4
            if lby2 < H:
                cv2.rectangle(rgb_ann, (lbx1, lby1-2), (lbx1+tw+6, lby2), (20,20,20), -1)
                for j, line in enumerate(lines):
                    cv2.putText(rgb_ann, line, (lbx1+3, lby1 + j*lh + lh - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
            else:
                # Draw above if no room below
                lby2 = y1 - 4
                lby1 = lby2 - lh * len(lines)
                if lby1 > 0:
                    cv2.rectangle(rgb_ann, (lbx1, lby1-2), (lbx1+tw+6, lby2+2), (20,20,20), -1)
                    for j, line in enumerate(lines):
                        cv2.putText(rgb_ann, line, (lbx1+3, lby1 + j*lh + lh - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)

            det_meta.append({"bbox": bbox, "depth_m": dims.depth_m})

            # ── Log to CSV ────────────────────────────────────────────────────
            self.logger.add({
                "timestamp":              datetime.now().isoformat(),
                "frame":                  frame_idx,
                "source_file":            source_name,
                "detection_id":           i,
                "class_name":             det["class_name"],
                "detector_confidence":    round(det["confidence"], 4),
                "bbox_x1":                x1, "bbox_y1": y1,
                "bbox_x2":                x2, "bbox_y2": y2,
                "depth_m":                dims.depth_m,
                "width_m":                dims.width_m,
                "length_m":               dims.length_m,
                "area_m2":                dims.area_m2,
                "volume_m3":              dims.volume_m3,
                "severity":               severity,
                "fuse_confidence":        dims.confidence,
                "method_weights":         json.dumps(dims.method_weights),
                "camera_height_m":        round(cam_h, 4),
                "pitch_deg":              round(np.degrees(pitch), 3),
                "roll_deg":               round(np.degrees(roll),  3),
                "depth_map_min_m":        dm_min,
                "depth_map_max_m":        dm_max,
                "depth_map_mean_m":       dm_mean,
                "depth_scaler_confidence": round(depth_conf, 3),
            })

        # ── HUD: bottom sensor bar ────────────────────────────────────────────
        cv2.rectangle(rgb_ann, (0, H-26), (W, H), (18, 18, 18), -1)
        bar = (f"frame={frame_idx}  t={timestamp:.1f}s  "
               f"h={cam_h:.2f}m  p={np.degrees(pitch):.1f}°  "
               f"r={np.degrees(roll):.1f}°  "
               f"detections={len(detections)}  "
               f"depth_conf={depth_conf:.2f}")
        cv2.putText(rgb_ann, bar, (6, H-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160,160,160), 1, cv2.LINE_AA)

        # ── HUD: top title ────────────────────────────────────────────────────
        cv2.rectangle(rgb_ann, (0, 0), (W, 30), (18, 18, 18), -1)
        cv2.putText(rgb_ann, "DEPTH & DIMENSION PIPELINE  |  RGB + DETECTION VIEW",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 1)

        # Pothole badge
        if detections:
            badge = f"DETECTIONS: {len(detections)}"
            (bw, bh), _ = cv2.getTextSize(badge, 0, 0.55, 2)
            cv2.rectangle(rgb_ann, (W-bw-22, 4), (W-4, bh+12), (0, 0, 180), -1)
            cv2.putText(rgb_ann, badge, (W-bw-12, bh+6),
                        0, 0.55, (255,255,255), 2, cv2.LINE_AA)

        # ── Depth panel ───────────────────────────────────────────────────────
        depth_panel = _build_depth_panel(viz_u8, abs_depth, W, H, det_meta)

        # Depth stats overlay on depth panel
        if abs_depth is not None:
            self.test_depth_orientation(frame_idx, abs_depth)
            stats_txt = (f"global min:{abs_depth.min():.1f}m  "
                         f"max:{abs_depth.max():.1f}m  "
                         f"med:{np.median(abs_depth):.1f}m  "
                         f"scaler_conf={depth_conf:.2f}")
            cv2.rectangle(depth_panel, (0, H-26), (W, H), (18, 18, 18), -1)
            cv2.putText(depth_panel, stats_txt, (6, H-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 220, 255), 1, cv2.LINE_AA)

        return rgb_ann, depth_panel

    # ── public entry points ───────────────────────────────────────────────────


    def test_depth_orientation(self, frame_idx: int, abs_depth: np.ndarray):
        """Test if depth is correctly oriented (closer objects = smaller depth values)"""
        h, w = abs_depth.shape
        
        # Sample depths at different vertical positions
        top_depth = np.median(abs_depth[0:h//4, :])
        mid_depth = np.median(abs_depth[h//4:3*h//4, :])
        bot_depth = np.median(abs_depth[3*h//4:, :])
        
        print(f"  [VERIFY] Frame {frame_idx}: Top={top_depth:.2f}m, Mid={mid_depth:.2f}m, Bottom={bot_depth:.2f}m")
        
        # In a driving scene, bottom of image should be closer (smaller depth)
        if bot_depth > top_depth:
            print(f"  [WARNING] Depth appears INVERTED! Bottom ({bot_depth:.2f}) > Top ({top_depth:.2f})")
        else:
            print(f"  [OK] Depth orientation correct: Bottom ({bot_depth:.2f}) < Top ({top_depth:.2f})")

    def process_image(self, image_path: str, timestamp: float = 0.0):
        img = cv2.imread(image_path)
        if img is None:
            print(f"  [PIPE] Cannot read: {image_path}"); return
        rgb_ann, depth_p = self._process_frame(
            img, timestamp, os.path.basename(image_path), 0)
        stem = Path(image_path).stem
        self._async_imwrite(os.path.join(self.run.annotated_dir, f"rgb_{stem}.jpg"),   rgb_ann)
        self._async_imwrite(os.path.join(self.run.annotated_dir, f"depth_{stem}.jpg"), depth_p, 88)
        print(f"  [PIPE] Image processed → {self.run.annotated_dir}")

    def process_image_folder(self, folder: str,
                             start_timestamp: float = 0.0,
                             interval_s: float = 1.0):
        exts  = {".jpg", ".jpeg", ".png"}
        paths = sorted(p for p in Path(folder).iterdir()
                       if p.suffix.lower() in exts)
        for i, p in enumerate(paths):
            self.process_image(str(p), start_timestamp + i * interval_s)
        print(f"  [PIPE] Folder done: {len(paths)} images")

    def process_video(self,
                      video_path:     str,
                      output_video:   Optional[str] = None,
                      frame_interval: int   = 15,
                      start_t:        float = 0.0) -> int:
        """
        Process a video file.

        Produces:
          • Dual-view output video: left = RGB+detections, right = TURBO depth map
          • dimensions.csv with per-detection measurements

        frame_interval : Full pipeline every N raw frames. Between processed
                         frames the last annotated pair is written to video
                         (smooth output at full fps).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [PIPE] Cannot open: {video_path}"); return 0

        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vid_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
        vid_name = Path(video_path).stem

        print(f"  [PIPE] Video: {vid_w}×{vid_h}  fps={fps:.1f}  "
              f"frames={total_fr}  rotation={rotation}°")

        # Rotation handling
        if rotation == 90:
            display_w, display_h = vid_h, vid_w
            rot_code = cv2.ROTATE_90_CLOCKWISE
        elif rotation == 270:
            display_w, display_h = vid_h, vid_w
            rot_code = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif rotation == 180:
            display_w, display_h = vid_w, vid_h
            rot_code = cv2.ROTATE_180
        else:
            display_w, display_h = vid_w, vid_h
            rot_code = None

        # Scale K if video resolution ≠ calibration resolution
        cal_wh = self.calibrator.image_shape
        sx     = display_w / cal_wh[0]
        sy     = display_h / cal_wh[1]
        if abs(sx-1.0) > 0.01 or abs(sy-1.0) > 0.01:
            K_orig   = self.calibrator.K
            K_scaled = K_orig.copy().astype(np.float64)
            K_scaled[0,0] *= sx;  K_scaled[0,2] *= sx
            K_scaled[1,1] *= sy;  K_scaled[1,2] *= sy
            self.undistorter = ImageUndistorter(
                K_scaled, self.calibrator.dist, (display_w, display_h), alpha=0.0)
            self._sync_K(K_scaled)
            print(f"  [PIPE] K scaled sx={sx:.3f} sy={sy:.3f}")

        self.undistorter.prepare((display_w, display_h))

        # Dual-view layout: [RGB | DEPTH] side by side + 50px header bar
        writer_w = display_w * 2
        writer_h = display_h + 50

        writer        = None
        actual_output = output_video

        if output_video:
            os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
            for vid_path, fcc in [
                (output_video,                               "mp4v"),
                (output_video,                               "XVID"),
                (output_video,                               "X264"),
                (output_video.rsplit(".", 1)[0] + "_fb.avi", "MJPG"),
            ]:
                fourcc   = cv2.VideoWriter_fourcc(*fcc)
                test_w   = cv2.VideoWriter(vid_path, fourcc, fps, (writer_w, writer_h))
                if test_w.isOpened():
                    writer        = test_w
                    actual_output = vid_path
                    print(f"  [PIPE] VideoWriter ({fcc}) → {vid_path}  "
                          f"size={writer_w}×{writer_h}")
                    break
                test_w.release()
            if writer is None:
                print("  [PIPE] WARNING: no VideoWriter codec — annotated JPEGs only.")

        ann_dir     = self.run.annotated_dir
        frame_idx   = 0
        saved       = 0
        t_start     = time.time()
        fps_disp    = 0.0
        fps_ctr     = 0
        fps_t0      = time.time()
        total_dets  = 0

        last_rgb:   Optional[np.ndarray] = None
        last_depth: Optional[np.ndarray] = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if rot_code is not None:
                frame = cv2.rotate(frame, rot_code)

            t = start_t + frame_idx / fps

            if frame_idx % frame_interval == 0:
                rgb_ann, depth_panel = self._process_frame(
                    frame, t, vid_name, frame_idx)
                # Count detections this frame
                total_dets += sum(1 for r in self.logger.records
                                  if r.get("frame") == frame_idx)
                last_rgb   = rgb_ann
                last_depth = depth_panel
                self._async_imwrite(
                    os.path.join(ann_dir, f"frame_{frame_idx:06d}.jpg"), rgb_ann)
                saved += 1

            fps_ctr += 1
            if time.time() - fps_t0 >= 1.0:
                fps_disp = fps_ctr; fps_ctr = 0; fps_t0 = time.time()

            # Write to video
            if writer and writer.isOpened() and last_rgb is not None and last_depth is not None:
                tv = cv2.resize(last_rgb,   (display_w, display_h), interpolation=cv2.INTER_LINEAR)
                dp = cv2.resize(last_depth, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

                # Header info bar
                info = np.zeros((50, writer_w, 3), np.uint8)
                info[:] = (15, 20, 35)
                cv2.putText(
                    info,
                    (f"DEPTH & DIMENSION PIPELINE  |  "
                     f"FPS:{fps_disp:.0f}  frame:{frame_idx}/{total_fr}  "
                     f"detections_total:{len(self.logger.records)}"),
                    (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 1)

                # Left/right labels
                cv2.putText(info, "◀  RGB + DETECTIONS",
                            (14, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
                cv2.putText(info, "ABSOLUTE DEPTH MAP  ▶",
                            (display_w + 14, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,220,255), 1)

                out_frame = np.vstack([info, np.hstack([tv, dp])])

                assert out_frame.shape[:2] == (writer_h, writer_w), \
                    f"Frame shape mismatch: {out_frame.shape} vs ({writer_h},{writer_w})"
                writer.write(out_frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                pct     = int(frame_idx / max(total_fr, 1) * 100)
                elapsed = time.time() - t_start
                print(f"  [PIPE] {frame_idx}/{total_fr} ({pct}%)  "
                      f"processed={saved}  records={len(self.logger.records)}  "
                      f"elapsed={elapsed:.0f}s")

        cap.release()
        if writer and writer.isOpened():
            writer.release()
            print(f"  [PIPE] Video written → {actual_output}")
        self._jpeg_pool.shutdown(wait=True)

        elapsed = time.time() - t_start
        print(f"  [PIPE] Done: {frame_idx} frames  {saved} processed  "
              f"{len(self.logger.records)} detections  {elapsed:.1f}s")
        return len(self.logger.records)

    # ── save results ──────────────────────────────────────────────────────────

    def save_results(self, input_video: Optional[str] = None) -> Dict:
        records = self.logger.records
        n       = len(records)

        # Summary statistics
        summary: Dict = {
            "processed_at":   datetime.now().isoformat(),
            "total_detections": n,
            "csv_path":       self.logger.csv_path,
        }

        if n:
            areas   = [r["area_m2"]   for r in records]
            vols    = [r["volume_m3"] for r in records]
            depths  = [r["depth_m"]   for r in records]
            widths  = [r["width_m"]   for r in records]
            lengths = [r["length_m"]  for r in records]
            sevs    = [r["severity"]  for r in records]

            summary.update({
                "depth_stats": {
                    "min_m":  round(float(np.min(depths)),  3),
                    "max_m":  round(float(np.max(depths)),  3),
                    "mean_m": round(float(np.mean(depths)), 3),
                },
                "width_stats": {
                    "min_m":  round(float(np.min(widths)),  4),
                    "max_m":  round(float(np.max(widths)),  4),
                    "mean_m": round(float(np.mean(widths)), 4),
                },
                "length_stats": {
                    "min_m":  round(float(np.min(lengths)),  4),
                    "max_m":  round(float(np.max(lengths)),  4),
                    "mean_m": round(float(np.mean(lengths)), 4),
                },
                "area_stats": {
                    "total_m2": round(float(np.sum(areas)), 4),
                    "mean_m2":  round(float(np.mean(areas)), 6),
                    "max_m2":   round(float(np.max(areas)), 6),
                },
                "volume_stats": {
                    "total_m3": round(float(np.sum(vols)), 6),
                    "mean_m3":  round(float(np.mean(vols)), 6),
                },
                "severity_counts": {s: sevs.count(s)
                                    for s in ("Minimal","Low","Medium","High")},
            })

        # Save summary JSON
        sum_path = os.path.join(self.run.data_dir, "summary.json")
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self._print_summary(summary)
        self.run.save_manifest(summary, video_path=input_video)
        return summary

    @staticmethod
    def _print_summary(s: Dict):
        n = s.get("total_detections", 0)
        print("\n  ══ SUMMARY ══════════════════════════════════════════")
        print(f"  Total detections : {n}")
        if n:
            ds = s.get("depth_stats", {})
            ws = s.get("width_stats", {})
            ls = s.get("length_stats", {})
            vs = s.get("volume_stats", {})
            as_ = s.get("area_stats", {})
            sc  = s.get("severity_counts", {})
            print(f"  Depth   : min={ds.get('min_m','?')}m  "
                  f"max={ds.get('max_m','?')}m  mean={ds.get('mean_m','?')}m")
            print(f"  Width   : min={ws.get('min_m','?')}m  "
                  f"max={ws.get('max_m','?')}m  mean={ws.get('mean_m','?')}m")
            print(f"  Length  : min={ls.get('min_m','?')}m  "
                  f"max={ls.get('max_m','?')}m  mean={ls.get('mean_m','?')}m")
            print(f"  Area    : total={as_.get('total_m2','?')}m²  "
                  f"mean={as_.get('mean_m2','?')}m²")
            print(f"  Volume  : total={vs.get('total_m3','?')}m³  "
                  f"mean={vs.get('mean_m3','?')}m³")
            print(f"  Severity: Min={sc.get('Minimal',0)}  Low={sc.get('Low',0)}  "
                  f"Med={sc.get('Medium',0)}  High={sc.get('High',0)}")
        print("  ════════════════════════════════════════════════════")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 – CALIBRATION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def run_calibration(image_folder: str,
                    output_path: str = "outputs/cal/calibration.npz",
                    chessboard_size: Tuple[int,int] = (8, 5),
                    square_size_m: float = 0.030) -> bool:
    cal   = CameraCalibrator(chessboard_size, square_size_m)
    exts  = {".jpg", ".jpeg", ".png"}
    paths = sorted(p for p in Path(image_folder).iterdir()
                   if p.suffix.lower() in exts)
    if not paths:
        print(f"  [CAL] No images found in {image_folder}")
        return False
    print(f"  [CAL] Processing {len(paths)} calibration images …")
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            cal.add_image(img)
    success = cal.calibrate()
    if success:
        cal.save(output_path)
    return success


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 – CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Depth & Dimension Estimation Pipeline")
    parser.add_argument("--mode",
                        choices=["calibrate", "image", "folder", "video"],
                        required=True)
    parser.add_argument("--input",          required=True,
                        help="Image / video / folder path")
    parser.add_argument("--calibration",    default="outputs/cal/calibration.npz")
    parser.add_argument("--sensor_csv",     default=None,
                        help="IMU CSV (AccelX/Y/Z, GyroX/Y/Z, Timestamp)")
    parser.add_argument("--pothole_model",  default="model/pothole_yolo.pt")
    parser.add_argument("--output_dir",     default="output")
    parser.add_argument("--output_video",   default=None,
                        help="Path for dual-view output video (.mp4)")
    parser.add_argument("--frame_interval", type=int,   default=1,
                        help="Run full pipeline every N raw frames "
                             "(output video stays at full fps)")
    parser.add_argument("--initial_height", type=float, default=2.3,
                        help="Camera height above ground [m]")
    parser.add_argument("--midas_model",    default="MiDaS_small",
                        choices=["MiDaS_small", "DPT_Hybrid", "DPT_Large"])
    parser.add_argument("--depth_every",    type=int,   default=1,
                        help="Run MiDaS every N processed frames "
                             "(1=every, 3=every 3rd → ~3× faster)")
    parser.add_argument("--pothole_conf",   type=float, default=0.35)
    parser.add_argument("--focal_mm",       type=float, default=4.0,
                        help="Camera focal length [mm]")
    parser.add_argument("--sensor_mm",      type=float, default=5.6,
                        help="Camera sensor width [mm]")
    parser.add_argument("--chessboard",     default="8,5",
                        help="[calibrate] inner corners WxH e.g. '8,5'")
    parser.add_argument("--square_size",    type=float, default=0.030,
                        help="[calibrate] square size [m]")
    args = parser.parse_args()

    if args.mode == "calibrate":
        cb = tuple(int(x) for x in args.chessboard.split(","))
        ok = run_calibration(
            image_folder    = args.input,
            output_path     = args.calibration,
            chessboard_size = cb,            # type: ignore[arg-type]
            square_size_m   = args.square_size,
        )
        print("  Calibration", "succeeded ✓" if ok else "FAILED ✗")
        raise SystemExit(0 if ok else 1)

    pipe = DepthDimensionPipeline(
        calibration_path  = args.calibration,
        pothole_model_path= args.pothole_model,
        base_output_dir   = args.output_dir,
        midas_model_type  = args.midas_model,
        initial_height_m  = args.initial_height,
        pothole_conf      = args.pothole_conf,
        focal_length_mm   = args.focal_mm,
        sensor_width_mm   = args.sensor_mm,
        depth_every_n     = args.depth_every,
    )

    if args.sensor_csv:
        pipe.load_sensor_csv(args.sensor_csv)

    if args.mode == "image":
        pipe.process_image(args.input, timestamp=0.0)

    elif args.mode == "folder":
        pipe.process_image_folder(args.input)

    elif args.mode == "video":
        pipe.process_video(
            video_path     = args.input,
            output_video   = args.output_video,
            frame_interval = 1,
        )

    pipe.save_results(input_video=args.input if args.mode == "video" else None)