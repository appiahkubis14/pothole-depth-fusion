"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        DEPTH & DIMENSION ESTIMATION PIPELINE                                ║
║        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                          ║
║  Data flow:                                                                 ║
║    RGB Frame ──► Undistort                                                  ║
║    IMU       ──► Pitch / Roll / Camera-height                               ║
║    Calib     ──► K, dist ──► Undistort ──► Homography metric anchors        ║
║    RGB Frame ──► Depth Anything V2 (relative inverse-depth  and metric depth)                             ║
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

# from pipeline.dav2e_utils import _YOLO_AVAILABLE, _PANDAS_AVAILABLE

# Depth availability flags (check if depth estimation modules are available)
# try:
#     from pipeline.dav2e import DepthAnythingV2Estimator
#     _DEPTHANYTHINGV2_AVAILABLE = True
# except ImportError:
#     _DEPTHANYTHINGV2_AVAILABLE = False
#     print("[WARN] DepthAnythingV2 not available")

# try:
#     from pipeline.depth_est import DepthEstimator  # or whatever your depth module is
#     _DEPTHANYTHING_AVAILABLE = True
# except ImportError:
#     _DEPTHANYTHING_AVAILABLE = False
# ══════════════════════════════════════════════════════════════════════════════
# RUN MANAGER
# ══════════════════════════════════════════════════════════════════════════════
from pipeline.runner import RunManager


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – CAMERA CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
from pipeline.cam_cal import CameraCalibrator

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – IMAGE UNDISTORTION
# ══════════════════════════════════════════════════════════════════════════════
from pipeline.img_undist import ImageUndistorter
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – MIDAS DEPTH ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – DEPTHANYTHING V2 DEPTH ESTIMATOR (Paper's Approach)
# ══════════════════════════════════════════════════════════════════════════════
from pipeline.dav2e import DepthAnythingV2Estimator
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – GEOMETRY-ANCHORED DEPTH SCALER
# ══════════════════════════════════════════════════════════════════════════════
from pipeline.geom_anchor import GeometryAnchoredDepthScaler
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – GROUND-PLANE HOMOGRAPHY
# ══════════════════════════════════════════════════════════════════════════════
from pipeline.homography import HomographyMapper
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – IMU PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════
from pipeline.imu_proc import IMUProcessor
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – DIMENSION FUSION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – MBTP: MINIMUM BOUNDING TRIANGULATED PIXEL (Paper's Method)
# ══════════════════════════════════════════════════════════════════════════════

# from mbtp_est import MBTPAreaEstimator

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7C – CDKF: CONFIDENCE AND DISTANCE-BASED KALMAN FILTER (Paper's Method)
# ══════════════════════════════════════════════════════════════════════════════

from pipeline.cdfk_est import ConfidenceDistanceKalmanFilter

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7B – UPDATED DIMENSION FUSION (Integrates MBTP)
# ══════════════════════════════════════════════════════════════════════════════
from pipeline.edfe_est import EnhancedDimensionFusionEngine

# ══════════════════════════════════════════════════════════════════════════════
# DIMENSION ESTIMATE DATACLASS
# ══════════════════════════════════════════════════════════════════════════════




# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – SEVERITY CLASSIFIER
# ════════════════════════════════════
def classify_pothole_severity(area_m2: float, volume_m3: float) -> str:
    al = 0 if area_m2   < 0.05  else 1 if area_m2   < 0.20  else 2 if area_m2   < 0.50  else 3
    vl = 0 if volume_m3 < 0.001 else 1 if volume_m3 < 0.005 else 2 if volume_m3 < 0.020 else 3
    return ["Minimal", "Low", "Medium", "High"][max(al, vl)]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – POTHOLE DETECTOR (YOLO or mock)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – ACSH-YOLOv8 DETECTOR (Paper's ACmix + Small Object Head)
# ══════════════════════════════════════════════════════════════════════════════

from pipeline.yolo_det import ACSHYOLOv8Detector


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9B – BoT-SORT TRACKER (Paper's State-of-the-Art Tracking)
# ══════════════════════════════════════════════════════════════════════════════
from pipeline.tracker import BoTSORTTracker

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – DIMENSIONS CSV LOGGER
# ══════════════════════════════════════════════════════════════════════════════

from pipeline.dim_logger import DimensionsLogger

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

        print("[4/7] DepthAnything V2 depth estimator …")
        self.depth_est = DepthAnythingV2Estimator(model_type="vits")  # or "vitb" for better accuracy


        print("[5/7] Homography mapper …")
        self.homography = HomographyMapper(self.calibrator.K, initial_height_m)

        print("[6/7] Geometry-anchored depth scaler …")
        self.depth_scaler = GeometryAnchoredDepthScaler(self.calibrator.K)

        # print("[7/7] Dimension fusion engine + detector + logger …")
        
        # print("[7/7] Enhanced dimension fusion + ACSH-YOLO + BoT-SORT …")
        # self.fuser = EnhancedDimensionFusionEngine(self.calibrator.K)
        # self.detector = ACSHYOLOv8Detector(pothole_model_path, conf=pothole_conf)
        # self.tracker = BoTSORTTracker(iou_threshold=0.3, max_lost_frames=30, use_cmc=True)
        # self.logger   = DimensionsLogger(self.run.data_dir)

        self.fuser = EnhancedDimensionFusionEngine(self.calibrator.K)
        self.detector = ACSHYOLOv8Detector(pothole_model_path, conf=pothole_conf)
        self.tracker = BoTSORTTracker(iou_threshold=0.3, max_lost_frames=30, use_cmc=True)
        self.cdkf = ConfidenceDistanceKalmanFilter()  # <-- ADD THIS
        self.logger = DimensionsLogger(self.run.data_dir)

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

        # for i, det in enumerate(detections):
        #     bbox = det["bbox"]
        #     dims = self.fuser.fuse(
        #         bbox              = bbox,
        #         homography_mapper = self.homography,
        #         pitch             = pitch,
        #         roll              = roll,
        #         yaw               = yaw,
        #         camera_height_m   = cam_h,
        #         abs_depth         = abs_depth,
        #     )
        #     severity = classify_pothole_severity(dims.area_m2, dims.volume_m3)
        # In _process_frame, inside the detection loop:
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            track_id = det.get("track_id", i)  # Get track ID from tracker
            detection_confidence = det["confidence"]
            
            # Get raw dimensions from fusion engine
            dims = self.fuser.fuse(
                bbox=bbox,
                homography_mapper=self.homography,
                pitch=pitch,
                roll=roll,
                yaw=yaw,
                camera_height_m=cam_h,
                abs_depth=abs_depth,
            )
            
            # Calculate distance to pothole (from depth map or homography)
            if abs_depth is not None:
                depth_m = DepthAnythingV2Estimator.bbox_metric_depth(abs_depth, bbox)
            else:
                depth_m = cam_h / max(np.cos(pitch), 0.1)
            
            # Apply CDKF smoothing to area
            dt = 1.0  # 1 frame delta
            smoothed_area = self.cdkf.update(
                track_id=track_id,
                measured_area_m2=dims.area_m2,
                detection_confidence=detection_confidence,
                distance_m=depth_m,
                dt=dt,
                frame_idx=frame_idx
            )
            
            # Get filter state for confidence
            filter_state = self.cdkf.get_state(track_id)
            filter_confidence = filter_state.confidence if filter_state else dims.confidence
            
            # Optionally adjust width/length proportionally based on area change
            if dims.area_m2 > 0:
                area_ratio = smoothed_area / dims.area_m2
                smoothed_width = dims.width_m * np.sqrt(area_ratio)
                smoothed_length = dims.length_m * np.sqrt(area_ratio)
            else:
                smoothed_width = dims.width_m
                smoothed_length = dims.length_m
            
            # Use smoothed values for display and logging
            severity = classify_pothole_severity(smoothed_area, dims.volume_m3)
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
            # lines = [
            #     f"[{severity}]  conf:{det['confidence']:.2f}",
            #     f"W:{dims.width_m:.3f}m  L:{dims.length_m:.3f}m",
            #     f"A:{dims.area_m2:.4f}m²  V:{dims.volume_m3:.5f}m³",
            #     f"D:{dims.depth_m:.2f}m  fuse:{dims.confidence:.2f}",
            # ]

            lines = [
                f"[{severity}]  conf:{det['confidence']:.2f}",
                f"W:{smoothed_width:.3f}m  L:{smoothed_length:.3f}m",
                f"A:{smoothed_area:.4f}m²  V:{dims.volume_m3:.5f}m³",
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

            # det_meta.append({"bbox": bbox, "depth_m": dims.depth_m})
            det_meta.append({"bbox": bbox, "depth_m": dims.depth_m, "area_smoothed": smoothed_area})

            

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

                "area_raw_m2": dims.area_m2,           # Raw MBTP area
                "area_smoothed_m2": smoothed_area,      # CDKF-smoothed area
                "cdkf_confidence": filter_confidence,   # Filter confidence
                "cdkf_uncertainty": filter_state.uncertainty if filter_state else 0,
                "pothole_distance_m": depth_m,          # Distance for reference
            })

        # Clean up old tracks
        active_track_ids = {det.get('track_id') for det in detections if 'track_id' in det}
        self.cdkf.cleanup(active_track_ids, max_age_frames=30, current_frame=frame_idx)

         # Predict areas for tracks without current detection
        for track_id, track in self.tracker.tracks.items():
            if track_id not in active_track_ids:
                predicted_area = self.cdkf.predict(track_id, dt=1.0, frame_idx=frame_idx)
                # Optional: Draw predicted pothole with dashed box
                if 'bbox' in track:
                    x1, y1, x2, y2 = track['bbox']
                    # Draw dashed rectangle for predicted detection
                    for i in range(x1, x2, 10):
                        cv2.line(rgb_ann, (i, y1), (min(i+5, x2), y1), (100, 100, 100), 1)
                        cv2.line(rgb_ann, (i, y2), (min(i+5, x2), y2), (100, 100, 100), 1)
                    for i in range(y1, y2, 10):
                        cv2.line(rgb_ann, (x1, i), (x1, min(i+5, y2)), (100, 100, 100), 1)
                        cv2.line(rgb_ann, (x2, i), (x2, min(i+5, y2)), (100, 100, 100), 1)
                    # Add predicted area text
                    cv2.putText(rgb_ann, f"Pred: {predicted_area:.3f}m²", 
                            (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # Clean up old CDKF tracks
        self.cdkf.cleanup(active_track_ids, max_age_frames=30, current_frame=frame_idx)

        # ── HUD: bottom sensor bar ────────────────────────────────────────────
        # cv2.rectangle(rgb_ann, (0, H-26), (W, H), (18, 18, 18), -1)


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