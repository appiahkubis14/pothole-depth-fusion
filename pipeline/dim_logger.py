
from typing import List, Dict, Optional
import csv
import os
import numpy as np
import cv2
# from pipeline.dav2e_utils import _PANDAS_AVAILABLE

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    print("[WARN] pandas not installed — CSV sensor loading disabled")

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

