
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
from pipeline.data_class_meth import DimensionEstimate
from pipeline.homography import HomographyMapper
from pipeline.mbtp_est import MBTPAreaEstimator
from pipeline.dav2e import DepthAnythingV2Estimator

class EnhancedDimensionFusionEngine:
    """
    Enhanced dimension fusion that prioritizes MBTP area estimation.
    Falls back to bounding box methods when MBTP fails.
    """
    
    def __init__(self, K: np.ndarray):
        self.K = K.copy().astype(np.float64)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.mbtp_estimator = MBTPAreaEstimator()
        
    
    def update_K(self, K: np.ndarray):
        self.K = K.copy().astype(np.float64)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
    
    def fuse(self,
             bbox: Tuple[int, int, int, int],
             homography_mapper: HomographyMapper,
             pitch: float, roll: float, yaw: float,
            rgb_image: np.ndarray,  # Add this parameter
             camera_height_m: float,
             abs_depth: Optional[np.ndarray]) -> DimensionEstimate:
        """
        Fuse dimensions with MBTP as primary method.
        """
        results: Dict[str, Tuple[float, float, float]] = {}
        depth_val: float = camera_height_m / max(np.cos(pitch), 0.1)
        
        # Method A: MBTP (Paper's method - PRIMARY)
        if abs_depth is not None:
            try:
                # mbtp = self.mbtp_estimator.estimate(abs_depth, bbox, self.K)
                mbtp = self.mbtp_estimator.estimate(
                    rgb_image,  # Pass RGB image
                    abs_depth, 
                    bbox, 
                    self.K
                )
                if mbtp.area_m2 > 0:
                    # Convert MBTP area to width/length approximation
                    # Assuming length/width ratio from bounding box
                    bbox_w = (bbox[2] - bbox[0]) * depth_val / self.fx
                    bbox_h = (bbox[3] - bbox[1]) * depth_val / self.fy
                    aspect_ratio = bbox_w / max(bbox_h, 0.001)
                    
                    if aspect_ratio > 1:
                        width_m = np.sqrt(mbtp.area_m2 * aspect_ratio)
                        length_m = mbtp.area_m2 / max(width_m, 0.001)
                    else:
                        length_m = np.sqrt(mbtp.area_m2 / max(aspect_ratio, 0.001))
                        width_m = mbtp.area_m2 / max(length_m, 0.001)
                    
                    results["mbtp"] = (width_m, length_m, mbtp.confidence)
                    depth_val = mbtp.depth_m
            except Exception as exc:
                print(f"  [FUSE] MBTP error: {exc}")
        
        # Method B: Homography (fallback)
        try:
            hd = homography_mapper.bbox_world_dims(bbox, pitch, roll, yaw)
            hc = homography_mapper.confidence(pitch, roll)
            if hd["width_m"] > 0 and hd["length_m"] > 0:
                results["homography"] = (hd["width_m"], hd["length_m"], hc * 0.7)  # Lower weight
        except Exception as exc:
            print(f"  [FUSE] Homography error: {exc}")
        
        # Method C: Depth map pinhole (fallback)
        if abs_depth is not None and "mbtp" not in results:
            d_m = DepthAnythingV2Estimator.bbox_metric_depth(abs_depth, bbox)
            depth_val = d_m
            dc = self._depth_confidence(d_m, camera_height_m)
            dw, dl = self._depth_dims(bbox, d_m)
            if dw > 0 and dl > 0:
                results["depth"] = (dw, dl, dc * 0.6)
        
        if not results:
            return DimensionEstimate(0.0, 0.0, 0.0, 0.0, depth_val, 0.0)
        
        total_conf = sum(v[2] for v in results.values()) or 1.0
        width_m = sum(v[0] * v[2] for v in results.values()) / total_conf
        length_m = sum(v[1] * v[2] for v in results.values()) / total_conf
        conf = min(1.0, total_conf / max(1, len(results)))
        
        # Use MBTP volume if available
        if abs_depth is not None:
            try:
                mbtp = self.mbtp_estimator.estimate(abs_depth, bbox, self.K)
                volume_m3 = mbtp.volume_m3
            except:
                volume_m3 = self._compute_volume(bbox, abs_depth)
        else:
            volume_m3 = 0.0
        
        weights = {k: round(v[2]/total_conf, 3) for k, v in results.items()}
        
        return DimensionEstimate(
            width_m=round(width_m, 4),
            length_m=round(length_m, 4),
            area_m2=round(width_m * length_m, 6),
            volume_m3=round(volume_m3, 6),
            depth_m=round(depth_val, 3),
            confidence=round(conf, 3),
            method_weights=weights
        )
    
    def _depth_dims(self, bbox, depth_m):
        x1, y1, x2, y2 = bbox
        return (max(1, x2-x1) * depth_m / self.fx,
                max(1, y2-y1) * depth_m / self.fy)
    
    def _depth_confidence(self, depth_m, cam_h):
        ratio = depth_m / max(cam_h, 0.1)
        if ratio < 2: return 0.95
        elif ratio < 5: return 0.85
        elif ratio < 15: return 0.70
        elif ratio < 30: return 0.50
        else: return 0.30
    
    def _compute_volume(self, bbox, abs_depth):
        """Fallback volume computation."""
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
        border[y1:y2, x1:x2] = False
        
        ground_d = abs_depth[border]
        ground_ref = float(np.nanmedian(ground_d)) if ground_d.size > 0 else np.median(abs_depth)
        
        roi = abs_depth[y1:y2, x1:x2].astype(np.float32)
        extra = np.clip(roi - ground_ref, 0.0, None)
        pix_area = (ground_ref / self.fx) * (ground_ref / self.fy)
        return round(float(np.sum(extra) * pix_area), 6)


