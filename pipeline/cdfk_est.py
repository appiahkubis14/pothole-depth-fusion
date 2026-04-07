
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
from pipeline.data_class_meth import CDKFState

class ConfidenceDistanceKalmanFilter:
    """
    CDKF: Kalman Filter with adaptive measurement noise based on:
    1. Detection confidence (higher confidence = lower noise)
    2. Distance to pothole (farther = higher uncertainty)
    
    Paper: Confidence and Distance-based Kalman Filter for robust pothole area estimation
    """
    
    def __init__(self, 
                 process_noise: float = 0.01,
                 base_measurement_noise: float = 0.05,
                 min_confidence: float = 0.2,
                 max_distance_m: float = 30.0):
        """
        Args:
            process_noise: Q - uncertainty in state prediction
            base_measurement_noise: R_base - base measurement uncertainty
            min_confidence: Minimum confidence to consider measurement valid
            max_distance_m: Distance at which uncertainty saturates
        """
        self.process_noise = process_noise
        self.base_measurement_noise = base_measurement_noise
        self.min_confidence = min_confidence
        self.max_distance = max_distance_m
        
        # Per-track filters
        self._filters: Dict[int, CDKFState] = {}
        
    def _compute_measurement_noise(self, confidence: float, distance_m: float) -> float:
        """
        Adaptive measurement noise based on confidence and distance.
        
        Lower confidence → higher noise (trust measurement less)
        Larger distance → higher noise (far away measurements less accurate)
        """
        # Confidence factor: low confidence = high noise
        conf_factor = max(0.1, 2.0 - confidence * 2)  # 0.9 conf → 0.2, 0.3 conf → 1.4
        
        # Distance factor: farther = higher noise
        dist_factor = 1.0 + min(3.0, distance_m / self.max_distance)
        
        return self.base_measurement_noise * conf_factor * dist_factor
    
    def _compute_kalman_gain(self, P: float, R: float) -> float:
        """Compute optimal Kalman gain."""
        return P / (P + R)
    
    def update(self, 
               track_id: int,
               measured_area_m2: float,
               detection_confidence: float,
               distance_m: float,
               dt: float = 1.0,
               frame_idx: int = 0) -> float:
        """
        Update Kalman filter with new measurement.
        
        Args:
            track_id: Unique pothole ID
            measured_area_m2: Raw area measurement from MBTP
            detection_confidence: YOLO detection confidence (0-1)
            distance_m: Distance from camera to pothole (meters)
            dt: Time delta (frames) since last update
            frame_idx: Current frame index
            
        Returns:
            Smoothed area estimate
        """
        # Get or create filter state
        if track_id not in self._filters:
            # Initialize new track
            self._filters[track_id] = CDKFState(
                area_m2=measured_area_m2,
                velocity_m2_per_frame=0.0,
                confidence=detection_confidence,
                uncertainty=0.1,  # Initial uncertainty
                last_update_frame=frame_idx,
                distance_m=distance_m
            )
            return measured_area_m2
        
        state = self._filters[track_id]
        
        # Skip if detection confidence is too low
        if detection_confidence < self.min_confidence:
            # Use prediction only (don't update with bad measurement)
            return self.predict(track_id, dt, frame_idx)
        
        # --- Prediction Step ---
        # Predict area based on previous velocity
        area_pred = state.area_m2 + state.velocity_m2_per_frame * dt
        P_pred = state.uncertainty + self.process_noise * dt
        
        # --- Measurement Noise (Adaptive) ---
        R = self._compute_measurement_noise(detection_confidence, distance_m)
        
        # --- Kalman Gain ---
        K = self._compute_kalman_gain(P_pred, R)
        
        # --- Update Step ---
        # Innovation (residual)
        innovation = measured_area_m2 - area_pred
        
        # Detect outliers (e.g., from depth inversion)
        is_outlier = abs(innovation) > (state.area_m2 * 0.5)  # 50% change is suspicious
        
        if is_outlier and detection_confidence < 0.6:
            # Outlier with low confidence - reject measurement
            smoothed_area = area_pred
            confidence_reduction = 0.7
        else:
            # Normal update
            smoothed_area = area_pred + K * innovation
            confidence_reduction = 1.0
        
        # Update state
        state.area_m2 = smoothed_area
        state.velocity_m2_per_frame = (smoothed_area - state.area_m2) / max(dt, 0.1)
        state.uncertainty = (1 - K) * P_pred
        state.confidence = state.confidence * confidence_reduction * detection_confidence
        state.confidence = max(0.1, min(1.0, state.confidence))
        state.last_update_frame = frame_idx
        state.distance_m = distance_m
        
        return smoothed_area
    
    def predict(self, track_id: int, dt: float = 1.0, frame_idx: int = 0) -> float:
        """
        Predict area when no measurement is available (e.g., detection missed).
        """
        if track_id not in self._filters:
            return 0.0
        
        state = self._filters[track_id]
        
        # Prediction only
        predicted_area = state.area_m2 + state.velocity_m2_per_frame * dt
        
        # Increase uncertainty when predicting without measurement
        state.uncertainty += self.process_noise * dt
        state.confidence *= 0.95  # Gradually decrease confidence
        state.last_update_frame = frame_idx
        
        return predicted_area
    
    def get_state(self, track_id: int) -> Optional[CDKFState]:
        """Get current filter state for a track."""
        return self._filters.get(track_id)
    
    def cleanup(self, active_tracks: set, max_age_frames: int = 30, current_frame: int = 0):
        """Remove old tracks that haven't been updated."""
        to_remove = []
        for tid, state in self._filters.items():
            if tid not in active_tracks:
                if current_frame - state.last_update_frame > max_age_frames:
                    to_remove.append(tid)
        
        for tid in to_remove:
            del self._filters[tid]
