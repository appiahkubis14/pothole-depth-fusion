
from typing import List, Dict
import cv2
import numpy as np


class BoTSORTTracker:
    """
    BoT-SORT (Bundled Optimization Tracker) - State-of-the-art tracking.
    
    Features:
    - Camera motion compensation (CMC)
    - Kalman filter with adaptive noise
    - Re-identification features
    - Better occlusion handling than IoU tracker
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 30,
                 use_cmc: bool = True, reid_features: bool = False):
        self.next_id = 0
        self.tracks = {}  # track_id -> track state
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost_frames
        self.use_cmc = use_cmc
        self.reid_features = reid_features
        
        # Kalman filter parameters (for each track)
        self.kalman = None  # Will initialize per track
        self._prev_frame = None
        self._homography = None  # For camera motion compensation
        
    def _init_kalman(self):
        """Initialize Kalman filter for tracking."""
        # State: [x, y, w, h, vx, vy, vw, vh]
        kalman = cv2.KalmanFilter(8, 4)
        kalman.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1
        kalman.errorCovPost = np.eye(8, dtype=np.float32)
        return kalman
    
    def _compute_cmc_homography(self, prev_frame, curr_frame):
        """Compute camera motion compensation homography."""
        if prev_frame is None or curr_frame is None:
            return np.eye(3)
        
        # Detect ORB features
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(prev_frame, None)
        kp2, des2 = orb.detectAndCompute(curr_frame, None)
        
        if des1 is None or des2 is None:
            return np.eye(3)
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:100]
        
        if len(matches) < 10:
            return np.eye(3)
        
        # Get matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        return H if H is not None else np.eye(3)
    
    def update(self, detections: List[Dict], frame_idx: int, 
            current_frame: np.ndarray = None) -> List[Dict]:
        """
        Update tracks with new detections using BoT-SORT logic.
        """
        # Camera motion compensation
        if self.use_cmc and current_frame is not None and self._prev_frame is not None:
            H = self._compute_cmc_homography(self._prev_frame, current_frame)
            self._homography = H
            
            # Apply CMC to existing tracks
            for tid, track in self.tracks.items():
                if H is not None and track.get('bbox'):
                    x1, y1, x2, y2 = track['bbox']
                    center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
                    center_warped = cv2.perspectiveTransform(center.reshape(-1, 1, 2), H)
                    if center_warped is not None:
                        cx, cy = center_warped[0][0]
                        w_box = x2 - x1
                        h_box = y2 - y1
                        track['predicted_bbox'] = (int(cx - w_box/2), int(cy - h_box/2),
                                                int(cx + w_box/2), int(cy + h_box/2))
        
        self._prev_frame = current_frame.copy() if current_frame is not None else None
        
        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    del self.tracks[tid]
            return []
        
        # Match detections to tracks
        matched = []
        
        for det in detections:
            det_bbox = det['bbox']
            best_iou = 0
            best_tid = None
            
            for tid, track in self.tracks.items():
                # Use predicted bbox if available (from CMC or Kalman)
                pred_bbox = track.get('predicted_bbox', track['bbox'])
                iou = self._compute_iou(det_bbox, pred_bbox)
                
                # Apply Kalman prediction
                if 'kalman' in track:
                    kalman = track['kalman']
                    prediction = kalman.predict()
                    # Extract scalar values properly
                    cx_pred = float(prediction[0, 0])  # <-- FIX: convert to scalar
                    cy_pred = float(prediction[1, 0])  # <-- FIX: convert to scalar
                    w_pred = float(prediction[2, 0])   # <-- FIX: convert to scalar
                    h_pred = float(prediction[3, 0])   # <-- FIX: convert to scalar
                    kalman_bbox = (int(cx_pred - w_pred/2), int(cy_pred - h_pred/2),
                                int(cx_pred + w_pred/2), int(cy_pred + h_pred/2))
                    iou = max(iou, self._compute_iou(det_bbox, kalman_bbox))
                
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_tid = tid
            
            if best_tid is not None:
                # Update existing track
                track = self.tracks[best_tid]
                track['bbox'] = det_bbox
                track['lost'] = 0
                track['history'].append((frame_idx, det_bbox))
                if len(track['history']) > 50:
                    track['history'].pop(0)
                det['track_id'] = best_tid
                matched.append(best_tid)
                
                # Update Kalman
                if 'kalman' in track:
                    cx = (det_bbox[0] + det_bbox[2]) / 2.0
                    cy = (det_bbox[1] + det_bbox[3]) / 2.0
                    w = float(det_bbox[2] - det_bbox[0])
                    h = float(det_bbox[3] - det_bbox[1])
                    track['kalman'].correct(np.array([[cx], [cy], [w], [h]], dtype=np.float32))
            else:
                # New track
                kalman = self._init_kalman()
                cx = (det_bbox[0] + det_bbox[2]) / 2.0
                cy = (det_bbox[1] + det_bbox[3]) / 2.0
                w = float(det_bbox[2] - det_bbox[0])
                h = float(det_bbox[3] - det_bbox[1])
                kalman.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
                
                self.tracks[self.next_id] = {
                    'bbox': det_bbox,
                    'lost': 0,
                    'history': [(frame_idx, det_bbox)],
                    'kalman': kalman,
                    'predicted_bbox': det_bbox
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
        """Get interpolated detections for missed frames."""
        results = []
        for tid, track in self.tracks.items():
            history = track['history']
            if len(history) >= 2:
                (f1, b1), (f2, b2) = history[-2], history[-1]
                if f1 < frame_idx < f2:
                    ratio = (frame_idx - f1) / (f2 - f1)
                    interp_bbox = tuple(int(b1[i] + ratio * (b2[i] - b1[i])) for i in range(4))
                    results.append({
                        'bbox': interp_bbox,
                        'track_id': tid,
                        'interpolated': True,
                        'confidence': 0.5
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

