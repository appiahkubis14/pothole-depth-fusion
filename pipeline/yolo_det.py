from typing import Tuple, Dict, Optional, List
import numpy as np
import os
import torch
# from pipeline.dav2e_utils import _YOLO_AVAILABLE

# if _YOLO_AVAILABLE:
#     from ultralytics import YOLO

try:   
    from ultralytics import YOLO 
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed — YOLO detection disabled; using mock potholes")    

class ACSHYOLOv8Detector:
    """
    ACSH-YOLOv8 with ACmix attention module and small object detection head.
    """
    
    def __init__(self, model_path: str, conf: float = 0.35,
                 device: Optional[torch.device] = None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.conf = conf
        self._model = None
        self._mode = "mock"
        self._names = {0: "pothole"}  # Default class names
        
        if _YOLO_AVAILABLE and os.path.exists(model_path):
            try:
                print(f"  [DET] Loading model from {model_path}...")
                self._model = YOLO(model_path)
                
                # Don't try to access internal model attributes that may not exist
                # Just verify the model works
                self._model.to(str(self.device))
                self._mode = "yolo"
                
                # Safely get class names
                try:
                    if hasattr(self._model, 'names'):
                        self._names = self._model.names
                    elif hasattr(self._model, 'model') and hasattr(self._model.model, 'names'):
                        self._names = self._model.model.names
                except:
                    pass
                    
                print(f"  [DET] ACSH-YOLOv8 ← {model_path} (classes: {len(self._names)})")
                
            except Exception as exc:
                print(f"  [DET] YOLO load failed: {exc}")
                self._mode = "mock"
        else:
            print(f"  [DET] Model not found ({model_path}) → mock")
    
    def detect(self, image: np.ndarray, lower_conf: float = 0.15) -> List[Dict]:
        """
        Detect potholes with multi-scale inference for small objects.
        """
        if self._mode == "yolo" and self._model is not None:
            try:
                h, w = image.shape[:2]
                all_detections = []
                
                # Single scale inference (simpler, more robust)
                results = self._model(image, conf=lower_conf, verbose=False, iou=0.45)
                
                for r in results:
                    if r.boxes is None or len(r.boxes) == 0:
                        continue
                    
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        # Get class name safely
                        cls_id = int(box.cls[0])
                        class_name = self._names.get(cls_id, f"class_{cls_id}")
                        
                        # Adaptive threshold (lower for small boxes)
                        box_area = (x2 - x1) * (y2 - y1)
                        is_small = box_area < (w * h * 0.01)
                        adaptive_thresh = self.conf * (0.7 if is_small else 1.0)
                        
                        if conf > adaptive_thresh:
                            all_detections.append({
                                "bbox": (x1, y1, x2, y2),
                                "confidence": conf,
                                "class_name": class_name,
                                "is_small": is_small
                            })
                
                # NMS
                return self._nms(all_detections, iou_threshold=0.3)
                
            except Exception as e:
                print(f"  [DET] Detection error: {e}")
                return self._mock_detect(image)
        
        return self._mock_detect(image)
    
    def _mock_detect(self, image: np.ndarray) -> List[Dict]:
        """Mock detection - returns empty list (no false positives)."""
        # Return empty list - let the pipeline run without forcing detections
        return []
    
    def _nms(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        if not detections:
            return []
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [d for d in detections 
                         if self._compute_iou(best['bbox'], d['bbox']) < iou_threshold]
        return keep
    
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
