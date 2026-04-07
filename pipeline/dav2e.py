
import os
from typing import Tuple, Optional

import cv2
import numpy as np
import torch

# from pipeline.dav2e_utils import _YOLO_AVAILABLE, _DEPTHANYTHING_AVAILABLE, _DEPTHANYTHINGV2_AVAILABLE

class DepthAnythingV2Estimator:
    """
    DepthAnything V2 - state-of-the-art monocular depth estimation.
    Better than MiDaS: sharper edges, better generalization, metric depth output.
    
    Paper: Depth Anything V2 (2024)
    """
    
    def __init__(self, model_type: str = "vits",  # vits, vitb, vitl, vitg
                 device: Optional[torch.device] = None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.available = False
        self._cache: Tuple[int, Optional[np.ndarray], Optional[np.ndarray]] = (-1, None, None)
        
        print(f"  [DEPTH] Loading DepthAnything V2 ({model_type}) on {self.device} …")
        
        try:
            # DepthAnything V2 requires installation:
            # pip install depth-anything-v2
            from depth_anything_v2.dpt import DepthAnythingV2
            
            # Model configurations
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
            }
            
            self._model = DepthAnythingV2(**model_configs[model_type])
            
            # Load pretrained weights
            checkpoint_urls = {
                'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
                'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
                'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
            }
            
            import urllib.request
            cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints/")
            os.makedirs(cache_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(cache_dir, f"depth_anything_v2_{model_type}.pth")
            if not os.path.exists(checkpoint_path):
                print(f"  [DEPTH] Downloading weights to {checkpoint_path}...")
                urllib.request.urlretrieve(checkpoint_urls[model_type], checkpoint_path)
            
            self._model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self._model = self._model.to(self.device).eval()
            
            self.available = True
            print("  [DEPTH] ✓ DepthAnything V2 loaded")
            
        except ImportError:
            print("  [DEPTH] ✗ depth_anything_v2 not installed. Install with: pip install depth-anything-v2")
            print("  [DEPTH] Falling back to MiDaS...")
            self._fallback_to_midas(model_type)
        except Exception as exc:
            print(f"  [DEPTH] ✗ Failed: {exc}")
            self._fallback_to_midas(model_type)
    
    def _fallback_to_midas(self, model_type):
        """Fallback to MiDaS if DepthAnything unavailable."""
        try:
            self._model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
            self._model.to(self.device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self._transform = midas_transforms.small_transform
            self.available = True
            self._is_midas_fallback = True
            print("  [DEPTH] ✓ Using MiDaS fallback")
        except:
            self.available = False
    
    def run(self, bgr_image: np.ndarray, frame_id: int = -1) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run depth estimation.
        Returns (depth_map_f32, viz_u8) where depth_map is in METERS (metric output!).
        """
        if frame_id >= 0 and frame_id == self._cache[0]:
            return self._cache[1], self._cache[2]
        
        if not self.available:
            return None, None
        
        try:
            h, w = bgr_image.shape[:2]
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            
            if hasattr(self, '_is_midas_fallback'):
                # MiDaS fallback
                inp = self._transform(rgb).to(self.device)
                with torch.no_grad():
                    pred = self._model(inp)
                    pred = torch.nn.functional.interpolate(
                        pred.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
                    ).squeeze()
                depth = pred.cpu().numpy().astype(np.float32)
                # Convert inverse depth to metric (rough estimate)
                depth = 1.0 / (depth + 0.01)
            else:
                # DepthAnything V2 - outputs metric depth directly
                # The correct method signature is infer_image(rgb) or infer_image(rgb, h, w)
                with torch.no_grad():
                    # Try different method signatures
                    if hasattr(self._model, 'infer_image'):
                        # Some versions take (image, height, width)
                        try:
                            depth = self._model.infer_image(rgb, h, w)
                        except TypeError:
                            # Some versions take only (image)
                            depth = self._model.infer_image(rgb)
                    else:
                        # Fallback to forward pass
                        import torch.nn.functional as F
                        input_tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
                        # Resize to model input size (518x518 typical)
                        input_tensor = F.interpolate(input_tensor, size=(518, 518), mode='bilinear')
                        depth = self._model(input_tensor)
                        depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode='bilinear').squeeze().cpu().numpy()
                
                # Ensure depth is numpy array
                if isinstance(depth, torch.Tensor):
                    depth = depth.cpu().numpy()
                
                depth = depth.astype(np.float32)
                
                # DepthAnything V2 outputs inverse depth? Check range
                # If values are large (>1000), it's likely inverse depth
                if np.median(depth) > 100:
                    depth = 1.0 / (depth + 0.01)
            
            # Normalize for visualization
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            self._cache = (frame_id, depth, depth_normalized)
            return depth, depth_normalized
            
        except Exception as exc:
            print(f"  [DEPTH] inference error: {exc}")
            import traceback
            traceback.print_exc()
            return None, None
    
    @staticmethod
    def bbox_metric_depth(depth_map: np.ndarray,
                          bbox: Tuple[int, int, int, int],
                          percentile: float = 20.0) -> float:
        """Extract metric depth from bounding box region."""
        x1, y1, x2, y2 = bbox
        hh, ww = depth_map.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(ww - 1, x2), min(hh - 1, y2)
        if x1 >= x2 or y1 >= y2:
            return float(np.nanmedian(depth_map))
        roi = depth_map[y1:y2, x1:x2]
        finite = roi[np.isfinite(roi) & (roi > 0)]
        return float(np.percentile(finite, percentile)) if finite.size else float(np.nanmedian(depth_map))
