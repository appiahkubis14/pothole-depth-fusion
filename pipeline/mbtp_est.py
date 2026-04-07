from typing import Tuple, List, Optional
import numpy as np
import cv2

from pipeline.data_class_meth import MBTPEstimate
from pipeline.rgbsegment import RobustPotholeSegmentor



class MBTPAreaEstimator:
    """
    Minimum Bounding Triangulated Pixel method for pothole area estimation.
    
    Paper: Uses triangulation of pothole contour pixels to compute accurate
    area regardless of shape irregularity. Better than bounding box for
    real-world potholes which are rarely rectangular.
    
    Steps:
    1. Segment pothole region from depth map (depth discontinuity)
    2. Extract contour of pothole
    3. Triangulate contour points (Delaunay triangulation)
    4. Sum triangle areas for total area
    5. Compute convex hull for shape irregularity metric
    """
    
    def __init__(self, 
                 depth_threshold_m: float = 0.02,  # 2cm depth threshold
                 min_area_pixels: int = 100,
                 contour_simplification_eps: float = 0.005,
                 use_rgb_segmentation: bool = True):  # <-- ADD THIS
        self.depth_threshold_m = depth_threshold_m
        self.min_area_pixels = min_area_pixels
        self.contour_eps = contour_simplification_eps
        self.use_rgb_segmentation = use_rgb_segmentation
        
        # Initialize robust segmentor
        self.segmentor = RobustPotholeSegmentor(
            depth_threshold_m=depth_threshold_m,
            min_area_pixels=min_area_pixels,
            use_rgb_fallback=use_rgb_segmentation
        )


    def segment_pothole_region(self, 
                                depth_map: np.ndarray, 
                                bbox: Tuple[int, int, int, int],
                                ground_ref: float) -> np.ndarray:
        """
        Segment pothole region from depth map using depth discontinuity.
        
        Returns binary mask where True = pothole pixels.
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape
        
        # Extract ROI
        roi = depth_map[y1:y2, x1:x2].copy()
        
        # Pothole = pixels deeper than ground reference by threshold
        mask = (ground_ref - roi) > self.depth_threshold_m
        
        # Clean mask: morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area_pixels:
                mask[labels == i] = 0
        
        return mask
    
    def compute_contour(self, mask: np.ndarray) -> List[np.ndarray]:
        """Extract and simplify contour from mask."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Take largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Simplify contour (Douglas-Peucker)
        epsilon = self.contour_eps * cv2.arcLength(largest, True)
        simplified = cv2.approxPolyDP(largest, epsilon, True)
        
        return [simplified]
    
    def triangulate_contour(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Delaunay triangulation on contour points.
        
        Returns:
            - triangles: List of triangle indices
            - triangulation_map: Visualization of triangulation
        """
        h, w = image_shape
        points = contour.reshape(-1, 2).astype(np.float32)
        
        if len(points) < 3:
            return np.array([]), None
        
        # Add centroid to ensure full coverage
        centroid = np.mean(points, axis=0)
        points = np.vstack([points, centroid])
        
        # Delaunay triangulation
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)
        for p in points:
            subdiv.insert((float(p[0]), float(p[1])))
        
        triangles = subdiv.getTriangleList()
        triangles = triangles.reshape(-1, 3, 2).astype(np.int32)
        
        # Filter triangles outside contour
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        valid_triangles = []
        for tri in triangles:
            # Check if triangle centroid is inside contour
            cx = int(np.mean(tri[:, 0]))
            cy = int(np.mean(tri[:, 1]))
            if 0 <= cx < w and 0 <= cy < h and mask[cy, cx] > 0:
                valid_triangles.append(tri)
        
        # Create visualization
        viz = np.zeros((h, w, 3), dtype=np.uint8)
        for tri in valid_triangles:
            cv2.polylines(viz, [tri], True, (0, 255, 0), 1)
            cv2.fillPoly(viz, [tri], (0, 100, 0))
        
        return np.array(valid_triangles), viz
    
    def compute_triangle_area(self, triangle: np.ndarray, 
                              depth_map: np.ndarray,
                              fx: float, fy: float,
                              ground_ref: float) -> float:
        """
        Compute real-world area of a triangle using pinhole projection.
        
        For each triangle vertex, compute:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth at that pixel
        
        Then compute 3D triangle area using cross product.
        """
        # Get depths at triangle vertices
        depths = []
        for pt in triangle:
            u, v = pt
            if 0 <= u < depth_map.shape[1] and 0 <= v < depth_map.shape[0]:
                z = depth_map[v, u]
                depths.append(z)
            else:
                depths.append(ground_ref)
        
        # Convert to 3D points
        points_3d = []
        for (u, v), z in zip(triangle, depths):
            x = (u - fx) * z / fx
            y = (v - fy) * z / fy
            points_3d.append([x, y, z])
        
        # Compute triangle area using cross product
        p1, p2, p3 = np.array(points_3d)
        v1 = p2 - p1
        v2 = p3 - p1
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        
        return area
    
    def estimate(self,
                 rgb_image: np.ndarray,
                 depth_map: np.ndarray,
                 bbox: Tuple[int, int, int, int],
                 K: np.ndarray,
                 ground_ref: float = None) -> MBTPEstimate:
        """
        Main MBTP estimation method with RGB fallback.
        """
        h, w = depth_map.shape
        fx, fy = K[0, 0], K[1, 1]
        x1, y1, x2, y2 = bbox
        
        # Compute ground reference if not provided
        if ground_ref is None:
            ring = 10
            rx1 = max(0, x1 - ring)
            ry1 = max(0, y1 - ring)
            rx2 = min(w - 1, x2 + ring)
            ry2 = min(h - 1, y2 + ring)
            
            border_mask = np.zeros(depth_map.shape, dtype=bool)
            border_mask[ry1:ry2+1, rx1:rx2+1] = True
            border_mask[y1:y2+1, x1:x2+1] = False
            
            ground_depths = depth_map[border_mask]
            if ground_depths.size == 0:
                ground_ref = float(np.median(depth_map))
            else:
                ground_ref = float(np.median(ground_depths))
        
        # Use robust segmentor
        mask, method = self.segmentor.segment_pothole(
            rgb_image, depth_map, bbox, ground_ref
        )
        
        # Log segmentation method
        if method != "depth":
            print(f"  [MBTP] Using {method} segmentation")
        
        if mask is None:
            print(f"  [MBTP] Segmentation failed ({method}), using bounding box fallback")
            return self._fallback_estimate(depth_map, bbox, K, ground_ref)
        
        # Extract contour
        contours = self.segmentor.get_contour(mask)
        if not contours:
            return self._fallback_estimate(depth_map, bbox, K, ground_ref)
        
        contour = contours[0]
        
        # Triangulate and compute area (rest of MBTP logic)
        triangles, tri_viz = self.triangulate_contour(contour, (h, w))
        
        total_area = 0.0
        for tri in triangles:
            area = self.compute_triangle_area(tri, depth_map, fx, fy, ground_ref)
            total_area += area
        
        # Compute convex hull and volume
        hull = cv2.convexHull(contour)
        hull_area_pixels = cv2.contourArea(hull)
        hull_area_m2 = self._pixel_area_to_m2(hull_area_pixels, depth_map, bbox, fx, fy)
        irregularity = 1.0 - (total_area / max(hull_area_m2, 0.001))
        
        # Volume integration
        roi_depth = depth_map[y1:y2, x1:x2]
        roi_mask = mask[y1:y2, x1:x2]
        depth_deficit = np.maximum(ground_ref - roi_depth, 0)
        pixel_area_m2 = (ground_ref / fx) * (ground_ref / fy)
        volume = float(np.sum(depth_deficit[roi_mask > 0]) * pixel_area_m2)
        
        # Confidence based on segmentation method
        method_confidence = {
            "depth": 0.9,
            "rgb_water": 0.7,
            "hybrid": 0.8,
            "bounding_box_fallback": 0.5
        }.get(method, 0.6)
        
        confidence = method_confidence * min(1.0, total_area / (hull_area_m2 + 0.001))
        
        return MBTPEstimate(
            area_m2=round(total_area, 6),
            perimeter_m=round(cv2.arcLength(contour, True) * (ground_ref / fx), 3),
            depth_m=round(np.mean(roi_depth[roi_mask > 0]) if np.any(roi_mask > 0) else ground_ref, 3),
            volume_m3=round(volume, 6),
            convex_hull_area_m2=round(hull_area_m2, 6),
            irregularity_score=round(irregularity, 3),
            confidence=round(confidence, 3),
            contour_points=contour.reshape(-1, 2).tolist(),
            triangulation_map=tri_viz
        )
    
    # def estimate(self,
    #              depth_map: np.ndarray,
    #              bbox: Tuple[int, int, int, int],
    #              K: np.ndarray,
    #              ground_ref: float = None) -> MBTPEstimate:
    #     """
    #     Main MBTP estimation method.
        
    #     Args:
    #         depth_map: Absolute metric depth map
    #         bbox: Initial bounding box from detector
    #         K: Camera intrinsics matrix
    #         ground_ref: Ground reference depth (if None, computed from border)
        
    #     Returns:
    #         MBTPEstimate with area, volume, confidence, etc.
    #     """
    #     h, w = depth_map.shape
    #     fx, fy = K[0, 0], K[1, 1]
        
    #     # Compute ground reference from bounding box border if not provided
    #     if ground_ref is None:
    #         x1, y1, x2, y2 = bbox
    #         ring = 10
    #         rx1, ry1 = max(0, x1-ring), max(0, y1-ring)
    #         rx2, ry2 = min(w-1, x2+ring), min(h-1, y2+ring)
            
    #         border_mask = np.zeros(depth_map.shape, dtype=bool)
    #         border_mask[ry1:ry2, rx1:rx2 + 1] = True
    #         border_mask[y1:y2, x1:x2 + 1] = False
            
    #         ground_depths = depth_map[border_mask]
    #         ground_ref = float(np.median(ground_depths)) if ground_depths.size > 0 else np.median(depth_map)
        
    #     # Step 1: Segment pothole region
    #     mask = self.segment_pothole_region(depth_map, bbox, ground_ref)
        
    #     if np.sum(mask) < self.min_area_pixels:
    #         # Fallback to bounding box method
    #         return self._fallback_estimate(depth_map, bbox, K, ground_ref)
        
    #     # Step 2: Extract contour
    #     contours = self.compute_contour(mask)
    #     if not contours:
    #         return self._fallback_estimate(depth_map, bbox, K, ground_ref)
        
    #     contour = contours[0]
        
    #     # Step 3: Triangulate contour
    #     triangles, tri_viz = self.triangulate_contour(contour, (h, w))
        
    #     # Step 4: Compute area from triangles
    #     total_area = 0.0
    #     for tri in triangles:
    #         area = self.compute_triangle_area(tri, depth_map, fx, fy, ground_ref)
    #         total_area += area
        
    #     # Step 5: Compute convex hull area for irregularity metric
    #     hull = cv2.convexHull(contour)
    #     hull_area_pixels = cv2.contourArea(hull)
    #     hull_area_m2 = self._pixel_area_to_m2(hull_area_pixels, depth_map, bbox, fx, fy)
    #     irregularity = 1.0 - (total_area / max(hull_area_m2, 0.001))
        
    #     # Step 6: Compute volume (integration of depth deficit)
    #     roi_depth = depth_map[y1:y2, x1:x2]
    #     roi_mask = mask[y1:y2, x1:x2]
    #     depth_deficit = np.maximum(ground_ref - roi_depth, 0)
    #     pixel_area_m2 = (ground_ref / fx) * (ground_ref / fy)
    #     volume = float(np.sum(depth_deficit[roi_mask > 0]) * pixel_area_m2)
        
    #     # Step 7: Compute mean depth
    #     pothole_depths = roi_depth[roi_mask > 0]
    #     mean_depth = float(np.mean(pothole_depths)) if len(pothole_depths) > 0 else ground_ref
        
    #     # Step 8: Confidence score
    #     confidence = min(1.0, total_area / (hull_area_m2 + 0.001))
    #     confidence *= min(1.0, np.sum(mask) / (self.min_area_pixels * 2))
        
    #     return MBTPEstimate(
    #         area_m2=round(total_area, 6),
    #         perimeter_m=round(cv2.arcLength(contour, True) * (ground_ref / fx), 3),
    #         depth_m=round(mean_depth, 3),
    #         volume_m3=round(volume, 6),
    #         convex_hull_area_m2=round(hull_area_m2, 6),
    #         irregularity_score=round(irregularity, 3),
    #         confidence=round(confidence, 3),
    #         contour_points=contour.reshape(-1, 2).tolist(),
    #         triangulation_map=tri_viz
    #     )
    
    def _pixel_area_to_m2(self, pixel_area: float, depth_map: np.ndarray, 
                          bbox: Tuple[int, int, int, int], fx: float, fy: float) -> float:
        """Convert pixel area to square meters using average depth."""
        x1, y1, x2, y2 = bbox
        roi = depth_map[y1:y2, x1:x2]
        avg_depth = np.median(roi[roi > 0]) if np.any(roi > 0) else 5.0
        return pixel_area * (avg_depth / fx) * (avg_depth / fy)
    
    def _fallback_estimate(self, depth_map: np.ndarray, 
                           bbox: Tuple[int, int, int, int],
                           K: np.ndarray, ground_ref: float) -> MBTPEstimate:
        """Fallback to bounding box method if segmentation fails."""
        print("  [MBTP] Segmentation failed, using bounding box fallback")
        x1, y1, x2, y2 = bbox
        fx, fy = K[0, 0], K[1, 1]
        
        # Use bounding box corners
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        depths = [depth_map[y, x] for x, y in corners]
        avg_depth = np.mean(depths)
        
        width_m = (x2 - x1) * avg_depth / fx
        length_m = (y2 - y1) * avg_depth / fy
        area = width_m * length_m
        
        # Volume approximation
        roi = depth_map[y1:y2, x1:x2]
        depth_deficit = np.maximum(ground_ref - roi, 0)
        pixel_area = (avg_depth / fx) * (avg_depth / fy)
        volume = float(np.sum(depth_deficit) * pixel_area)
        
        return MBTPEstimate(
            area_m2=round(area, 6),
            perimeter_m=round(2 * (width_m + length_m), 3),
            depth_m=round(avg_depth, 3),
            volume_m3=round(volume, 6),
            convex_hull_area_m2=round(area, 6),
            irregularity_score=0.0,
            confidence=0.5,
            contour_points=[],
            triangulation_map=None
        )

