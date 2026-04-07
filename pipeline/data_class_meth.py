

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import numpy as np



@dataclass
class MBTPEstimate:
    """MBTP area estimation results."""
    area_m2: float
    perimeter_m: float
    depth_m: float
    volume_m3: float
    convex_hull_area_m2: float
    irregularity_score: float
    confidence: float
    contour_points: List[Tuple[int, int]]
    triangulation_map: Optional[np.ndarray] = None




@dataclass
class CDKFState:
    """State of CDKF for a tracked pothole."""
    area_m2: float           # Smoothed area
    velocity_m2_per_frame: float  # Rate of area change
    confidence: float        # Filter confidence (0-1)
    uncertainty: float       # State uncertainty (P)
    last_update_frame: int   # Last frame updated
    distance_m: float        # Last known distance to pothole



@dataclass
class DimensionEstimate:
    """Final dimension estimate after fusion."""
    width_m: float
    length_m: float
    area_m2: float
    volume_m3: float
    depth_m: float
    confidence: float
    method_weights: Dict[str, float] = field(default_factory=dict)
