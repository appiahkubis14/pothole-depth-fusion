from datetime import datetime
import os
from typing import List, Dict
import numpy as np
import pandas as pd

# from pipeline.dav2e_utils import _PANDAS_AVAILABLE

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    print("[WARN] pandas not installed — CSV sensor loading disabled")

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


