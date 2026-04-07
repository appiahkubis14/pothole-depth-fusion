"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        UNIFIED TRANSPORTATION INTELLIGENCE PIPELINE                         ║
║        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                          ║
║  Combines:                                                                   ║
║    • Camera Calibration (chessboard intrinsics + distortion)                ║
║    • MiDaS Monocular Depth Estimation (metric-anchored)                     ║
║    • Homography + IMU Fusion (pitch/roll-corrected ground-plane)            ║
║    • GPS Track (per-frame geo-location + GeoJSON/GPX export)                ║
║    • Pothole Detection & Dimension/Volume Estimation (YOLO)                 ║
║    • Multi-Class Multi-Object Tracking (MCMOT via ByteTrack)                ║
║    • Real-World Object Dimension Calculator (depth + pinhole)               ║
║    • Risk Assessment (collision probability per tracked object)             ║
║    • Dual-View Annotated Video Export                                       ║
║    • Unified CSV / JSON / GeoJSON logging                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dimension fusion strategy (potholes)
──────────────────────────────────────
For each detected pothole bounding box three independent measurements:
  A) Homography (H):  width_H, length_H  (reliable pitch/roll < 20°)
  B) Depth anchor:    width_D, length_D  (MiDaS depth scaled via camera height)
  C) Known-height:    width_K            (focal-length/pixel ratio)
Confidence-weighted mean → final dimension.  Volume from depth-map ROI.

MCMOT strategy
──────────────
ByteTrack keeps object identities across frames.  Per-track state:
  • Class-consistent validation (prevents car→person flips)
  • Smoothed depth (EMA) → real-world width/height via pinhole projection
  • Velocity EMA → future-position prediction
  • Risk score = f(depth, class importance, bounding-box size, track quality)
  • Trajectory history drawn on annotated frames

No values are hard-coded: every camera-intrinsic and scene parameter is either
loaded from calibration files or computed on-the-fly.
"""

from __future__ import annotations

# ── standard library ─────────────────────────────────────────────────────────
import csv
import json
import os
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from math import radians, cos, sin, sqrt, atan2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── optional heavy imports ────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed — YOLO detection/tracking disabled")

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    print("[WARN] pandas not installed — CSV sensor loading disabled")

try:
    from scipy.ndimage import median_filter as _scipy_median_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    def _scipy_median_filter(arr, size):  # type: ignore[misc]
        return arr  # no-op fallback


# ── Resolve ByteTrack config to absolute path (avoids cwd-resolution issues) ─
def _resolve_bytetrack_yaml() -> str:
    """Return the absolute path to bytetrack.yaml inside the ultralytics package."""
    try:
        import ultralytics as _ul
        _p = os.path.join(os.path.dirname(_ul.__file__),
                          "cfg", "trackers", "bytetrack.yaml")
        if os.path.isfile(_p):
            return _p
    except Exception:
        pass
    return "bytetrack.yaml"   # fall back to name (ultralytics resolves internally)

_BYTETRACK_YAML = _resolve_bytetrack_yaml()



# ══════════════════════════════════════════════════════════════════════════════
# RUN MANAGER  –  per-run project directory & manifest
# ══════════════════════════════════════════════════════════════════════════════

class RunManager:
    """
    Creates a timestamped project directory for each pipeline run.

    Layout
    ──────
    <base_output_dir>/
      run_YYYYMMDD_HHMMSS/            ← self.run_dir
        annotated/                    ← per-frame JPEGs
        data/
          unified_log.csv             ← all events (potholes + tracked objects)
          unified_log.json
          potholes.csv                ← pothole events only
          tracked_objects.csv         ← MCMOT events only
          potholes.geojson            ← GIS-ready pothole map
          track.gpx                   ← GPS route
        reports/
          summary.json                ← machine-readable statistics
          report.html                 ← human-readable HTML report
        manifest.json                 ← run settings, file inventory
    """

    def __init__(self, base_dir: str = "output"):
        ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id  = f"run_{ts}"
        self.run_dir = os.path.join(base_dir, self.run_id)

        # Sub-directories
        self.annotated_dir = os.path.join(self.run_dir, "annotated")
        self.data_dir      = os.path.join(self.run_dir, "data")
        self.reports_dir   = os.path.join(self.run_dir, "reports")

        for d in (self.annotated_dir, self.data_dir, self.reports_dir):
            os.makedirs(d, exist_ok=True)

        self._settings: Dict = {}
        self._start_time     = time.time()
        print(f"  [RUN] Project → {self.run_dir}")

    def set_settings(self, **kwargs):
        self.settings = {**self._settings, **kwargs,
                         "run_id": self.run_id,
                         "started_at": datetime.now().isoformat()}

    def save_manifest(self, summary: Dict, video_path: Optional[str] = None):
        """Write manifest.json with run settings + file inventory."""
        inventory = []
        for root, dirs, files in os.walk(self.run_dir):
            dirs.sort()
            for fname in sorted(files):
                full = os.path.join(root, fname)
                rel  = os.path.relpath(full, self.run_dir)
                sz   = os.path.getsize(full)
                inventory.append({"path": rel, "bytes": sz})

        manifest = {
            "run_id":         self.run_id,
            "started_at":     getattr(self, "settings", {}).get("started_at", ""),
            "finished_at":    datetime.now().isoformat(),
            "elapsed_s":      round(time.time() - self._start_time, 1),
            "settings":       getattr(self, "settings", {}),
            "input_video":    video_path or "",
            "summary":        summary,
            "output_files":   inventory,
        }
        path = os.path.join(self.run_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"  [RUN] Manifest → {path}")

        # Print inventory
        print(f"\n  ── Run {self.run_id} output files {'─'*30}")
        for item in inventory:
            sz = item['bytes']
            if   sz > 1_000_000: label = f"{sz/1_000_000:6.1f} MB"
            elif sz > 1_000:     label = f"{sz/1_000:6.1f} KB"
            else:                label = f"{sz:6d}  B"
            print(f"    {item['path']:<50} {label}")
        print(f"  {'─'*68}")

    @staticmethod
    def save_split_csv(records: List[Dict], path: str, event_type: str,
                       fields: List[str]):
        """Write a filtered CSV for one event type."""
        subset = [r for r in records if r.get("road_event") == event_type]
        if not subset:
            # Write header-only file so the file always exists
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(subset)

    @staticmethod
    def save_html_report(run_dir: str, run_id: str, summary: Dict,
                         settings: Dict, records: List[Dict]):
        """Generate a self-contained HTML report for transport decision-making."""
        pots = [r for r in records if r.get("road_event") == "pothole"]
        objs = [r for r in records if r.get("road_event") == "tracked_object"]

        sev = summary.get("potholes", {}).get("severity_counts", {})
        risk = summary.get("tracked_objects", {}).get("risk_counts", {})
        cls_counts = summary.get("tracked_objects", {}).get("class_counts", {})

        def sev_bar(label, count, total, color):
            pct = int(count/max(total,1)*100)
            return (f'<div class="bar-row"><span class="bar-label">{label}</span>'
                    f'<div class="bar-bg"><div class="bar-fill" '
                    f'style="width:{pct}%;background:{color}">{count}</div></div></div>')

        pothole_rows = ""
        for r in pots[:200]:
            sev_c = {"Minimal":"#22c55e","Low":"#84cc16","Medium":"#f97316","High":"#ef4444"}.get(r.get("severity",""), "#888")
            pothole_rows += (f'<tr>'
                f'<td>{r.get("frame","")}</td>'
                f'<td><span class="badge" style="background:{sev_c}">{r.get("severity","")}</span></td>'
                f'<td>{r.get("area_m2","")}</td>'
                f'<td>{r.get("volume_m3","")}</td>'
                f'<td>{r.get("width_m","")} × {r.get("length_m","")}</td>'
                f'<td>{r.get("depth_m","")}</td>'
                f'<td>{r.get("latitude","")}, {r.get("longitude","")}</td>'
                f'</tr>')

        obj_rows = ""
        for r in objs[:200]:
            risk_c = {"SAFE":"#22c55e","CAUTION":"#eab308","WARNING":"#f97316","CRITICAL":"#ef4444"}.get(r.get("risk_level",""), "#888")
            obj_rows += (f'<tr>'
                f'<td>{r.get("frame","")}</td>'
                f'<td>#{r.get("track_id","")}</td>'
                f'<td>{r.get("class_name","")}</td>'
                f'<td>{r.get("obj_depth_m","")}</td>'
                f'<td>{r.get("obj_width_m","")} × {r.get("obj_height_m","")}</td>'
                f'<td><span class="badge" style="background:{risk_c}">{r.get("risk_level","")}</span></td>'
                f'<td>{r.get("velocity_x","")}, {r.get("velocity_y","")}</td>'
                f'</tr>')

        cls_pills = "".join(f'<span class="pill">{k}: {v}</span>' for k,v in cls_counts.items())
        n_pot = summary.get("potholes", {}).get("total", 0)
        n_obj = summary.get("tracked_objects", {}).get("total_detections", 0)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Transportation Intelligence Report · {run_id}</title>
<style>
  :root{{--bg:#0f172a;--card:#1e293b;--border:#334155;--text:#e2e8f0;--muted:#94a3b8;
    --accent:#38bdf8;--green:#22c55e;--orange:#f97316;--red:#ef4444;--yellow:#eab308}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);padding:2rem}}
  h1{{font-size:1.6rem;font-weight:700;color:var(--accent);margin-bottom:.25rem}}
  h2{{font-size:1.1rem;font-weight:600;color:var(--accent);margin:1.5rem 0 .75rem}}
  .subtitle{{color:var(--muted);font-size:.85rem;margin-bottom:2rem}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-bottom:2rem}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:.75rem;padding:1.25rem}}
  .card .num{{font-size:2rem;font-weight:800;color:var(--accent)}}
  .card .lbl{{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-top:.25rem}}
  .section{{background:var(--card);border:1px solid var(--border);border-radius:.75rem;padding:1.5rem;margin-bottom:1.5rem}}
  .bar-row{{display:flex;align-items:center;gap:.75rem;margin:.4rem 0}}
  .bar-label{{width:70px;font-size:.8rem;color:var(--muted);flex-shrink:0}}
  .bar-bg{{flex:1;background:#0f172a;border-radius:999px;height:22px;overflow:hidden}}
  .bar-fill{{height:100%;border-radius:999px;display:flex;align-items:center;padding:0 .5rem;
             font-size:.75rem;font-weight:700;color:#fff;min-width:2rem;transition:width .4s}}
  table{{width:100%;border-collapse:collapse;font-size:.78rem}}
  th{{background:#0f172a;color:var(--muted);text-transform:uppercase;font-size:.68rem;
      letter-spacing:.05em;padding:.6rem .75rem;text-align:left;border-bottom:1px solid var(--border)}}
  td{{padding:.55rem .75rem;border-bottom:1px solid var(--border);color:var(--text)}}
  tr:hover td{{background:#0f172a}}
  .badge{{padding:.2rem .6rem;border-radius:.375rem;font-size:.72rem;font-weight:700;color:#fff}}
  .pill{{display:inline-block;background:#1e3a5f;border:1px solid #38bdf880;border-radius:999px;
         padding:.2rem .7rem;font-size:.78rem;color:var(--accent);margin:.2rem}}
  .settings{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:.5rem}}
  .setting-row{{display:flex;gap:.5rem;font-size:.8rem}}
  .sk{{color:var(--muted);min-width:160px}}
  .sv{{color:var(--text);font-family:monospace}}
  @media(max-width:600px){{body{{padding:1rem}}h1{{font-size:1.2rem}}}}
</style>
</head>
<body>
<h1>🚦 Transportation Intelligence Report</h1>
<div class="subtitle">Run ID: {run_id} &nbsp;·&nbsp; {summary.get("processed_at","")}</div>

<div class="grid">
  <div class="card"><div class="num">{n_pot}</div><div class="lbl">Potholes Detected</div></div>
  <div class="card"><div class="num">{n_obj}</div><div class="lbl">Object Detections</div></div>
  <div class="card"><div class="num">{summary.get("potholes",{}).get("total_area_m2","0")}</div><div class="lbl">Total Damaged Area (m²)</div></div>
  <div class="card"><div class="num">{summary.get("potholes",{}).get("total_volume_m3","0")}</div><div class="lbl">Total Pothole Volume (m³)</div></div>
  <div class="card"><div class="num">{summary.get("total_records",0)}</div><div class="lbl">Total Log Records</div></div>
</div>

<div class="section">
  <h2>🕳️ Pothole Severity Distribution</h2>
  {sev_bar("Minimal", sev.get("Minimal",0), n_pot, "#22c55e")}
  {sev_bar("Low",     sev.get("Low",0),     n_pot, "#84cc16")}
  {sev_bar("Medium",  sev.get("Medium",0),  n_pot, "#f97316")}
  {sev_bar("High",    sev.get("High",0),    n_pot, "#ef4444")}
</div>

<div class="section">
  <h2>🚗 Object Risk Distribution</h2>
  {sev_bar("SAFE",     risk.get("SAFE",0),     n_obj, "#22c55e")}
  {sev_bar("CAUTION",  risk.get("CAUTION",0),  n_obj, "#eab308")}
  {sev_bar("WARNING",  risk.get("WARNING",0),  n_obj, "#f97316")}
  {sev_bar("CRITICAL", risk.get("CRITICAL",0), n_obj, "#ef4444")}
  <div style="margin-top:1rem">{cls_pills}</div>
</div>

<div class="section">
  <h2>🕳️ Pothole Detections (top 200)</h2>
  <div style="overflow-x:auto">
  <table>
    <tr><th>Frame</th><th>Severity</th><th>Area m²</th><th>Volume m³</th>
        <th>W×L (m)</th><th>Depth (m)</th><th>GPS</th></tr>
    {pothole_rows if pothole_rows else '<tr><td colspan="7" style="color:var(--muted);text-align:center">No potholes detected</td></tr>'}
  </table></div>
</div>

<div class="section">
  <h2>🚗 Tracked Objects (top 200)</h2>
  <div style="overflow-x:auto">
  <table>
    <tr><th>Frame</th><th>Track</th><th>Class</th><th>Depth (m)</th>
        <th>W×H (m)</th><th>Risk</th><th>Velocity px/f</th></tr>
    {obj_rows if obj_rows else '<tr><td colspan="7" style="color:var(--muted);text-align:center">No tracked objects (general YOLO model may not have been loaded)</td></tr>'}
  </table></div>
</div>

<div class="section">
  <h2>⚙️ Run Settings</h2>
  <div class="settings">
    {''.join(f'<div class="setting-row"><span class="sk">{k}</span><span class="sv">{v}</span></div>' for k,v in settings.items())}
  </div>
</div>

</body></html>"""

        rpt_path = os.path.join(run_dir, "reports", "report.html")
        with open(rpt_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  [RUN] HTML report → {rpt_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – CAMERA CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

class CameraCalibrator:
    """
    Chessboard camera calibration.
    Returns intrinsic matrix K and distortion coefficients d.
    """

    def __init__(self,
                 chessboard_size: Tuple[int, int] = (8, 5),
                 square_size_m: float = 0.030):
        self.chessboard_size = chessboard_size
        self.square_size     = square_size_m
        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

        tpl = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        tpl[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self._tpl = tpl * square_size_m

        self._obj_pts:  List[np.ndarray] = []
        self._img_pts:  List[np.ndarray] = []
        self.image_shape: Optional[Tuple[int, int]] = None   # (W, H)

        self.K:                   Optional[np.ndarray] = None
        self.dist:                Optional[np.ndarray] = None
        self.reprojection_error:  Optional[float]      = None

    # ── corner detection ──────────────────────────────────────────────────────

    def find_corners(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq  = clahe.apply(gray)
        size     = self.chessboard_size

        def _refine(g, c):
            return cv2.cornerSubPix(g, c, (11, 11), (-1, -1), self._criteria)

        flags_list = [
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        ]
        for g in (gray_eq, gray):
            for flags in flags_list:
                found, c = cv2.findChessboardCorners(g, size, flags)
                if found:
                    return True, _refine(g, c)

        for g in (gray_eq, gray):
            found, c = cv2.findChessboardCornersSB(g, size)
            if found:
                return True, c

        h, w = gray_eq.shape
        for scale in (0.75, 0.5, 1.25):
            rs = cv2.resize(gray_eq, (int(w * scale), int(h * scale)))
            found, c = cv2.findChessboardCorners(
                rs, size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
            )
            if found:
                return True, _refine(gray_eq, c / scale)

        nc, nr = size
        for dc, dr in [(-1,0),(0,-1),(-1,-1),(1,0),(0,1),(1,1),(-1,1),(1,-1)]:
            alt = (nc + dc, nr + dr)
            if alt[0] < 2 or alt[1] < 2:
                continue
            for g in (gray_eq, gray):
                found, c = cv2.findChessboardCorners(
                    g, alt,
                    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
                )
                if found:
                    print(f"  [CAL] Auto-corrected chessboard size {size} → {alt}")
                    self.chessboard_size = alt
                    self._tpl = np.zeros((alt[0] * alt[1], 3), np.float32)
                    self._tpl[:, :2] = np.mgrid[0:alt[0], 0:alt[1]].T.reshape(-1, 2)
                    self._tpl *= self.square_size
                    return True, _refine(g, c)

        return False, None

    def add_image(self, image: np.ndarray) -> bool:
        if self.image_shape is None:
            h, w = image.shape[:2]
            self.image_shape = (w, h)
        found, corners = self.find_corners(image)
        if found:
            self._obj_pts.append(self._tpl.copy())
            self._img_pts.append(corners)
            print(f"  [CAL] ✓  Corners found  ({len(self._obj_pts)} images total)")
        else:
            print("  [CAL] ✗  No corners found")
        return found

    def calibrate(self) -> bool:
        n = len(self._img_pts)
        if n < 5:
            print(f"  [CAL] Need ≥ 5 images, have {n}")
            return False
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self._obj_pts, self._img_pts, self.image_shape, None, None
        )
        if not ret:
            return False
        self.K    = K
        self.dist = dist
        self._rvecs = rvecs
        self._tvecs = tvecs
        self.reprojection_error = self._mean_reproj_error(rvecs, tvecs)
        print(f"  [CAL] fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  "
              f"cx={K[0,2]:.2f}  cy={K[1,2]:.2f}  "
              f"err={self.reprojection_error:.4f}px")
        return True

    def _mean_reproj_error(self, rvecs, tvecs) -> float:
        total = 0.0
        for i in range(len(self._obj_pts)):
            proj, _ = cv2.projectPoints(self._obj_pts[i], rvecs[i], tvecs[i], self.K, self.dist)
            total += cv2.norm(self._img_pts[i], proj, cv2.NORM_L2) / len(proj)
        return total / len(self._obj_pts)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "outputs/cal/calibration.npz"):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, K=self.K, dist=self.dist,
                 image_shape=np.array(self.image_shape),
                 reprojection_error=np.array([self.reprojection_error]))
        print(f"  [CAL] Saved → {path}")

    def load(self, path: str = "outputs/cal/calibration.npz"):
        data = np.load(path, allow_pickle=True)
        self.K    = data["K"]
        self.dist = data["dist"]
        self.image_shape = tuple(data["image_shape"].tolist())
        if "reprojection_error" in data:
            self.reprojection_error = float(data["reprojection_error"][0])
        print(f"  [CAL] Loaded from {path}  "
              f"fx={self.K[0,0]:.2f}  fy={self.K[1,1]:.2f}  "
              f"err={self.reprojection_error:.4f}px")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – IMAGE UNDISTORTION
# ══════════════════════════════════════════════════════════════════════════════

class ImageUndistorter:
    """
    Pre-builds remap arrays for fast per-frame undistortion.
    Uses alpha=0 (no black borders).  Scales K to match actual resolution.
    """

    def __init__(self, K: np.ndarray, dist: np.ndarray,
                 cal_wh: Tuple[int, int], alpha: float = 0.0):
        self._K_cal   = K.copy()
        self._dist    = dist.copy()
        self._cal_wh  = cal_wh
        self._alpha   = alpha
        self._map1    = None
        self._map2    = None
        self._new_K   = None
        self._last_wh: Optional[Tuple[int, int]] = None

    def _scale_K(self, target_wh: Tuple[int, int]) -> np.ndarray:
        sx = target_wh[0] / self._cal_wh[0]
        sy = target_wh[1] / self._cal_wh[1]
        K  = self._K_cal.copy().astype(np.float64)
        K[0, 0] *= sx;  K[0, 2] *= sx
        K[1, 1] *= sy;  K[1, 2] *= sy
        return K

    def prepare(self, wh: Tuple[int, int]):
        if self._last_wh == wh:
            return
        self._last_wh = wh
        K_scaled = self._scale_K(wh)
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K_scaled, self._dist, wh, self._alpha, wh
        )
        self._new_K = new_K
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            K_scaled, self._dist, None, new_K, wh, cv2.CV_32FC1
        )

    def undistort(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        self.prepare((w, h))
        return cv2.remap(image, self._map1, self._map2, cv2.INTER_LINEAR)

    @property
    def current_K(self) -> Optional[np.ndarray]:
        return self._new_K


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – MIDAS DEPTH ESTIMATOR  (shared by both sub-systems)
# ══════════════════════════════════════════════════════════════════════════════

class MiDaSDepthEstimator:
    """
    Monocular depth via MiDaS.

    Converts MiDaS inverse-depth to metric using a per-frame ground-plane
    anchor: the bottom-centre strip of the image is assumed to be flat ground
    at depth = camera_height / cos(pitch).

    Also supports raw inverse-depth queries for MCMOT object-depth estimation.
    """

    def __init__(self, model_type: str = "MiDaS_small",
                 device: Optional[torch.device] = None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.available    = False
        self._last_inv    = None   # raw MiDaS output (kept for MCMOT)
        self._last_viz    = None   # uint8 colormap array
        self._last_metric = None   # metric depth array (for pothole pipeline)
        self._frame_ctr   = 0

        print(f"  [MIDAS] Loading {model_type} on {self.device} …")
        try:
            self._model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
            self._model.to(self.device).eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self._transform = (transforms.dpt_transform
                               if "DPT" in model_type
                               else transforms.small_transform)
            self.available = True
            print("  [MIDAS] ✓ Loaded")
        except Exception as exc:
            print(f"  [MIDAS] ✗ Failed: {exc}")

    # ── raw MiDaS output ──────────────────────────────────────────────────────

    def _raw_inverse_depth(self, bgr_image: np.ndarray) -> Optional[np.ndarray]:
        if not self.available:
            return None
        try:
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            inp = self._transform(rgb).to(self.device)
            with torch.no_grad():
                pred = self._model(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=rgb.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            return pred.cpu().numpy().astype(np.float32)
        except Exception as exc:
            print(f"  [MIDAS] inference error: {exc}")
            return None

    # ── metric-anchored estimate (pothole sub-system) ─────────────────────────

    def estimate(self,
                 bgr_image: np.ndarray,
                 camera_height_m: float,
                 pitch_rad: float,
                 K: np.ndarray,
                 force_update: bool = True
                 ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns (depth_metric [m], depth_viz [uint8 colourmap]).
        If force_update=False, returns cached result every other frame.
        """
        self._frame_ctr += 1
        if (not force_update
                and self._frame_ctr % 3 != 0
                and self._last_metric is not None):
            return self._last_metric, self._last_viz

        inv_depth = self._raw_inverse_depth(bgr_image)
        if inv_depth is None:
            return self._last_metric, self._last_viz

        self._last_inv = inv_depth

        h, w = inv_depth.shape
        strip_h = max(1, h // 8)
        cx_lo, cx_hi = w // 4, 3 * w // 4
        strip_inv = inv_depth[h - strip_h:, cx_lo:cx_hi]

        median_inv = (float(np.median(strip_inv[strip_inv > 0]))
                      if strip_inv.size else 0.0)
        if median_inv <= 0:
            median_inv = float(np.median(inv_depth[inv_depth > 0]))

        expected_z = camera_height_m / max(np.cos(pitch_rad), 0.1)
        scale      = expected_z * median_inv

        with np.errstate(divide="ignore", invalid="ignore"):
            depth_m = np.where(inv_depth > 0, scale / inv_depth, np.inf).astype(np.float32)
        depth_m = np.clip(depth_m, 0.1, 200.0)

        d_norm    = cv2.normalize(inv_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_viz = cv2.applyColorMap(d_norm, cv2.COLORMAP_MAGMA)

        self._last_metric = depth_m
        self._last_viz    = depth_viz
        return depth_m, depth_viz

    # ── MCMOT: object depth from raw inverse-depth map ────────────────────────

    def estimate_depth_map_raw(self,
                                bgr_image: np.ndarray,
                                force_update: bool = True
                                ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns (raw_inverse_depth_map, uint8_viz) without metric anchoring.
        Used by MCMOT for relative distance ordering.
        Caches result; use force_update=False to reuse last frame's map.
        """
        if (not force_update
                and self._last_inv is not None):
            return self._last_inv, self._last_viz

        inv = self._raw_inverse_depth(bgr_image)
        if inv is None:
            return self._last_inv, self._last_viz

        self._last_inv = inv
        d_norm         = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self._last_viz = d_norm   # single channel; colourmap applied at display time
        return inv, d_norm

    # ── shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def bbox_median_depth(depth_map: np.ndarray,
                          bbox: Tuple[int, int, int, int],
                          percentile: float = 20.0) -> float:
        """
        Robust depth for a bounding-box region.
        percentile=20 gives the *near* portion, avoiding background leakage.
        Works on both metric and raw inverse-depth maps.
        """
        x1, y1, x2, y2 = bbox
        hh, ww = depth_map.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(ww - 1, x2), min(hh - 1, y2)
        if x1 >= x2 or y1 >= y2:
            return float(np.nanmedian(depth_map))
        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            return float(np.nanmedian(depth_map))
        finite = roi[np.isfinite(roi)]
        return float(np.percentile(finite, percentile)) if finite.size else float(np.nanmedian(depth_map))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – GROUND-PLANE HOMOGRAPHY
# ══════════════════════════════════════════════════════════════════════════════

class HomographyMapper:
    """
    Maps image pixels → ground-plane world coordinates (X right, Y forward)
    using a pinhole camera model with pitch and roll correction.
    """

    def __init__(self, K: np.ndarray, camera_height_m: float):
        self.K = K.copy().astype(np.float64)
        self.h = camera_height_m

    def update_height(self, h: float):
        self.h = h

    def compute_H(self, pitch: float, roll: float, yaw: float = 0.0) -> np.ndarray:
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll),  np.sin(roll)
        cy, sy = np.cos(yaw),   np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        R  = Rz @ Ry @ Rx
        K_inv   = np.linalg.inv(self.K)
        r1, r2  = R[:, 0], R[:, 1]
        t_cam   = np.array([0.0, 0.0, self.h])
        A = np.column_stack([r1, r2, t_cam])
        return A @ K_inv

    def pixel_to_world(self, px: float, py: float,
                       pitch: float, roll: float, yaw: float = 0.0
                       ) -> Tuple[float, float]:
        H     = self.compute_H(pitch, roll, yaw)
        p_hom = np.array([px, py, 1.0])
        w_hom = H @ p_hom
        if abs(w_hom[2]) < 1e-9:
            return (0.0, 0.0)
        return (w_hom[0] / w_hom[2], w_hom[1] / w_hom[2])

    def bbox_world_dims(self,
                        bbox: Tuple[int, int, int, int],
                        pitch: float, roll: float, yaw: float = 0.0) -> Dict:
        x1, y1, x2, y2 = bbox
        corners_px = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        corners_w  = [self.pixel_to_world(px, py, pitch, roll, yaw)
                      for px, py in corners_px]
        Xs = [c[0] for c in corners_w]
        Ys = [c[1] for c in corners_w]
        return {
            "width_m":       round(abs(max(Xs) - min(Xs)), 4),
            "length_m":      round(abs(max(Ys) - min(Ys)), 4),
            "area_m2":       round(abs(max(Xs)-min(Xs)) * abs(max(Ys)-min(Ys)), 6),
            "world_corners": corners_w,
        }

    def confidence(self, pitch: float, roll: float) -> float:
        tilt     = np.sqrt(pitch**2 + roll**2)
        max_tilt = np.radians(30)
        return float(np.clip(1.0 - tilt / max_tilt, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – IMU PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class IMUProcessor:
    """
    Fuses accelerometer + gyroscope to track camera orientation over time.
    Gravity-vector tilt for pitch/roll (no integration drift).
    """

    def __init__(self, initial_height_m: float = 1.2):
        self.initial_height = initial_height_m
        self.current_height = initial_height_m
        self._t:      List[float] = []
        self._pitch:  List[float] = []
        self._roll:   List[float] = []
        self._yaw:    List[float] = []
        self._height: List[float] = []
        self._start_time: Optional[float] = None
        self.orientation = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}

    def load_from_csv(self, csv_path: str):
        if not _PANDAS_AVAILABLE:
            print("  [IMU] pandas not available")
            return
        if not os.path.exists(csv_path):
            print(f"  [IMU] File not found: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        def col(*names):
            return next((n for n in names if n in df.columns), None)

        ts_col = col("Timestamp", "timestamp", "Time", "time")
        ax_col = col("AccelX", "accel_x", "ax")
        ay_col = col("AccelY", "accel_y", "ay")
        az_col = col("AccelZ", "accel_z", "az")
        gx_col = col("GyroX",  "gyro_x",  "gx")
        gy_col = col("GyroY",  "gyro_y",  "gy")
        gz_col = col("GyroZ",  "gyro_z",  "gz")

        FORMATS = [
            "%a %b %d %H:%M:%S GMT %Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ]

        def _parse(raw) -> float:
            for fmt in FORMATS:
                try:
                    return datetime.strptime(str(raw).strip(), fmt).timestamp()
                except ValueError:
                    pass
            try:
                return float(raw)
            except Exception:
                return 0.0

        n     = len(df)
        epoch = df[ts_col].apply(_parse).to_numpy(float) if ts_col else np.arange(n, dtype=float)
        t_sec = epoch - epoch[0]
        ax    = df[ax_col].to_numpy(float) if ax_col else np.zeros(n)
        ay    = df[ay_col].to_numpy(float) if ay_col else np.zeros(n)
        az    = df[az_col].to_numpy(float) if az_col else np.full(n, 9.81)
        gx    = df[gx_col].to_numpy(float) if gx_col else np.zeros(n)
        gy    = df[gy_col].to_numpy(float) if gy_col else np.zeros(n)
        gz    = df[gz_col].to_numpy(float) if gz_col else np.zeros(n)
        self._process_arrays(t_sec, ax, ay, az, gx, gy, gz)
        print(f"  [IMU] {n} rows  duration={t_sec[-1]:.1f}s")

    def _process_arrays(self,
                        t, ax, ay, az, gx, gy, gz):
        n = len(t)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        static   = gyro_mag < 0.08
        if static.sum() < 5:
            static = np.ones(n, bool)

        g_med  = np.array([np.median(ax[static]),
                           np.median(ay[static]),
                           np.median(az[static])])
        g_norm = float(np.linalg.norm(g_med))
        if g_norm < 7.0:
            g_norm = 9.81

        fracs      = np.abs(g_med) / g_norm
        grav_axis  = int(np.argmax(fracs))
        grav_sign  = 1.0 if g_med[grav_axis] > 0 else -1.0

        ax_s = _scipy_median_filter(ax, 5)
        ay_s = _scipy_median_filter(ay, 5)
        az_s = _scipy_median_filter(az, 5)

        if grav_axis == 0:
            pitches = np.arctan2(az_s, ax_s * grav_sign)
            rolls   = np.arctan2(ay_s, ax_s * grav_sign)
        elif grav_axis == 2:
            pitches = np.arctan2(ax_s, az_s * grav_sign)
            rolls   = np.arctan2(ay_s, az_s * grav_sign)
        else:
            pitches = np.arctan2(ax_s, ay_s * grav_sign)
            rolls   = np.zeros(n)

        MAX_TILT = np.radians(20)
        pitches  = np.clip(pitches, -MAX_TILT, MAX_TILT)
        rolls    = np.clip(rolls,   -MAX_TILT, MAX_TILT)
        yaws     = np.zeros(n)

        vert     = [ax, ay, az][grav_axis]
        vert_lin = vert - grav_sign * g_norm
        vert_std = float(np.std(vert_lin))
        if   vert_std < 1.5: h_est = 1.7
        elif vert_std < 2.5: h_est = 1.3
        elif vert_std < 3.5: h_est = 1.0
        else:                h_est = 0.8

        configured = self.initial_height
        deviation  = abs(h_est - configured) / max(configured, 0.1)
        h_final    = h_est if deviation > 0.25 else configured
        heights    = np.full(n, h_final)

        self._t      = t.tolist()
        self._pitch  = pitches.tolist()
        self._roll   = rolls.tolist()
        self._yaw    = yaws.tolist()
        self._height = heights.tolist()
        self.current_height = h_final
        self.orientation = {
            "pitch": float(np.median(pitches)),
            "roll":  float(np.median(rolls)),
            "yaw":   0.0,
        }
        self._start_time = 0.0
        print(f"  [IMU] Height={h_final:.3f}m  "
              f"pitch [{np.degrees(pitches.min()):.1f}°, {np.degrees(pitches.max()):.1f}°]  "
              f"roll  [{np.degrees(rolls.min()) :.1f}°, {np.degrees(rolls.max()) :.1f}°]")

    def _interp(self, arr: List[float], t_query: float) -> float:
        if not self._t:
            return arr[0] if arr else 0.0
        return float(np.interp(t_query, self._t, arr))

    def height_at(self, t: float) -> float:
        return self._interp(self._height, t) if self._height else self.current_height

    def orientation_at(self, t: float) -> Dict[str, float]:
        if not self._t:
            return dict(self.orientation)
        return {
            "pitch": self._interp(self._pitch, t),
            "roll":  self._interp(self._roll,  t),
            "yaw":   self._interp(self._yaw,   t),
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – GPS MAPPER
# ══════════════════════════════════════════════════════════════════════════════

class GPSMapper:
    """Stores GPS track; exports GeoJSON (potholes) and GPX (route)."""

    def __init__(self):
        self.locations: List[Dict] = []

    def add(self, lat: float, lon: float, t: float,
            alt: float = 0.0, meta: Optional[Dict] = None):
        self.locations.append({"lat": lat, "lon": lon,
                                "t": t, "alt": alt, "meta": meta or {}})

    def nearest(self, t: float) -> Optional[Dict]:
        if not self.locations:
            return None
        return min(self.locations, key=lambda l: abs(l["t"] - t))

    def load_csv(self, path: str):
        if not _PANDAS_AVAILABLE or not os.path.exists(path):
            return
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            try:
                lat = float(row.get("Latitude",  row.get("latitude",  row.get("lat", 0.0))))
                lon = float(row.get("Longitude", row.get("longitude", row.get("lon", 0.0))))
                alt = float(row.get("GPSAltitude", row.get("altitude", row.get("alt", 0.0))))
                t   = float(row.get("timestamp_seconds", row.get("t_sec", 0.0)))
                if lat == 0.0 and lon == 0.0:
                    continue
                self.add(lat, lon, t, alt)
            except Exception:
                pass
        print(f"  [GPS] {len(self.locations)} fixes  bbox={self.bounding_box()}")

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2) -> float:
        R    = 6_371_000.0
        phi1, phi2 = radians(lat1), radians(lat2)
        dphi = radians(lat2 - lat1)
        dlam = radians(lon2 - lon1)
        a    = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlam/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    def track_length_m(self) -> float:
        if len(self.locations) < 2:
            return 0.0
        return sum(
            self.haversine(self.locations[i-1]["lat"], self.locations[i-1]["lon"],
                           self.locations[i]["lat"],   self.locations[i]["lon"])
            for i in range(1, len(self.locations))
        )

    def bounding_box(self) -> Optional[Dict]:
        if not self.locations:
            return None
        lats = [l["lat"] for l in self.locations]
        lons = [l["lon"] for l in self.locations]
        return {"min_lat": min(lats), "max_lat": max(lats),
                "min_lon": min(lons), "max_lon": max(lons)}

    def save_geojson(self, detections: List[Dict], path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        features = []
        for d in detections:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [d.get("longitude", 0.0),
                                             d.get("latitude",  0.0)]},
                "properties": {k: v for k, v in d.items()
                               if k not in ("world_corners",)},
            })
        with open(path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features},
                      f, indent=2, default=str)
        print(f"  [GPS] GeoJSON → {path}  ({len(detections)} features)")

    def save_gpx(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gpx version="1.1" creator="UnifiedTransportationPipeline">',
            "  <trk><name>Survey Track</name><trkseg>",
        ]
        for loc in self.locations:
            lines.append(
                f'    <trkpt lat="{loc["lat"]}" lon="{loc["lon"]}">'
                f'<ele>{loc["alt"]}</ele></trkpt>'
            )
        lines += ["  </trkseg></trk>", "</gpx>"]
        with open(path, "w") as f:
            f.write("\n".join(lines))
        note = "no GPS data" if not self.locations else f"{len(self.locations)} pts"
        print(f"  [GPS] GPX → {path}  ({note})")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – POTHOLE DIMENSION FUSION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DimensionEstimate:
    width_m:        float
    length_m:       float
    area_m2:        float
    volume_m3:      float
    depth_m:        float
    confidence:     float
    method_weights: Dict[str, float] = field(default_factory=dict)


class DimensionFusionEngine:
    """
    Fuses three dimension-estimation strategies for potholes:
      A) Homography (H)  – projects bbox corners to ground plane.
      B) Depth-anchor (D)– uses metric MiDaS depth + pinhole projection.
      C) Known-height (K)– focal length + camera height + pixel bbox size.
    Confidence-weighted mean.  Volume from depth-map ROI.
    """

    def __init__(self, K: np.ndarray):
        self.K  = K.copy().astype(np.float64)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])

    def update_K(self, K: np.ndarray):
        self.K  = K.copy().astype(np.float64)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])

    def _depth_dims(self, bbox: Tuple[int,int,int,int],
                    depth_m: float) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (max(1, x2-x1) * depth_m / self.fx,
                max(1, y2-y1) * depth_m / self.fy)

    def _known_height_dims(self, bbox: Tuple[int,int,int,int],
                           camera_height_m: float,
                           pitch: float) -> Tuple[float, float]:
        z_approx = camera_height_m / max(np.cos(pitch), 0.05)
        return self._depth_dims(bbox, z_approx)

    def _compute_volume(self,
                        bbox: Tuple[int,int,int,int],
                        depth_map: Optional[np.ndarray],
                        width_m: float, length_m: float,
                        image_shape: Tuple[int,int]) -> float:
        if depth_map is None:
            return 0.0
        x1, y1, x2, y2 = bbox
        H, W = depth_map.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W-1, x2), min(H-1, y2)
        if x1 >= x2 or y1 >= y2:
            return 0.0
        ring = 10
        rx1, ry1 = max(0, x1-ring), max(0, y1-ring)
        rx2, ry2 = min(W-1, x2+ring), min(H-1, y2+ring)
        border_mask = np.zeros(depth_map.shape, bool)
        border_mask[ry1:ry2, rx1:rx2] = True
        border_mask[y1:y2, x1:x2]     = False
        ground_depths = depth_map[border_mask]
        if ground_depths.size == 0:
            return 0.0
        ground_ref = float(np.nanmedian(ground_depths))
        roi   = depth_map[y1:y2, x1:x2].astype(np.float32)
        if roi.size == 0:
            return 0.0
        extra = np.clip(roi - ground_ref, 0.0, None)
        pix_area_m2 = (ground_ref / self.fx) * (ground_ref / self.fy)
        return round(float(np.sum(extra) * pix_area_m2), 6)

    def fuse(self,
             bbox: Tuple[int,int,int,int],
             homography_mapper: HomographyMapper,
             pitch: float, roll: float, yaw: float,
             camera_height_m: float,
             depth_map: Optional[np.ndarray],
             image_shape: Tuple[int,int]) -> DimensionEstimate:

        results   = {}
        depth_val = 999.0

        try:
            hom_dims = homography_mapper.bbox_world_dims(bbox, pitch, roll, yaw)
            hom_conf = homography_mapper.confidence(pitch, roll)
            if hom_dims["width_m"] > 0 and hom_dims["length_m"] > 0:
                results["homography"] = (hom_dims["width_m"], hom_dims["length_m"], hom_conf)
        except Exception as exc:
            print(f"  [FUSE] Homography error: {exc}")

        if depth_map is not None:
            raw_depth  = MiDaSDepthEstimator.bbox_median_depth(depth_map, bbox)
            depth_val  = raw_depth
            depth_conf = self._depth_confidence(raw_depth, camera_height_m)
            dw, dl     = self._depth_dims(bbox, raw_depth)
            if dw > 0 and dl > 0:
                results["depth"] = (dw, dl, depth_conf)

        kh_conf = 0.5
        kw, kl  = self._known_height_dims(bbox, camera_height_m, pitch)
        if kw > 0 and kl > 0:
            results["known_height"] = (kw, kl, kh_conf)

        if not results:
            return DimensionEstimate(0.0, 0.0, 0.0, 0.0, depth_val, 0.0)

        total_conf = sum(v[2] for v in results.values()) or 1.0
        width_m    = sum(v[0]*v[2] for v in results.values()) / total_conf
        length_m   = sum(v[1]*v[2] for v in results.values()) / total_conf
        area_m2    = width_m * length_m
        conf       = min(1.0, total_conf / max(1, len(results)))
        volume_m3  = self._compute_volume(bbox, depth_map, width_m, length_m, image_shape)
        weights    = {k: round(v[2]/total_conf, 3) for k, v in results.items()}

        return DimensionEstimate(
            width_m    = round(width_m,   4),
            length_m   = round(length_m,  4),
            area_m2    = round(area_m2,   6),
            volume_m3  = volume_m3,
            depth_m    = round(depth_val, 3),
            confidence = round(conf,      3),
            method_weights = weights,
        )

    @staticmethod
    def _depth_confidence(depth_m: float, cam_h: float) -> float:
        ratio = depth_m / max(cam_h, 0.1)
        if   ratio < 2:  return 0.95
        elif ratio < 5:  return 0.85
        elif ratio < 15: return 0.70
        elif ratio < 30: return 0.50
        else:            return 0.30


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – REAL-WORLD OBJECT DIMENSION CALCULATOR  (MCMOT)
# ══════════════════════════════════════════════════════════════════════════════

# Known real-world heights / widths (metres) per COCO class name
KNOWN_OBJECT_HEIGHTS: Dict[str, float] = {
    "person": 1.70, "car": 1.50, "bicycle": 1.20, "motorcycle": 1.20,
    "bus": 3.00, "truck": 2.50, "traffic light": 1.00, "stop sign": 0.80,
    "default": 1.50,
}
KNOWN_OBJECT_WIDTHS: Dict[str, float] = {
    "person": 0.50, "car": 1.80, "bicycle": 0.60, "motorcycle": 0.70,
    "bus": 2.50, "truck": 2.20, "default": 1.00,
}


class RealWorldObjectDimCalculator:
    """
    Calculates real-world width / height for tracked objects via pinhole model:
        real_size = (pixel_size * depth_m) / focal_length_px
    Blends depth-based measurement with known-class priors weighted by
    depth reliability.

    Camera params can be supplied directly (focal_length_px) or derived from
    phone hardware spec (focal_length_mm, sensor_width_mm, image_width_px).
    """

    def __init__(self,
                 focal_length_px:  Optional[float] = None,
                 focal_length_mm:  float = 4.0,
                 sensor_width_mm:  float = 5.6,
                 image_width_px:   int   = 1920,
                 image_height_px:  int   = 1080,
                 known_heights:    Optional[Dict[str,float]] = None,
                 known_widths:     Optional[Dict[str,float]] = None):

        if focal_length_px is not None:
            self.focal_length_px = focal_length_px
        else:
            # f_px = (f_mm * image_width_px) / sensor_width_mm
            self.focal_length_px = (focal_length_mm * image_width_px) / sensor_width_mm

        self.image_height_px = image_height_px
        self.image_width_px  = image_width_px
        self.known_heights   = known_heights or KNOWN_OBJECT_HEIGHTS
        self.known_widths    = known_widths  or KNOWN_OBJECT_WIDTHS
        print(f"  [RWDIM] focal_px={self.focal_length_px:.1f}")

    def calculate(self,
                  bbox: Tuple[int,int,int,int],
                  depth_m: float,
                  class_name: str) -> Dict:
        x1, y1, x2, y2   = bbox
        w_px = max(1, x2 - x1)
        h_px = max(1, y2 - y1)

        # Pinhole
        rw = (w_px * depth_m) / self.focal_length_px
        rh = (h_px * depth_m) / self.focal_length_px

        kh = self.known_heights.get(class_name, self.known_heights["default"])
        kw = self.known_widths.get(class_name,  self.known_widths["default"])

        # Depth confidence
        if   depth_m <   2: dc = 0.95
        elif depth_m <   5: dc = 0.90
        elif depth_m <  15: dc = 0.80
        elif depth_m <  30: dc = 0.60
        else:               dc = 0.40

        if depth_m < 999 and depth_m > 0:
            fw = rw * dc + kw * (1 - dc)
            fh = rh * dc + kh * (1 - dc)
            method = "depth_blended"
        else:
            fw, fh = kw, kh
            method = "known_class"
            dc     = 0.5

        return {
            "width_m":            round(fw,   3),
            "height_m":           round(fh,   3),
            "width_px":           w_px,
            "height_px":          h_px,
            "depth_m":            round(depth_m, 2),
            "method":             method,
            "confidence":         round(dc, 2),
        }

    def distance_from_known_size(self, bbox_height_px: int, class_name: str) -> float:
        kh = self.known_heights.get(class_name, self.known_heights["default"])
        if bbox_height_px > 0:
            return min((kh * self.focal_length_px) / bbox_height_px, 100.0)
        return 999.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – SEVERITY CLASSIFIER  (potholes)
# ══════════════════════════════════════════════════════════════════════════════

def classify_pothole_severity(area_m2: float, volume_m3: float) -> str:
    if   area_m2 < 0.05: al = 0
    elif area_m2 < 0.20: al = 1
    elif area_m2 < 0.50: al = 2
    else:                al = 3

    if   volume_m3 < 0.001: vl = 0
    elif volume_m3 < 0.005: vl = 1
    elif volume_m3 < 0.020: vl = 2
    else:                   vl = 3

    return ["Minimal", "Low", "Medium", "High"][max(al, vl)]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – CLASS-AWARE MULTI-OBJECT TRACKER  (MCMOT state machine)
# ══════════════════════════════════════════════════════════════════════════════

class ClassAwareTracker:
    """
    Per-track state machine for MCMOT:
      • Class-consistency validation (prevents spurious class switches)
      • Smoothed EMA velocity (for trajectory & prediction)
      • Smoothed EMA depth
      • Real-world dimensions cache
      • Track quality score (age × confidence)
    """

    MOTION_PARAMS: Dict[str, Dict] = {
        "car":        {"max_speed": 30, "smoothness": 0.95, "pred_h": 10},
        "person":     {"max_speed":  5, "smoothness": 0.70, "pred_h": 15},
        "bicycle":    {"max_speed":  8, "smoothness": 0.80, "pred_h": 12},
        "motorcycle": {"max_speed": 12, "smoothness": 0.85, "pred_h": 12},
        "bus":        {"max_speed": 20, "smoothness": 0.98, "pred_h":  8},
        "truck":      {"max_speed": 25, "smoothness": 0.97, "pred_h":  8},
        "default":    {"max_speed": 15, "smoothness": 0.90, "pred_h": 10},
    }

    CLASS_IMPORTANCE: Dict[str, int] = {
        "person": 10, "bicycle": 9, "motorcycle": 9,
        "car": 7, "truck": 8, "bus": 8, "default": 5,
    }

    CLASS_COLORS: Dict[str, Tuple[int,int,int]] = {
        "person":     (0, 255, 0),
        "car":        (255, 0, 0),
        "bicycle":    (0, 255, 255),
        "motorcycle": (255, 255, 0),
        "bus":        (0, 0, 255),
        "truck":      (128, 0, 255),
        "default":    (128, 128, 128),
    }

    TRAJ_LEN: Dict[str, int] = {
        "person": 40, "bicycle": 35, "motorcycle": 35,
        "car": 20, "truck": 20, "bus": 20, "default": 25,
    }

    ALLOWED_TRANSITIONS: Dict[str, List[str]] = {
        "car":        ["car", "truck", "bus"],
        "truck":      ["truck", "car", "bus"],
        "bus":        ["bus", "truck", "car"],
        "person":     ["person", "bicycle", "motorcycle"],
        "bicycle":    ["bicycle", "person"],
        "motorcycle": ["motorcycle", "person"],
    }

    def __init__(self):
        self._class_history:  Dict[int, List[str]]          = defaultdict(list)
        self._conf_history:   Dict[int, List[float]]         = defaultdict(list)
        self._age:            Dict[int, int]                 = defaultdict(int)
        self._last_seen:      Dict[int, int]                 = defaultdict(int)
        self._velocity:       Dict[int, Tuple[float,float]]  = defaultdict(lambda: (0.0, 0.0))
        self._depth:          Dict[int, float]               = defaultdict(lambda: 10.0)
        self._dimensions:     Dict[int, Dict]                = {}
        self._prev_pos:       Dict[int, Tuple[int,int]]      = {}

    # ── class validation ──────────────────────────────────────────────────────

    def validated_class(self, track_id: int, new_cls_id: int, cls_names: Dict) -> int:
        new_name = cls_names.get(new_cls_id, "unknown")
        hist     = self._class_history[track_id]
        if hist:
            from collections import Counter
            dominant = Counter(hist).most_common(1)[0][0]
            allowed  = self.ALLOWED_TRANSITIONS.get(dominant, [dominant])
            if new_name not in allowed:
                for cid, cname in cls_names.items():
                    if cname == dominant:
                        return cid
        hist.append(new_name)
        if len(hist) > 10:
            hist.pop(0)
        return new_cls_id

    # ── quality & velocity ────────────────────────────────────────────────────

    def update(self, track_id: int, conf: float, center: Tuple[int,int], frame: int):
        self._conf_history[track_id].append(conf)
        if len(self._conf_history[track_id]) > 20:
            self._conf_history[track_id].pop(0)
        self._age[track_id]       += 1
        self._last_seen[track_id]  = frame

        prev = self._prev_pos.get(track_id)
        if prev:
            vx = center[0] - prev[0]
            vy = center[1] - prev[1]
            ov_x, ov_y = self._velocity[track_id]
            self._velocity[track_id] = (ov_x * 0.7 + vx * 0.3,
                                        ov_y * 0.7 + vy * 0.3)
        self._prev_pos[track_id] = center

    def quality(self, track_id: int) -> float:
        h = self._conf_history.get(track_id)
        if not h:
            return 0.5
        avg_conf   = float(np.mean(h))
        age_factor = min(1.0, self._age[track_id] / 50)
        return avg_conf * 0.7 + age_factor * 0.3

    def predict_pos(self, track_id: int, center: Tuple[int,int]) -> Tuple[int,int]:
        vx, vy = self._velocity[track_id]
        return (int(center[0] + vx), int(center[1] + vy))

    # ── depth / dimensions ────────────────────────────────────────────────────

    def update_depth(self, track_id: int, depth: float):
        old = self._depth[track_id]
        self._depth[track_id] = old * 0.8 + depth * 0.2

    def get_depth(self, track_id: int) -> float:
        return self._depth[track_id]

    def update_dims(self, track_id: int, dims: Dict):
        self._dimensions[track_id] = dims

    def get_dims(self, track_id: int) -> Dict:
        return self._dimensions.get(track_id, {})

    # ── helpers ───────────────────────────────────────────────────────────────

    def traj_len(self, cls: str) -> int:
        return self.TRAJ_LEN.get(cls, 25)

    def color(self, cls: str) -> Tuple[int,int,int]:
        return self.CLASS_COLORS.get(cls, self.CLASS_COLORS["default"])

    def importance(self, cls: str) -> int:
        return self.CLASS_IMPORTANCE.get(cls, 5)

    def cleanup(self, active: set, max_age: int, current_frame: int):
        dead = [tid for tid in self._last_seen
                if tid not in active
                and current_frame - self._last_seen[tid] > max_age]
        for tid in dead:
            for d in (self._class_history, self._conf_history, self._age,
                      self._last_seen, self._velocity, self._depth,
                      self._dimensions, self._prev_pos):
                d.pop(tid, None)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 – RISK ASSESSOR  (MCMOT)
# ══════════════════════════════════════════════════════════════════════════════

class RiskAssessor:
    """Collision-risk score per tracked object."""

    LEVEL_TABLE = [
        (0.70, "CRITICAL", (0,   0, 255)),
        (0.40, "WARNING",  (0, 165, 255)),
        (0.20, "CAUTION",  (0, 255, 255)),
        (0.00, "SAFE",     (0, 255,   0)),
    ]

    def __init__(self, frame_w: int, frame_h: int):
        self.fw = frame_w
        self.fh = frame_h

    def assess(self,
               bbox: Tuple[int,int,int,int],
               class_name: str,
               importance: int,
               track_quality: float,
               depth_m: float = 10.0) -> Dict:
        x1, y1, x2, y2 = bbox
        cy = (y1 + y2) / 2
        bh = y2 - y1

        depth_risk  = max(0, min(1, 20.0 / (depth_m + 2))) if depth_m < 999 else 1.0 - cy/self.fh
        class_risk  = importance / 10.0
        size_risk   = min(1.0, (bh / self.fh) * 5)
        total       = (depth_risk*0.5 + class_risk*0.3 + size_risk*0.2) * track_quality

        for thresh, level, color in self.LEVEL_TABLE:
            if total >= thresh:
                return {"score": total, "level": level, "color": color,
                        "depth_risk": depth_risk}
        return {"score": total, "level": "SAFE", "color": (0,255,0),
                "depth_risk": depth_risk}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 – UNIFIED DETECTOR  (pothole + MCMOT in one pass)
# ══════════════════════════════════════════════════════════════════════════════

class UnifiedDetector:
    """
    Single YOLO wrapper that:
      • In DETECTION mode  – runs model.predict() for pothole detection.
      • In TRACKING mode   – runs model.track() (ByteTrack) for MCMOT.
      • Can operate both modes simultaneously on the same frame if separate
        model weights are supplied.
    Falls back to a mock pothole detector if no YOLO model is available.
    """

    def __init__(self,
                 pothole_model_path: str,
                 general_model_path: str = "yolov8s.pt",
                 pothole_conf: float = 0.35,
                 general_conf: float = 0.30,
                 device: Optional[torch.device] = None,
                 tracker_cfg: str = "bytetrack.yaml"):

        self.device      = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.p_conf      = pothole_conf
        self.g_conf      = general_conf
        self.tracker_cfg = tracker_cfg

        # ── Pothole model ──────────────────────────────────────────────────
        self._pot_model = None
        self._pot_mode  = "mock"
        if _YOLO_AVAILABLE and os.path.exists(pothole_model_path):
            try:
                self._pot_model = YOLO(pothole_model_path)
                self._pot_model.to(str(self.device))
                self._pot_mode  = "yolo"
                print(f"  [DET] Pothole YOLO ← {pothole_model_path}")
            except Exception as exc:
                print(f"  [DET] Pothole YOLO load failed: {exc} → mock")
        else:
            print(f"  [DET] Pothole model not found ({pothole_model_path}) → mock")

        # ── General tracking model ─────────────────────────────────────────
        # yolov8s.pt auto-downloads on first run if not present locally.
        # Use the absolute bytetrack path resolved at import time.
        self._gen_model = None
        self.tracker_cfg = _BYTETRACK_YAML   # always use absolute path
        if _YOLO_AVAILABLE:
            try:
                self._gen_model = YOLO(general_model_path)
                self._gen_model.to(str(self.device))
                print(f"  [DET] General YOLO ← {general_model_path}  "
                      f"({len(self._gen_model.names)} classes)")
            except Exception as exc:
                print(f"  [DET] General YOLO load failed: {exc}")
                print(f"  [DET] MCMOT tracking disabled — only pothole detection active")

    # ── pothole detection ──────────────────────────────────────────────────

    def detect_potholes(self, image: np.ndarray) -> List[Dict]:
        if self._pot_mode == "yolo" and self._pot_model is not None:
            return self._yolo_detect(self._pot_model, image, self.p_conf)
        return self._mock_pothole(image)

    @staticmethod
    def _yolo_detect(model, image, conf) -> List[Dict]:
        results = model(image, conf=conf, verbose=False)
        out = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                out.append({
                    "bbox":        (x1, y1, x2, y2),
                    "confidence":  float(box.conf[0]),
                    "class_id":    int(box.cls[0]),
                    "class_name":  model.names[int(box.cls[0])],
                })
        return out

    @staticmethod
    def _mock_pothole(image: np.ndarray) -> List[Dict]:
        h, w = image.shape[:2]
        cx, cy = w // 2, int(h * 0.70)
        hw, hh = w // 12, h // 14
        return [{"bbox": (cx-hw, cy-hh, cx+hw, cy+hh),
                 "confidence": 0.72,
                 "class_id": 0,
                 "class_name": "pothole"}]

    # ── MCMOT tracking ─────────────────────────────────────────────────────

    def track_objects(self, image: np.ndarray,
                      conf: Optional[float] = None,
                      iou:  float = 0.45,
                      max_det: int = 100) -> List[Dict]:
        """
        Runs ByteTrack on the general COCO model.
        Returns list of dicts: bbox, track_id, class_id, class_name, confidence.

        Ultralytics 8.x boxes.data layout with persist=True:
            7 cols: [x1, y1, x2, y2, track_id, conf, cls]
            6 cols: [x1, y1, x2, y2, conf, cls]   (no id assigned yet)
        track_id may be NaN on the very first detection — these are skipped.
        """
        if self._gen_model is None:
            return []
        c = conf if conf is not None else self.g_conf
        try:
            results = self._gen_model.track(
                image, conf=c, iou=iou, max_det=max_det,
                tracker=self.tracker_cfg, persist=True, verbose=False
            )
            out = []
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                # Prefer using r.boxes attributes (type-safe) over raw .data
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()          # (N,4)
                confs      = r.boxes.conf.cpu().numpy()           # (N,)
                clss       = r.boxes.cls.cpu().numpy().astype(int) # (N,)
                ids        = (r.boxes.id.cpu().numpy().astype(int)
                              if r.boxes.id is not None
                              else np.full(len(boxes_xyxy), -1, int))   # (N,)

                for (x1,y1,x2,y2), tid, cf, cid in zip(
                        boxes_xyxy, ids, confs, clss):
                    if tid < 0:          # unassigned track — skip
                        continue
                    out.append({
                        "bbox":       (int(x1), int(y1), int(x2), int(y2)),
                        "track_id":   int(tid),
                        "class_id":   int(cid),
                        "class_name": self._gen_model.names.get(int(cid), str(cid)),
                        "confidence": float(cf),
                    })
            return out
        except Exception as exc:
            print(f"  [DET] Tracking error: {exc}")
            import traceback; traceback.print_exc()
            return []

    @property
    def general_class_names(self) -> Dict[int, str]:
        if self._gen_model is not None:
            return self._gen_model.names
        return {}

    # ── drawing helpers ────────────────────────────────────────────────────

    @staticmethod
    def draw_potholes(image: np.ndarray, detections: List[Dict],
                      severity_map: Optional[Dict[int, str]] = None) -> np.ndarray:
        out = image.copy()
        SEV_COLORS = {"Minimal": (0,255,0), "Low": (0,255,128),
                      "Medium": (0,165,255), "High": (0,0,255)}
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d["bbox"]
            sev   = (severity_map or {}).get(i, "Low")
            color = SEV_COLORS.get(sev, (0,200,255))
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
            label = f"{d['class_name']} {d['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(out, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 – DRAWING UTILITIES  (MCMOT overlays)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_trajectory(im: np.ndarray,
                     pts: List[Tuple[int,int]],
                     color: Tuple[int,int,int],
                     thickness: int = 2,
                     dashed: bool = False):
    if len(pts) < 2:
        return
    if dashed:
        for i in range(len(pts) - 1):
            if i % 2 == 0:
                cv2.line(im, pts[i], pts[i+1], color, thickness)
    else:
        pts_arr = np.array(pts, dtype=np.int32).reshape((-1,1,2))
        cv2.polylines(im, [pts_arr], isClosed=False, color=color, thickness=thickness)


def _draw_dashed_rect(im, x1, y1, x2, y2, color, thickness=2):
    step = 10
    for i in range(x1, x2, step):
        cv2.line(im, (i, y1), (min(i+5, x2), y1), color, thickness)
        cv2.line(im, (i, y2), (min(i+5, x2), y2), color, thickness)
    for i in range(y1, y2, step):
        cv2.line(im, (x1, i), (x1, min(i+5, y2)), color, thickness)
        cv2.line(im, (x2, i), (x2, min(i+5, y2)), color, thickness)


def _build_depth_colormap(depth_raw_u8: Optional[np.ndarray],
                          w: int, h: int) -> np.ndarray:
    """
    Build a JET colour-mapped depth panel for the dual-view display.
    Accepts single-channel uint8 inverse-depth OR 3-channel BGR viz map.
    None -> placeholder panel.
    """
    if depth_raw_u8 is None:
        panel = np.zeros((h, w, 3), np.uint8)
        cv2.putText(panel, "Depth initializing...",
                    (w // 2 - 110, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
        return panel
    # Normalise to single-channel uint8
    if len(depth_raw_u8.shape) == 3:
        gray = cv2.cvtColor(depth_raw_u8, cv2.COLOR_BGR2GRAY)
    else:
        gray = depth_raw_u8.copy()
    resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
    panel   = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
    # Colour scale bar (right edge)
    bar_w, bar_h = 22, h - 100
    bx, by = w - bar_w - 15, 50
    for i in range(bar_h):
        val = int(255 * (1 - i / bar_h))
        col = cv2.applyColorMap(np.uint8([[val]]), cv2.COLORMAP_JET)[0][0]
        cv2.line(panel, (bx, by + i), (bx + bar_w, by + i),
                 (int(col[0]), int(col[1]), int(col[2])), 1)
    cv2.putText(panel, "Near", (bx - 28, by + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    cv2.putText(panel, "Far",  (bx - 24, by + bar_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 – UNIFIED CSV / JSON LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class UnifiedLogger:
    """
    Writes two streams to one CSV file:
      • Pothole detections (road_event = "pothole")
      • MCMOT tracked objects (road_event = "tracked_object")
    Also maintains in-memory list for JSON / GeoJSON export.
    """

    FIELDS = [
        # common
        "timestamp", "frame", "source_file", "road_event",
        "latitude", "longitude", "altitude",
        # pothole-specific
        "width_m", "length_m", "area_m2", "volume_m3",
        "depth_m", "severity", "method_weights",
        "camera_height_m", "pitch_deg", "roll_deg",
        "pothole_confidence", "fuse_confidence",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        # MCMOT-specific
        "track_id", "class_name", "detector_confidence",
        "obj_width_m", "obj_height_m", "obj_depth_m",
        "dim_method", "dim_confidence",
        "center_x", "center_y", "velocity_x", "velocity_y",
        "risk_score", "risk_level",
    ]

    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self._csv  = os.path.join(output_dir, "unified_log.csv")
        self._json = os.path.join(output_dir, "unified_log.json")
        self._records: List[Dict] = []
        with open(self._csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def add(self, record: Dict):
        self._records.append(record)
        with open(self._csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS,
                           extrasaction="ignore").writerow(record)

    @property
    def records(self) -> List[Dict]:
        return self._records

    @property
    def pothole_records(self) -> List[Dict]:
        return [r for r in self._records if r.get("road_event") == "pothole"]

    def flush_json(self):
        with open(self._json, "w") as f:
            json.dump(self._records, f, indent=2, default=str)
        print(f"  [LOG] {len(self._records)} records → {self._json}")








class GeometryAnchoredDepthScaler:
    """
    Fuses Camera Calibration + IMU + Homography + MiDaS to produce
    absolute dense depth maps with minimal error.
    
    Strategy:
    1. Get sparse metric anchors from geometry (homography + IMU)
    2. Solve for optimal scale/shift of MiDaS relative depth
    3. Apply to full depth map
    4. Temporal smoothing for stability
    """
    
    def __init__(self, K: np.ndarray, min_anchors: int = 10):
        self.K = K.copy()
        self.min_anchors = min_anchors
        self._last_scale = 1.0
        self._last_shift = 0.0
        self._history = []  # rolling window for stability
        
    def compute_metric_anchors_from_geometry(
        self,
        frame: np.ndarray,
        homography_mapper: HomographyMapper,
        pitch: float,
        roll: float,
        yaw: float,
        camera_height_m: float
    ) -> List[Tuple[int, int, float]]:
        """
        Extract sparse metric depth points from homography + IMU.
        
        Returns list of (u, v, depth_m) for pixels on ground plane.
        """
        H, W = frame.shape[:2]
        anchors = []
        
        # Sample points on ground plane region (bottom half of image)
        # These are where homography is most reliable
        grid_h = 15  # points vertically
        grid_w = 20  # points horizontally
        
        y_start = H // 2  # only bottom half (ground region)
        for iy in range(grid_h):
            y = y_start + (iy * (H - y_start) // grid_h)
            for ix in range(grid_w):
                x = ix * W // grid_w
                
                # Convert pixel to world coordinates using homography
                world_x, world_y = homography_mapper.pixel_to_world(
                    x, y, pitch, roll, yaw
                )
                
                # Depth from homography: sqrt(X² + Y² + camera_height²)
                depth_m = np.sqrt(world_x**2 + world_y**2 + camera_height_m**2)
                
                # Only accept plausible depths
                if 0.5 < depth_m < 50.0:
                    anchors.append((x, y, depth_m))
                    
        return anchors
    
    def compute_metric_anchors_from_vio(
        self,
        tracked_features: List[Dict],
        current_frame: int
    ) -> List[Tuple[int, int, float]]:
        """
        Alternative: Use VIO feature tracking for non-planar regions.
        Requires visual-inertial odometry to provide metric depth at features.
        """
        anchors = []
        for feat in tracked_features:
            if feat.get('depth_metric', 0) > 0:
                anchors.append((
                    feat['x'], feat['y'], feat['depth_metric']
                ))
        return anchors
    
    def solve_scale_shift(
        self,
        rel_depth_map: np.ndarray,  # MiDaS relative depth
        anchors: List[Tuple[int, int, float]]  # (u, v, metric_depth)
    ) -> Tuple[float, float, float]:
        """
        Solve for scale 'a' and shift 'b' where:
            D_metric = a * D_rel + b
        
        Using RANSAC to reject outlier anchors.
        """
        if len(anchors) < self.min_anchors:
            # Fall back to previous values
            return self._last_scale, self._last_shift, 0.0
        
        H, W = rel_depth_map.shape
        valid_anchors = []
        
        for u, v, z_metric in anchors:
            if 0 <= u < W and 0 <= v < H:
                z_rel = rel_depth_map[v, u]
                if z_rel > 0 and np.isfinite(z_rel):
                    valid_anchors.append((z_rel, z_metric))
        
        if len(valid_anchors) < self.min_anchors // 2:
            return self._last_scale, self._last_shift, 0.0
        
        # RANSAC to find best linear fit
        best_scale = self._last_scale
        best_shift = self._last_shift
        best_inliers = 0
        
        # Convert to numpy for RANSAC
        points = np.array(valid_anchors)
        
        # Try multiple hypotheses
        for _ in range(50):
            # Pick 2 random points
            idx = np.random.choice(len(points), 2, replace=False)
            z_rel1, z_met1 = points[idx[0]]
            z_rel2, z_met2 = points[idx[1]]
            
            if abs(z_rel2 - z_rel1) < 1e-6:
                continue
                
            scale = (z_met2 - z_met1) / (z_rel2 - z_rel1)
            shift = z_met1 - scale * z_rel1
            
            # Count inliers
            pred = scale * points[:, 0] + shift
            errors = np.abs(pred - points[:, 1])
            inliers = np.sum(errors < 0.1 * np.median(points[:, 1]))  # 10% error threshold
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_scale = scale
                best_shift = shift
        
        # Refine with all inliers
        if best_inliers > 0:
            pred = best_scale * points[:, 0] + best_shift
            errors = np.abs(pred - points[:, 1])
            inlier_mask = errors < 0.1 * np.median(points[:, 1])
            if np.sum(inlier_mask) >= 2:
                inlier_points = points[inlier_mask]
                # Least squares fit on inliers
                A = np.vstack([inlier_points[:, 0], np.ones(len(inlier_points))]).T
                b = inlier_points[:, 1]
                best_scale, best_shift = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Temporal smoothing
        self._history.append((best_scale, best_shift))
        if len(self._history) > 10:
            self._history.pop(0)
        
        # Exponential moving average
        alpha = 0.7
        self._last_scale = alpha * best_scale + (1 - alpha) * self._last_scale
        self._last_shift = alpha * best_shift + (1 - alpha) * self._last_shift
        
        confidence = min(1.0, best_inliers / len(valid_anchors))
        
        return self._last_scale, self._last_shift, confidence
    
    def apply_to_depth_map(
        self,
        rel_depth_map: np.ndarray,
        scale: float,
        shift: float
    ) -> np.ndarray:
        """Apply solved scale/shift to full depth map."""
        with np.errstate(divide='ignore', invalid='ignore'):
            abs_depth = scale * rel_depth_map + shift
            abs_depth = np.clip(abs_depth, 0.1, 100.0)
        return abs_depth









# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15 – MAIN UNIFIED PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class UnifiedTransportationPipeline:
    """
    Master pipeline integrating the complete transportation intelligence stack:

    Camera Calibration → Image Undistortion → IMU → MiDaS Depth →
    Homography Mapper → Dimension Fusion (potholes) →
    Real-World Dim Calc (objects) → Unified Detector (pothole + MCMOT) →
    Risk Assessor → GPS Mapper → Unified Logger → Dual-View Video Export

    Every run creates a new timestamped project directory under base_output_dir.
    """

    def __init__(self,
                 calibration_path:   str,
                 pothole_model_path: str,
                 general_model_path: str   = "yolov8s.pt",
                 base_output_dir:    str   = "output",
                 midas_model_type:   str   = "MiDaS_small",
                 initial_height_m:   float = 1.2,
                 pothole_conf:       float = 0.35,
                 general_conf:       float = 0.30,
                 # Camera hardware spec for MCMOT dimension calc
                 focal_length_mm:    float = 4.0,
                 sensor_width_mm:    float = 5.6,
                 depth_every_n:      int   = 1):   # run MiDaS every N processed frames

        print("=" * 72)
        print("  UNIFIED TRANSPORTATION INTELLIGENCE PIPELINE")
        print("=" * 72)

        # ── Run manager (creates timestamped project dir) ──────────────────
        self.run = RunManager(base_output_dir)
        self.output_dir   = self.run.run_dir   # all outputs go here
        self.depth_every  = max(1, depth_every_n)
        self._depth_ctr   = 0

        # 1. Camera calibration ────────────────────────────────────────────
        print("\n[1/9] Camera calibration …")
        self.calibrator = CameraCalibrator()
        self.calibrator.load(calibration_path)
        K0, dist  = self.calibrator.K, self.calibrator.dist
        cal_wh    = self.calibrator.image_shape

        # 2. Undistorter ───────────────────────────────────────────────────
        print("[2/9] Undistorter …")
        self.undistorter = ImageUndistorter(K0, dist, cal_wh, alpha=0.0)

        # 3. IMU ───────────────────────────────────────────────────────────
        print("[3/9] IMU processor …")
        self.imu = IMUProcessor(initial_height_m=initial_height_m)

        # 4. Shared MiDaS estimator ────────────────────────────────────────
        print("[4/9] MiDaS depth estimator …")
        self.depth_est = MiDaSDepthEstimator(model_type=midas_model_type)

        # 5. Homography + pothole fusion ────────────────────────────────────
        print("[5/9] Homography mapper …")
        self.homography = HomographyMapper(K0, initial_height_m)

        print("[6/9] Pothole dimension fusion …")
        self.fuser = DimensionFusionEngine(K0)

        # 6. Real-world object dimension calc (MCMOT) ──────────────────────
        print("[7/9] Real-world object dimension calculator …")
        self._focal_mm  = focal_length_mm
        self._sens_mm   = sensor_width_mm
        # Will be (re-)built once we know the actual frame size
        self.obj_dim_calc: Optional[RealWorldObjectDimCalculator] = None

        # 7. Unified detector + tracker ────────────────────────────────────
        print("[8/9] Unified detector + tracker …")
        self.detector = UnifiedDetector(
            pothole_model_path = pothole_model_path,
            general_model_path = general_model_path,
            pothole_conf       = pothole_conf,
            general_conf       = general_conf,
            # tracker_cfg resolved to absolute bytetrack.yaml inside UnifiedDetector
        )

        # 8. MCMOT state machines ──────────────────────────────────────────
        self.mcmot_tracker  = ClassAwareTracker()
        self.risk_assessor: Optional[RiskAssessor] = None   # init on first frame
        self._track_history: Dict[int, List[Tuple[int,int]]] = {}

        # 9. GPS + logger ──────────────────────────────────────────────────
        print("[9/9] GPS mapper + unified logger …")
        self.gps    = GPSMapper()
        self.logger = UnifiedLogger(self.run.data_dir)   # data/ sub-directory

        # Store run settings for manifest
        self.run.set_settings(
            calibration_path   = calibration_path,
            pothole_model      = pothole_model_path,
            general_model      = general_model_path,
            midas_model        = midas_model_type,
            initial_height_m   = initial_height_m,
            pothole_conf       = pothole_conf,
            general_conf       = general_conf,
            focal_length_mm    = focal_length_mm,
            sensor_width_mm    = sensor_width_mm,
            depth_every_n      = depth_every_n,
        )

        print("\n  ✓ Pipeline ready\n" + "=" * 72)

    # ── sensor data ───────────────────────────────────────────────────────────

    def load_sensor_csv(self, path: str):
        """Load combined IMU + GPS CSV (Android phone format)."""
        self.imu.load_from_csv(path)
        self.gps.load_csv(path)

    # ── internal K / height sync ──────────────────────────────────────────────

    def _sync_K(self, new_K: Optional[np.ndarray]):
        if new_K is not None:
            self.homography.K = new_K.copy()
            self.fuser.update_K(new_K)

    def _ensure_components(self, frame_w: int, frame_h: int):
        """Lazily initialise frame-size-dependent components."""
        if self.risk_assessor is None:
            self.risk_assessor = RiskAssessor(frame_w, frame_h)
        if self.obj_dim_calc is None:
            self.obj_dim_calc = RealWorldObjectDimCalculator(
                focal_length_mm  = self._focal_mm,
                sensor_width_mm  = self._sens_mm,
                image_width_px   = frame_w,
                image_height_px  = frame_h,
            )

    # ── core per-frame processing ─────────────────────────────────────────────

    def _process_frame(self,
                       raw_frame:   np.ndarray,
                       timestamp:   float,
                       source_name: str,
                       frame_idx:   int
                       ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Full pipeline for one frame.
        Returns (tracking_annotated, depth_colourmap, record_list).
        """
        # ── Step 1: Undistort ────────────────────────────────────────────────
        frame = self.undistorter.undistort(raw_frame)
        H, W  = frame.shape[:2]
        self._ensure_components(W, H)

        new_K = self.undistorter.current_K
        self._sync_K(new_K)
        K_eff = new_K if new_K is not None else self.calibrator.K

        # ── Step 2: IMU state ────────────────────────────────────────────────
        cam_h = self.imu.height_at(timestamp)
        ori   = self.imu.orientation_at(timestamp)
        pitch, roll, yaw = ori["pitch"], ori["roll"], ori["yaw"]
        self.homography.update_height(cam_h)

        # ── Step 3: MiDaS depth (shared) ─────────────────────────────────────
        self._depth_ctr += 1
        force = (self._depth_ctr % self.depth_every == 0)
        depth_metric, depth_viz = self.depth_est.estimate(
            frame, cam_h, pitch, K_eff, force_update=force
        )
        # For MCMOT we use the same raw inverse-depth map (already cached)
        depth_raw = self.depth_est._last_inv   # raw MiDaS output

        # ── Step 4: GPS ───────────────────────────────────────────────────────
        gps_fix = self.gps.nearest(timestamp)
        lat = gps_fix["lat"] if gps_fix else 0.0
        lon = gps_fix["lon"] if gps_fix else 0.0
        alt = gps_fix["alt"] if gps_fix else 0.0

        records      = []
        tracking_vis = frame.copy()
        severity_map: Dict[int, str] = {}

        # ═════════════════════════════════════════════════════════════════════
        # SUB-PIPELINE A: POTHOLE DETECTION & DIMENSION FUSION
        # ═════════════════════════════════════════════════════════════════════
        potholes = self.detector.detect_potholes(frame)

        for i, det in enumerate(potholes):
            bbox = det["bbox"]
            dims = self.fuser.fuse(
                bbox             = bbox,
                homography_mapper = self.homography,
                pitch            = pitch,
                roll             = roll,
                yaw              = yaw,
                camera_height_m  = cam_h,
                depth_map        = depth_metric,
                image_shape      = (H, W),
            )
            severity = classify_pothole_severity(dims.area_m2, dims.volume_m3)
            severity_map[i] = severity

            SEV_COLORS = {"Minimal": (0,255,0), "Low": (0,255,128),
                          "Medium": (0,165,255), "High": (0,0,255)}
            col = SEV_COLORS.get(severity, (0,200,255))
            x1, y1, x2, y2 = bbox
            cv2.rectangle(tracking_vis, (x1,y1), (x2,y2), col, 2)

            lines = [
                f"POTHOLE  {det['confidence']:.2f}  [{severity}]",
                f"W:{dims.width_m:.3f}m  L:{dims.length_m:.3f}m",
                f"A:{dims.area_m2:.4f}m²  V:{dims.volume_m3:.5f}m³",
                f"D:{dims.depth_m:.2f}m  conf={dims.confidence:.2f}",
            ]
            for j, line in enumerate(lines):
                y_txt = y2 + 15 + j * 15
                if y_txt < H:
                    cv2.putText(tracking_vis, line, (x1, y_txt),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

            record: Dict = {
                "timestamp":         datetime.now().isoformat(),
                "frame":             frame_idx,
                "source_file":       source_name,
                "road_event":        "pothole",
                "latitude":          round(lat, 8),
                "longitude":         round(lon, 8),
                "altitude":          round(alt, 3),
                "width_m":           dims.width_m,
                "length_m":          dims.length_m,
                "area_m2":           dims.area_m2,
                "volume_m3":         dims.volume_m3,
                "depth_m":           dims.depth_m,
                "severity":          severity,
                "method_weights":    json.dumps(dims.method_weights),
                "camera_height_m":   round(cam_h,  4),
                "pitch_deg":         round(np.degrees(pitch), 3),
                "roll_deg":          round(np.degrees(roll),  3),
                "pothole_confidence": round(det["confidence"], 4),
                "fuse_confidence":   dims.confidence,
                "bbox_x1":           x1, "bbox_y1": y1,
                "bbox_x2":           x2, "bbox_y2": y2,
            }
            self.logger.add(record)
            records.append(record)

        # ═════════════════════════════════════════════════════════════════════
        # SUB-PIPELINE B: MCMOT (ByteTrack multi-class tracking)
        # ═════════════════════════════════════════════════════════════════════
        tracked = self.detector.track_objects(frame)
        cls_names = self.detector.general_class_names
        active_ids: set = set()
        class_counts: Dict[str, int] = defaultdict(int)

        for det in tracked:
            track_id = det.get("track_id", -1)
            if track_id < 0:
                continue

            class_id = self.mcmot_tracker.validated_class(
                track_id, det["class_id"], cls_names
            )
            class_name = cls_names.get(class_id, det["class_name"])
            conf_sc    = det["confidence"]
            bbox       = det["bbox"]
            x1, y1, x2, y2 = bbox

            center = ((x1+x2)//2, (y1+y2)//2)
            self.mcmot_tracker.update(track_id, conf_sc, center, frame_idx)
            active_ids.add(track_id)
            class_counts[class_name] += 1

            # Depth for this object (from raw MiDaS map)
            raw_d = (MiDaSDepthEstimator.bbox_median_depth(depth_raw, bbox)
                     if depth_raw is not None else 999.0)
            self.mcmot_tracker.update_depth(track_id, raw_d)
            smooth_d = self.mcmot_tracker.get_depth(track_id)

            # Real-world dimensions
            dims_obj = self.obj_dim_calc.calculate(bbox, smooth_d, class_name)
            self.mcmot_tracker.update_dims(track_id, dims_obj)

            # Velocity
            velocity = self.mcmot_tracker._velocity[track_id]

            # Risk
            tq       = self.mcmot_tracker.quality(track_id)
            imp      = self.mcmot_tracker.importance(class_name)
            risk     = self.risk_assessor.assess(bbox, class_name, imp, tq, smooth_d)

            # Trajectory
            traj = self._track_history.setdefault(track_id, [])
            traj.append(center)
            max_traj = self.mcmot_tracker.traj_len(class_name)
            if len(traj) > max_traj:
                traj.pop(0)

            # ── Draw ─────────────────────────────────────────────────────────
            color    = self.mcmot_tracker.color(class_name)
            dashed_p = class_name in ("person", "bicycle", "motorcycle")
            _draw_trajectory(tracking_vis, traj, color, 2, dashed=dashed_p)

            if class_name == "person":
                _draw_dashed_rect(tracking_vis, x1, y1, x2, y2, color, 2)
            else:
                cv2.rectangle(tracking_vis, (x1,y1), (x2,y2), color, 2)

            label = (f"{class_name}#{track_id}  "
                     f"D:{smooth_d:.0f}m  "
                     f"{dims_obj['width_m']:.1f}×{dims_obj['height_m']:.1f}m")
            (tw, th), _ = cv2.getTextSize(label, 0, 0.46, 1)
            cv2.rectangle(tracking_vis,
                          (x1+2, y1+2), (x1+tw+6, y1+th+8), color, -1)
            cv2.putText(tracking_vis, label, (x1+4, y1+th+4),
                        0, 0.46, (255,255,255), 1, cv2.LINE_AA)

            # Dimension line below box
            dim_y = y2 + 13
            if dim_y < H:
                cv2.putText(tracking_vis,
                            f"W:{dims_obj['width_m']:.1f}m H:{dims_obj['height_m']:.1f}m",
                            (x1, dim_y), 0, 0.38, (255,255,0), 1)

            # Risk badge
            if risk["score"] > 0.40:
                badge = risk["level"][:3]
                cv2.rectangle(tracking_vis,
                              (x1-2, y1-22), (x1+36, y1-4), risk["color"], -1)
                cv2.putText(tracking_vis, badge, (x1, y1-9),
                            0, 0.40, (255,255,255), 1, cv2.LINE_AA)

            # Predicted position dot
            pred = self.mcmot_tracker.predict_pos(track_id, center)
            cv2.circle(tracking_vis, pred, 4, (255,255,255), -1)
            cv2.circle(tracking_vis, pred, 2, color, -1)

            # Log MCMOT record
            mcmot_rec: Dict = {
                "timestamp":          datetime.now().isoformat(),
                "frame":              frame_idx,
                "source_file":        source_name,
                "road_event":         "tracked_object",
                "latitude":           round(lat, 8),
                "longitude":          round(lon, 8),
                "altitude":           round(alt, 3),
                "track_id":           track_id,
                "class_name":         class_name,
                "detector_confidence": round(conf_sc, 4),
                "obj_width_m":        dims_obj["width_m"],
                "obj_height_m":       dims_obj["height_m"],
                "obj_depth_m":        dims_obj["depth_m"],
                "dim_method":         dims_obj["method"],
                "dim_confidence":     dims_obj["confidence"],
                "center_x":           center[0],
                "center_y":           center[1],
                "velocity_x":         round(velocity[0], 2),
                "velocity_y":         round(velocity[1], 2),
                "risk_score":         round(risk["score"], 3),
                "risk_level":         risk["level"],
                # fill pothole fields as empty
                "camera_height_m":    round(cam_h, 4),
                "pitch_deg":          round(np.degrees(pitch), 3),
                "roll_deg":           round(np.degrees(roll),  3),
            }
            self.logger.add(mcmot_rec)
            records.append(mcmot_rec)

        # ── Clean up lost tracks ──────────────────────────────────────────────
        self.mcmot_tracker.cleanup(active_ids, max_age=30, current_frame=frame_idx)
        for tid in list(self._track_history):
            if tid not in active_ids:
                del self._track_history[tid]

        # ── HUD overlays ──────────────────────────────────────────────────────
        # Class counts panel (top-left)
        y_off = 55
        if class_counts:
            cv2.putText(tracking_vis, "Tracked objects:",
                        (8, y_off), 0, 0.55, (255,255,255), 1, cv2.LINE_AA)
            y_off += 18
            for cname, cnt in sorted(class_counts.items(),
                                     key=lambda x: x[1], reverse=True)[:8]:
                col = self.mcmot_tracker.color(cname)
                cv2.putText(tracking_vis, f"  {cname}: {cnt}",
                            (8, y_off), 0, 0.45, col, 1, cv2.LINE_AA)
                y_off += 16

        # Pothole count badge (top-right corner)
        n_pot = len(potholes)
        if n_pot:
            badge_txt = f"POTHOLES: {n_pot}"
            (bw, bh), _ = cv2.getTextSize(badge_txt, 0, 0.6, 2)
            cv2.rectangle(tracking_vis,
                          (W-bw-20, 6), (W-4, bh+14), (0,0,200), -1)
            cv2.putText(tracking_vis, badge_txt,
                        (W-bw-12, bh+8), 0, 0.6, (255,255,255), 2, cv2.LINE_AA)

        # Bottom sensor bar
        cv2.rectangle(tracking_vis, (0, H-26), (W, H), (20,20,20), -1)
        bar = (f"frame={frame_idx}  t={timestamp:.1f}s  h={cam_h:.2f}m  "
               f"p={np.degrees(pitch):.1f}°  r={np.degrees(roll):.1f}°  "
               f"tracks={len(active_ids)}  potholes={n_pot}")
        cv2.putText(tracking_vis, bar, (6, H-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160,160,160), 1, cv2.LINE_AA)

        # ── Depth panel (right 20%) ────────────────────────────────────────────
        panel_w = max(1, W // 5)
        if depth_viz is not None:
            dv_r = cv2.resize(depth_viz, (panel_w, H))
            tracking_vis[:, W-panel_w:] = (
                tracking_vis[:, W-panel_w:].astype(np.float32)*0.35
                + dv_r.astype(np.float32)*0.65
            ).astype(np.uint8)
            cv2.putText(tracking_vis, "DEPTH", (W-panel_w+4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        # ── Dual-view depth colourmap panel ──────────────────────────────────
        # Use the raw inverse-depth float32 map (self.depth_est._last_inv)
        # normalised to uint8.  This gives _build_depth_colormap a proper
        # single-channel map to apply JET to (avoids BGR->gray artefacts).
        raw_inv = self.depth_est._last_inv
        if raw_inv is not None:
            dv_u8 = cv2.normalize(raw_inv, None, 0, 255,
                                  cv2.NORM_MINMAX).astype(np.uint8)
        else:
            dv_u8 = None
        depth_panel = _build_depth_colormap(dv_u8, W, H)

        # ── Overlay depth-panel title ─────────────────────────────────────────
        cv2.rectangle(depth_panel, (0,0), (W,36), (0,0,0), -1)
        cv2.putText(depth_panel, "REAL-TIME DEPTH MAP",
                    (14, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0,255,255), 2)

        return tracking_vis, depth_panel, records

    # ── public entry points ───────────────────────────────────────────────────

    def process_image(self, image_path: str, timestamp: float = 0.0) -> List[Dict]:
        img = cv2.imread(image_path)
        if img is None:
            print(f"  [PIPE] Cannot read: {image_path}")
            return []
        ann, depth_p, records = self._process_frame(
            img, timestamp, os.path.basename(image_path), 0
        )
        ann_dir  = self.run.annotated_dir
        stem     = Path(image_path).stem
        cv2.imwrite(os.path.join(ann_dir, f"ann_{stem}.jpg"),
                    ann, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(ann_dir, f"depth_{stem}.jpg"),
                    depth_p, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  [PIPE] {len(records)} record(s) → {ann_dir}")
        return records

    def process_image_folder(self, folder: str,
                             start_timestamp: float = 0.0,
                             interval_s: float = 1.0) -> int:
        exts  = {".jpg", ".jpeg", ".png"}
        paths = sorted(p for p in Path(folder).iterdir()
                       if p.suffix.lower() in exts)
        total = 0
        for i, p in enumerate(paths):
            recs   = self.process_image(str(p), timestamp=start_timestamp + i*interval_s)
            total += len(recs)
        print(f"  [PIPE] Folder done: {len(paths)} images  {total} records")
        return total

    def process_video(self,
                      video_path:     str,
                      output_video:   Optional[str]  = None,
                      frame_interval: int   = 15,
                      start_t:        float = 0.0,
                      dual_view:      bool  = True) -> int:
        """
        Process a video file.

        Args:
            video_path     : Input video.
            output_video   : Optional annotated (dual-view) output video path.
            frame_interval : Process every Nth frame (others written through).
            start_t        : Start timestamp to sync with sensor CSV.
            dual_view      : If True writes side-by-side tracking+depth video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [PIPE] Cannot open video: {video_path}")
            return 0

        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vid_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
        vid_name = Path(video_path).stem

        print(f"  [PIPE] Video: {vid_w}×{vid_h}  fps={fps:.1f}  "
              f"frames={total_fr}  rotation={rotation}°")

        # Only rotate when the container metadata explicitly signals it.
        # Never rotate based on aspect ratio — landscape videos are landscape.
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

        # Scale K to video resolution
        cal_wh = self.calibrator.image_shape
        sx = display_w / cal_wh[0]
        sy = display_h / cal_wh[1]
        if abs(sx-1.0) > 0.01 or abs(sy-1.0) > 0.01:
            K_orig   = self.calibrator.K
            K_scaled = K_orig.copy().astype(np.float64)
            K_scaled[0,0] *= sx;  K_scaled[0,2] *= sx
            K_scaled[1,1] *= sy;  K_scaled[1,2] *= sy
            self.undistorter = ImageUndistorter(
                K_scaled, self.calibrator.dist,
                (display_w, display_h), alpha=0.0
            )
            self._sync_K(K_scaled)
            print(f"  [PIPE] K scaled sx={sx:.3f} sy={sy:.3f}")

        self.undistorter.prepare((display_w, display_h))

        # ── Video writer ─────────────────────────────────────────────────────
        writer       = None
        writer_w     = 0
        writer_h     = 0
        actual_output_video = output_video  # may be changed to .avi below
        if output_video:
            out_dir = os.path.dirname(os.path.abspath(output_video))
            os.makedirs(out_dir, exist_ok=True)

            # Dual-view layout: [info_bar(60px)] / [tracking | depth]
            # Both panels are resized to display_w × display_h before hstack.
            writer_w = display_w * 2 if dual_view else display_w
            writer_h = display_h + 60 if dual_view else display_h

            # Codec priority: mp4v (always works) → MJPG (.avi fallback)
            codec_attempts = [
                (output_video,                    "mp4v"),
                (output_video,                    "XVID"),
                (output_video,                    "X264"),
                (output_video.rsplit(".", 1)[0] + "_fb.avi", "MJPG"),
            ]
            for vid_path, fcc in codec_attempts:
                fourcc = cv2.VideoWriter_fourcc(*fcc)
                test_w = cv2.VideoWriter(vid_path, fourcc, fps,
                                         (writer_w, writer_h))
                if test_w.isOpened():
                    writer = test_w
                    actual_output_video = vid_path
                    print(f"  [PIPE] VideoWriter ({fcc}) → {vid_path}  "
                          f"size={writer_w}×{writer_h}")
                    break
                test_w.release()
            if writer is None:
                print("  [PIPE] WARNING: Could not open any VideoWriter codec — "
                      "annotated frames will be saved as JPEGs only.")

        ann_dir = self.run.annotated_dir   # already created by RunManager

        frame_idx  = 0
        saved      = 0
        total_recs = 0
        t_start    = time.time()
        fps_disp   = 0.0
        fps_ctr    = 0
        fps_t0     = time.time()

        last_tracking_vis: Optional[np.ndarray] = None
        last_depth_panel:  Optional[np.ndarray] = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if rot_code is not None:
                frame = cv2.rotate(frame, rot_code)

            t = start_t + frame_idx / fps

            if frame_idx % frame_interval == 0:
                tracking_vis, depth_panel, recs = self._process_frame(
                    frame, t, vid_name, frame_idx
                )
                total_recs += len(recs)
                last_tracking_vis = tracking_vis
                last_depth_panel  = depth_panel

                ann_path = os.path.join(ann_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(ann_path, tracking_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved += 1

            # FPS counter
            fps_ctr += 1
            if time.time() - fps_t0 >= 1.0:
                fps_disp = fps_ctr
                fps_ctr  = 0
                fps_t0   = time.time()

            # Write to video
            if writer and writer.isOpened():
                if dual_view and last_tracking_vis is not None and last_depth_panel is not None:
                    # Force-resize both panels to exactly display_w × display_h
                    tv = cv2.resize(last_tracking_vis, (display_w, display_h),
                                    interpolation=cv2.INTER_LINEAR)
                    dp = cv2.resize(last_depth_panel,  (display_w, display_h),
                                    interpolation=cv2.INTER_LINEAR)
                    info_bar = np.zeros((60, writer_w, 3), np.uint8)
                    info_bar[:] = (25, 25, 45)
                    cv2.putText(
                        info_bar,
                        (f"UNIFIED TRANSPORTATION PIPELINE  |  "
                         f"FPS:{fps_disp:.0f}  frame:{frame_idx}  "
                         f"tracks:{len(self._track_history)}  records:{total_recs}"),
                        (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 2
                    )
                    top       = np.hstack([tv, dp])         # display_w*2 × display_h
                    out_frame = np.vstack([info_bar, top])  # writer_w × writer_h
                    assert out_frame.shape[1] == writer_w and out_frame.shape[0] == writer_h, \
                        f"Frame size mismatch: {out_frame.shape} vs {writer_w}×{writer_h}"
                    writer.write(out_frame)
                else:
                    # Single-view: use last processed or current raw frame
                    src = last_tracking_vis if last_tracking_vis is not None else frame
                    f_out = cv2.resize(src, (writer_w, writer_h),
                                       interpolation=cv2.INTER_LINEAR)
                    writer.write(f_out)

            frame_idx += 1
            if frame_idx % 100 == 0:
                pct = int(frame_idx / max(total_fr, 1) * 100)
                elapsed = time.time() - t_start
                print(f"  [PIPE] {frame_idx}/{total_fr} ({pct}%)  "
                      f"processed={saved}  records={total_recs}  "
                      f"elapsed={elapsed:.0f}s")

        cap.release()
        if writer and writer.isOpened():
            writer.release()
            print(f"  [PIPE] Video written → {actual_output_video}")
        elapsed = time.time() - t_start
        print(f"  [PIPE] Done: {frame_idx} frames  {saved} processed  "
              f"{total_recs} records  {elapsed:.1f}s")
        return total_recs

    # ── results output ────────────────────────────────────────────────────────

    def save_results(self, input_video: Optional[str] = None) -> Dict:
        """
        Flush and persist all outputs into the run's project directory:
          data/unified_log.csv + .json  – every event
          data/potholes.csv              – pothole events only
          data/tracked_objects.csv       – MCMOT events only
          data/potholes.geojson          – GIS-ready pothole map
          data/track.gpx                 – GPS route
          reports/summary.json           – machine-readable stats
          reports/report.html            – human-readable HTML dashboard
          manifest.json                  – run config + file inventory
        """
        dd = self.run.data_dir
        rd = self.run.reports_dir

        # 1. Flush unified CSV + JSON (incrementally written; finalize JSON)
        self.logger.flush_json()

        # 2. Split CSVs
        POTHOLE_FIELDS = [
            "timestamp", "frame", "source_file", "road_event",
            "latitude", "longitude", "altitude",
            "width_m", "length_m", "area_m2", "volume_m3",
            "depth_m", "severity", "method_weights",
            "camera_height_m", "pitch_deg", "roll_deg",
            "pothole_confidence", "fuse_confidence",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        ]
        OBJ_FIELDS = [
            "timestamp", "frame", "source_file", "road_event",
            "latitude", "longitude", "altitude",
            "track_id", "class_name", "detector_confidence",
            "obj_width_m", "obj_height_m", "obj_depth_m",
            "dim_method", "dim_confidence",
            "center_x", "center_y", "velocity_x", "velocity_y",
            "risk_score", "risk_level",
            "camera_height_m", "pitch_deg", "roll_deg",
        ]
        RunManager.save_split_csv(
            self.logger.records, os.path.join(dd, "potholes.csv"),
            "pothole", POTHOLE_FIELDS
        )
        RunManager.save_split_csv(
            self.logger.records, os.path.join(dd, "tracked_objects.csv"),
            "tracked_object", OBJ_FIELDS
        )
        print(f"  [PIPE] Split CSVs → {dd}/potholes.csv  tracked_objects.csv")

        # 3. GeoJSON + GPX (always written, empty when no data)
        self.gps.save_geojson(
            self.logger.pothole_records,
            os.path.join(dd, "potholes.geojson")
        )
        self.gps.save_gpx(os.path.join(dd, "track.gpx"))

        # 4. Summary JSON
        summary  = self._build_summary(self.logger.records)
        sum_path = os.path.join(rd, "summary.json")
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  [PIPE] summary.json → {sum_path}")

        # 5. HTML report
        RunManager.save_html_report(
            run_dir   = self.run.run_dir,
            run_id    = self.run.run_id,
            summary   = summary,
            settings  = getattr(self.run, "settings", {}),
            records   = self.logger.records,
        )

        self._print_summary(summary)

        # 6. Manifest + file inventory
        self.run.save_manifest(summary, video_path=input_video)

        return summary

    @staticmethod
    def _build_summary(records: List[Dict]) -> Dict:
        pots  = [r for r in records if r.get("road_event") == "pothole"]
        objs  = [r for r in records if r.get("road_event") == "tracked_object"]
        n_pot = len(pots)
        n_obj = len(objs)

        pot_summary: Dict = {"total": n_pot}
        if n_pot:
            areas = [r["area_m2"]   for r in pots]
            vols  = [r["volume_m3"] for r in pots]
            sevs  = [r["severity"]  for r in pots]
            pot_summary.update({
                "total_area_m2":   round(sum(areas), 4),
                "mean_area_m2":    round(sum(areas)/n_pot, 6),
                "max_area_m2":     round(max(areas), 6),
                "total_volume_m3": round(sum(vols), 6),
                "severity_counts": {s: sevs.count(s)
                                    for s in ("Minimal","Low","Medium","High")},
            })

        obj_summary: Dict = {"total_detections": n_obj}
        if n_obj:
            from collections import Counter
            cls_ctr  = Counter(r["class_name"] for r in objs)
            risk_ctr = Counter(r["risk_level"]  for r in objs)
            depths   = [r["obj_depth_m"] for r in objs if r.get("obj_depth_m")]
            obj_summary.update({
                "class_counts": dict(cls_ctr),
                "risk_counts":  dict(risk_ctr),
                "mean_depth_m": round(float(np.mean(depths)), 2) if depths else None,
            })

        return {
            "processed_at":    datetime.now().isoformat(),
            "total_records":   len(records),
            "potholes":        pot_summary,
            "tracked_objects": obj_summary,
        }

    @staticmethod
    def _print_summary(s: Dict):
        print("\n  ══ SUMMARY ══════════════════════════════════════════")
        print(f"  Total records : {s.get('total_records', 0)}")
        pot = s.get("potholes", {})
        if pot.get("total", 0):
            print(f"  Potholes      : {pot['total']}  "
                  f"area={pot.get('total_area_m2','?')}m²  "
                  f"vol={pot.get('total_volume_m3','?')}m³")
            sc = pot.get("severity_counts", {})
            print(f"  Severity      : Min={sc.get('Minimal',0)}  "
                  f"Low={sc.get('Low',0)}  "
                  f"Med={sc.get('Medium',0)}  "
                  f"High={sc.get('High',0)}")
        obj = s.get("tracked_objects", {})
        if obj.get("total_detections", 0):
            print(f"  MCMOT detects : {obj['total_detections']}")
            print(f"  Classes       : {obj.get('class_counts', {})}")
            print(f"  Risk counts   : {obj.get('risk_counts', {})}")
            print(f"  Mean depth    : {obj.get('mean_depth_m', '?')}m")
        print("  ════════════════════════════════════════════════════")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 16 – CALIBRATION HELPER
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
# SECTION 17 – CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified Transportation Intelligence Pipeline"
    )
    parser.add_argument("--mode",
                        choices=["calibrate", "image", "folder", "video"],
                        required=True)
    parser.add_argument("--input",           required=True,
                        help="Image/video/folder path")
    parser.add_argument("--calibration",     default="outputs/cal/calibration.npz")
    parser.add_argument("--sensor_csv",      default=None,
                        help="Combined IMU+GPS CSV")
    parser.add_argument("--pothole_model",   default="model/pothole_yolo.pt")
    parser.add_argument("--general_model",   default="yolov8s.pt",
                        help="YOLO model for MCMOT (COCO-trained)")
    parser.add_argument("--output_dir",      default="output")
    parser.add_argument("--output_video",    default=None)
    parser.add_argument("--frame_interval",  type=int, default=15)
    parser.add_argument("--initial_height",  type=float, default=1.2)
    parser.add_argument("--midas_model",     default="MiDaS_small",
                        choices=["MiDaS_small", "DPT_Hybrid", "DPT_Large"])
    parser.add_argument("--depth_every",     type=int, default=1,
                        help="Run MiDaS every N processed frames (>1 = faster)")
    parser.add_argument("--pothole_conf",    type=float, default=0.35)
    parser.add_argument("--general_conf",    type=float, default=0.30)
    # tracker is always bytetrack (resolved via absolute path automatically)
    parser.add_argument("--focal_mm",        type=float, default=4.0,
                        help="Camera focal length in mm (for MCMOT dim calc)")
    parser.add_argument("--sensor_mm",       type=float, default=5.6,
                        help="Camera sensor width in mm")
    parser.add_argument("--no_dual_view",    action="store_true",
                        help="Disable side-by-side depth panel in output video")
    parser.add_argument("--chessboard",      default="8,5",
                        help="[calibrate] inner corners WxH e.g. '8,5'")
    parser.add_argument("--square_size",     type=float, default=0.030,
                        help="[calibrate] chessboard square size in metres")
    args = parser.parse_args()

    # ── Calibration ──────────────────────────────────────────────────────────
    if args.mode == "calibrate":
        cb = tuple(int(x) for x in args.chessboard.split(","))
        ok = run_calibration(
            image_folder    = args.input,
            output_path     = args.calibration,
            chessboard_size = cb,
            square_size_m   = args.square_size,
        )
        print("  Calibration", "succeeded ✓" if ok else "FAILED ✗")
        exit(0 if ok else 1)

    # ── Processing ───────────────────────────────────────────────────────────
    pipe = UnifiedTransportationPipeline(
        calibration_path   = args.calibration,
        pothole_model_path = args.pothole_model,
        general_model_path = args.general_model,
        base_output_dir    = args.output_dir,
        midas_model_type   = args.midas_model,
        initial_height_m   = args.initial_height,
        pothole_conf       = args.pothole_conf,
        general_conf       = args.general_conf,
        focal_length_mm    = args.focal_mm,
        sensor_width_mm    = args.sensor_mm,
        depth_every_n      = args.depth_every,
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
            frame_interval = args.frame_interval,
            dual_view      = not args.no_dual_view,
        )

    # Pass input path to save_results so manifest records the source
    pipe.save_results(input_video=args.input if args.mode == "video" else None)