<!-- # Step 1 – calibrate once
python unified_pothole_pipeline.py --mode calibrate \
  --input ./calib_images --chessboard 8,5 --square_size 0.030

# Step 2 – process video
python unified_pothole_pipeline.py --mode video --input '/home/mrtenkorang/Downloads/videos/306280.mp4' --sensor_csv 'data/imu&gps/2026-03-12_05-12-08_Tech-Adum_Road Conditions Assessment .csv' --yolo_model 'model/road_defects_detection_model.pt' --output_video 'outputs/dets/output_video out.mp4'




 -->


# Unified Transportation Intelligence Pipeline

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                UNIFIED TRANSPORTATION INTELLIGENCE PIPELINE                  │
├───────────────────────────┬─────────────────────────────────────────────────┤
│   SHARED INFRASTRUCTURE   │           PER-FRAME FLOW                        │
│                           │                                                 │
│  CameraCalibrator         │  raw_frame                                      │
│    └→ K, dist, cal_wh     │       │                                         │
│                           │       ▼                                         │
│  ImageUndistorter         │  ImageUndistorter.undistort()                   │
│    └→ remap arrays        │       │  new_K (scaled to frame size)           │
│                           │       ▼                                         │
│  IMUProcessor             │  IMU.height_at(t) + orientation_at(t)          │
│    └→ pitch/roll/height   │       │  cam_h, pitch, roll, yaw               │
│       per timestamp       │       ▼                                         │
│                           │  MiDaSDepthEstimator.estimate()  ◄──────────┐  │
│  MiDaSDepthEstimator      │       │  depth_metric [m]                    │  │
│    └→ shared by both      │       │  depth_viz [uint8]                   │  │
│       sub-systems         │       │  _last_inv  (raw, for MCMOT)         │  │
│                           │       │                                      │  │
│  GPSMapper                │       ├──────────────────────────────────────┘  │
│    └→ nearest(t)          │       │                                         │
│                           │  ┌────┴────────────────────────────────────┐   │
│  UnifiedLogger            │  │  SUB-PIPELINE A: POTHOLE                │   │
│    └→ unified_log.csv     │  │                                         │   │
│       unified_log.json    │  │  UnifiedDetector.detect_potholes()      │   │
│       potholes.geojson    │  │       │  bbox list                      │   │
│       track.gpx           │  │       ▼                                 │   │
│       summary.json        │  │  HomographyMapper.bbox_world_dims()     │   │
│                           │  │  DimensionFusionEngine.fuse()           │   │
│                           │  │    Strategy A: Homography               │   │
│                           │  │    Strategy B: Depth-anchor (MiDaS)     │   │
│                           │  │    Strategy C: Known-height + pinhole   │   │
│                           │  │       │  DimensionEstimate              │   │
│                           │  │       ▼                                 │   │
│                           │  │  classify_pothole_severity()            │   │
│                           │  │       │  severity: Minimal/Low/Med/High │   │
│                           │  │       ▼                                 │   │
│                           │  │  UnifiedLogger.add(road_event=pothole)  │   │
│                           │  └─────────────────────────────────────────┘   │
│                           │                                                 │
│                           │  ┌─────────────────────────────────────────┐   │
│                           │  │  SUB-PIPELINE B: MCMOT                  │   │
│                           │  │                                         │   │
│                           │  │  UnifiedDetector.track_objects()        │   │
│                           │  │    (ByteTrack / YOLOv8)                 │   │
│                           │  │       │  track_id, bbox, class, conf    │   │
│                           │  │       ▼                                 │   │
│                           │  │  ClassAwareTracker                      │   │
│                           │  │    .validated_class()  ← consistency    │   │
│                           │  │    .update()           ← velocity EMA   │   │
│                           │  │    .update_depth()     ← depth EMA      │   │
│                           │  │       │                                 │   │
│                           │  │       ▼                                 │   │
│                           │  │  RealWorldObjectDimCalculator           │   │
│                           │  │    real_size = px_size × depth / f_px   │   │
│                           │  │    blended with known-class prior       │   │
│                           │  │       │                                 │   │
│                           │  │       ▼                                 │   │
│                           │  │  RiskAssessor.assess()                  │   │
│                           │  │    risk = f(depth, class, size, quality)│   │
│                           │  │    SAFE / CAUTION / WARNING / CRITICAL  │   │
│                           │  │       │                                 │   │
│                           │  │       ▼                                 │   │
│                           │  │  Draw trajectory + bbox + risk badge    │   │
│                           │  │  UnifiedLogger.add(road_event=tracked)  │   │
│                           │  └─────────────────────────────────────────┘   │
│                           │                                                 │
│                           │  ┌─────────────────────────────────────────┐   │
│                           │  │  DUAL-VIEW VIDEO ASSEMBLY               │   │
│                           │  │                                         │   │
│                           │  │  [ Info bar: FPS | tracks | records ]   │   │
│                           │  │  [ Tracking vis + depth panel  ]        │   │
│                           │  │  [ Depth colourmap (full panel) ]       │   │
│                           │  └─────────────────────────────────────────┘   │
└───────────────────────────┴─────────────────────────────────────────────────┘
```

## Installation

```bash
pip install ultralytics torch torchvision opencv-python numpy pandas scipy
# MiDaS is loaded via torch.hub (internet required on first run)
```

## Modes

### 1. Calibrate camera
```bash
python unified_transportation_pipeline.py \
  --mode calibrate \
  --input /path/to/chessboard_images/ \
  --calibration outputs/cal/calibration.npz \
  --chessboard 8,5 \
  --square_size 0.030
```

### 2. Process single image
```bash
python unified_transportation_pipeline.py \
  --mode image \
  --input frame.jpg \
  --calibration outputs/cal/calibration.npz \
  --pothole_model model/pothole_yolo.pt \
  --general_model yolov8s.pt
```

### 3. Process folder of images
```bash
python unified_transportation_pipeline.py \
  --mode folder \
  --input /path/to/frames/ \
  --calibration outputs/cal/calibration.npz \
  --sensor_csv sensor_data.csv
```

### 4. Process video (full dual-view output)
```bash
python unified_transportation_pipeline.py \
  --mode video \
  --input data/videos/test1.mp4\
  --output_video out/annotated.mp4 \
  --calibration outputs/cal/calibration.npz \
  --sensor_csv 'data/imu&gps/2026-03-12_05-12-08_Tech-Adum_Road Conditions Assessment .csv' \
  --pothole_model 'model/road_defects_detection_model.pt' \
  --general_model yolov8s.pt \
  --frame_interval 11 \
  --midas_model MiDaS_small \
  --depth_every 1 \
  --focal_mm 4.0 \
  --sensor_mm 5.6
```

## Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--pothole_model` | `model/pothole_yolo.pt` | Custom YOLO for potholes |
| `--general_model` | `yolov8s.pt` | COCO YOLO for MCMOT tracking |
| `--frame_interval` | `15` | Process every Nth frame |
| `--depth_every` | `1` | Run MiDaS every N processed frames (>1 = faster) |
| `--initial_height` | `1.2` | Camera height prior in metres |
| `--focal_mm` | `4.0` | Smartphone focal length (mm) |
| `--sensor_mm` | `5.6` | Smartphone sensor width (mm) |
| `--no_dual_view` | off | Disable side-by-side video output |

## Sensor CSV Format

The `--sensor_csv` file should contain one row per IMU/GPS sample with columns:

```
Timestamp, AccelX, AccelY, AccelZ, GyroX, GyroY, GyroZ, Latitude, Longitude, GPSAltitude
```

Column names are flexible (case-insensitive, common aliases accepted).

## Output Files

| File | Contents |
|---|---|
| `output/unified_log.csv` | All detections (potholes + tracked objects) |
| `output/unified_log.json` | Same as CSV in JSON |
| `output/potholes.geojson` | GeoJSON FeatureCollection of potholes |
| `output/track.gpx` | GPX route from GPS data |
| `output/summary.json` | Aggregate statistics |
| `output/annotated/` | Per-frame annotated JPEGs |
| `output_video` | Dual-view annotated MP4 |

## Pothole Severity Thresholds

| Level | Area (m²) | Volume (m³) |
|---|---|---|
| Minimal | < 0.05 | < 0.001 |
| Low | 0.05–0.20 | 0.001–0.005 |
| Medium | 0.20–0.50 | 0.005–0.020 |
| High | > 0.50 | > 0.020 |

## MCMOT Risk Levels

| Level | Score threshold | Colour |
|---|---|---|
| SAFE | < 0.20 | Green |
| CAUTION | 0.20–0.40 | Yellow |
| WARNING | 0.40–0.70 | Orange |
| CRITICAL | > 0.70 | Red |

Risk score = 0.5×depth_risk + 0.3×class_importance + 0.2×size_risk, weighted by track quality.












# calibrate
python unified_pipeline.py --mode calibrate --input ./calib_images

# process video (MiDaS every 2nd processed frame for speed)
python unified_pipeline.py --mode video --input '/home/mrtenkorang/Downloads/videos/306225.mp4' \
  --camera_height 1.3 --frame_interval 10 --depth_every 2 \
  --gps_csv 'data/imu&gps/2026-03-12_05-12-08_Tech-Adum_Road Conditions Assessment .csv'

















  # Video
python depth_est_imp.py \
  --mode video \
  --input 'data/videos/5.webm' \
  --calibration 'outputs/cal/calibration.npz' \
  --pothole_model 'model/road_defects_detection_model.pt' \
  --output_video 'output/annotated_v1.mp4' \
  --sensor_csv 'data/imu&gps/2026-03-12_05-12-08_Tech-Adum_Road Conditions Assessment .csv' \
  --depth_every 1

# Single image
python depth_dimension_pipeline.py \
  --mode image --input frame.jpg \
  --calibration outputs/cal/calibration.npz \
  --pothole_model model/pothole_yolo.pt