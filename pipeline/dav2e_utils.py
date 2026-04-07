
# try:
#     from ultralytics import YOLO
#     _YOLO_AVAILABLE = True
# except ImportError:
#     _YOLO_AVAILABLE = False
#     print("[WARN] ultralytics not installed — YOLO detection disabled; using mock potholes")

# try:
#     import pandas as pd
#     _PANDAS_AVAILABLE = True
# except ImportError:
#     _PANDAS_AVAILABLE = False
#     print("[WARN] pandas not installed — CSV sensor loading disabled")

# try:
#     from scipy.ndimage import median_filter as _scipy_median_filter
#     _SCIPY_AVAILABLE = True
# except ImportError:
#     _SCIPY_AVAILABLE = False
#     def _scipy_median_filter(arr, size):
#         return arr

