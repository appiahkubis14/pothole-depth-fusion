
from datetime import datetime
import os
import time
import json
from typing import Dict, Optional
class RunManager:
    """
    Creates a timestamped project directory for each pipeline run.

    Layout
    ──────
    <base_output_dir>/
      run_YYYYMMDD_HHMMSS/
        annotated/          ← per-frame JPEGs (written async)
        data/
          dimensions.csv    ← all pothole/object dimension records
        manifest.json
    """

    def __init__(self, base_dir: str = "output"):
        ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id  = f"run_{ts}"
        self.run_dir = os.path.join(base_dir, self.run_id)

        self.annotated_dir = os.path.join(self.run_dir, "annotated")
        self.data_dir      = os.path.join(self.run_dir, "data")

        for d in (self.annotated_dir, self.data_dir):
            os.makedirs(d, exist_ok=True)

        self._settings:   Dict  = {}
        self._start_time: float = time.time()
        print(f"  [RUN] Project → {self.run_dir}")

    def set_settings(self, **kwargs):
        self.settings = {**self._settings, **kwargs,
                         "run_id":     self.run_id,
                         "started_at": datetime.now().isoformat()}

    def save_manifest(self, summary: Dict, video_path: Optional[str] = None):
        inventory = []
        for root, dirs, files in os.walk(self.run_dir):
            dirs.sort()
            for fname in sorted(files):
                full = os.path.join(root, fname)
                rel  = os.path.relpath(full, self.run_dir)
                sz   = os.path.getsize(full)
                inventory.append({"path": rel, "bytes": sz})

        manifest = {
            "run_id":       self.run_id,
            "started_at":   getattr(self, "settings", {}).get("started_at", ""),
            "finished_at":  datetime.now().isoformat(),
            "elapsed_s":    round(time.time() - self._start_time, 1),
            "settings":     getattr(self, "settings", {}),
            "input_video":  video_path or "",
            "summary":      summary,
            "output_files": inventory,
        }
        path = os.path.join(self.run_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"  [RUN] Manifest → {path}")
        print(f"\n  ── Run {self.run_id} output files {'─'*30}")
        for item in inventory:
            sz = item["bytes"]
            if   sz > 1_000_000: label = f"{sz/1_000_000:6.1f} MB"
            elif sz > 1_000:     label = f"{sz/1_000:6.1f} KB"
            else:                label = f"{sz:6d}  B"
            print(f"    {item['path']:<50} {label}")
        print(f"  {'─'*68}")
