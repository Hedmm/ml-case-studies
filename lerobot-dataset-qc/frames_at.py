"""Top-camera frames at given times: python frames_at.py <out_dir> dataset:episode:seconds ... (seconds from episode start;
'mid', 'start', 'end' are also accepted)."""
import glob
import os
import subprocess
import sys

import pandas as pd

CAM = "observation.images.top"
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for spec in sys.argv[2:]:
    ds, ep, t = spec.split(":")
    p = f"data/{ds}"
    e = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(p + "/meta/episodes/*/*.parquet"))]).set_index("episode_index").loc[int(ep)]
    dur = e.length / 30
    t = {"start": 0.3, "mid": dur / 2, "end": dur - 0.3}.get(t, None) if not t.replace(".", "").isdigit() else float(t)
    vp = f"{p}/videos/{CAM}/chunk-{int(e[f'videos/{CAM}/chunk_index']):03d}/file-{int(e[f'videos/{CAM}/file_index']):03d}.mp4"
    name = os.path.join(out, f"{ds}_{ep}_{spec.split(':')[2]}.jpg")
    subprocess.run(["nice", "-n", "10", "ffmpeg", "-v", "error", "-y", "-ss", f"{e[f'videos/{CAM}/from_timestamp'] + t:.3f}", "-i", vp,
                    "-frames:v", "1", "-vf", "scale=480:360", "-q:v", "3", name], check=True)
    print(name)
