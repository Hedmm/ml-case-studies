"""Demo 3: export what the portfolio images need (runs after qc.py).

out/vis/trace_<dataset>_<ep>.json : per-frame leader/follower gripper, arm speed, grasp spans, trim range
out/vis/<dataset>_<ep>_<k>.jpg   : top-camera frames at key moments of that episode
out/vis/fail_<dataset>_<ep>.jpg  : middle frame of every FAIL episode
Usage: python export_vis.py svla_so100_sorting:45 svla_so100_pickplace:1
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from qc import ARM, grasp_spans

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out", "vis")
CAM = "observation.images.top"


def load(ds):
    p = os.path.join(ROOT, "data", ds)
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(p + "/meta/episodes/*/*.parquet"))]).set_index("episode_index")
    data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(p + "/data/*/*.parquet"))]).sort_values("index").reset_index(drop=True)
    return p, eps, data


def frame(p, e, t_rel, out):
    vp = f"{p}/videos/{CAM}/chunk-{int(e[f'videos/{CAM}/chunk_index']):03d}/file-{int(e[f'videos/{CAM}/file_index']):03d}.mp4"
    t = e[f"videos/{CAM}/from_timestamp"] + t_rel
    subprocess.run(["nice", "-n", "10", "ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", vp, "-frames:v", "1",
                    "-vf", "scale=480:360", "-q:v", "3", out], check=True)


os.makedirs(OUT, exist_ok=True)
for spec in sys.argv[1:]:
    ds, ep = spec.split(":")
    ep = int(ep)
    p, eps, data = load(ds)
    e = eps.loc[ep]
    x = data.iloc[int(e.dataset_from_index):int(e.dataset_to_index)]
    a, s = np.stack(x["action"].values), np.stack(x["observation.state"].values)
    fps = 30
    speed = np.r_[0, np.linalg.norm(np.diff(a[:, ARM], axis=0), axis=1) * fps]
    speed = np.convolve(speed, np.ones(9) / 9, mode="same")
    g = s[:, 5]
    grasps = grasp_spans(a[:, 5], g, fps)
    q = pd.read_csv(os.path.join(ROOT, "out", "qc_episodes.csv")).set_index(["dataset", "episode"]).loc[(ds, ep)]
    json.dump({"dataset": ds, "episode": ep, "fps": fps, "leader_gripper": a[:, 5].round(2).tolist(),
               "follower_gripper": g.round(2).tolist(), "arm_speed": speed.round(1).tolist(), "grasps": grasps,
               "trim": [int(q.trim_start), int(q.trim_end)], "verdict": q.verdict, "reasons": q.reasons if isinstance(q.reasons, str) else ""},
              open(os.path.join(OUT, f"trace_{ds}_{ep}.json"), "w"))
    # key moments: before motion, each grasp (middle), each release, end of motion
    moments = [("start", q.trim_start / fps)]
    for k, (b, en) in enumerate(grasps):
        moments += [(f"grasp {k + 1}", (b + en) / 2 / fps), (f"release {k + 1}", min(len(x) - 1, en + 8) / fps)]
    moments.append(("end", (q.trim_end - 1) / fps))
    for k, (label, t) in enumerate(moments):
        frame(p, e, t, os.path.join(OUT, f"{ds}_{ep}_{k}.jpg"))
    json.dump(moments, open(os.path.join(OUT, f"moments_{ds}_{ep}.json"), "w"))
    print(ds, ep, "grasps", grasps, "moments", [m[0] for m in moments], flush=True)

qc = pd.read_csv(os.path.join(ROOT, "out", "qc_episodes.csv"))
for r in qc[qc.verdict == "FAIL"].itertuples():
    p, eps, _ = load(r.dataset)
    frame(p, eps.loc[r.episode], r.seconds / 2, os.path.join(OUT, f"fail_{r.dataset}_{r.episode}.jpg"))
print("fail frames", int((qc.verdict == "FAIL").sum()))
