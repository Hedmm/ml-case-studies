"""Demo 3: quality control for LeRobot v3 robot-demonstration datasets (SO-100 arm, leader-follower teleop).

Every episode of every dataset gets these checks and a verdict (FAIL > WARN > PASS):
  timing  - monotonic timestamps, frame interval = 1/fps (drops / duplicates), episode-length outliers
  signals - NaN/inf, follower steps faster than the servo can move, stalled joints, leader-follower tracking error
  motion  - idle time at the start/end (trim ranges), long pauses inside the episode
  task    - grasps: the follower gripper is held partly open by the object while the leader squeezes further;
            more grasps than the dataset's usual count = re-grasp (operator retry, worth a review)
  video   - frame count vs data rows, black / over-exposed frames, frozen camera while the arm moves, blur (both cameras)
Outputs (out/): qc_episodes.csv, qc_summary.json, keep_episodes.json (episodes to keep + trim ranges,
usable as LeRobotDataset(repo_id, episodes=[...]) plus a frame slice), video_frames.npz (per-frame video stats)
Usage (CPU only): python qc.py data/svla_so100_pickplace data/svla_so100_sorting data/svla_so100_stacking
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
ARM = slice(0, 5)  # shoulder_pan .. wrist_roll; index 5 is the gripper
SERVO_MAX_STEP = 12.0  # deg per frame at 30 fps = 360 deg/s, above the STS3215 servo's rated speed
IDLE_SPEED, MARGIN, PAUSE_MIN = 6.0, 15, 60  # deg/s; frames kept around the motion; pause = idle run >= 2 s
GRIP_GAP = 0.8  # follower minus leader gripper while holding: 1.2-4 deg on these datasets; a free gripper tracks within 0.2-0.4
STALL_DEG = 10.0  # follower still for >= 1 s while the leader moves this far; 5-7 deg is the normal servo deadband under load
VW, VH = 160, 120  # video analysed at reduced size, grayscale


def runs(mask):
    """(start, end) of runs of True values, end exclusive."""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def grasp_spans(leader, follower, fps):
    """Frames where an object is in the gripper: the follower is held at a steady, partly open value while the
    leader is squeezed further closed (an empty close goes down to ~0, an open hand tracks the leader).
    Spans split by a short blip with no opening in between count as one grasp."""
    std = pd.Series(follower).rolling(9, center=True, min_periods=1).std().values
    holding = (follower > 6) & (follower < 19) & (std < 0.3) & (follower - leader >= GRIP_GAP)
    spans = []
    for b, en in runs(holding):
        if spans and b - spans[-1][1] < fps and follower[spans[-1][1]:b].max() < follower[spans[-1][0]:spans[-1][1]].max() + 3:
            spans[-1] = (spans[-1][0], en)
        else:
            spans.append((b, en))
    return [(int(b), int(en)) for b, en in spans if en - b >= int(0.5 * fps)]


def robust_z(x):
    x = np.asarray(x, float)
    mad = np.median(np.abs(x - np.median(x))) or 1e-9
    return 0.6745 * (x - np.median(x)) / mad


def video_stats(path):
    """Mean brightness, mean abs difference to the previous frame and Laplacian variance for every frame."""
    cmd = ["nice", "-n", "10", "ffmpeg", "-v", "error", "-threads", "4", "-i", path,
           "-vf", f"scale={VW}:{VH},format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = VW * VH
    mean, diff, lap, prev = [], [], [], None
    while True:
        buf = p.stdout.read(n * 512)
        if not buf:
            break
        f = np.frombuffer(buf, np.uint8).reshape(-1, VH, VW).astype(np.float32)
        mean.append(f.mean((1, 2)))
        lp = 4 * f[:, 1:-1, 1:-1] - f[:, :-2, 1:-1] - f[:, 2:, 1:-1] - f[:, 1:-1, :-2] - f[:, 1:-1, 2:]
        lap.append(lp.var((1, 2)))
        d = np.abs(np.diff(np.concatenate([f[:1] if prev is None else prev[None], f]), axis=0)).mean((1, 2))
        if prev is None:
            d[0] = np.nan
        diff.append(d)
        prev = f[-1]
    p.wait()
    return np.concatenate(mean), np.concatenate(diff), np.concatenate(lap)


def check_dataset(path):
    name = os.path.basename(path.rstrip("/"))
    info = json.load(open(os.path.join(path, "meta", "info.json")))
    fps = info["fps"]
    cams = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(path, "meta/episodes/*/*.parquet")))])
    data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(path, "data/*/*.parquet")))])
    data = data.sort_values("index").reset_index(drop=True)

    vids = {}
    for cam in cams:
        for ch, fi in eps[[f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"]].drop_duplicates().values:
            vp = os.path.join(path, info["video_path"].format(video_key=cam, chunk_index=ch, file_index=fi))
            cache = os.path.join(OUT, "cache", f"{name}__{cam}__{ch}_{fi}.npy")
            if not os.path.exists(cache):
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                np.save(cache, np.stack(video_stats(vp)))
            vids[(cam, ch, fi)] = tuple(np.load(cache))
            print(name, cam, os.path.basename(vp), len(vids[(cam, ch, fi)][0]), "frames decoded", flush=True)
    lap_median = {cam: np.nanmedian(np.concatenate([v[2] for k, v in vids.items() if k[0] == cam])) for cam in cams}

    rows, frames = [], {}
    for _, e in eps.iterrows():
        ei, L = int(e.episode_index), int(e.length)
        x = data.iloc[int(e.dataset_from_index):int(e.dataset_to_index)]
        a, s = np.stack(x["action"].values), np.stack(x["observation.state"].values)
        ts = x["timestamp"].values.astype(np.float64)
        r = {"dataset": name, "episode": ei, "frames": L, "seconds": round(L / fps, 2)}
        fail, warn = [], []

        # timing
        dt = np.diff(ts)
        r["dropped_frames"] = int((dt > 1.5 / fps).sum())
        r["duplicate_frames"] = int((dt < 0.5 / fps).sum())
        if (dt <= 0).any() or r["dropped_frames"] or r["duplicate_frames"] or not np.array_equal(x.frame_index.values, np.arange(L)):
            fail.append("timing")

        # signals
        r["nan_values"] = int((~np.isfinite(a)).sum() + (~np.isfinite(s)).sum())
        if r["nan_values"]:
            fail.append("nan")
        step = np.abs(np.diff(s, axis=0))
        r["servo_speed_violations"] = int((step > SERVO_MAX_STEP).any(1).sum())
        if r["servo_speed_violations"]:
            warn.append("state jump")
        stalls = 0
        for j in range(5):
            for b, en in runs(step[:, j] < 0.01):
                if en - b >= fps and np.ptp(a[b:en + 1, j]) > STALL_DEG:
                    stalls += 1
        r["stalled_joint_runs"] = stalls
        if stalls:
            fail.append("stalled joint")
        r["tracking_err_p95"] = round(float(np.quantile(np.abs(a[:, ARM] - s[:, ARM]), 0.95)), 2)

        # motion
        speed = np.r_[0, np.linalg.norm(np.diff(a[:, ARM], axis=0), axis=1) * fps]
        speed = np.convolve(speed, np.ones(9) / 9, mode="same")
        moving = np.where(speed >= IDLE_SPEED)[0]
        first, last = (int(moving[0]), int(moving[-1])) if len(moving) else (0, L - 1)
        r["trim_start"], r["trim_end"] = max(0, first - MARGIN), min(L, last + MARGIN + 1)
        r["idle_start_s"], r["idle_end_s"] = round(first / fps, 2), round((L - 1 - last) / fps, 2)
        r["trimmable_frames"] = r["trim_start"] + (L - r["trim_end"])
        pauses = [(b, en) for b, en in runs(speed[first:last + 1] < IDLE_SPEED) if en - b >= PAUSE_MIN]
        r["pauses"], r["pause_s"] = len(pauses), round(sum(en - b for b, en in pauses) / fps, 2)
        if r["pause_s"] > 5:
            warn.append("long pause")

        # task
        grasps = grasp_spans(a[:, 5], s[:, 5], fps)
        r["grasps"], r["holding_s"] = len(grasps), round(sum(en - b for b, en in grasps) / fps, 2)
        if not grasps:
            fail.append("no grasp")

        # video, both cameras
        for cam in cams:
            c = cam.split(".")[-1]
            m, d, lp = vids[(cam, e[f"videos/{cam}/chunk_index"], e[f"videos/{cam}/file_index"])]
            f0 = int(round(e[f"videos/{cam}/from_timestamp"] * fps))
            f1 = int(round(e[f"videos/{cam}/to_timestamp"] * fps))
            m, d, lp = m[f0:f1], d[f0:f1], lp[f0:f1]
            r[f"{c}_frame_mismatch"] = int(abs((f1 - f0) - L)) if f1 <= len(vids[(cam, e[f"videos/{cam}/chunk_index"], e[f"videos/{cam}/file_index"])][0]) else L
            r[f"{c}_dark_or_overexposed"] = int(((m < 8) | (m > 247)).sum())
            n = min(len(d), L)
            frozen = [(b, en) for b, en in runs((d[:n] < 0.3) & (speed[:n] > 20)) if en - b >= 15]
            r[f"{c}_frozen_while_moving"] = len(frozen)
            r[f"{c}_blurry_pct"] = round(100 * float(np.mean(lp < 0.4 * lap_median[cam])), 1)
            if r[f"{c}_frame_mismatch"] > 1:
                fail.append(f"{c} video length")
            if r[f"{c}_dark_or_overexposed"] > 5:
                fail.append(f"{c} exposure")
            if frozen:
                fail.append(f"{c} frozen")
            if r[f"{c}_blurry_pct"] > 25:
                warn.append(f"{c} blur")
            frames[f"{name}/{ei}/{c}"] = np.stack([m, d, lp])

        r["_fail"], r["_warn"] = fail, warn
        rows.append(r)

    df = pd.DataFrame(rows)
    for col, label, two_sided in (("frames", "length outlier", True), ("tracking_err_p95", "tracking outlier", False)):
        z = robust_z(df[col])
        for i in np.where((np.abs(z) if two_sided else z) > 3.5)[0]:
            df.at[i, "_warn"] = df.at[i, "_warn"] + [label]
    expected = int(df.grasps[df.grasps > 0].mode().iloc[0])  # the task's usual number of grasps, from the data itself
    for i in np.where(df.grasps > expected)[0]:
        df.at[i, "_warn"] = df.at[i, "_warn"] + ["re-grasp"]
    df["expected_grasps"] = expected
    df["verdict"] = ["FAIL" if f else "WARN" if w else "PASS" for f, w in zip(df._fail, df._warn)]
    df["reasons"] = ["; ".join(f + w) for f, w in zip(df._fail, df._warn)]
    return df.drop(columns=["_fail", "_warn"]), frames, fps


def main():
    os.makedirs(OUT, exist_ok=True)
    dfs, frames, keep = [], {}, {}
    for path in sys.argv[1:]:
        df, fr, fps = check_dataset(path)
        dfs.append(df)
        frames.update(fr)
        keep[df.dataset.iloc[0]] = [{"episode": int(r.episode), "frames": [int(r.trim_start), int(r.trim_end)]}
                                    for r in df.itertuples() if r.verdict != "FAIL"]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(OUT, "qc_episodes.csv"), index=False)
    json.dump(keep, open(os.path.join(OUT, "keep_episodes.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "video_frames.npz"), **{k.replace("/", "__"): v for k, v in frames.items()})

    summary = {"episodes": int(len(df)), "frames": int(df.frames.sum()), "hours": round(df.frames.sum() / fps / 3600, 3),
               "verdicts": df.verdict.value_counts().to_dict(),
               "trimmable_frames": int(df.trimmable_frames.sum()),
               "trimmable_pct": round(100 * df.trimmable_frames.sum() / df.frames.sum(), 1),
               "per_dataset": {}}
    for ds, g in df.groupby("dataset"):
        reasons = pd.Series([x for r in g.reasons if r for x in r.split("; ")]).value_counts().to_dict()
        summary["per_dataset"][ds] = {"episodes": int(len(g)), "verdicts": g.verdict.value_counts().to_dict(),
                                      "reasons": reasons, "median_seconds": float(g.seconds.median()),
                                      "grasps_median": float(g.grasps.median()),
                                      "trimmable_pct": round(100 * g.trimmable_frames.sum() / g.frames.sum(), 1)}
    json.dump(summary, open(os.path.join(OUT, "qc_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
