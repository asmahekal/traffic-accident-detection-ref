# ===== Per-condition analysis (day/night, weather, video quality) -- reviewer comment R1-9 =====
# Inputs: ACCIDENT dataset + the previous version of THIS notebook (system_eval) as input. No GPU needed.
import glob, os, sys, json, shutil, pandas as pd, numpy as np
WORK = "/kaggle/working"; os.chdir(WORK)
caches = glob.glob("/kaggle/input/**/results_eval/tracks_*.json", recursive=True)
metas = glob.glob("/kaggle/input/**/metadata-real.csv", recursive=True)
print(len(caches), "cached videos |", len(metas), "metadata file")
assert metas, "ACCIDENT dataset is not attached: Add Input > picekl/accident"
assert caches, "system_eval output is not attached: Add Input > Your Work > Notebook > system_eval"

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ------------------------------------------------------------------ temporal confirmation
class AccidentConfirmer:
    """Confirms an accident when a candidate persists for N consecutive analysed frames
    for the same track identities. Pure logic: no model, no I/O (unit-testable)."""

    def __init__(self, names, conf, confirm_frames, iou_thr, use_overlap, road_users, cooldown_s):
        self.names = names
        self.acc_ids = {i for i, n in names.items() if "accident" in str(n).lower()} or set(names)
        self.road_ids = {i for i, n in names.items() if n in road_users}
        self.conf, self.n, self.iou_thr = conf, confirm_frames, iou_thr
        self.use_overlap, self.cooldown_s = use_overlap, cooldown_s
        self.reset()

    def reset(self):
        self.run = {}            # key -> consecutive count
        self.first_t = {}        # key -> time (s) of first frame of the current run
        self.last_alert_t = None

    def candidates(self, tracks):
        keys = set()
        for tid, cls, cf, box in tracks:
            if cls in self.acc_ids and cf >= self.conf:
                keys.add(("acc", tid if tid is not None else f"c{cls}"))
        if self.use_overlap:
            road = [(tid, box) for tid, cls, cf, box in tracks
                    if cls in self.road_ids and tid is not None and cf >= self.conf]
            for i in range(len(road)):
                for j in range(i + 1, len(road)):
                    if iou(road[i][1], road[j][1]) >= self.iou_thr:
                        keys.add(("ovl",) + tuple(sorted((road[i][0], road[j][0]))))
        return keys

    def update(self, t, tracks):
        """t = time of the frame (s). Returns a confirmation dict or None."""
        keys = self.candidates(tracks)
        for k in list(self.run):
            if k not in keys:
                del self.run[k]; self.first_t.pop(k, None)
        confirmed = None
        for k in keys:
            if k not in self.run:
                self.run[k] = 0; self.first_t[k] = t
            self.run[k] += 1
            if self.run[k] == self.n and confirmed is None:
                in_cooldown = self.last_alert_t is not None and t - self.last_alert_t < self.cooldown_s
                if not in_cooldown:
                    confirmed = {"key": "/".join(map(str, k)), "first_candidate_t": self.first_t[k], "confirm_t": t}
                    self.last_alert_t = t
        return confirmed



class _A: pass
A = _A(); A.AccidentConfirmer = AccidentConfirmer

meta = pd.read_csv(glob.glob("/kaggle/input/**/metadata-real.csv", recursive=True)[0])
meta["video"] = meta["path"].map(lambda p: os.path.basename(str(p)))
onsets = pd.read_csv(glob.glob("/kaggle/input/**/eval_videos/onsets.csv", recursive=True)[0]) \
    if glob.glob("/kaggle/input/**/eval_videos/onsets.csv", recursive=True) else None

N, IOU, TOL, WIN, COOLDOWN = 5, 0.30, 1.0, 10.0, 30      # deployed setting
rows = []
for f in caches:
    c = json.load(open(f))
    names = {int(k): v for k, v in c["names"].items()}
    m = meta[meta.video == c["video"]]
    if m.empty:
        continue
    m = m.iloc[0]; onset = float(m["accident_time"]); dur = c["frames"] / c["fps"]
    a0 = A.AccidentConfirmer(names, 0.19, N, IOU, True, ["car", "bike", "person"], 0)
    ad = A.AccidentConfirmer(names, 0.19, N, IOU, True, ["car", "bike", "person"], COOLDOWN)
    allc, dep = [], []
    for fr in c["frames_data"]:
        e = a0.update(fr["t"], fr["tracks"]); e and allc.append(e["confirm_t"])
        e = ad.update(fr["t"], fr["tracks"]); e and dep.append(e["confirm_t"])
    hits = [t for t in allc if onset - TOL <= t <= onset + WIN]
    free = max(0.0, min(dur, onset - TOL))
    rows.append({"video": c["video"], "day_time": m.get("day_time"), "weather": m.get("weather"),
                 "quality": m.get("quality"), "scene_layout": m.get("scene_layout"), "type": m.get("type"),
                 "detected": bool(hits), "delay_s": max(0.0, hits[0] - onset) if hits else np.nan,
                 "false_alerts": sum(t < free for t in dep), "accident_free_s": free})
df = pd.DataFrame(rows); df.to_csv("per_video_results.csv", index=False)

def summary(by):
    g = df.groupby(by).agg(videos=("video", "count"), detected=("detected", "sum"),
                           median_delay_s=("delay_s", "median"), false_alerts=("false_alerts", "sum"),
                           free_h=("accident_free_s", lambda s: s.sum() / 3600)).reset_index()
    g["detection_rate_%"] = (100 * g.detected / g.videos).round(1)
    g["false_alerts_per_hour"] = (g.false_alerts / g.free_h).round(1)
    g["median_delay_s"] = g.median_delay_s.round(2)
    return g.drop(columns="free_h")

out = []
for by in ["day_time", "weather", "quality", "scene_layout", "type"]:
    s = summary(by); s.insert(0, "factor", by); s = s.rename(columns={by: "value"}); out.append(s)
    print(s.to_string(index=False), "\n")
res = pd.concat(out); res.to_csv("per_condition_results.csv", index=False)
os.makedirs("conditions_results", exist_ok=True)
for f in ["per_video_results.csv", "per_condition_results.csv"]:
    shutil.copy(f, "conditions_results/")
shutil.make_archive("conditions_results", "zip", "conditions_results")
print("Done -> conditions_results.zip")
