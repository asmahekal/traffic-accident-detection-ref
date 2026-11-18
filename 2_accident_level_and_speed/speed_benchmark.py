# ===== Speed / memory benchmark (NVIDIA T4, 640x640, batch 1) =====
# In a notebook, first run:  !pip install -q ultralytics roboflow
import glob, os, time, cv2, numpy as np, pandas as pd, torch
from ultralytics import YOLO
from roboflow import Roboflow
from kaggle_secrets import UserSecretsClient

WORK = "/kaggle/working"; os.chdir(WORK)
rf = Roboflow(api_key=UserSecretsClient().get_secret("ROBOFLOW_API_KEY"))
src = rf.workspace("deteccion-h92uo").project("deteccion_accidentes").version(2) \
        .download("yolov8", location=f"{WORK}/raw/deteccion_accidentes").location
files = sorted(glob.glob(f"{src}/test/images/*"))[:320]
frames = [cv2.imread(f) for f in files]                 # disk I/O excluded from timing
print(len(frames), "frames, GPU:", torch.cuda.get_device_name(0))

def find_w(m):
    h = glob.glob(f"/kaggle/input/**/deteccion_accidentes_{m}.pt", recursive=True) or \
        glob.glob(f"/kaggle/input/**/runs_group_split/deteccion_accidentes/{m}/weights/best.pt", recursive=True)
    return h[0] if h else None

def sync(): torch.cuda.synchronize()
WARM, rows = 20, []
for m in ["YOLOv5", "YOLOv8", "YOLOv9", "YOLOv11", "YOLOv12"]:
    w = find_w(m)
    if not w: print("missing", m); continue
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
    det = YOLO(w)
    pre, inf, post, t_det = [], [], [], []
    for i, fr in enumerate(frames):                     # detection only
        sync(); t0 = time.perf_counter()
        r = det.predict(fr, imgsz=640, conf=0.19, device=0, verbose=False)[0]
        sync(); dt = (time.perf_counter() - t0) * 1000
        if i >= WARM:
            t_det.append(dt); pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
    trk = YOLO(w); t_trk = []
    for i, fr in enumerate(frames):                     # detection + ByteTrack (complete per-frame pipeline)
        sync(); t0 = time.perf_counter()
        trk.track(fr, imgsz=640, conf=0.19, device=0, persist=True, tracker="bytetrack.yaml", verbose=False)
        sync(); dt = (time.perf_counter() - t0) * 1000
        if i >= WARM: t_trk.append(dt)
    med, p95 = np.median, lambda x: np.percentile(x, 95)
    rows.append({"model": m,
        "preprocess_ms": round(med(pre), 1), "inference_med_ms": round(med(inf), 1), "inference_p95_ms": round(p95(inf), 1),
        "postprocess_ms": round(med(post), 1), "tracking_overhead_ms": round(med(t_trk) - med(t_det), 1),
        "pipeline_med_ms": round(med(t_trk), 1), "pipeline_p95_ms": round(p95(t_trk), 1),
        "FPS": round(1000 / np.mean(t_trk), 1), "peak_GPU_mem_GB": round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
        "params_M": round(sum(p.numel() for p in det.model.parameters()) / 1e6, 1), "frames_timed": len(t_trk)})
    print(rows[-1])
df = pd.DataFrame(rows); df.to_csv(f"{WORK}/speed_benchmark.csv", index=False)
print(df.to_string(index=False))
