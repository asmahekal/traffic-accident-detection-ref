"""
accident_system.py -- Real-time accident detection, confirmation and alerting
(upgraded version of the original GUI program; same model, same interface).

What it does (and what the manuscript describes):
  1. Detection      : YOLO model (Dataset 1 weights: road users + accident types).
  2. Tracking       : ByteTrack (model.track, persist=True) -> persistent track IDs.
  3. Confirmation   : an accident is CONFIRMED only when a candidate persists for
                      CONFIRM_FRAMES consecutive analysed frames for the same track
                      identities. Candidates are
                        (a) a tracked accident-class box with confidence >= CONF, or
                        (b) two tracked road users whose boxes overlap with IoU >= IOU_THR
                            (can be switched off with "use_overlap_rule": false).
  4. Alerting       : ONE e-mail per confirmed event (cooldown), sent in a separate
                      thread with bounded retries; video analysis never waits for SMTP.
  5. Robust input   : live camera read in its own thread with a bounded buffer
                      (only the newest frames), read timeout and reconnection.
  6. Measurement    : every alert writes a row to alerts_log.csv with all latency stages
                      (onset -> confirmed detection -> alert prepared -> SMTP accepted
                      -> inbox receipt).

Run modes
  python accident_system.py                       # GUI (image / video / live camera)
  python accident_system.py batch  VIDEO_FOLDER   # process all videos, cache tracks, log alerts
  python accident_system.py sweep                 # sensitivity of CONFIRM_FRAMES and IOU_THR
                                                    (uses cached tracks, no re-detection)

Settings are read from config.json next to this file (see the template).
The e-mail password is read from the environment variable ACC_SENDER_PASSWORD,
never stored in the code.
"""
import csv
import json
import os
import queue
import smtplib
import ssl
import sys
import threading
import time
import uuid
from email.message import EmailMessage

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ configuration
DEFAULTS = {
    "model_path": "deteccion_accidentes_YOLOv12.pt",
    "conf": 0.19,                    # detection confidence threshold (paper: tau_c)
    "confirm_frames": 5,             # N consecutive analysed frames to confirm an accident
    "iou_thr": 0.30,                 # overlap of two road users counted as a collision candidate
    "use_overlap_rule": True,        # rule (b) on/off
    "road_user_classes": ["car", "bike", "person"],
    "frame_skip": 1,                 # 1 = analyse every frame
    "alert_cooldown_s": 30,          # one alert per event: no new alert within this time
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 465,
    "smtp_timeout_s": 15,
    "smtp_max_retries": 3,
    "sender_email": "",
    "recipients": [],
    "measure_inbox": True,           # poll the sender's own inbox (IMAP) to time inbox receipt
    "imap_host": "imap.gmail.com",
    "inbox_wait_max_s": 120,
    "camera_index": 0,
    "camera_lat": None,              # fixed coordinates of the camera (recommended)
    "camera_lon": None,
    "read_timeout_s": 5,
    "reconnect_attempts": 3,
    "display": True,
    "background_image": "",
    "output_dir": "results",
}


def load_config():
    cfg = dict(DEFAULTS)
    path = os.environ.get("ACC_CONFIG") or os.path.join(HERE, "config.json")
    if os.path.exists(path):
        cfg.update(json.load(open(path, encoding="utf-8")))
    cfg["sender_password"] = os.environ.get("ACC_SENDER_PASSWORD", "")
    os.makedirs(os.path.join(HERE, cfg["output_dir"]), exist_ok=True)
    return cfg


CFG = load_config()
OUT = os.path.join(HERE, CFG["output_dir"])


# ------------------------------------------------------------------ geometry
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


# ------------------------------------------------------------------ location
def get_location():
    if CFG["camera_lat"] is not None and CFG["camera_lon"] is not None:
        return CFG["camera_lat"], CFG["camera_lon"], "camera coordinates"
    try:
        import requests
        d = requests.get("http://ip-api.com/json", timeout=3).json()
        if d.get("lat") and d.get("lon"):
            return d["lat"], d["lon"], "approximate (IP-based)"
    except Exception:
        pass
    return 30.0444, 31.2357, "default"


# ------------------------------------------------------------------ alerting thread
class AlertWorker(threading.Thread):
    """Sends alerts in the background. The analysis loop only puts a job in the queue."""
    FIELDS = ["alert_id", "source", "event_key", "onset_s", "first_candidate_s", "confirm_s",
              "t_confirm_wall", "t_prepared_wall", "t_smtp_wall", "t_inbox_wall",
              "onset_to_confirm_s", "confirm_to_prepared_s", "prepared_to_smtp_s", "smtp_to_inbox_s",
              "onset_to_smtp_s", "onset_to_inbox_s", "smtp_attempts", "status", "location_source"]

    def __init__(self, send_email=True):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=10)
        self.pending = 0
        self.send_email = send_email
        self.log = os.path.join(OUT, "alerts_log.csv")
        if not os.path.exists(self.log):
            with open(self.log, "w", newline="") as f:
                csv.writer(f).writerow(self.FIELDS)
        self.start()

    def submit(self, frame, info):
        info["t_confirm_wall"] = time.time()
        try:
            self.q.put_nowait((frame.copy(), info)); self.pending += 1
        except queue.Full:
            print("Alert queue full: alert dropped")

    def run(self):
        while True:
            frame, info = self.q.get()
            try:
                self.process(frame, info)
            except Exception as e:
                print("Alert error:", e)
            finally:
                self.pending -= 1

    def process(self, frame, info):
        aid = uuid.uuid4().hex[:8]
        lat, lon, src = get_location()
        ok, jpg = cv2.imencode(".jpg", frame)
        msg = EmailMessage()
        msg["Subject"] = f"Emergency: Accident Detected [{aid}]"
        msg["From"] = CFG["sender_email"]
        to = list(CFG["recipients"])
        if CFG["measure_inbox"] and CFG["sender_email"] and CFG["sender_email"] not in to:
            to.append(CFG["sender_email"])           # copy to self, used to time inbox receipt
        msg["To"] = ", ".join(to)
        msg.set_content(f"Accident detected by camera '{info['source']}'.\n\n"
                        f"Location ({src}): https://www.google.com/maps?q={lat},{lon}\n"
                        f"Detection confidence: {info.get('conf', 0):.2f}\n\n"
                        "The snapshot from the detection system is attached.")
        msg.add_attachment(jpg.tobytes(), maintype="image", subtype="jpeg", filename="accident.jpg")
        t_prep = time.time()

        t_smtp, attempts, status = None, 0, "not sent"
        if self.send_email and CFG["sender_email"] and CFG["sender_password"] and CFG["recipients"]:
            for attempts in range(1, CFG["smtp_max_retries"] + 1):
                try:
                    ctx = ssl.create_default_context()
                    with smtplib.SMTP_SSL(CFG["smtp_host"], CFG["smtp_port"], context=ctx,
                                          timeout=CFG["smtp_timeout_s"]) as s:
                        s.login(CFG["sender_email"], CFG["sender_password"])
                        s.send_message(msg)
                    t_smtp, status = time.time(), "sent"
                    break
                except Exception as e:
                    status = f"failed: {e}"
                    time.sleep(min(2 * attempts, 5))
        elif self.send_email:
            status = "skipped: e-mail settings missing"

        t_inbox = self.wait_inbox(aid) if (t_smtp and CFG["measure_inbox"]) else None
        self.write(aid, info, t_prep, t_smtp, t_inbox, attempts, status, src)
        print(f"Alert {aid}: {status}")

    def wait_inbox(self, aid):
        import imaplib
        t0 = time.time()
        try:
            m = imaplib.IMAP4_SSL(CFG["imap_host"])
            m.login(CFG["sender_email"], CFG["sender_password"])
            while time.time() - t0 < CFG["inbox_wait_max_s"]:
                m.select("INBOX")
                typ, data = m.search(None, f'(SUBJECT "{aid}")')
                if typ == "OK" and data and data[0].split():
                    t = time.time(); m.logout(); return t
                time.sleep(0.25)
            m.logout()
        except Exception as e:
            print("Inbox timing failed:", e)
        return None

    def write(self, aid, info, t_prep, t_smtp, t_inbox, attempts, status, src):
        onset = info.get("onset_s")
        base = onset if onset is not None else info["first_candidate_s"]
        oc = info["confirm_s"] - base                       # in video/stream time
        r = lambda x: round(x, 3) if x is not None else ""
        row = {"alert_id": aid, "source": info["source"], "event_key": info["key"],
               "onset_s": r(onset), "first_candidate_s": r(info["first_candidate_s"]), "confirm_s": r(info["confirm_s"]),
               "t_confirm_wall": r(info["t_confirm_wall"]), "t_prepared_wall": r(t_prep),
               "t_smtp_wall": r(t_smtp), "t_inbox_wall": r(t_inbox),
               "onset_to_confirm_s": r(oc),
               "confirm_to_prepared_s": r(t_prep - info["t_confirm_wall"]),
               "prepared_to_smtp_s": r(t_smtp - t_prep) if t_smtp else "",
               "smtp_to_inbox_s": r(t_inbox - t_smtp) if (t_inbox and t_smtp) else "",
               "onset_to_smtp_s": r(oc + t_smtp - info["t_confirm_wall"]) if t_smtp else "",
               "onset_to_inbox_s": r(oc + t_inbox - info["t_confirm_wall"]) if t_inbox else "",
               "smtp_attempts": attempts, "status": status, "location_source": src}
        with open(self.log, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)


# ------------------------------------------------------------------ model helpers
_model = None


def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        p = CFG["model_path"] if os.path.isabs(CFG["model_path"]) else os.path.join(HERE, CFG["model_path"])
        _model = YOLO(p)
    return _model


def new_confirmer(model, n=None, iou_thr=None, use_overlap=None):
    return AccidentConfirmer(model.names, CFG["conf"], n or CFG["confirm_frames"],
                             CFG["iou_thr"] if iou_thr is None else iou_thr,
                             CFG["use_overlap_rule"] if use_overlap is None else use_overlap,
                             CFG["road_user_classes"], CFG["alert_cooldown_s"])


def track_frame(model, frame):
    r = model.track(frame, conf=CFG["conf"], persist=True, tracker="bytetrack.yaml", verbose=False)[0]
    b = r.boxes
    ids = b.id.int().tolist() if b.id is not None else [None] * len(b)
    tracks = [(ids[i], int(b.cls[i]), float(b.conf[i]), [float(v) for v in b.xyxy[i].tolist()])
              for i in range(len(b))]
    return r, tracks


def show(win, img):
    if not CFG["display"]:
        return True
    h, w = img.shape[:2]
    cv2.imshow(win, cv2.resize(img, (600, int(600 * h / w))))
    return (cv2.waitKey(1) & 0xFF) != ord("q")


# ------------------------------------------------------------------ video / camera processing
def process_video(path, alerts, onset_s=None, cache_tracks=True):
    model = get_model()
    model.predictor = None                                   # fresh tracker state per video
    conf = new_confirmer(model)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    name = os.path.basename(path)
    cache, idx, t_start, n_alerts = [], 0, time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % CFG["frame_skip"] == 0:
            t = idx / fps                                    # video time of this frame
            r, tracks = track_frame(model, frame)
            cache.append({"i": idx, "t": round(t, 4), "tracks": tracks})
            ann = r.plot()
            ev = conf.update(t, tracks)
            if ev:
                n_alerts += 1
                cv2.putText(ann, "ACCIDENT CONFIRMED", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cf = max([c for _, cls, c, _ in tracks if cls in conf.acc_ids] or [0.0])
                alerts.submit(ann, {**ev, "source": name, "onset_s": onset_s, "conf": cf,
                                    "first_candidate_s": ev["first_candidate_t"], "confirm_s": ev["confirm_t"]})
            if not show("Accident Detection - Video", ann):
                break
        idx += 1
    cap.release()
    proc = time.time() - t_start
    analysed = len(cache)
    if cache_tracks:
        json.dump({"video": name, "fps": fps, "frames": idx, "frame_skip": CFG["frame_skip"],
                   "names": model.names, "frames_data": cache},
                  open(os.path.join(OUT, f"tracks_{os.path.splitext(name)[0]}.json"), "w"))
    print(f"{name}: {idx} frames, {analysed} analysed in {proc:.1f}s "
          f"({analysed / max(proc, 1e-9):.1f} FPS), {n_alerts} alert(s)")
    return {"video": name, "frames": idx, "fps": fps, "duration_s": idx / fps, "analysed": analysed,
            "processing_fps": analysed / max(proc, 1e-9), "alerts": n_alerts}


class CameraReader(threading.Thread):
    """Reads frames in its own thread; keeps only the newest frames (bounded buffer)."""
    def __init__(self, index):
        super().__init__(daemon=True)
        self.index, self.buf, self.stop_flag, self.ok = index, queue.Queue(maxsize=2), False, True
        self.cap = cv2.VideoCapture(index)
        self.start()

    def run(self):
        fails = 0
        while not self.stop_flag:
            ok, f = self.cap.read()
            if not ok:
                fails += 1
                if fails > CFG["reconnect_attempts"]:
                    self.ok = False; break
                time.sleep(1); self.cap.release(); self.cap = cv2.VideoCapture(self.index); continue
            fails = 0
            if self.buf.full():
                try: self.buf.get_nowait()
                except queue.Empty: pass
            self.buf.put((time.time(), f))
        self.cap.release()

    def read(self):
        try:
            return self.buf.get(timeout=CFG["read_timeout_s"])
        except queue.Empty:
            return None, None


def live_camera(alerts):
    model = get_model()
    model.predictor = None
    conf = new_confirmer(model)
    cam = CameraReader(CFG["camera_index"])
    t0 = None
    while True:
        tw, frame = cam.read()
        if frame is None:
            print("Camera: no frame within the read timeout" + ("" if cam.ok else " (camera lost)"))
            if not cam.ok:
                break
            continue
        t0 = t0 or tw
        r, tracks = track_frame(model, frame)
        ann = r.plot()
        ev = conf.update(tw - t0, tracks)
        if ev:
            cv2.putText(ann, "ACCIDENT CONFIRMED", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            alerts.submit(ann, {**ev, "source": f"camera {CFG['camera_index']}", "onset_s": None,
                                "first_candidate_s": ev["first_candidate_t"], "confirm_s": ev["confirm_t"]})
        if not show("Accident Detection - Live", ann):
            break
    cam.stop_flag = True
    cv2.destroyAllWindows()


def analyze_image(path, alerts):
    model = get_model()
    img = cv2.imread(path)
    r = model.predict(img, conf=CFG["conf"], verbose=False)[0]
    ann = r.plot()
    acc = [float(c) for cls, c in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())
           if "accident" in str(model.names[int(cls)]).lower()]
    if acc:                                               # one alert per image
        cv2.putText(ann, "Accident Detected", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        alerts.submit(ann, {"key": "image", "source": os.path.basename(path), "onset_s": None, "conf": max(acc),
                            "first_candidate_s": 0.0, "confirm_s": 0.0})
    if CFG["display"]:
        show("Accident Detection - Image", ann); cv2.waitKey(0); cv2.destroyAllWindows()


# ------------------------------------------------------------------ batch + sweep (for the paper)
def read_onsets(path):
    """onsets.csv: video,onset_s   (onset_s empty or 'none' for videos WITHOUT an accident)"""
    d = {}
    if path and os.path.exists(path):
        for row in csv.DictReader(open(path, encoding="utf-8")):
            v = (row.get("onset_s") or "").strip().lower()
            d[row["video"].strip()] = None if v in ("", "none", "-") else float(v)
    return d


def batch(folder, onsets_csv=None, send_email=True):
    CFG["display"] = False
    onsets = read_onsets(onsets_csv or os.path.join(HERE, "onsets.csv"))
    alerts = AlertWorker(send_email=send_email)
    vids = sorted(f for f in os.listdir(folder) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")))
    rows = [process_video(os.path.join(folder, v), alerts, onsets.get(v)) for v in vids]
    t_wait = time.time()
    while alerts.pending > 0 and time.time() - t_wait < 600:            # let the last alerts finish
        time.sleep(1)
    with open(os.path.join(OUT, "videos_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "settings_used.json"), "w") as f:
        json.dump({k: v for k, v in CFG.items() if "password" not in k}, f, indent=2)
    print("Done. See", OUT)


def sweep(onsets_csv=None, ns=(1, 3, 5, 8, 10, 15), ious=(0.1, 0.2, 0.3, 0.5), early_tol_s=1.0, window_s=10.0):
    """Re-applies the confirmation rule to cached tracks for many (N, IoU) settings.
    For every accident video (onset annotated):
      * detection : a confirmation (of any candidate) in [onset - early_tol_s, onset + window_s];
                    delay = first such confirmation - onset (>= 0)
      * deployed  : the FIRST alert of the deployed system (with the one-alert-per-event cooldown)
                    falls in that window
      * false alerts : confirmations before onset - early_tol_s (accident-free footage),
                    counted with the deployed cooldown; rate per hour of accident-free footage
    Videos without an accident (onset 'none') contribute only false alerts."""
    onsets = read_onsets(onsets_csv or os.path.join(HERE, "onsets.csv"))
    caches = [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.startswith("tracks_")]
    if not caches:
        print("No cached tracks: run 'batch' first."); return
    settings = [(n, None, False) for n in ns] + [(n, i, True) for n in ns for i in ious]
    out = []
    for n, i, ovl in settings:
        n_acc = det = det_first = vids_fa = fa = 0
        delays, clean_s = [], 0.0
        for c in caches:
            names = {int(k): v for k, v in c["names"].items()}
            dur = c["frames"] / c["fps"]
            onset = onsets.get(c["video"])
            free_end = dur if onset is None else max(0.0, min(dur, onset - early_tol_s))
            clean_s += free_end
            ac0 = AccidentConfirmer(names, CFG["conf"], n, i or 0.0, ovl, CFG["road_user_classes"], 0)
            acd = AccidentConfirmer(names, CFG["conf"], n, i or 0.0, ovl, CFG["road_user_classes"], CFG["alert_cooldown_s"])
            all_c, dep = [], []
            for fr in c["frames_data"]:
                e0 = ac0.update(fr["t"], fr["tracks"])
                if e0: all_c.append(e0["confirm_t"])
                ed = acd.update(fr["t"], fr["tracks"])
                if ed: dep.append(ed["confirm_t"])
            # false alerts in accident-free footage (deployed behaviour, with cooldown)
            f_here = [t for t in dep if t < free_end] if onset is not None else dep
            fa += len(f_here); vids_fa += bool(f_here)
            if onset is None:
                continue
            n_acc += 1
            hits = [t for t in all_c if onset - early_tol_s <= t <= onset + window_s]
            if hits:
                det += 1; delays.append(max(0.0, hits[0] - onset))
            if dep and onset - early_tol_s <= dep[0] <= onset + window_s:
                det_first += 1
        h = clean_s / 3600
        out.append({"confirm_frames": n, "overlap_rule": ovl, "iou_thr": i if ovl else "",
                    "accident_videos": n_acc, "detected": det,
                    "detection_rate_%": round(100 * det / n_acc, 1) if n_acc else "",
                    "median_delay_s": round(float(np.median(delays)), 2) if delays else "",
                    "mean_delay_s": round(float(np.mean(delays)), 2) if delays else "",
                    "first_alert_correct_%": round(100 * det_first / n_acc, 1) if n_acc else "",
                    "videos_with_false_alert": vids_fa, "false_alerts": fa,
                    "accident_free_hours": round(h, 3),
                    "false_alerts_per_hour": round(fa / h, 1) if h else ""})
    with open(os.path.join(OUT, "sensitivity_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    for r in out:
        print(r)
    print("Saved", os.path.join(OUT, "sensitivity_sweep.csv"))


# ------------------------------------------------------------------ GUI
def main_window():
    import tkinter as tk
    from tkinter import filedialog
    alerts = AlertWorker()
    root = tk.Tk(); root.title("Accident Detection Interface"); root.geometry("500x300")
    bg = CFG.get("background_image")
    if bg and os.path.exists(bg):
        from PIL import Image, ImageTk
        photo = ImageTk.PhotoImage(Image.open(bg).resize((500, 300)))
        lbl = tk.Label(root, image=photo); lbl.image = photo; lbl.place(relwidth=1, relheight=1)
    tk.Label(root, text="Welcome to Our Accident Detection Project!", font=("Helvetica", 14, "bold"),
             bg="white").pack(pady=20)

    def pick(kind):
        root.withdraw()
        types = [("Image Files", "*.jpg;*.jpeg;*.png")] if kind == "image" else [("Video Files", "*.mp4;*.avi;*.mov")]
        p = filedialog.askopenfilename(filetypes=types)
        if p:
            analyze_image(p, alerts) if kind == "image" else process_video(p, alerts, cache_tracks=False)
            cv2.destroyAllWindows()
        root.deiconify()

    tk.Button(root, text="Upload Image", width=20, height=2, command=lambda: pick("image")).pack(pady=8)
    tk.Button(root, text="Upload Video", width=20, height=2, command=lambda: pick("video")).pack(pady=8)
    tk.Button(root, text="Use Live Camera", width=20, height=2, command=lambda: live_camera(alerts)).pack(pady=8)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "batch":
        batch(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "videos"),
              sys.argv[3] if len(sys.argv) > 3 else None,
              send_email="--no-email" not in sys.argv)
    elif len(sys.argv) >= 2 and sys.argv[1] == "sweep":
        sweep(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        main_window()
