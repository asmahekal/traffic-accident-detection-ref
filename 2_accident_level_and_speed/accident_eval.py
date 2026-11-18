"""
accident_eval.py -- accident-level evaluation (what the alerting system needs)

All accident classes are merged into ONE class "accident"; road-user classes
(car, bike, person) are ignored. Two levels are reported for every model:

  1. Box level   : AP50, AP50-95 of the merged "accident" class, and P/R/F1 at a
                   confidence threshold chosen on the VALIDATION subset (max F1)
                   and then applied unchanged to the TEST subset.
  2. Image level : an image is "accident" if it contains at least one accident box.
                   The model flags an image if it predicts an accident with
                   confidence >= threshold (threshold chosen on validation, max F1).
                   -> P/R/F1/accuracy of "is there an accident in this frame?"

Usage (after the split exists in data_group_split/):
  python accident_eval.py --datasets deteccion_accidentes Accident
Weights are searched automatically (Kaggle inputs or local runs).
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import yaml

WORK = os.path.abspath(".")
SPLITS_ROOT = os.path.join(WORK, "data_group_split")
OUT = os.path.join(WORK, "accident_results")
MODELS = ["YOLOv5", "YOLOv8", "YOLOv9", "YOLOv11", "YOLOv12"]
IOUS = np.linspace(0.5, 0.95, 10)


# ------------------------------------------------------------------ metrics (pure numpy)
def iou_matrix(a, b):
    """a: (n,4) b: (m,4) xyxy -> (n,m)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def match_image(gt, pred_boxes, pred_conf):
    """Greedy matching by confidence; returns tp (n_pred, 10)."""
    order = np.argsort(-pred_conf)
    ious = iou_matrix(pred_boxes[order], gt)
    tp = np.zeros((len(pred_conf), len(IOUS)), bool)
    for k, t in enumerate(IOUS):
        used = np.zeros(len(gt), bool)
        for i in range(len(order)):
            if len(gt) == 0:
                break
            cand = np.where((ious[i] >= t) & ~used)[0]
            if len(cand):
                j = cand[np.argmax(ious[i, cand])]
                used[j] = True
                tp[order[i], k] = True
    return tp


def ap_101(recall, precision):
    """COCO 101-point AP: mean over r in {0,.01,..,1} of max precision at recall >= r."""
    env = np.flip(np.maximum.accumulate(np.flip(precision)))
    x = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, x, side="left")
    return float(np.mean([env[i] if i < len(env) else 0.0 for i in idx]))


def box_metrics(images):
    """images: list of dict(gt=(n,4), boxes=(m,4), conf=(m,))"""
    tps, confs, n_gt = [], [], 0
    for im in images:
        n_gt += len(im["gt"])
        if len(im["conf"]):
            tps.append(match_image(im["gt"], im["boxes"], im["conf"]))
            confs.append(im["conf"])
    if not confs or n_gt == 0:
        return None
    tp, conf = np.concatenate(tps), np.concatenate(confs)
    o = np.argsort(-conf)
    tp, conf = tp[o], conf[o]
    tpc = np.cumsum(tp, 0); fpc = np.cumsum(~tp, 0)
    rec = tpc / n_gt; prec = tpc / (tpc + fpc)
    ap = [ap_101(rec[:, k], prec[:, k]) for k in range(len(IOUS))]
    return {"AP50": ap[0], "AP50_95": float(np.mean(ap)), "conf": conf,
            "prec50": prec[:, 0], "rec50": rec[:, 0], "n_gt": n_gt}


def prf_at(bm, thr):
    k = np.searchsorted(-bm["conf"], -thr, side="right") - 1
    if k < 0:
        return 0.0, 0.0, 0.0
    p, r = bm["prec50"][k], bm["rec50"][k]
    return float(p), float(r), float(2 * p * r / (p + r + 1e-12))


def best_thr_box(bm):
    f1 = 2 * bm["prec50"] * bm["rec50"] / (bm["prec50"] + bm["rec50"] + 1e-12)
    return float(bm["conf"][int(np.argmax(f1))])


def image_level(images, thr):
    y = np.array([len(im["gt"]) > 0 for im in images])
    s = np.array([im["conf"].max() if len(im["conf"]) else 0.0 for im in images])
    yhat = s >= thr
    tp = int((y & yhat).sum()); fp = int((~y & yhat).sum()); fn = int((y & ~yhat).sum()); tn = int((~y & ~yhat).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r + 1e-12), "accuracy": (tp + tn) / len(y),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn, "positives": int(y.sum()), "negatives": int((~y).sum())}


def best_thr_image(images):
    cands = np.unique(np.round(np.concatenate([im["conf"] for im in images if len(im["conf"])] + [np.array([1.0])]), 3))
    best = max(cands, key=lambda t: image_level(images, t)["f1"])
    return float(best)


# ------------------------------------------------------------------ data
def accident_ids(names):
    ids = [i for i, n in names.items() if "accident" in str(n).lower()]
    return ids if ids else list(names)          # Accident dataset: every class is an accident (severity)


def load_gt(label_path, acc):
    out = []
    if os.path.exists(label_path):
        for line in open(label_path):
            p = line.split()
            if len(p) >= 5 and int(float(p[0])) in acc:
                x, y, w, h = map(float, p[1:5])
                out.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
    return np.array(out).reshape(-1, 4)


def collect(model, img_dir, acc):
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))
    images = []
    for k in range(0, len(files), 64):
        batch = [os.path.join(img_dir, f) for f in files[k:k + 64]]
        for f, r in zip(batch, model.predict(batch, conf=0.001, iou=0.7, max_det=300, imgsz=640, verbose=False)):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            keep = np.isin(cls, acc)
            lbl = os.path.join(os.path.dirname(img_dir), "labels", os.path.splitext(os.path.basename(f))[0] + ".txt")
            images.append({"gt": load_gt(lbl, acc), "boxes": r.boxes.xyxyn.cpu().numpy()[keep],
                           "conf": r.boxes.conf.cpu().numpy()[keep]})
    return images


def find_weights(ds, m):
    for p in [os.path.join("/kaggle/input", "**", f"{ds}_{m}.pt"),          # uploaded & renamed weights
              os.path.join(WORK, "**", "runs_group_split", ds, m, "weights", "best.pt"),
              os.path.join("/kaggle/input", "**", "runs_group_split", ds, m, "weights", "best.pt")]:
        hit = glob.glob(p, recursive=True)
        if hit:
            return hit[0]
    return None


def main(a):
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for ds in a.datasets:
        names = yaml.safe_load(open(os.path.join(SPLITS_ROOT, ds, "data.yaml")))["names"]
        names = names if isinstance(names, dict) else dict(enumerate(names))
        acc = accident_ids(names)
        print(ds, "accident classes:", [names[i] for i in acc])
        for m in a.models:
            w = find_weights(ds, m)
            if not w:
                print("  missing weights:", ds, m); continue
            model = YOLO(w)
            val = collect(model, os.path.join(SPLITS_ROOT, ds, "valid", "images"), acc)
            test = collect(model, os.path.join(SPLITS_ROOT, ds, "test", "images"), acc)
            bv, bt = box_metrics(val), box_metrics(test)
            tb = best_thr_box(bv); ti = best_thr_image(val)
            p, r, f = prf_at(bt, tb)
            il = image_level(test, ti)
            row = {"dataset": ds, "model": m,
                   "box_AP50": round(100 * bt["AP50"], 1), "box_AP50_95": round(100 * bt["AP50_95"], 1),
                   "box_thr(valid)": round(tb, 3), "box_P": round(100 * p, 1), "box_R": round(100 * r, 1), "box_F1": round(100 * f, 1),
                   "img_thr(valid)": round(ti, 3), "img_P": round(100 * il["precision"], 1), "img_R": round(100 * il["recall"], 1),
                   "img_F1": round(100 * il["f1"], 1), "img_acc": round(100 * il["accuracy"], 1),
                   "img_TP": il["TP"], "img_FP": il["FP"], "img_FN": il["FN"], "img_TN": il["TN"],
                   "test_accident_boxes": bt["n_gt"], "test_images_with_accident": il["positives"],
                   "test_images_without_accident": il["negatives"]}
            rows.append(row); print(row)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "accident_level.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["deteccion_accidentes", "Accident"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    main(ap.parse_args())
