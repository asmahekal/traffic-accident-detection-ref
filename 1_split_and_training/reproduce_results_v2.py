"""
reproduce_results_v2.py  --  group-aware, class-stratified split + training + evaluation

Changes from v1:
  * split: groups are assigned so that EVERY class appears in train, valid and test
    (~70/15/15 of the instances of each class), while no source image / group is shared
    between subsets (no leakage). Groups that are too coarse for a rare class are
    subdivided into smaller contiguous chunks (reported in the output).
  * training: AdamW, lr0=0.001, 150 epochs, patience 30  (v1: Adam lr0=0.01, 100 epochs)
  * models: YOLOv5, YOLOv8, YOLOv9, YOLOv11, YOLOv12
  * weights are found wherever Ultralytics saved them (fixes the v1 "missing weights" bug)

Commands:
  python reproduce_results_v2.py split    --dataset NAME --src ROBOFLOW_EXPORT --manifest split_manifest.csv
  python reproduce_results_v2.py train    --datasets ... --models ... --device 0,1
  python reproduce_results_v2.py evaluate --datasets ... --models ...
"""
import argparse, csv, glob, json, os, random, re, shutil
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import yaml

WORK = os.path.abspath(".")
SPLITS_ROOT = os.path.join(WORK, "data_group_split")
RUNS_ROOT = os.path.join(WORK, "runs_group_split")
OUT = os.path.join(WORK, "paper_results")
DATASETS = ["deteccion_accidentes", "Accident"]          # manuscript Dataset 1, Dataset 2
MODELS = {"YOLOv5": "yolov5su.pt", "YOLOv8": "yolov8s.pt", "YOLOv9": "yolov9s.pt",
          "YOLOv11": "yolo11s.pt", "YOLOv12": "yolo12s.pt"}
TRAIN_ARGS = dict(imgsz=640, epochs=150, patience=30, batch=16, optimizer="AdamW",
                  lr0=0.001, momentum=0.9, seed=42, deterministic=True)
RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}
SUBSETS = list(RATIOS)
MIN_GROUPS_PER_CLASS = 6     # a class must be spread over >= this many groups to reach all subsets
CHUNK = 20                   # max size (distinct source images) of sub-blocks used when a group is subdivided
MIN_CHUNK = 5                # never cut blocks smaller than this
THRESHOLDS = [0.10, 0.19, 0.30, 0.50, 0.70]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SEED = 42


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def find_weights(ds, model):
    pats = [os.path.join(RUNS_ROOT, ds, model, "weights", "best.pt"),
            os.path.join(WORK, "runs", "**", "runs_group_split", ds, model, "weights", "best.pt"),
            os.path.join("/kaggle/input", "**", "runs_group_split", ds, model, "weights", "best.pt")]
    for p in pats:
        hit = glob.glob(p, recursive=True)
        if hit:
            return hit[0]
    return None


def data_yaml(ds):
    return os.path.join(SPLITS_ROOT, ds, "data.yaml")


# ============================================================ 1. stratified group split
def read_classes(label_file):
    cls = Counter()
    if os.path.exists(label_file):
        for line in open(label_file):
            parts = line.split()
            if parts:
                cls[int(float(parts[0]))] += 1
    return cls


def cmd_split(a):
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    for r in rows:
        r["cls"] = read_classes(os.path.join(a.src, r["label_path"]))
    names = yaml.safe_load(open(os.path.join(a.src, "data.yaml")))["names"]
    names = names if isinstance(names, dict) else dict(enumerate(names))

    # --- group -> rows ; subdivide groups that make a class too concentrated
    group_rows = defaultdict(list)
    for r in rows:
        group_rows[r["provisional_source_group"]].append(r)

    def class_groups():
        cg = defaultdict(set)
        for g, rs in group_rows.items():
            for r in rs:
                for c in r["cls"]:
                    cg[c].add(g)
        return cg

    subdivided = []
    img_cls = defaultdict(set)
    for r in rows:
        img_cls[r["original_image_group"]].update(r["cls"])
    imgs_per_class = Counter(c for cs in img_cls.values() for c in cs)
    for c in sorted(imgs_per_class, key=lambda c: imgs_per_class[c]):
        gs = class_groups()[c]
        if len(gs) >= MIN_GROUPS_PER_CLASS:
            continue
        size = max(MIN_CHUNK, min(CHUNK, -(-imgs_per_class[c] // MIN_GROUPS_PER_CLASS)))
        for g in sorted(gs):
            imgs = sorted({r["original_image_group"] for r in group_rows[g]}, key=natural_key)
            if len(imgs) <= size:
                continue
            rs = group_rows.pop(g)
            for k in range(0, len(imgs), size):
                chunk = set(imgs[k:k + size])
                group_rows[f"{g}#{size}:{k // size}"] = [r for r in rs if r["original_image_group"] in chunk]
            subdivided.append((names.get(c, c), g, len(imgs), size))

    # --- greedy iterative stratification at group level
    gstats = {g: (sum((r["cls"] for r in rs), Counter()), len(rs)) for g, rs in group_rows.items()}
    total_c = sum((s[0] for s in gstats.values()), Counter())
    total_n = sum(s[1] for s in gstats.values())
    have_c = {s: Counter() for s in SUBSETS}
    have_n = Counter()
    rng = random.Random(SEED)
    order = sorted(gstats, key=natural_key)
    rng.shuffle(order)
    order.sort(key=lambda g: (min((total_c[c] for c in gstats[g][0]), default=10**9), -gstats[g][1]))
    assign = {}
    for g in order:
        cc, n = gstats[g]
        best, best_score = None, None
        for s in SUBSETS:
            # deficit of each class in this subset, relative to its target; rare classes weigh more
            need = sum((RATIOS[s] * total_c[c] - have_c[s][c]) / (RATIOS[s] * total_c[c]) / np.log1p(total_c[c])
                       for c in cc) if cc else 0.0
            cap = (RATIOS[s] * total_n - have_n[s]) / (RATIOS[s] * total_n)
            score = (round(need, 9), cap)
            if best_score is None or score > best_score:
                best, best_score = s, score
        assign[g] = best
        have_c[best].update(cc)
        have_n[best] += n

    # --- write the split
    dst = os.path.join(SPLITS_ROOT, a.dataset)
    shutil.rmtree(dst, ignore_errors=True)
    for s in SUBSETS:
        os.makedirs(os.path.join(dst, s, "images"))
        os.makedirs(os.path.join(dst, s, "labels"))
    missing = 0
    out_rows = []
    for g in sorted(group_rows, key=natural_key):
        rs, s = group_rows[g], assign[g]
        for r in rs:
            stem = os.path.splitext(os.path.basename(r["label_path"]))[0]
            src_lbl = os.path.join(a.src, r["label_path"])
            img = [p for p in glob.glob(os.path.join(a.src, r["source_split"], "images", glob.escape(stem) + ".*"))
                   if p.lower().endswith(IMG_EXT)]
            if not img or not os.path.exists(src_lbl):
                missing += 1
                continue
            shutil.copy2(src_lbl, os.path.join(dst, s, "labels", stem + ".txt"))
            shutil.copy2(img[0], os.path.join(dst, s, "images", os.path.basename(img[0])))
            out_rows.append({"image": os.path.basename(img[0]), "source_image": r["original_image_group"],
                             "group": g, "subset": s})

    # --- leakage checks
    df = pd.DataFrame(out_rows)
    for key in ("source_image", "group"):
        bad = df.groupby(key)["subset"].nunique()
        assert (bad <= 1).all(), f"LEAKAGE: {key} in more than one subset: {list(bad[bad > 1].index[:5])}"

    yaml.safe_dump({"path": dst, "train": "train/images", "val": "valid/images", "test": "test/images",
                    "names": names}, open(os.path.join(dst, "data.yaml"), "w"))
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, f"split_assignment_{a.dataset}.csv"), index=False)
    t1 = pd.DataFrame([{"subset": s, "images": int((df.subset == s).sum()),
                        "distinct_source_images": df[df.subset == s].source_image.nunique(),
                        "groups": df[df.subset == s].group.nunique()} for s in SUBSETS])
    t1.to_csv(os.path.join(OUT, f"table1_split_{a.dataset}.csv"), index=False)
    cov = pd.DataFrame({s: {names.get(c, c): have_c[s][c] for c in sorted(total_c)} for s in SUBSETS})
    cov.to_csv(os.path.join(OUT, f"class_instances_per_subset_{a.dataset}.csv"))
    print(t1.to_string(index=False)); print("\nInstances per class and subset:\n", cov.to_string())
    if subdivided:
        print("\nSubdivided groups (class, group, #source images, new block size):", subdivided)
    empty = [(c, s) for s in SUBSETS for c, v in cov[s].items() if v == 0]
    print("\nCLASS COVERAGE:", "OK - every class is in every subset" if not empty else f"MISSING {empty}")
    if missing:
        print(f"WARNING: {missing} of {len(rows)} manifest entries not found in {a.src}")


# ============================================================ 2. train
def cmd_train(a):
    from ultralytics import YOLO
    for ds in a.datasets:
        for m in a.models:
            if find_weights(ds, m) and not a.force:
                print("skip (exists):", ds, m); continue
            YOLO(MODELS[m]).train(data=data_yaml(ds), project=os.path.join(RUNS_ROOT, ds), name=m,
                                  exist_ok=True, device=a.device, **TRAIN_ARGS)
            print("weights:", find_weights(ds, m))


# ============================================================ 3. evaluate
def cmd_evaluate(a):
    from ultralytics import YOLO
    figs = os.path.join(OUT, "figures")
    os.makedirs(figs, exist_ok=True)
    overall, per_class, best = [], [], {}
    for ds in a.datasets:
        for m in a.models:
            w = find_weights(ds, m)
            if not w:
                print("missing weights, skipped:", ds, m); continue
            r = YOLO(w).val(data=data_yaml(ds), split="test", imgsz=640, batch=16, plots=True,
                            project=figs, name=f"{ds}_{m}", exist_ok=True)
            b = r.box; p, rc = float(b.mp), float(b.mr)
            overall.append({"dataset": ds, "model": m, "precision": round(100 * p, 1), "recall": round(100 * rc, 1),
                            "f1": round(200 * p * rc / (p + rc + 1e-16), 1), "mAP50": round(100 * float(b.map50), 1),
                            "mAP50_95": round(100 * float(b.map), 1)})
            for i, c in enumerate(b.ap_class_index):
                per_class.append({"dataset": ds, "model": m, "class": r.names[int(c)],
                                  "mAP50_95": round(100 * float(b.all_ap[i].mean()), 1)})
            px = np.asarray(b.px)
            pc, rcv, fc = (np.asarray(x).mean(0) for x in (b.p_curve, b.r_curve, b.f1_curve))
            rows = [{"threshold": t, "precision": round(100 * pc[k], 1), "recall": round(100 * rcv[k], 1),
                     "f1": round(100 * fc[k], 1)} for t in THRESHOLDS for k in [int(np.argmin(abs(px - t)))]]
            pd.DataFrame(rows).to_csv(os.path.join(OUT, f"table7_threshold_{ds}_{m}.csv"), index=False)
            kb = int(np.argmax(fc))
            best[f"{ds}_{m}"] = {"best_conf": round(float(px[kb]), 3), "best_f1": round(100 * fc[kb], 1)}
            run_dir = os.path.dirname(os.path.dirname(w))
            dst = os.path.join(figs, f"{ds}_{m}")
            os.makedirs(dst, exist_ok=True)
            for f in ("results.png", "results.csv", "args.yaml"):
                if os.path.exists(os.path.join(run_dir, f)):
                    shutil.copy2(os.path.join(run_dir, f), os.path.join(dst, f))
    pd.DataFrame(overall).to_csv(os.path.join(OUT, "table5_overall.csv"), index=False)
    if per_class:
        (pd.DataFrame(per_class).pivot_table(index=["dataset", "class"], columns="model", values="mAP50_95")
           .to_csv(os.path.join(OUT, "table6_per_class.csv")))
    json.dump(best, open(os.path.join(OUT, "best_f1.json"), "w"), indent=2)
    print(pd.DataFrame(overall).to_string(index=False)); print(best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--dataset", required=True); s.add_argument("--src", required=True)
    s.add_argument("--manifest", required=True)
    t = sub.add_parser("train"); e = sub.add_parser("evaluate")
    for q in (t, e):
        q.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
        q.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    t.add_argument("--device", default="0"); t.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"split": cmd_split, "train": cmd_train, "evaluate": cmd_evaluate}[a.cmd](a)
