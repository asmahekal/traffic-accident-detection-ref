# Real-time traffic accident detection, severity classification and e-mail alerting

Code, data splits and results for the article

> A. A. Hekal, M. A. Shawky, E. S. Abass, S. T. Shah, A. G. Abdellatif,
> **"YOLOv12-Based Real-Time Detection and Severity Classification of Traffic Accidents with Automated Emergency Alerting"**,
> *International Journal of Intelligent Systems* (under revision, manuscript 1862071).

**Authors**

| Author | ORCID |
|---|---|
| Asmaa A. Hekal | [0009-0007-7234-4696](https://orcid.org/0009-0007-7234-4696) |
| Mahmoud A. Shawky | [0000-0003-3393-8460](https://orcid.org/0000-0003-3393-8460) |
| Eman S. Abass | |
| Syed T. Shah | [0000-0003-4722-1786](https://orcid.org/0000-0003-4722-1786) |
| Ahmed Gamal Abdellatif | [0000-0002-3440-8448](https://orcid.org/0000-0002-3440-8448) |

Archived version: Zenodo, DOI **[NEW_DOI]**

The repository contains everything needed to reproduce every number in the article:
the leakage-free data split, the training and evaluation of five YOLO detectors, the
accident-level evaluation and speed benchmark, and the complete alerting system
(tracking, temporal confirmation, e-mail alerts) with its evaluation on real CCTV videos.

## Contents

| Folder | What it does | Article |
|---|---|---|
| `1_split_and_training/` | Class-stratified, group-aware split of both datasets; training and test evaluation of YOLOv5, v8, v9, v11 and v12 | Sections 3.1.3, 3.3, 4.1, 4.5; Tables 1, 2, 6, 7, 11 |
| `2_accident_level_and_speed/` | Accident-level evaluation (all accident classes merged) and speed/memory benchmark on an NVIDIA T4 | Sections 4.1, 4.4; Tables 8, 9 |
| `3_system/` | The complete system (GUI, video, live camera) and its evaluation on 150 real CCTV videos of the ACCIDENT benchmark | Sections 3.1, 4.4, 4.6; Tables 10, 12, 13 |
| `results/` | All result files produced by the runs reported in the article | all tables |

## Data (not included)

The datasets are not redistributed here. Download them from their sources:

| Name in the article | Source | License |
|---|---|---|
| Dataset 1 (7,630 images, 10 classes) | [deteccion_accidentes, version 2](https://universe.roboflow.com/deteccion-h92uo/deteccion_accidentes/dataset/2) | CC BY 4.0 |
| Dataset 2 (7,869 images; mild, moderate, severe) | [Accident, version 2](https://universe.roboflow.com/ann-qwhjm/accident-yqljb/dataset/2) | CC BY 4.0 |
| ACCIDENT benchmark (real CCTV clips) | [kaggle.com/datasets/picekl/accident](https://www.kaggle.com/datasets/picekl/accident) | terms set by its authors |

The notebooks download Datasets 1 and 2 with the Roboflow API. A free Roboflow API key is needed.

## Requirements

All experiments were run on [Kaggle](https://www.kaggle.com) notebooks with an NVIDIA T4 GPU.
Locally:

```bash
pip install -r requirements.txt
```

## Reproducing the results

The easiest way is to run the three Kaggle notebooks in order. Each notebook explains its
inputs and settings in its first cell.

**Credentials.** No password or key is stored in this repository. On Kaggle, add them under
*Add-ons → Secrets*:

| Secret | Needed for |
|---|---|
| `ROBOFLOW_API_KEY` | downloading Datasets 1 and 2 (steps 1 and 2) |
| `ACC_SENDER_EMAIL`, `ACC_SENDER_PASSWORD`, `ACC_RECIPIENT` | only the e-mail latency test in step 3 (use a Gmail *app password*, never your normal password) |

### Step 1: data split, training and evaluation

Notebook: `1_split_and_training/kaggle_split_train_evaluate.ipynb`
(add `reproduce_results_v2.py` and the folder `split_manifests/` as a Kaggle dataset input).

The same steps on the command line:

```bash
cd 1_split_and_training
# 1. split (deterministic, seed 42); must print "CLASS COVERAGE: OK"
python reproduce_results_v2.py split --dataset deteccion_accidentes \
       --src PATH_TO_ROBOFLOW_EXPORT --manifest split_manifests/deteccion_accidentes/split_manifest.csv
python reproduce_results_v2.py split --dataset Accident \
       --src PATH_TO_ROBOFLOW_EXPORT --manifest split_manifests/Accident/split_manifest.csv
# 2. training (AdamW, lr0 0.001, 150 epochs, patience 30, batch 16, 640x640, seed 42)
python reproduce_results_v2.py train --models YOLOv5 YOLOv8 YOLOv9 YOLOv11 YOLOv12 --device 0
# 3. evaluation on the test subsets -> paper_results/
python reproduce_results_v2.py evaluate
```

On Kaggle one model takes about 6-8 hours, so the notebook trains one model per session.

**How the split works.** `split_manifests/*.csv` assign every image to a provisional source
group: all augmented versions of one source image are always one group, and frames whose file
names indicate a common sequence are merged. Groups are assigned to train/validation/test
(70/15/15 %) as indivisible units, stratified by class, so that no source image and none of its
augmented versions appears in more than one subset. The resulting assignment of every image is
in `results/split/split_assignment_*.csv`; re-running the split reproduces these files exactly.

### Step 2: accident-level evaluation and speed benchmark

Notebook: `2_accident_level_and_speed/kaggle_accident_level_and_speed.ipynb`
(inputs: the step-1 files and the ten trained weight files).

```bash
python accident_eval.py --datasets deteccion_accidentes Accident   # Table 8
```

`speed_benchmark.py` is the last notebook cell (Table 9: 300 timed frames after warm-up,
640x640, batch 1, including ByteTrack tracking).

### Step 3: the complete system and its evaluation on real CCTV videos

Notebook: `3_system/kaggle_system_evaluation.ipynb`
(inputs: the ACCIDENT dataset and the YOLOv12 weights for Dataset 1).

It draws 150 real clips at random (seed 42), runs detection, tracking, temporal confirmation
and alerting on every frame, and writes:

* `videos_summary.csv` and `alerts_log.csv` for the deployed setting (Section 4.6),
* `sensitivity_sweep.csv` for N = 1, 3, 5, 8, 10, 15 and IoU 0.1-0.5 (Table 12),
* real e-mail alerts on 10 clips with every latency stage time-stamped (Table 10).

`conditions_analysis.py` is an extra notebook cell that breaks the results down by time of day,
weather, video quality, scene and collision type from the cached tracks (Table 13).

**Evaluation rule.** An accident counts as detected if the system confirms it between 1 s before
and 10 s after the annotated impact time. Every alert earlier than 1 s before impact is a
false alert, reported per hour of accident-free footage.

### Running the system yourself

```bash
cd 3_system
cp config.example.json config.json         # edit e-mails, camera coordinates, model path
export ACC_SENDER_PASSWORD="your-gmail-app-password"
python accident_system.py                  # GUI: image, video or live camera
python accident_system.py batch VIDEO_FOLDER onsets.csv --no-email   # batch evaluation
python accident_system.py sweep onsets.csv                           # sensitivity analysis
```

Deployed settings (Section 3.1): confidence threshold 0.19; an accident is confirmed when a
tracked accident-class box, or two tracked road users overlapping with IoU >= 0.30, persist for
5 consecutive frames; one alert per event (30 s cooldown); e-mail sent from a separate thread
with up to 3 SMTP attempts. The location in the alert is `camera_lat`/`camera_lon` from the
config or, if not set, an approximate location from the IP address of the computer.
`config.json` is listed in `.gitignore` so that real addresses are never committed.

## Result files and article tables

The file names in `results/detectors/` come from an earlier table numbering. They correspond to the article as follows:

| File | Article |
|---|---|
| `results/split/table1_split_*.csv`, `class_instances_per_subset_*.csv` | Tables 1 and 2 |
| `results/detectors/*/table5_overall.csv` | Table 6 (overall P, R, F1, mAP) |
| `results/detectors/*/table6_per_class.csv` | Table 7 (per-class mAP@50-95) |
| `results/detectors/*/table7_threshold_*.csv` | Table 11 (confidence threshold) |
| `results/detectors/*/figures/` | Figures 8-17 and 20-23 |
| `results/accident_level/accident_level.csv` | Table 8 |
| `results/speed_benchmark_table9.csv` | Table 9 |
| `results/system/alerts_log_latency.csv` | Table 10 |
| `results/system/sensitivity_sweep.csv` | Table 12 |
| `results/system/per_condition/per_condition_results.csv` | Table 13 |

`Accident` = Dataset 2 and `deteccion_accidentes` = Dataset 1.

## Main results

| | Dataset 1 | Dataset 2 |
|---|---|---|
| Test mAP@50-95, five detectors | 32.7-36.4 % | 34.1-39.1 % |

Complete system on 150 real CCTV videos (YOLOv12, N = 5, IoU 0.30): 57.3 % of accidents detected,
median delay 0.37 s after impact, alert in the inbox about 2.9 s after impact, false alert
before the accident in 53 of 150 videos (267 per hour of accident-free footage). See the article
for the full results and limitations.


## License

Code: MIT (see `LICENSE`). The datasets keep their own licenses and are not part of this repository.

## Citation

See `CITATION.cff`, or cite the article above.
