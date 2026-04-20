
# IoT Drone Face Recognition — Benchmarking

This project benchmarks multiple **face recognition models** under drone-like constraints.

- **Detector (UltraFace) is fixed**
- Only the **recognizer models** are compared
- Focus: **accuracy, latency, and deployability**

We evaluate models on:
- **VGG Face2** (general face recognition)
- **DroneFace** (real drone conditions with height/distance variations)

---

## 🚀 Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
````

### Run one model (DroneFace)

```bash
python benchmark_recognizers.py \
  --model mobilefacenet \
  --dataset-root open_data_set \
  --dataset-type droneface
```

### Run one model (VGG Face2)

```bash
python benchmark_recognizers.py \
  --model mobilefacenet \
  --dataset-root "archive (1)" \
  --dataset-type vggface2
```

### Run all models

```bash
python benchmark_recognizers.py \
  --all \
  --dataset-root open_data_set \
  --dataset-type droneface
```

---

## 🧪 Constrained Benchmarking

Simulates drone hardware limitations.

```bash
python benchmark_recognizers.py \
  --model mobilefacenet \
  --dataset-root open_data_set \
  --dataset-type droneface \
  --constrained
```

* **Mac:** CPU constraint only
* **Windows:** CPU + RAM constraint

> Accuracy does not change under constraints — only latency and feasibility.

---

## 📊 Models in the Benchmark

| Model         | Type                                     |
| ------------- | ---------------------------------------- |
| mobilefacenet | Lightweight (edge)                       |
| sface         | Lightweight (OpenCV)                     |
| facenet       | Medium                                   |
| facenet_casia | Medium                                   |
| arcface_r50   | Heavy                                    |
| arcface_r100  | Very heavy                               |
| arcface_r18   | Lightweight (optional, requires weights) |

---

## 🎯 Goal

The goal is to evaluate:

* Which models perform best on **real drone data**
* How performance changes under **domain shift** (VGG Face2 → DroneFace)
* Which models remain **deployable under hardware constraints**

Not all models are expected to run under constraints —
**failure to load or high latency is part of the evaluation.**

---

## 📁 Datasets

* `archive (1)/` — VGG Face2 subset
* `open_data_set/` — DroneFace dataset

DroneFace includes metadata for:

* **Height**
* **Distance**

The benchmark automatically reports accuracy breakdowns by these conditions.

---

## 📈 Results

Results are saved in:

```
benchmark_results/
```

Each run produces:

* JSON output
* Printed summary table

To display results:

```bash
python show_results.py
```

---

## 📂 Key Files

* `benchmark_recognizers.py` — main benchmarking pipeline
* `show_results.py` — prints summary tables
* `drone_constraints.py` — hardware constraint simulation
* `requirements.txt` — dependencies

---

## ⚠️ Notes

* **SFace** → auto-downloads, works immediately
* **MobileFaceNet / ArcFace models** → auto-download via InsightFace
* **ArcFace R18** → requires manual weights (`weights/arcface_r18.pth`)

---

## 💡 Summary

This project focuses on **real-world feasibility**, not just accuracy.

Lightweight models (e.g., MobileFaceNet, SFace) are more suitable for deployment,
while larger models serve as high-accuracy baselines.

```
```
