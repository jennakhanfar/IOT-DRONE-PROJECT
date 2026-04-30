# IoT Drone Face Recognition — Benchmarks

Benchmarks face-recognition models on a drone-style hardware budget.

The detector is held fixed; only the **recognizer** is swapped. Each model is
evaluated for accuracy, latency, model size, and feasibility under a 128 MB
RAM / single-core CPU throttle.

Datasets:
- **VGG Face2** — general face recognition reference.
- **DroneFace** — real drone footage with height and distance metadata.

## Layout

```
.
├── src/                            main benchmark pipeline
│   ├── benchmark_recognizers.py    run one model (or --all) on a dataset
│   ├── drone_constraints.py        128 MB RAM + CPU-throttle context
│   ├── run_all_benchmarks.py       full sweep orchestrator
│   ├── aggregate_results.py        per-table CSVs from result JSONs
│   ├── show_results.py             pretty summary tables
│   └── measure_baselines.py        per-model baseline RSS in fresh subprocesses
├── docs/CONSTRAINED_SETUP.md       setup + run reference
├── experiments/compression/        isolated edge-compression sandbox
│                                   (mobilenet/efficientnet, quant + prune)
├── benchmark_results/              JSON outputs land here
├── droneface_groups.csv            subject -> gender map for breakdowns
└── requirements.txt
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Datasets go in the project root: `archive (1)/` (VGG Face2) and
`open_data_set/` (DroneFace). Both are gitignored.

## Run

All commands assume project root as CWD.

```bash
# Single model on DroneFace
python src/benchmark_recognizers.py \
    --model mobilefacenet \
    --dataset-root open_data_set \
    --dataset-type droneface

# All models on VGG Face2
python src/benchmark_recognizers.py \
    --all \
    --dataset-root "archive (1)" \
    --dataset-type vggface2

# Drone simulation (128 MB RAM cap + ~15% of one CPU core)
python src/benchmark_recognizers.py \
    --model mobilefacenet \
    --dataset-root open_data_set \
    --dataset-type droneface \
    --constrained --cpu-mhz 400

# Full sweep (resumes — skips JSONs that already exist)
python src/run_all_benchmarks.py \
    --droneface-root open_data_set \
    --vggface2-root  "archive (1)"

# Print summary tables
python src/show_results.py

# Aggregate per-axis CSVs into benchmark_results/tables/
python src/aggregate_results.py
```

Under `--constrained`, accuracy is unchanged; only latency and RAM headroom
move. Heavier models may be killed by the RAM watchdog at 128 MB — that is a
recorded, valid result.

## Models

| Name            | Architecture / training data            | Size   | Notes                                     |
|-----------------|-----------------------------------------|--------|-------------------------------------------|
| `sface`         | SFace, OpenCV Zoo                       | ~1.1 M | Auto-downloads ONNX                       |
| `mobilefacenet` | MobileFaceNet, WebFace600K              | ~1.2 M | InsightFace `buffalo_sc`, auto-downloads  |
| `arcface_r18`   | ArcFace IResNet-18, MS1MV2              | ~12 M  | Needs `weights/arcface_r18.pth` manually  |
| `facenet`       | InceptionResnetV1, VGGFace2             | ~23 M  | `facenet-pytorch`                         |
| `facenet_casia` | InceptionResnetV1, CASIA-WebFace        | ~23 M  | `facenet-pytorch`                         |
| `arcface_r50`   | ArcFace ResNet-50, WebFace600K          | ~43 M  | InsightFace `buffalo_l`, auto-downloads   |
| `arcface_r100`  | ArcFace GlintR100, Glint360K            | ~65 M  | InsightFace `antelopev2`, auto-downloads  |

For ArcFace R18, download a pretrained checkpoint from
[`insightface/arcface_torch`](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)
and save it as `weights/arcface_r18.pth`. Without it, the model loads with
random weights — latency and size are still valid, but accuracy is meaningless.

## Output

Each per-model run produces
`benchmark_results/bench_<model>_<dataset>_<tag>.json`. A combined
`benchmark_combined_…json` is also written when `--all` is used. The
DroneFace runs include accuracy breakdowns by height, distance, and (via
`droneface_groups.csv`) gender.

## Edge-compression sandbox

`experiments/compression/` is a separate, self-contained workspace for
exploring quantization, pruning, and distillation on lightweight torchvision
backbones (`mobilenet_v3_small`, `efficientnet_b0`). It does **not** share
runtime state with the main benchmark. See
[`experiments/compression/README.md`](experiments/compression/README.md).

## Further reading

- [`docs/CONSTRAINED_SETUP.md`](docs/CONSTRAINED_SETUP.md) — full setup walkthrough including the constrained runs.
- [`experiments/compression/COMPRESSION_SUMMARY.md`](experiments/compression/COMPRESSION_SUMMARY.md) — historical notes on the compression experiments.
