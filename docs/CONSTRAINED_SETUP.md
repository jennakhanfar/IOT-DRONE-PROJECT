# Running Constrained Benchmarks

`src/drone_constraints.py` simulates the drone CPU/RAM budget around model
inference: a single core pinned to ~15% duty cycle (~400 MHz of a typical dev
laptop) and a 128 MB hard RAM cap on the inference delta. Both Windows and
macOS/Linux are supported; the RAM monitor is enforced via a watchdog thread.

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place datasets in the project folder:
#    archive (1)/      VGG Face2 subset
#    open_data_set/    DroneFace

# 4. InsightFace packs auto-download on first run to:
#    ~/.insightface/models/buffalo_sc/    (MobileFaceNet)
#    ~/.insightface/models/buffalo_l/     (ArcFace R50)
#    ~/.insightface/models/antelopev2/    (ArcFace R100)

# 5. For ArcFace R18, place weights at:
#    weights/arcface_r18.pth
#    Source: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
```

All commands below assume you run from the project root.

## Unconstrained (baseline)

```bash
python src/benchmark_recognizers.py --model mobilefacenet --dataset-root open_data_set --dataset-type droneface
python src/benchmark_recognizers.py --model sface         --dataset-root open_data_set --dataset-type droneface
python src/benchmark_recognizers.py --model arcface_r18   --dataset-root open_data_set --dataset-type droneface
python src/benchmark_recognizers.py --model facenet       --dataset-root open_data_set --dataset-type droneface
python src/benchmark_recognizers.py --model arcface_r50   --dataset-root open_data_set --dataset-type droneface
python src/benchmark_recognizers.py --model arcface_r100  --dataset-root open_data_set --dataset-type droneface
```

## Constrained (drone simulation)

Add `--constrained` to enforce 128 MB RAM + ~15% CPU on inference:

```bash
python src/benchmark_recognizers.py --model mobilefacenet --dataset-root open_data_set --dataset-type droneface --constrained
python src/benchmark_recognizers.py --model sface         --dataset-root open_data_set --dataset-type droneface --constrained
python src/benchmark_recognizers.py --model arcface_r18   --dataset-root open_data_set --dataset-type droneface --constrained
python src/benchmark_recognizers.py --model facenet       --dataset-root open_data_set --dataset-type droneface --constrained
python src/benchmark_recognizers.py --model arcface_r50   --dataset-root open_data_set --dataset-type droneface --constrained
python src/benchmark_recognizers.py --model arcface_r100  --dataset-root open_data_set --dataset-type droneface --constrained
```

CPU target can be overridden with `--cpu-mhz 250` (or 650, etc.) for ablation.
Larger models (R50, R100) may be killed by the RAM watchdog under 128 MB —
that's a valid recorded result.

## Sweep + aggregation

```bash
# Run the full plan (per-model x dataset x cpu_mhz). Skips finished JSONs.
python src/run_all_benchmarks.py --droneface-root open_data_set --vggface2-root archive

# Build per-table CSVs in benchmark_results/tables/
python src/aggregate_results.py
```

## View results

```bash
python src/show_results.py
```

Results land in `benchmark_results/` as JSON.
