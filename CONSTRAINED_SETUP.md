# Running Constrained Benchmarks (Windows)

On Windows, `drone_constraints.py` can enforce **both** RAM limit (128MB via Job Objects) and CPU throttle (15% of one core). macOS can only do CPU throttle.

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure datasets are in the project folder:
#    archive (1)/      — VGG Face2
#    open_data_set/     — DroneFace
#
# 4. Make sure InsightFace model packs are downloaded:
#    ~/.insightface/models/buffalo_sc/    (MobileFaceNet)
#    ~/.insightface/models/buffalo_l/     (ArcFace R50)
#    ~/.insightface/models/antelopev2/    (ArcFace R100)
#    If missing, the script auto-downloads them on first run.
#
# 5. For ArcFace R18: download backbone weights from
#    https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
#    Save as: weights/arcface_r18.pth
```

## Run unconstrained (baseline)

```bash
python benchmark_recognizers.py --model mobilefacenet --dataset-root open_data_set --dataset-type droneface
python benchmark_recognizers.py --model sface --dataset-root open_data_set --dataset-type droneface
python benchmark_recognizers.py --model arcface_r18 --dataset-root open_data_set --dataset-type droneface
python benchmark_recognizers.py --model facenet --dataset-root open_data_set --dataset-type droneface
python benchmark_recognizers.py --model arcface_r50 --dataset-root open_data_set --dataset-type droneface
python benchmark_recognizers.py --model arcface_r100 --dataset-root open_data_set --dataset-type droneface
```

## Run constrained (drone simulation)

Add `--constrained` to enforce 128MB RAM + 15% CPU:

```bash
python benchmark_recognizers.py --model mobilefacenet --dataset-root open_data_set --dataset-type droneface --constrained
python benchmark_recognizers.py --model sface --dataset-root open_data_set --dataset-type droneface --constrained
python benchmark_recognizers.py --model arcface_r18 --dataset-root open_data_set --dataset-type droneface --constrained
python benchmark_recognizers.py --model facenet --dataset-root open_data_set --dataset-type droneface --constrained
python benchmark_recognizers.py --model arcface_r50 --dataset-root open_data_set --dataset-type droneface --constrained
python benchmark_recognizers.py --model arcface_r100 --dataset-root open_data_set --dataset-type droneface --constrained
```

Bigger models (R50, R100) may crash with MemoryError under 128MB — that's expected and is a valid result.

## View results

```bash
python show_results.py
```

Results are saved as JSON in `benchmark_results/`.
