"""
measure_baselines.py
--------------------
For each model, spawn a FRESH subprocess that:
  1. imports python + torch/onnxruntime
  2. loads the model via the factory
  3. measures RSS at that point (= baseline)

This is the exact baseline the DroneInferenceContext would have captured.
Saves results to benchmark_results/baselines.json so the aggregator can
compute an accurate delta for runs that didn't save it directly.

Usage:
    python measure_baselines.py
    python aggregate_results.py   # will now use measured baselines
"""

import json
import os
import subprocess
import sys
from pathlib import Path


MODELS = ["sface", "mobilefacenet", "arcface_r18", "facenet", "facenet_casia",
          "arcface_r50", "arcface_r100"]

OUT_PATH = Path("benchmark_results/baselines.json")


# Child process script: load one model, print baseline RSS in MB
CHILD_SCRIPT = """
import os, sys, psutil
sys.path.insert(0, os.getcwd())
from benchmark_recognizers import MODEL_REGISTRY
name = sys.argv[1]
factory, _ = MODEL_REGISTRY[name]
model = factory()
# Touch the model once so lazy inits finish (some ONNX backends defer until first run)
import torch, numpy as np
try:
    dummy = torch.zeros(1, 3, model.input_size, model.input_size)
    _ = model.get_embeddings(dummy)
except Exception as e:
    sys.stderr.write("warmup failed: %s\\n" % e)
rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
print("BASELINE_MB=%.2f" % rss_mb)
"""


def measure(model_name):
    proc = subprocess.Popen(
        [sys.executable, "-c", CHILD_SCRIPT, model_name],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    out, err = proc.communicate()
    out = out.decode("utf-8", "ignore")
    for line in out.splitlines():
        if line.startswith("BASELINE_MB="):
            return float(line.split("=", 1)[1])
    sys.stderr.write("  [warn] no baseline captured for %s\n%s\n" % (model_name, err.decode("utf-8", "ignore")))
    return None


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if OUT_PATH.exists():
        with open(str(OUT_PATH), "r") as f:
            results = json.load(f)

    print("Measuring baselines (one fresh subprocess per model)...")
    for name in MODELS:
        print("  [%s]" % name, end=" ", flush=True)
        baseline = measure(name)
        if baseline is not None:
            results[name] = round(baseline, 2)
            print("baseline = %.1f MB" % baseline)
        else:
            print("FAILED")

    with open(str(OUT_PATH), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: %s" % OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
