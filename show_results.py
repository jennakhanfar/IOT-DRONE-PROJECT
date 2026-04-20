#!/usr/bin/env python3
"""Print a complete summary of all benchmark results."""
import glob
import json
import os

RESULTS_DIR = "benchmark_results"
# Only the 5 face-recognition models we care about
MODELS = ["sface", "mobilefacenet", "arcface_r18", "facenet", "facenet_casia", "arcface_r50", "arcface_r100"]

def _candidate_paths(model, dataset, run_tag):
    return [
        os.path.join(RESULTS_DIR, "bench_%s_%s_%s.json" % (model, dataset, run_tag)),
        os.path.join(RESULTS_DIR, run_tag, "bench_%s_%s_%s.json" % (model, dataset, run_tag)),
    ]


def load(model, dataset, run_tag="unconstrained"):
    for path in _candidate_paths(model, dataset, run_tag):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def latest_constrained_tag():
    matches = glob.glob(os.path.join(RESULTS_DIR, "bench_*_*_docker_*mb_cpu*.json"))
    matches += glob.glob(os.path.join(RESULTS_DIR, "docker_*mb_cpu*", "bench_*_*_docker_*mb_cpu*.json"))
    if not matches:
        return "constrained"

    latest = max(matches, key=os.path.getmtime)
    name = os.path.basename(latest)
    for dataset in ("vggface2", "droneface"):
        needle = "_" + dataset + "_"
        if needle in name:
            return name.split(needle, 1)[1].rsplit(".json", 1)[0]
    return "constrained"

def fmt_acc(v):
    return "%.2f%%" % (v * 100) if v is not None else "—"

def fmt_ms(v):
    return "%.1f" % v if v is not None else "—"

def fmt_fps(v):
    return "%.1f" % v if v is not None else "—"

def fmt_size(v):
    return "%.0f" % v if v is not None else "—"

def fmt_params(v):
    if v is None:
        return "—"
    if v >= 1e6:
        return "%.1fM" % (v / 1e6)
    return "%.0fK" % (v / 1e3)

SEP = "=" * 95
THIN = "-" * 95

print()
print(SEP)
print("  FACE RECOGNITION BENCHMARK — FULL RESULTS")
print(SEP)

constrained_tag = latest_constrained_tag()
print("  Latest constrained tag detected: %s" % constrained_tag)

# ── VGG Face2 ──
print()
print("  Dataset: VGG Face2  (60 identities, 21 295 images)")
print(THIN)
print("  %-20s %8s %10s %10s %8s %10s %10s" % (
    "Model", "Acc", "Lat(ms)", "P95(ms)", "FPS", "Size(MB)", "Params"))
print(THIN)

for m in MODELS:
    r = load(m, "vggface2")
    if r is None:
        continue
    print("  %-20s %8s %10s %10s %8s %10s %10s" % (
        r["model_name"],
        fmt_acc(r.get("accuracy")),
        fmt_ms(r.get("avg_latency_ms")),
        fmt_ms(r.get("p95_latency_ms")),
        fmt_fps(r.get("fps")),
        fmt_size(r.get("model_size_mb")),
        fmt_params(r.get("parameter_count")),
    ))

print(SEP)

# ── DroneFace ──
print()
print("  Dataset: DroneFace  (11 identities, 1 364 images)")
print(THIN)
print("  %-20s %8s %10s %10s %8s %10s %10s" % (
    "Model", "Acc", "Lat(ms)", "P95(ms)", "FPS", "Size(MB)", "Params"))
print(THIN)

for m in MODELS:
    r = load(m, "droneface")
    if r is None:
        continue
    print("  %-20s %8s %10s %10s %8s %10s %10s" % (
        r["model_name"],
        fmt_acc(r.get("accuracy")),
        fmt_ms(r.get("avg_latency_ms")),
        fmt_ms(r.get("p95_latency_ms")),
        fmt_fps(r.get("fps")),
        fmt_size(r.get("model_size_mb")),
        fmt_params(r.get("parameter_count")),
    ))

print(SEP)

# ── DroneFace per-condition breakdown ──
print()
print("  DroneFace — Accuracy by Height")
print(THIN)
heights = ["1.5m", "3.0m", "4.0m", "5.0m"]
print("  %-20s %8s %8s %8s %8s" % ("Model", *heights))
print(THIN)
for m in MODELS:
    r = load(m, "droneface")
    if r is None or "conditions" not in r:
        continue
    by_h = r["conditions"].get("accuracy_by_height", {})
    vals = [fmt_acc(by_h.get(h)) for h in heights]
    print("  %-20s %8s %8s %8s %8s" % (r["model_name"], *vals))

print(SEP)

print()
print("  DroneFace — Accuracy by Distance")
print(THIN)
distances = ["2m", "4m", "6m", "8m", "10m", "12m", "14m", "16m"]
print("  %-20s %6s %6s %6s %6s %6s %6s %6s %6s" % ("Model", *distances))
print(THIN)
for m in MODELS:
    r = load(m, "droneface")
    if r is None or "conditions" not in r:
        continue
    by_d = r["conditions"].get("accuracy_by_distance", {})
    vals = [fmt_acc(by_d.get(d)) for d in distances]
    print("  %-20s %6s %6s %6s %6s %6s %6s %6s %6s" % (r["model_name"], *vals))

print(SEP)
print()
