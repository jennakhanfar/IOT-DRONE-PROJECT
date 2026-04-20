# Compression Summary

This file summarizes the edge-model compression experiments and drone deployment
readiness for the TA feedback section.

## Deployment target

The HULA drone (Ryze/DJI Tello-class, ~100g) has an estimated ARM-based SoC
running at 80-400MHz with 64-128MB of RAM. The project must run on Python 3.6.7
because that is the version supported by `pyhula`, the SDK used to send flight
commands.

## Final recommendation

**Deploy `mobilenet_v3_small` with dynamic int8 quantization.** Under the
`hula_high` preset it is the only variant flagged as `best_edge_candidate`, at
4.14MB and ~7 FPS on dev hardware. EfficientNet-B0 is ruled out on every preset
(too large, too slow). Pruning at 0.2 degrades latency without saving size, so
it is not carried forward. Distillation is prepared as a fallback if real-
hardware testing shows accuracy drops from quantization are unacceptable.

---

## Benchmark results (dev hardware, unthrottled)

Constraint preset: `generic_edge`

### mobilenet_v3_small

| Variant | Size (MB) | Avg Latency (ms) | FPS | RAM Budget | Recommendation |
|---|---:|---:|---:|---|---|
| baseline | 5.512 | 161.782 | 6.181 | True | usable_with_tradeoffs |
| quantized | 4.143 | 171.268 | 5.839 | True | usable_with_tradeoffs |
| pruned | 5.512 | 225.539 | 4.434 | True | too_heavy_for_strict_drone_budget |

### efficientnet_b0

| Variant | Size (MB) | Avg Latency (ms) | FPS | RAM Budget | Recommendation |
|---|---:|---:|---:|---|---|
| baseline | 23.108 | 582.278 | 1.717 | True | too_heavy_for_strict_drone_budget |
| quantized | 17.484 | 417.838 | 2.393 | True | too_heavy_for_strict_drone_budget |
| pruned | 23.108 | 481.794 | 2.076 | True | too_heavy_for_strict_drone_budget |

Constraint preset: `hula_high`

### mobilenet_v3_small

| Variant | Size (MB) | Avg Latency (ms) | FPS | RAM Budget | Recommendation |
|---|---:|---:|---:|---|---|
| baseline | 5.512 | 143.797 | 6.954 | True | usable_with_tradeoffs |
| quantized | 4.143 | 140.991 | 7.093 | True | best_edge_candidate |
| pruned | 5.512 | 142.051 | 7.04 | True | usable_with_tradeoffs |

> **Caveat on the "RAM Budget" column above.** The original profiler compared
> only the saved `.pth` file size to the RAM ceiling, which is a weak proxy.
> `edge_tools.profile_model` has been updated to report `runtime_footprint_mb`
> (weights + incremental RSS during inference) and to use that value for the
> budget check. The numbers above predate that fix; re-run the profiler to
> refresh them.

---

## Drone-constraint deployment checks

| Check | Status | Notes |
|---|---|---|
| Model size < 128MB (`hula_high` RAM) | PASS | Quantized MobileNet is 4.14MB. |
| Runtime footprint < 128MB | PENDING | Run `throttled_benchmark.py`; reports `runtime_footprint_mb`. |
| Single-core CPU + RAM cap + ~15% CPU simulation | PENDING | Enforced process-wide by `drone_constraints.apply_drone_constraints()` (called at the top of `throttled_benchmark.py`). Works on Windows/Mac/Linux. |
| Accuracy retained after quantization | PENDING | Compare baseline vs quantized on validation set. |
| K-fold mean accuracy on face dataset | PENDING | `sklearn.model_selection.KFold`, report mean +/- std. |
| Fine-tune on drone-face dataset | PENDING | After normal-face fine-tune, run a second fine-tune pass on drone data. |
| Edge-case evaluation | PENDING | Sun-glare, low-light, overhead-angle, low-contrast, darker-skin-in-low-light subsets. |
| Python 3.6.7 dependency freeze | DONE | See `requirements.txt` (pyhula 1.1.8, torch 1.9.0, timm 0.4.12). |

---

## How to reproduce

1. `python -m pip install -r requirements.txt` (in a Python 3.6.7 venv).
2. Generate throttled numbers:
   ```
   python throttled_benchmark.py --preset hula_high --variant quantized
   ```
   `drone_constraints.apply_drone_constraints()` is called at the top of the
   script, which caps the whole process at 128MB RAM and ~15% of one CPU core
   (pinned to core 0). The JSON output lands in
   `edge_compare_throttled_hula_high_quantized.json`.
3. To run any other script under the same drone simulation, add these two
   lines at the very top (before any heavy imports):
   ```python
   from drone_constraints import apply_drone_constraints
   apply_drone_constraints()
   ```
4. Replace the PENDING rows above with the values from the new JSON files.

---

## Known risks for the TA write-up

1. The dev-machine latency numbers (140-170ms) will degrade significantly on
   the real ARM SoC. Expect 3-10x slower end-to-end. Design the flight loop
   around pessimistic latency, not the dev numbers.
2. Dynamic quantization can drop accuracy on face embedding tasks. If the
   accuracy drop is larger than ~2%, the distillation path in
   `distillation_tools.py` is the intended next step.
3. The `pyhula` 3.6.7 pin forces older versions of torch/timm. Newer
   quantization APIs (e.g. static PTQ with `torch.ao.quantization`) are not
   reliably available in torch 1.9 and were not used.
