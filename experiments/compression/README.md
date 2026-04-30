# Edge-compression experiments

Sandbox for compressing **lightweight torchvision/timm backbones**
(`mobilenet_v3_small`, `efficientnet_b0`, …) under drone-style budgets.
Separate from the main face-recognition benchmark in `src/`.

## Files

- `model.py` — `EdgeEmbeddingModel`, `ArcFaceLoss`, `create_edge_backbone`,
  `apply_dynamic_quantization`, `apply_global_pruning`.
- `edge_tools.py` — `profile_model`, `compare_edge_variants`,
  `DRONE_PRESETS` (`hula_high`, `generic_edge`).
- `evaluate_edge.py` — CLI: profile one variant, or `--compare-variants` for
  baseline / quantized / pruned side-by-side.
- `throttled_benchmark.py` — runs `profile_model` for one preset/variant and
  writes a JSON summary.
- `summarize_edge_results.py` — turns a list of edge-compare JSON files into
  a markdown table.
- `adversarial_training.py` — FGSM / PGD adversarial fine-tuning of an
  `EdgeEmbeddingModel` (no external libs). Uses `torchvision.ImageFolder` if
  no project-specific dataset module is on the path.
- `COMPRESSION_SUMMARY.md` — write-up of past results and the deployment
  recommendation (`mobilenet_v3_small` + dynamic int8).

## Run from this folder

```bash
cd experiments/compression

# baseline vs quantized vs pruned (mobilenet_v3_small, hula_high preset)
python evaluate_edge.py --compare-variants --preset hula_high \
       --model mobilenet_v3_small --output edge_compare.json

# single variant
python throttled_benchmark.py --preset hula_high --variant quantized

# adversarial fine-tune (needs an ImageFolder-style dataset)
python adversarial_training.py --data-root /path/to/face/folders --attack pgd
```

These scripts target torchvision backbones, **not** the InsightFace / FaceNet
models in `src/benchmark_recognizers.py`. They share no runtime state with
the main benchmark.
