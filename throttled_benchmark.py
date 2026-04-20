"""
Throttled benchmark for drone deployment simulation.

Wraps edge_tools.profile_model with the process-wide constraints defined in
drone_constraints.py (128MB RAM hard limit, ~15% of one CPU core pinned to
core 0). This simulates the HULA drone's ARM Cortex-A ~400MHz SoC on a normal
dev machine across Windows, macOS, and Linux.

Usage:
    python throttled_benchmark.py
    python throttled_benchmark.py --preset hula_high --variant quantized
    python throttled_benchmark.py --variant baseline --model mobilenet_v3_small

Output: prints the profile dict and writes it to
    edge_compare_throttled_<preset>_<variant>.json
"""

# IMPORTANT: apply_drone_constraints() MUST run before any heavy imports
# (torch, timm, etc.) so the RAM cap and CPU pin cover model loading too.
from drone_constraints import apply_drone_constraints
apply_drone_constraints()

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from edge_tools import (
    DRONE_PRESETS,
    build_profiled_model,
    constraints_from_preset,
    profile_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=list(DRONE_PRESETS.keys()),
        default="hula_high",
        help="Which drone constraint preset to use (default: hula_high).",
    )
    parser.add_argument(
        "--variant",
        choices=["baseline", "quantized", "pruned"],
        default="quantized",
        help="Which compression variant to benchmark (default: quantized).",
    )
    parser.add_argument(
        "--model",
        default="mobilenet_v3_small",
        help="timm model name (default: mobilenet_v3_small).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write the JSON result into.",
    )
    args = parser.parse_args()

    # Build constraints + model variant.
    overrides = {"model_name": args.model}
    if args.variant == "quantized":
        overrides["quantize"] = True
    elif args.variant == "pruned":
        overrides["prune_amount"] = 0.2

    constraints = constraints_from_preset(args.preset, **overrides)
    model = build_profiled_model(constraints)

    # Run the profile under the already-applied drone constraints.
    t0 = time.perf_counter()
    profile = profile_model(model, constraints)
    wallclock_s = time.perf_counter() - t0

    profile["throttle_meta"] = {
        "constraints_source": "drone_constraints.apply_drone_constraints",
        "wallclock_seconds": round(wallclock_s, 3),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
    }
    profile["variant"] = args.variant

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"edge_compare_throttled_{args.preset}_{args.variant}.json"
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print("\n--- Throttled benchmark result ---")
    print(f"model           : {args.model} ({args.variant})")
    print(f"preset          : {args.preset}")
    print(f"model size (MB) : {profile['model_size_mb']}")
    print(f"runtime footprint (MB): {profile['runtime_footprint_mb']}")
    print(f"avg latency (ms): {profile['avg_latency_ms']}")
    print(f"p95 latency (ms): {profile['p95_latency_ms']}")
    print(f"fps             : {profile['fps']}")
    print(f"meets RAM budget: {profile['meets_ram_budget']}")
    print(f"CPU status est. : {profile['estimated_cpu_status']}")
    print(f"recommendation  : {profile['recommendation']}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
