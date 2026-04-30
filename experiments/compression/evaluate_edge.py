import argparse
import json
from pathlib import Path

from .edge_tools import (
    DeploymentConstraints,
    build_profiled_model,
    compare_edge_variants,
    constraints_from_preset,
    profile_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile lightweight face-recognition backbones under drone-like edge constraints."
    )
    parser.add_argument("--model", default="mobilenet_v3_small", help="Backbone name.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--preset",
        default="generic_edge",
        help="Constraint preset name. Example: generic_edge or hula_high",
    )
    parser.add_argument("--input-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ram-limit-mb", type=int, default=1024)
    parser.add_argument("--cpu-percent-limit", type=float, default=10.0)
    parser.add_argument("--power-budget-watts", type=float, default=15.0)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--timed-runs", type=int, default=25)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--prune-amount", type=float, default=0.0)
    parser.add_argument(
        "--compare-variants",
        action="store_true",
        help="Run baseline vs quantized vs pruned comparison.",
    )
    parser.add_argument("--output", default="edge_profile.json")
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.compare_variants:
        results = compare_edge_variants(
            args.model,
            output_path=str(output_path),
            pretrained=args.pretrained,
            preset_name=args.preset,
        )
        print(json.dumps(results, indent=2))
        return

    constraints = constraints_from_preset(
        args.preset,
        cpu_percent_limit=args.cpu_percent_limit,
        ram_limit_mb=args.ram_limit_mb,
        power_budget_watts=args.power_budget_watts,
        input_size=args.input_size,
        batch_size=args.batch_size,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
        model_name=args.model,
        pretrained=args.pretrained,
        quantize=args.quantize,
        prune_amount=args.prune_amount,
        device=args.device,
    )
    model = build_profiled_model(constraints)
    result = profile_model(model, constraints)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
