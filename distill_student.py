import argparse
import json
from pathlib import Path

from .distillation_tools import DistillationConfig, export_distillation_plan, train_distilled_student


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a lightweight student model using teacher-student distillation."
    )
    parser.add_argument("--train-root", help="Training dataset root. Example: split/train")
    parser.add_argument("--output-dir", default="distillation_runs")
    parser.add_argument("--teacher-model", default="efficientnet_b0")
    parser.add_argument("--student-model", default="mobilenet_v3_small")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write out the distillation plan without starting training.",
    )
    parser.add_argument("--output-json", default="distillation_summary.json")
    args = parser.parse_args()

    config = DistillationConfig(
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        temperature=args.temperature,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        pretrained=args.pretrained,
        device=args.device,
    )

    if args.plan_only:
        export_distillation_plan(Path(args.output_dir) / "distillation_plan.txt", config)
        print("Distillation plan written.")
        return

    if not args.train_root:
        raise ValueError("--train-root is required unless --plan-only is used.")

    result = train_distilled_student(args.train_root, args.output_dir, config)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
