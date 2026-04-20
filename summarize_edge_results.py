import argparse
import json
from pathlib import Path
from typing import Dict


def load_result(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_variant_table(model_name: str, results: Dict[str, object]) -> str:
    lines = [
        f"## {model_name}",
        "",
        "| Variant | Size (MB) | Avg Latency (ms) | FPS | RAM Budget | Recommendation |",
        "|---|---:|---:|---:|---|---|",
    ]
    for variant, payload in results.items():
        recommendation = payload.get("recommendation", "legacy_profile")
        lines.append(
            "| "
            + f"{variant} | {payload['model_size_mb']} | {payload['avg_latency_ms']} | "
            + f"{payload['fps']} | {payload['meets_ram_budget']} | {recommendation} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a report-ready markdown summary from edge profiling JSON files.")
    parser.add_argument("json_files", nargs="+")
    parser.add_argument("--output", default="COMPRESSION_SUMMARY.md")
    args = parser.parse_args()

    sections = [
        "# Compression Summary",
        "",
        "This file summarizes the edge-model compression experiments for the TA feedback section.",
        "",
    ]

    for file_name in args.json_files:
        path = Path(file_name)
        payload = load_result(path)
        first_variant = next(iter(payload.values()))
        model_name = first_variant["constraints"]["model_name"]
        preset_name = first_variant["constraints"].get("preset_name", "legacy_profile")
        sections.append(f"Constraint preset: `{preset_name}`")
        sections.append("")
        sections.append(render_variant_table(model_name, payload))

    Path(args.output).write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
