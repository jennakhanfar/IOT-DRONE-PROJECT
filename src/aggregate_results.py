"""
aggregate_results.py
--------------------
Reads all JSON files in benchmark_results/ and prints / writes the tables
the TA asked for:

  Table A - Model comparison   : each model @ 400 MHz on DroneFace
  Table B - CPU ablation       : models across 250/400/650 MHz
  Table C - Dataset comparison : same models on DroneFace vs VGGFace2
  Table D - Gender breakdown   : accuracy by gender (DroneFace)

Outputs:
  - Pretty printed tables to stdout
  - CSVs under benchmark_results/tables/ for easy pasting into the report

Usage:
    python aggregate_results.py
    python aggregate_results.py --results-dir benchmark_results
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_all(results_dir):
    """Load every bench_*.json (per-model result files)."""
    rows = []
    for p in sorted(Path(results_dir).glob("bench_*.json")):
        # Skip combined files
        if p.name.startswith("bench_combined_") or p.name.startswith("benchmark_combined_"):
            continue
        try:
            with open(str(p), "r") as f:
                rows.append(json.load(f))
        except Exception as e:
            print("  [agg] WARN: could not load %s: %s" % (p.name, e))
    return rows


def fmt_pct(x):
    return "%.2f" % (x * 100) if x is not None else "-"


def fmt_num(x, nd=1):
    return ("%.*f" % (nd, x)) if x is not None else "-"


_BASELINES = None


def load_baselines(results_dir):
    global _BASELINES
    if _BASELINES is not None:
        return _BASELINES
    p = Path(results_dir) / "baselines.json"
    if p.exists():
        import json as _json
        with open(str(p), "r") as f:
            _BASELINES = _json.load(f)
    else:
        _BASELINES = {}
    return _BASELINES


def delta_mb(row, baselines):
    """Return measured delta if present, else compute from peak - baseline."""
    if row.get("peak_inference_delta_ram_mb") is not None:
        return row["peak_inference_delta_ram_mb"]
    peak = row.get("peak_inference_ram_mb")
    name = row.get("model_name")
    if peak is None or name not in baselines:
        return None
    return max(0.0, round(peak - baselines[name], 1))


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("  [agg] wrote %s" % path)


# ── Table A: Model comparison (DroneFace @ 400 MHz) ──────────────────────────

def table_model_comparison(rows, out_dir):
    sel = [r for r in rows if r.get("dataset") == "droneface"
           and r.get("cpu_mhz_target") == 400]
    sel.sort(key=lambda r: -r.get("accuracy", 0))
    baselines = load_baselines(out_dir.parent)

    print("\n" + "=" * 95)
    print("  TABLE A: Model comparison (DroneFace @ 400 MHz)")
    print("=" * 95)
    print("%-16s %8s %10s %10s %8s %10s %12s %10s" % (
        "Model", "Acc%", "Lat(ms)", "P95(ms)", "FPS", "Size(MB)", "Params", "DeltaRAM"))
    print("-" * 95)
    csv_rows = []
    for r in sel:
        print("%-16s %8s %10s %10s %8s %10s %12s %10s" % (
            r["model_name"],
            fmt_pct(r["accuracy"]),
            fmt_num(r["avg_latency_ms"]),
            fmt_num(r["p95_latency_ms"]),
            fmt_num(r["fps"]),
            fmt_num(r["model_size_mb"], 3),
            "{:,}".format(r["parameter_count"]),
            fmt_num(delta_mb(r, baselines)),
        ))
        csv_rows.append([
            r["model_name"],
            round(r["accuracy"] * 100, 2),
            r["avg_latency_ms"], r["p95_latency_ms"], r["fps"],
            r["model_size_mb"], r["parameter_count"],
            delta_mb(r, baselines),
        ])
    write_csv(out_dir / "table_A_model_comparison.csv",
              ["model", "accuracy_pct", "avg_latency_ms", "p95_latency_ms",
               "fps", "model_size_mb", "params", "delta_ram_mb"],
              csv_rows)


# ── Table B: CPU ablation (DroneFace, varying MHz) ───────────────────────────

def table_cpu_ablation(rows, out_dir):
    sel = [r for r in rows if r.get("dataset") == "droneface"
           and r.get("constrained") and r.get("cpu_mhz_target") is not None]
    by_model = defaultdict(dict)  # model -> mhz -> row
    for r in sel:
        by_model[r["model_name"]][r["cpu_mhz_target"]] = r

    mhz_set = sorted({r["cpu_mhz_target"] for r in sel})
    if not mhz_set:
        return

    print("\n" + "=" * 95)
    print("  TABLE B: CPU ablation (DroneFace, latency per MHz)")
    print("=" * 95)
    header = "%-16s" % "Model"
    for mhz in mhz_set:
        header += " | %4dMHz Acc%%  Lat(ms)" % mhz
    print(header)
    print("-" * len(header))

    csv_rows = []
    for model in sorted(by_model):
        line = "%-16s" % model
        row = [model]
        for mhz in mhz_set:
            r = by_model[model].get(mhz)
            if r:
                line += " | %8s  %7s" % (fmt_pct(r["accuracy"]), fmt_num(r["avg_latency_ms"]))
                row += [round(r["accuracy"] * 100, 2), r["avg_latency_ms"]]
            else:
                line += " | %8s  %7s" % ("-", "-")
                row += ["", ""]
        print(line)
        csv_rows.append(row)

    csv_header = ["model"]
    for mhz in mhz_set:
        csv_header += ["acc_%dmhz_pct" % mhz, "lat_%dmhz_ms" % mhz]
    write_csv(out_dir / "table_B_cpu_ablation.csv", csv_header, csv_rows)


# ── Table C: Dataset comparison ──────────────────────────────────────────────

def table_dataset_comparison(rows, out_dir):
    sel = [r for r in rows if r.get("cpu_mhz_target") == 400]
    by_model = defaultdict(dict)
    for r in sel:
        by_model[r["model_name"]][r["dataset"]] = r

    datasets = sorted({r["dataset"] for r in sel})
    if len(datasets) < 2:
        return

    print("\n" + "=" * 95)
    print("  TABLE C: Dataset comparison (@ 400 MHz)")
    print("=" * 95)
    header = "%-16s" % "Model"
    for d in datasets:
        header += " | %-10s Acc%%  Lat(ms)" % d
    print(header)
    print("-" * len(header))

    csv_rows = []
    for model in sorted(by_model):
        line = "%-16s" % model
        row = [model]
        for d in datasets:
            r = by_model[model].get(d)
            if r:
                line += " | %15s  %7s" % (fmt_pct(r["accuracy"]), fmt_num(r["avg_latency_ms"]))
                row += [round(r["accuracy"] * 100, 2), r["avg_latency_ms"]]
            else:
                line += " | %15s  %7s" % ("-", "-")
                row += ["", ""]
        print(line)
        csv_rows.append(row)

    csv_header = ["model"]
    for d in datasets:
        csv_header += ["acc_%s_pct" % d, "lat_%s_ms" % d]
    write_csv(out_dir / "table_C_dataset_comparison.csv", csv_header, csv_rows)


# ── Table D: Gender breakdown (DroneFace @ 400 MHz) ──────────────────────────

def table_gender_breakdown(rows, out_dir):
    sel = [r for r in rows if r.get("dataset") == "droneface"
           and r.get("cpu_mhz_target") == 400
           and r.get("conditions", {}).get("accuracy_by_group", {}).get("gender")]
    if not sel:
        return

    # Determine gender keys
    gkeys = set()
    for r in sel:
        gkeys.update(r["conditions"]["accuracy_by_group"]["gender"].keys())
    gkeys = sorted(gkeys)

    print("\n" + "=" * 95)
    print("  TABLE D: Gender breakdown (DroneFace @ 400 MHz)")
    print("=" * 95)
    header = "%-16s %10s" % ("Model", "Overall%")
    for g in gkeys:
        header += " | %s (n)  Acc%% " % g
    print(header)
    print("-" * len(header))

    csv_rows = []
    for r in sorted(sel, key=lambda x: -x["accuracy"]):
        line = "%-16s %10s" % (r["model_name"], fmt_pct(r["accuracy"]))
        row = [r["model_name"], round(r["accuracy"] * 100, 2)]
        gdata = r["conditions"]["accuracy_by_group"]["gender"]
        for g in gkeys:
            stats = gdata.get(g)
            if stats:
                line += " | %s (%d)  %7s" % (g, stats["n"], fmt_pct(stats["accuracy"]))
                row += [stats["n"], round(stats["accuracy"] * 100, 2)]
            else:
                line += " | %s       -" % g
                row += ["", ""]
        print(line)
        csv_rows.append(row)

    csv_header = ["model", "overall_acc_pct"]
    for g in gkeys:
        csv_header += ["%s_n" % g, "%s_acc_pct" % g]
    write_csv(out_dir / "table_D_gender_breakdown.csv", csv_header, csv_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="benchmark_results")
    args = ap.parse_args()

    rows = load_all(args.results_dir)
    print("[agg] loaded %d result files" % len(rows))
    if not rows:
        return 1

    out_dir = Path(args.results_dir) / "tables"
    table_model_comparison(rows, out_dir)
    table_cpu_ablation(rows, out_dir)
    table_dataset_comparison(rows, out_dir)
    table_gender_breakdown(rows, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
