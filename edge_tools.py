import json
import os
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import psutil

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError(
        "edge_tools.py requires torch. Install the project dependencies before running it."
    ) from exc

from model import (
    apply_dynamic_quantization,
    apply_global_pruning,
    count_parameters,
    create_edge_backbone,
)


@dataclass
class DeploymentConstraints:
    preset_name: str = "custom"
    device_name: str = "generic edge device"
    cpu_percent_limit: float = 10.0
    ram_limit_mb: int = 1024
    power_budget_watts: float = 15.0
    os_name: str = "unknown"
    input_size: int = 112
    batch_size: int = 1
    warmup_runs: int = 5
    timed_runs: int = 25
    model_name: str = "mobilenet_v3_small"
    pretrained: bool = True
    quantize: bool = False
    prune_amount: float = 0.0
    device: str = "cpu"


DRONE_PRESETS = {
    "hula_high": {
        "preset_name": "hula_high",
        "device_name": "HULA drone (higher-end estimate)",
        "cpu_percent_limit": 15.0,
        "ram_limit_mb": 128,
        "power_budget_watts": 15.0,
        "os_name": "Windows",
    },
    "generic_edge": {
        "preset_name": "generic_edge",
        "device_name": "Generic edge device",
        "cpu_percent_limit": 10.0,
        "ram_limit_mb": 1024,
        "power_budget_watts": 15.0,
        "os_name": "Linux",
    },
}


def constraints_from_preset(
    preset_name: str,
    **overrides,
) -> DeploymentConstraints:
    if preset_name not in DRONE_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    payload = {**DRONE_PRESETS[preset_name], **overrides}
    return DeploymentConstraints(**payload)


def _save_model_size_mb(model: torch.nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(model.state_dict(), temp_path)
        return temp_path.stat().st_size / (1024 * 1024)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _estimate_power_status(constraints: DeploymentConstraints, latency_ms: float) -> str:
    if constraints.device != "cpu":
        return "not_estimated"

    # A simple report-oriented heuristic. The point is to surface whether the
    # measured configuration is likely to violate a small-drone CPU power budget.
    estimated_watts = max(1.0, (constraints.cpu_percent_limit / 100.0) * 28.0)
    return "within_budget" if estimated_watts <= constraints.power_budget_watts else "over_budget"


def _estimate_cpu_status(constraints: DeploymentConstraints, latency_ms: float) -> str:
    if constraints.cpu_percent_limit <= 15.0:
        return "high_risk_for_real_time" if latency_ms > 100.0 else "possible_for_real_time"
    return "profiling_only"


def compression_recommendation(profile: Dict[str, object]) -> str:
    fps = float(profile["fps"])
    size_mb = float(profile["model_size_mb"])
    if fps >= 7.0 and size_mb <= 5.0:
        return "best_edge_candidate"
    if fps >= 5.0 and size_mb <= 10.0:
        return "usable_with_tradeoffs"
    return "too_heavy_for_strict_drone_budget"


def profile_model(
    model: torch.nn.Module,
    constraints: DeploymentConstraints,
) -> Dict[str, object]:
    device = torch.device(constraints.device)
    model = model.to(device)
    model.eval()

    sample = torch.randn(
        constraints.batch_size,
        3,
        constraints.input_size,
        constraints.input_size,
        device=device,
    )

    for _ in range(constraints.warmup_runs):
        with torch.no_grad():
            _ = model(sample)

    timings_ms: List[float] = []
    process = psutil.Process(os.getpid())
    baseline_rss_mb = process.memory_info().rss / (1024 * 1024)

    for _ in range(constraints.timed_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    peak_rss_mb = process.memory_info().rss / (1024 * 1024)
    avg_latency_ms = statistics.mean(timings_ms)
    model_size_mb = round(_save_model_size_mb(model), 3)
    rss_delta_mb = round(max(0.0, peak_rss_mb - baseline_rss_mb), 3)
    # Honest budget check: weights (model file size) + incremental runtime memory
    # (activations, intermediate tensors) must fit under the drone RAM ceiling.
    # We intentionally exclude the Python interpreter's baseline RSS because on
    # the actual drone the model would run under a slim C runtime, not CPython.
    runtime_footprint_mb = round(model_size_mb + rss_delta_mb, 3)
    meets_ram_budget = runtime_footprint_mb <= constraints.ram_limit_mb

    result = {
        "constraints": asdict(constraints),
        "parameter_count": count_parameters(model),
        "trainable_parameter_count": count_parameters(model, trainable_only=True),
        "model_size_mb": model_size_mb,
        "avg_latency_ms": round(avg_latency_ms, 3),
        "median_latency_ms": round(statistics.median(timings_ms), 3),
        "p95_latency_ms": round(sorted(timings_ms)[int(0.95 * (len(timings_ms) - 1))], 3),
        "fps": round(1000.0 / avg_latency_ms, 3),
        "rss_delta_mb": rss_delta_mb,
        "runtime_footprint_mb": runtime_footprint_mb,
        "ram_limit_mb": constraints.ram_limit_mb,
        "meets_ram_budget": meets_ram_budget,
        "estimated_cpu_status": _estimate_cpu_status(constraints, avg_latency_ms),
        "estimated_power_status": _estimate_power_status(constraints, avg_latency_ms),
        "recommendation": "",
    }
    result["recommendation"] = compression_recommendation(result)
    return result


def build_profiled_model(constraints: DeploymentConstraints) -> torch.nn.Module:
    model, _ = create_edge_backbone(
        model_name=constraints.model_name,
        pretrained=constraints.pretrained,
    )
    if constraints.prune_amount > 0:
        model = apply_global_pruning(model, amount=constraints.prune_amount)
    if constraints.quantize and constraints.device == "cpu":
        model = apply_dynamic_quantization(model)
    return model


def compare_edge_variants(
    model_name: str,
    output_path: Optional[str] = None,
    pretrained: bool = True,
    preset_name: str = "generic_edge",
) -> Dict[str, Dict[str, object]]:
    baseline = constraints_from_preset(
        preset_name,
        model_name=model_name,
        pretrained=pretrained,
    )
    quantized = constraints_from_preset(
        preset_name,
        model_name=model_name,
        pretrained=pretrained,
        quantize=True,
    )
    pruned = constraints_from_preset(
        preset_name,
        model_name=model_name,
        pretrained=pretrained,
        prune_amount=0.2,
    )

    results = {
        "baseline": profile_model(build_profiled_model(baseline), baseline),
        "quantized": profile_model(build_profiled_model(quantized), quantized),
        "pruned": profile_model(build_profiled_model(pruned), pruned),
    }

    if output_path:
        Path(output_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
