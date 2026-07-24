"""Kineto full-step profiling using the established Qwen training workflow."""

import json
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


def _module_category(name: str, module: torch.nn.Module) -> str | None:
    class_name = type(module).__name__
    if class_name in {"DeepseekV4HyperConnection", "DeepseekV4HyperHead"}:
        return "mhc"
    if "RMSNorm" in class_name:
        return "rmsnorm"
    if class_name == "DeepseekV4Indexer":
        return "indexer"
    if class_name in {"DeepseekV4CompressedKV", "DeepseekV4HeavilyCompressedKV"}:
        return "compressor"
    if class_name == "DeepseekV4Attention":
        return "attention"
    if class_name == "GGUFGroupedLinear":
        return "grouped_output_a"
    if class_name == "DeepseekV4GGUFMoeLora":
        return "routed_experts"
    if class_name in {"DeepseekV4GGUFLoraLinear", "DeepseekV4LoraLinear"}:
        if ".shared_experts." in name:
            return "shared_expert"
        if name.endswith(".o_b_proj"):
            return "output_b"
        return "ordinary_lora"
    return None


@contextmanager
def deepseek_v4_module_ranges(model: torch.nn.Module):
    """Install the same forward pre/post record_function hooks used by Qwen profiling."""

    handles: list[Any] = []
    active_ranges: dict[int, list[Any]] = {}
    for name, module in model.named_modules():
        category = _module_category(name, module)
        if category is None:
            continue

        label = f"module/{category}/{name}"
        module_id = id(module)

        def pre_hook(_module, _args, *, module_id=module_id, label=label):
            context = torch.autograd.profiler.record_function(label)
            context.__enter__()
            active_ranges.setdefault(module_id, []).append(context)

        def post_hook(_module, _args, output, *, module_id=module_id):
            active_ranges[module_id].pop().__exit__(None, None, None)
            return output

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    yield

    for contexts in active_ranges.values():
        while contexts:
            contexts.pop().__exit__(None, None, None)
    for handle in handles:
        handle.remove()


def _event_times(event: Any) -> tuple[float, float]:
    cpu_us = float(getattr(event, "self_cpu_time_total", 0.0))
    device_us = float(
        getattr(
            event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0)
        )
    )
    return cpu_us, device_us


def _summarize_events(profiler: torch.profiler.profile) -> dict[str, Any]:
    rows = []
    module_totals: dict[str, dict[str, float | int]] = {}
    for event in profiler.key_averages():
        cpu_us, device_us = _event_times(event)
        rows.append(
            {
                "name": event.key,
                "calls": int(event.count),
                "self_cpu_ms": cpu_us / 1000,
                "self_device_ms": device_us / 1000,
                "cpu_memory_bytes": int(getattr(event, "cpu_memory_usage", 0)),
                "device_memory_bytes": int(
                    getattr(
                        event,
                        "device_memory_usage",
                        getattr(event, "cuda_memory_usage", 0),
                    )
                ),
            }
        )
        if event.key.startswith("module/"):
            category = event.key.split("/", 2)[1]
            aggregate = module_totals.setdefault(
                category,
                {"calls": 0, "self_cpu_ms": 0.0, "self_device_ms": 0.0},
            )
            aggregate["calls"] += int(event.count)
            aggregate["self_cpu_ms"] += cpu_us / 1000
            aggregate["self_device_ms"] += device_us / 1000
    return {
        "top_device_self_time": sorted(
            rows, key=lambda row: row["self_device_ms"], reverse=True
        )[:100],
        "top_cpu_self_time": sorted(
            rows, key=lambda row: row["self_cpu_ms"], reverse=True
        )[:100],
        "module_self_time": module_totals,
    }


def profile_warmed_training_update(
    model: torch.nn.Module,
    update: Callable[[str], dict[str, Any]],
    *,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Warm one update, then trace the next with synchronized CPU+GPU Kineto."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output_path.with_suffix(".trace.json")

    warm = update("profile_warmup")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with deepseek_v4_module_ranges(model):
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            traced = update("profile_traced")
            profiler.step()
    torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    report = {
        "method": "warm_one_update_then_trace_one_update",
        "activities": ["cpu", "gpu"],
        "warmup": warm,
        "traced": traced,
        "trace_path": str(trace_path),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "events": _summarize_events(profiler),
        "metadata": metadata or {},
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
