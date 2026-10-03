"""Shared Kineto reporting for architecture-specific full-step audits."""

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

ModuleCategorizer = Callable[[str, torch.nn.Module], str | None]
RoleFragment = tuple[str, str]


def lora_module_category(
    name: str,
    class_name: str,
    *,
    packed_lora_classes: frozenset[str],
    generic_lora_classes: frozenset[str],
    expert_lora_classes: frozenset[str],
    role_fragments: tuple[RoleFragment, ...] = (),
) -> str | None:
    """Classify the LoRA wrappers and packed LM head shared by both models.

    Qwen3.5-MoE and DeepSeek V4 profiles use the same category vocabulary and
    the same precedence so their reports stay comparable:
    - the routed-expert wrapper owns `routed_experts`;
    - a shared-expert projection owns `shared_expert` regardless of which
      ordinary wrapper class PEFT selected for it;
    - the remaining ordinary wrappers split into `packed_ordinary_lora` for
      the native packed-base path and `ordinary_lora` for the generic path;
    - the frozen LM head is `packed_lm_head`.

    Only the architecture-owned class names and role fragments are supplied by
    the caller. `role_fragments` is applied in order, so an architecture can
    add a name-owned role such as DeepSeek's `o_b_proj` without redefining
    the shared precedence in its own categorizer.
    """

    if class_name in expert_lora_classes:
        return "routed_experts"
    if class_name in packed_lora_classes or class_name in generic_lora_classes:
        for fragment, category in role_fragments:
            if fragment in name:
                return category
        if class_name in packed_lora_classes:
            return "packed_ordinary_lora"
        return "ordinary_lora"
    if name.endswith("lm_head"):
        return "packed_lm_head"
    return None


@contextmanager
def module_ranges(
    model: torch.nn.Module, categorize: ModuleCategorizer
) -> Iterator[None]:
    """Annotate selected module forwards with architecture-owned categories."""

    handles: list[Any] = []
    active_ranges: dict[int, list[Any]] = {}
    for name, module in model.named_modules():
        category = categorize(name, module)
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

    try:
        yield
    finally:
        for contexts in active_ranges.values():
            while contexts:
                contexts.pop().__exit__(None, None, None)
        for handle in handles:
            handle.remove()


def _event_times(event: Any) -> tuple[float, float, float, float]:
    self_cpu_us = float(getattr(event, "self_cpu_time_total", 0.0))
    self_device_us = float(
        getattr(
            event,
            "self_device_time_total",
            getattr(event, "self_cuda_time_total", 0.0),
        )
    )
    total_cpu_us = float(getattr(event, "cpu_time_total", self_cpu_us))
    total_device_us = float(
        getattr(
            event,
            "device_time_total",
            getattr(event, "cuda_time_total", self_device_us),
        )
    )
    return self_cpu_us, self_device_us, total_cpu_us, total_device_us


def summarize_events(profiler: torch.profiler.profile) -> dict[str, Any]:
    """Rank operators and aggregate categories by self and inclusive time."""

    rows = []
    module_totals: dict[str, dict[str, float | int]] = {}
    for event in profiler.key_averages():
        self_cpu_us, self_device_us, total_cpu_us, total_device_us = _event_times(event)
        rows.append(
            {
                "name": event.key,
                "calls": int(event.count),
                "self_cpu_ms": self_cpu_us / 1000,
                "self_device_ms": self_device_us / 1000,
                "total_cpu_ms": total_cpu_us / 1000,
                "total_device_ms": total_device_us / 1000,
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
                {
                    "calls": 0,
                    "self_cpu_ms": 0.0,
                    "self_device_ms": 0.0,
                    "total_cpu_ms": 0.0,
                    "total_device_ms": 0.0,
                },
            )
            aggregate["calls"] += int(event.count)
            aggregate["self_cpu_ms"] += self_cpu_us / 1000
            aggregate["self_device_ms"] += self_device_us / 1000
            aggregate["total_cpu_ms"] += total_cpu_us / 1000
            aggregate["total_device_ms"] += total_device_us / 1000
    category_rows = [
        {"category": category, **times} for category, times in module_totals.items()
    ]
    return {
        "top_device_self_time": sorted(
            rows, key=lambda row: row["self_device_ms"], reverse=True
        )[:100],
        "top_cpu_self_time": sorted(
            rows, key=lambda row: row["self_cpu_ms"], reverse=True
        )[:100],
        "top_device_total_time": sorted(
            rows, key=lambda row: row["total_device_ms"], reverse=True
        )[:100],
        "top_cpu_total_time": sorted(
            rows, key=lambda row: row["total_cpu_ms"], reverse=True
        )[:100],
        "module_self_time": module_totals,
        "module_time_by_device": sorted(
            category_rows,
            key=lambda row: row["total_device_ms"],
            reverse=True,
        ),
        "module_time_by_cpu": sorted(
            category_rows,
            key=lambda row: row["total_cpu_ms"],
            reverse=True,
        ),
    }


def profile_warmed_training_update(
    model: torch.nn.Module,
    update: Callable[[str], dict[str, Any]],
    *,
    output_path: str | Path,
    categorize: ModuleCategorizer,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Warm one update, then trace a synchronized CPU+GPU update with Kineto."""

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
    with (
        module_ranges(model, categorize),
        torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler,
    ):
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
        "events": summarize_events(profiler),
        "metadata": metadata or {},
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
