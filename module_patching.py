"""Shared instance-local patching protocol for module forwards.

`qwen3_5_fused_norms.py`, `deepseek_v4_liger_rmsnorm.py`, `deepseek_v4_lora.py`, and
`fast_moe_ranking.py` install a replacement forward on selected module instances and then gate the
run on a fixed inventory. They need the same protocol, so it lives here and each family module only
supplies what is actually family-specific:
- match modules by type, in `named_modules()` order.
- count every match, patched or not, so the inventory gate sees the whole model.
- validate the site (device, dtype, geometry) before touching it, and prepare it if needed.
- optionally freeze the replaced weight, which sits outside the fixed LoRA target contract.
- set a marker attribute and, when a forward is supplied, replace `forward` with `MethodType`, so
  a second call only counts.
- report the inventory counters, per-family names, `patched`, `already_patched`, and
  `patched_names`.
- fail closed through `require_complete_inventory` when the inventory is not exactly covered.

Only the forward, the site hooks, and the skip rule are per-family. `matches` selects which
modules of `module_type` a spec owns, which lets several specs share one module class and route by
an attribute (for example the attention layer family). A spec with `forward=None` only validates
and marks, for sites whose execution is owned elsewhere. Kernel-level code (for example the
frozen-weight backward of the DeepSeek norms) stays in the family module. This layer never touches
the math.

Marker convention: an installed forward is marked with a `_patched_<family>` attribute so a second
pass can count it and a debugger can see what owns the module. Flags that record a choice rather
than an installation (`fast_lora._GENERIC_PACKED_FORWARD_ATTR`, `_aiter_expert_prior`) deliberately
do not use the prefix.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MethodType
from typing import Any, Generic, TypeVar

import torch

PATCH_MARKER = "_patched_fused_norm"

ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)


@dataclass(frozen=True)
class ModulePatchSpec(Generic[ModuleT]):
    """One module type, the forward that replaces its eager path, and its counters."""

    module_type: type[ModuleT]
    forward: Callable[..., Any] | None
    handled_key: str
    matches: Callable[[str, ModuleT], bool] | None = None
    validate: Callable[[str, ModuleT], None] | None = None
    prepare: Callable[[str, ModuleT], None] | None = None
    accept: Callable[[str, ModuleT], bool] | None = None
    skip_key: str | None = None
    marker: str = PATCH_MARKER
    freeze_weight: bool = True


def patch_module_forwards(
    model: torch.nn.Module,
    specs: Sequence[ModulePatchSpec[Any]],
    *,
    declared_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Install every spec on its matching modules and return the counted report.

    `matches` narrows a spec to the modules of `module_type` it owns. A module it does not match is
    offered to the next spec instead of failing. `validate` raises for a site the family cannot
    serve. `prepare` is the hook for per-match side effects that must run on every pass, including
    already-patched modules (freezing controls, converting parameters, binding an expert prior).
    `declared_keys` carries inventory keys that belong to the gate but have no spec producing them,
    so a family can require that a reserved exclusion counter stays at zero.
    """

    counts: dict[str, int] = dict.fromkeys(declared_keys, 0)
    for spec in specs:
        counts.setdefault(spec.handled_key, 0)
        if spec.skip_key is not None:
            counts.setdefault(spec.skip_key, 0)
    handled_by_key: dict[str, list[str]] = {key: [] for key in counts}

    patched = 0
    already_patched = 0
    patched_names: list[str] = []

    for name, module in model.named_modules():
        for spec in specs:
            if not isinstance(module, spec.module_type):
                continue
            if spec.matches is not None and not spec.matches(name, module):
                continue
            if spec.accept is not None and not spec.accept(name, module):
                if spec.skip_key is None:
                    raise RuntimeError(
                        f"module patch {spec.handled_key!r} rejects {name!r} "
                        "without a skip key"
                    )
                counts[spec.skip_key] += 1
                break
            counts[spec.handled_key] += 1
            handled_by_key[spec.handled_key].append(name)
            if spec.validate is not None:
                spec.validate(name, module)
            if spec.prepare is not None:
                spec.prepare(name, module)
            if spec.freeze_weight:
                weight = getattr(module, "weight", None)
                if weight is not None:
                    weight.requires_grad_(False)
            if getattr(module, spec.marker, False):
                already_patched += 1
                break
            if spec.forward is not None:
                module.forward = MethodType(spec.forward, module)
            setattr(module, spec.marker, True)
            patched += 1
            patched_names.append(name)
            break

    return {
        **counts,
        "patched": patched,
        "already_patched": already_patched,
        "patched_names": patched_names,
        "handled_by_key": handled_by_key,
    }


def require_complete_inventory(
    report: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    subject: str,
) -> None:
    """Fail closed unless every expected inventory value matches the report.

    Values are compared by equality, so a gate can require counts as well as a fixed flag such as
    the installed implementation name.
    """

    mismatches = {
        key: (value, report.get(key))
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"incomplete {subject} configuration: {mismatches}")


def require_cuda_weight(name: str, module: torch.nn.Module, *, subject: str) -> None:
    """Reject a weighted module that an accelerator-only kernel cannot serve."""

    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise RuntimeError(f"{subject} {name!r} has no floating weight")
    if weight.device.type != "cuda":
        raise RuntimeError(
            f"{subject} {name!r} requires a CUDA/ROCm weight, got {weight.device}"
        )
