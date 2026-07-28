from dataclasses import dataclass, field
from typing import Any

import torch

_ROUTE_WEIGHT_SAMPLE_ROWS = 256


def validate_deepseek_v4_routes(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> None:
    if hidden_states.ndim != 2 or hidden_states.shape[-1] != 4096:
        raise ValueError(
            f"DeepSeek V4 routed hidden states must be [tokens,4096], got {tuple(hidden_states.shape)}."
        )
    expected = (hidden_states.shape[0], 6)
    if tuple(top_k_index.shape) != expected or tuple(top_k_weights.shape) != expected:
        raise ValueError(
            f"DeepSeek V4 routes must both have shape {expected}, got "
            f"{tuple(top_k_index.shape)} and {tuple(top_k_weights.shape)}."
        )
    if top_k_index.dtype != torch.long:
        raise TypeError(
            f"DeepSeek V4 expert indices must be int64, got {top_k_index.dtype}."
        )
    if bool(torch.any((top_k_index < 0) | (top_k_index >= 256)).item()):
        raise ValueError("DeepSeek V4 expert indices must be in [0,256).")
    if not bool(torch.isfinite(top_k_weights).all().item()):
        raise ValueError("DeepSeek V4 expert weights must be finite.")
    if bool(torch.any(top_k_weights < 0).item()):
        raise ValueError("DeepSeek V4 expert weights must be nonnegative.")


def _sample_sorted_route_weights(top_k_weights: torch.Tensor) -> torch.Tensor:
    sorted_weights = top_k_weights.detach().float().sort(dim=-1).values
    stride = max(sorted_weights.shape[0] // _ROUTE_WEIGHT_SAMPLE_ROWS, 1)
    return sorted_weights[::stride][:_ROUTE_WEIGHT_SAMPLE_ROWS].cpu()


def summarize_deepseek_v4_routes(
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    *,
    layer: int | None = None,
    router_kind: str | None = None,
) -> dict[str, Any]:
    if top_k_index.ndim != 2 or top_k_index.shape[-1] != 6:
        raise ValueError(
            f"DeepSeek V4 route indices must be [tokens,6], got {tuple(top_k_index.shape)}."
        )
    if top_k_weights.shape != top_k_index.shape:
        raise ValueError(
            "DeepSeek V4 route weights must match route indices, got "
            f"{tuple(top_k_weights.shape)} and {tuple(top_k_index.shape)}."
        )
    counts = torch.bincount(top_k_index.detach().reshape(-1), minlength=256).cpu()
    active = counts[counts > 0]
    sorted_counts = active.sort().values
    median = (
        float(sorted_counts.float().median().item()) if sorted_counts.numel() else 0.0
    )
    weights = top_k_weights.detach().float()
    row_sums = weights.sum(dim=-1)
    weight_sample = _sample_sorted_route_weights(weights)
    return {
        "layer": layer,
        "router_kind": router_kind,
        "tokens": top_k_index.shape[0],
        "routes": top_k_index.numel(),
        "active_experts": active.numel(),
        "rows_min": int(active.min().item()) if active.numel() else 0,
        "rows_max": int(active.max().item()) if active.numel() else 0,
        "rows_mean": float(active.float().mean().item()) if active.numel() else 0.0,
        "rows_median": median,
        "groups_at_least_16": int(torch.count_nonzero(active >= 16).item()),
        "groups_at_least_32": int(torch.count_nonzero(active >= 32).item()),
        "groups_at_least_64": int(torch.count_nonzero(active >= 64).item()),
        "groups_at_least_128": int(torch.count_nonzero(active >= 128).item()),
        "rows_per_expert": counts.tolist(),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_rms": float(weights.square().mean().sqrt()),
        "weight_sum_mean": float(row_sums.mean()),
        "weight_sum_max_error": float((row_sums - row_sums.mean()).abs().max()),
        "sorted_weight_sample": weight_sample.tolist(),
    }


def compare_deepseek_v4_route_weights(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, float | int]:
    """Compare selected score weights without comparing expert identities."""

    if len(reference) != len(candidate):
        raise ValueError(
            f"route summary count mismatch: {len(reference)} != {len(candidate)}"
        )
    reference_samples = []
    candidate_samples = []
    for reference_entry, candidate_entry in zip(reference, candidate, strict=True):
        reference_key = (
            reference_entry.get("layer"),
            reference_entry.get("router_kind"),
            reference_entry.get("tokens"),
        )
        candidate_key = (
            candidate_entry.get("layer"),
            candidate_entry.get("router_kind"),
            candidate_entry.get("tokens"),
        )
        if reference_key != candidate_key:
            raise ValueError(
                f"route summary identity mismatch: {reference_key} != {candidate_key}"
            )
        reference_sample = torch.tensor(
            reference_entry["sorted_weight_sample"], dtype=torch.float32
        )
        candidate_sample = torch.tensor(
            candidate_entry["sorted_weight_sample"], dtype=torch.float32
        )
        if reference_sample.shape != candidate_sample.shape:
            raise ValueError(
                "route weight sample shape mismatch: "
                f"{tuple(reference_sample.shape)} != {tuple(candidate_sample.shape)}"
            )
        reference_samples.append(reference_sample.reshape(-1))
        candidate_samples.append(candidate_sample.reshape(-1))

    reference_weights = torch.cat(reference_samples)
    candidate_weights = torch.cat(candidate_samples)
    delta = candidate_weights - reference_weights
    reference_rms = reference_weights.square().mean().sqrt()
    candidate_rms = candidate_weights.square().mean().sqrt()
    denominator = reference_weights.norm() * candidate_weights.norm()
    cosine = (
        torch.dot(reference_weights, candidate_weights) / denominator
        if float(denominator) != 0.0
        else torch.ones(())
    )
    return {
        "summaries": len(reference),
        "sampled_weights": reference_weights.numel(),
        "cosine": float(cosine),
        "rmse": float(delta.square().mean().sqrt()),
        "relative_rmse": float(delta.square().mean().sqrt() / (reference_rms + 1e-20)),
        "max_abs": float(delta.abs().max()),
        "reference_rms": float(reference_rms),
        "candidate_rms": float(candidate_rms),
    }


@dataclass
class DeepseekV4RouteCollector:
    """Forward-hook collector used only during validation and profiling."""

    records: list[tuple[int | None, str, torch.Tensor, torch.Tensor]] = field(
        default_factory=list
    )
    _handles: list[Any] = field(default_factory=list)

    def install(self, model: torch.nn.Module):
        self.remove()
        for name, module in model.named_modules():
            if not (name.endswith(".mlp.gate") and hasattr(module, "top_k")):
                continue
            layer_text = name.split(".layers.", 1)[-1].split(".", 1)[0]
            layer = int(layer_text) if layer_text.isdigit() else None
            router_kind = "hash" if hasattr(module, "tid2eid") else "learned"

            def hook(_module, _inputs, output, *, layer=layer, router_kind=router_kind):
                self.records.append(
                    (layer, router_kind, output[2].detach(), output[1].detach())
                )

            self._handles.append(module.register_forward_hook(hook))
        return self

    def summaries(self) -> list[dict[str, Any]]:
        return [
            summarize_deepseek_v4_routes(
                indices,
                weights,
                layer=layer,
                router_kind=router_kind,
            )
            for layer, router_kind, indices, weights in self.records
        ]

    def clear(self) -> None:
        self.records.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
