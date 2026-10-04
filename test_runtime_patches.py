from typing import Any

import pytest
import torch
from transformers import Trainer

import gguf_dequant_compile
from bf16_adapter_trainer import BF16AdapterTrainer


@pytest.fixture
def patch_marker_cleared():
    """Hand the test an unpatched module and leave it that way, whatever the test does to it."""

    gguf_dequant_compile.gguf_dequant.__dict__.pop(
        gguf_dequant_compile._PATCH_MARKER, None
    )
    yield
    gguf_dequant_compile.gguf_dequant.__dict__.pop(
        gguf_dequant_compile._PATCH_MARKER, None
    )


def test_compiled_dequant_patch_updates_every_runtime_binding(
    monkeypatch, patch_marker_cleared
) -> None:
    modules = (
        gguf_dequant_compile.gguf_dequant,
        gguf_dequant_compile.gguf_quantized_parameter,
        gguf_dequant_compile.gguf_kernels,
    )
    originals = {module: module.dequantize for module in modules}
    for module, original in originals.items():
        monkeypatch.setitem(module.__dict__, "dequantize", original)
    monkeypatch.setattr(
        torch._dynamo.config, "recompile_limit", torch._dynamo.config.recompile_limit
    )
    compile_calls: list[tuple[Any, dict[str, Any]]] = []

    def fake_compile(function: Any, **kwargs: Any) -> Any:
        compile_calls.append((function, kwargs))

        def compiled(*args: Any, **call_kwargs: Any) -> Any:
            return function(*args, **call_kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    assert gguf_dequant_compile.configure_compiled_gguf_dequantize() is True
    assert gguf_dequant_compile.configure_compiled_gguf_dequantize() is False
    compiled: Any = modules[0].dequantize
    assert all(module.dequantize is compiled for module in modules)
    assert compiled._eager_dequantize is originals[modules[0]]
    assert compile_calls == [
        (
            originals[modules[0]],
            {
                "fullgraph": True,
                "mode": "max-autotune-no-cudagraphs",
                "recompile_limit": 64,
            },
        )
    ]


def test_checkpoint_restore_forces_bf16_adapter_load(
    monkeypatch,
) -> None:
    calls = []

    class AdapterModel(torch.nn.Module):
        def load_adapter(self, *args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

    model = AdapterModel()
    trainer = object.__new__(BF16AdapterTrainer)
    trainer.model = model

    def parent_load(
        self: Trainer, resume_from_checkpoint: str, model: Any = None
    ) -> str:
        del self
        assert model is not None
        assert "load_adapter" in model.__dict__
        model.load_adapter(resume_from_checkpoint, autocast_adapter_dtype=True)
        return "loaded"

    monkeypatch.setattr(Trainer, "_load_from_checkpoint", parent_load)

    result = trainer._load_from_checkpoint("checkpoint", model=model)
    assert result == "loaded"
    assert calls == [(("checkpoint",), {"autocast_adapter_dtype": False})]
