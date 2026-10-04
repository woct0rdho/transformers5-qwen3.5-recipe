"""The disk-backed PLE table must return exactly the rows the resident payload would.

Every test here runs on CPU with a synthetic IQ4_NL table in a temporary file, so the suite does not need
the 61.85 GiB checkpoint: the reader's bytes are compared against the file, the patched embedding against a
module that holds the same payload in memory, and the failure modes against the reader's own validation.
"""

import hashlib
from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from transformers.integrations.gguf.gguf_quantized_parameter import (
    GgufQuantizedParameter,
)
from transformers.integrations.gguf.modules import GgufEmbedding

import ple_disk_residency as ple
from ple_disk_residency import (
    NgramIdSource,
    PleDiskError,
    PleDiskState,
    PleRowReader,
    PleTableGeometry,
    _PleDiskSource,
    placeholder_for,
)

_ROWS = 512
_COLUMNS = 160
_ROW_BYTES = _COLUMNS // ple.BLOCK_ELEMENTS * ple.BLOCK_BYTES
_PREFIX = 1234  # a non-zero, non-aligned offset, as a real GGUF region has


def _payload(rows: int = _ROWS, seed: int = 7) -> bytes:
    """A valid IQ4_NL payload: an fp16 scale of 1.0 followed by sixteen 4-bit code bytes per block."""
    rng = np.random.default_rng(seed)
    block = np.concatenate(
        [
            np.array([0x00, 0x3C], dtype=np.uint8),
            rng.integers(0, 256, 16, dtype=np.uint8),
        ]
    )
    return np.tile(block, (rows * _COLUMNS // ple.BLOCK_ELEMENTS, 1)).tobytes()


def _table(tmp_path, rows: int = _ROWS) -> tuple[PleTableGeometry, bytes]:
    payload = _payload(rows)
    path = tmp_path / "table.bin"
    path.write_bytes(b"\x5a" * _PREFIX + payload + b"\xa5" * 64)
    geometry = PleTableGeometry(
        path=path,
        offset=_PREFIX,
        nbytes=len(payload),
        rows=rows,
        columns=_COLUMNS,
        row_bytes=_ROW_BYTES,
    )
    return geometry, payload


def _state(geometry: PleTableGeometry) -> PleDiskState:
    reader = PleRowReader(geometry, workers=4)
    return PleDiskState(
        geometry=geometry, reader=reader, identity=reader.sample_identity()
    )


def _module(
    geometry: PleTableGeometry, payload: bytes, *, resident: bool
) -> GgufEmbedding:
    module = GgufEmbedding(geometry.rows, geometry.columns, compute_dtype=torch.float32)
    if resident:
        tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(
            geometry.rows, geometry.row_bytes
        )
        module.weight = GgufQuantizedParameter(
            tensor.clone(),
            quant_type=geometry.ggml_type,
            logical_shape=(geometry.rows, geometry.columns),
        )
    else:
        module.weight = placeholder_for(geometry)
        cast(Any, module).forward = MethodType(ple._ple_embedding_forward, module)
    return module


def test_reader_returns_the_file_bytes(tmp_path) -> None:
    geometry, payload = _table(tmp_path)
    rows = np.array([3, 511, 0, 128, 129, 42], dtype=np.int64)
    gathered = PleRowReader(geometry, workers=4).gather(rows)
    assert gathered.shape == (rows.size, _ROW_BYTES)
    for position, row in enumerate(rows):
        start = int(row) * _ROW_BYTES
        assert bytes(gathered[position].numpy()) == payload[start : start + _ROW_BYTES]


def test_reader_matches_a_plain_read_of_many_rows(tmp_path) -> None:
    geometry, payload = _table(tmp_path)
    rows = (
        np.random.default_rng(3)
        .choice(geometry.rows, size=300, replace=False)
        .astype(np.int64)
    )
    gathered = PleRowReader(geometry, workers=8).gather(rows).numpy()
    expected = np.frombuffer(payload, dtype=np.uint8).reshape(
        geometry.rows, _ROW_BYTES
    )[rows]
    assert np.array_equal(gathered, expected)


def test_reader_rejects_rows_it_cannot_serve(tmp_path) -> None:
    geometry, _ = _table(tmp_path)
    reader = PleRowReader(geometry, workers=2)
    with pytest.raises(PleDiskError, match="outside"):
        reader.gather(np.array([geometry.rows], dtype=np.int64))
    with pytest.raises(PleDiskError, match="deduplicated"):
        reader.gather(np.array([1, 1], dtype=np.int64))
    with pytest.raises(PleDiskError, match="1-D"):
        reader.gather(np.zeros((2, 2), dtype=np.int64))
    assert reader.gather(np.zeros(0, dtype=np.int64)).shape == (0, _ROW_BYTES)


def test_identity_samples_the_file(tmp_path) -> None:
    geometry, payload = _table(tmp_path)
    reader = PleRowReader(geometry, workers=2)
    identity = reader.sample_identity(chunk=1024)
    middle = max((len(payload) - 1024) // 2, 0)
    expected = payload[:1024] + payload[middle : middle + 1024] + payload[-1024:]
    assert identity["sample_sha256"] == hashlib.sha256(expected).hexdigest()
    assert identity["bytes"] == geometry.nbytes
    assert identity["logical_shape"] == [geometry.rows, geometry.columns]


def test_geometry_reads_a_header_and_fails_closed(tmp_path) -> None:
    geometry, payload = _table(tmp_path)
    header = SimpleNamespace(
        path=str(geometry.path),
        data_start=geometry.offset,
        tensors=[
            SimpleNamespace(
                name=ple.GGUF_TENSOR,
                ggml_type=ple.GGUF_QUANT_TYPE,
                shape=(_ROWS, _COLUMNS),
                nbytes=len(payload),
                offset=0,
            )
        ],
    )
    parsed = PleTableGeometry.from_header(header)
    assert (parsed.offset, parsed.rows, parsed.columns, parsed.row_bytes) == (
        geometry.offset,
        geometry.rows,
        geometry.columns,
        geometry.row_bytes,
    )
    with pytest.raises(PleDiskError, match="IQ4_NL"):
        PleTableGeometry.from_header(
            SimpleNamespace(
                path="x",
                data_start=0,
                tensors=[
                    SimpleNamespace(
                        name=ple.GGUF_TENSOR,
                        ggml_type=2,
                        shape=(4, 160),
                        nbytes=1,
                        offset=0,
                    )
                ],
            )
        )
    with pytest.raises(PleDiskError, match="whole IQ4_NL blocks"):
        PleTableGeometry.from_header(
            SimpleNamespace(
                path="x",
                data_start=0,
                tensors=[
                    SimpleNamespace(
                        name=ple.GGUF_TENSOR,
                        ggml_type=ple.GGUF_QUANT_TYPE,
                        shape=(4, 100),
                        nbytes=1,
                        offset=0,
                    )
                ],
            )
        )
    with pytest.raises(PleDiskError, match="not in"):
        PleTableGeometry.from_header(
            SimpleNamespace(path="x", data_start=0, tensors=[])
        )


def test_placeholder_holds_no_payload_and_keeps_its_geometry(tmp_path) -> None:
    geometry, _ = _table(tmp_path)
    weight = placeholder_for(geometry)
    assert isinstance(weight, torch.nn.Parameter)
    assert weight.numel() == 0
    assert weight.dtype == torch.uint8
    assert tuple(weight.shape) == (0, geometry.row_bytes)
    assert weight.requires_grad is False
    # the loader rebuilds the parameter with `.to(...)`, which a packed subclass would refuse for an empty payload
    moved = weight.to(device="cpu", dtype=torch.uint8)
    assert moved.numel() == 0


def test_source_materializes_the_placeholder_only(tmp_path) -> None:
    geometry, _ = _table(tmp_path)
    source = _PleDiskSource(geometry, ple.GGUF_TENSOR)
    weight = source[...]
    assert weight.numel() == 0
    assert tuple(weight.shape) == (0, geometry.row_bytes)
    with pytest.raises(PleDiskError, match="only readable as a whole tensor"):
        source[0]


def test_patched_forward_equals_the_resident_payload(tmp_path, monkeypatch) -> None:
    geometry, payload = _table(tmp_path)
    ids = torch.randint(0, geometry.rows, (2, 64, 16), dtype=torch.long)
    resident = _module(geometry, payload, resident=True)
    expected = resident(ids)

    state = _state(geometry)
    monkeypatch.setattr(ple, "_STATE", state)
    patched = _module(geometry, payload, resident=False)
    assert torch.equal(patched(ids), expected)
    # the same lookup again must come out of the step's cache, not off the disk
    assert torch.equal(patched(ids), expected)
    assert state.reader.gathers == 1
    assert state.cache_hits == 1
    # a different input must be gathered
    other = torch.randint(0, geometry.rows, (1, 32, 16), dtype=torch.long)
    assert torch.equal(patched(other), resident(other))
    assert state.reader.gathers == 2


def test_patched_forward_refuses_a_resident_payload(tmp_path, monkeypatch) -> None:
    geometry, payload = _table(tmp_path)
    monkeypatch.setattr(ple, "_STATE", _state(geometry))
    module = _module(geometry, payload, resident=True)
    cast(Any, module).forward = MethodType(ple._ple_embedding_forward, module)
    with pytest.raises(PleDiskError, match="still holds a payload"):
        module(torch.zeros(1, 4, 16, dtype=torch.long))


def test_patching_a_model_installs_once_and_validates(tmp_path, monkeypatch) -> None:
    geometry, _ = _table(tmp_path)
    monkeypatch.setitem(ple._PENDING, "geometry", geometry)
    monkeypatch.setattr(ple, "_STATE", None)
    layer, _recorder = _layer(rows=2048)
    cast(Any, layer).ngram_embedding = _module(geometry, _payload(), resident=False)
    model = torch.nn.Module()
    model.add_module("ple_embedding", layer)
    report: dict[str, Any] = ple.require_ple_disk_residency(
        model, expected_rows=geometry.rows, expected_dim=geometry.columns
    )
    assert report[ple.HANDLED_KEY] == 1
    assert report["disk_backed"] is True
    assert report["payload_bytes"] == geometry.nbytes
    assert report["identity"]["bytes"] == geometry.nbytes
    second = ple.require_ple_disk_residency(
        model, expected_rows=geometry.rows, expected_dim=geometry.columns
    )
    assert second["already_patched"] == 1

    # a table whose shape is not the one the patch was configured for must not be served
    other_layer, _recorder = _layer(rows=2048)
    cast(Any, other_layer).ngram_embedding = _module(
        geometry, _payload(), resident=False
    )
    other = torch.nn.Module()
    other.add_module("ple_embedding", other_layer)
    with pytest.raises(PleDiskError, match="expected"):
        ple.require_ple_disk_residency(other, expected_rows=1, expected_dim=2)


def test_prefetch_fills_the_cache(tmp_path) -> None:
    geometry, payload = _table(tmp_path)
    state = _state(geometry)
    rows = np.array([7, 9, 11], dtype=np.int64)
    assert state.prefetch(rows) == 3
    cached = state.cached(rows)
    assert cached is not None
    expected = np.frombuffer(payload, dtype=np.uint8).reshape(
        geometry.rows, _ROW_BYTES
    )[rows]
    assert np.array_equal(cached.numpy(), expected)
    assert state.cached(np.array([1, 2, 3], dtype=np.int64)) is None
    report = state.report()
    assert report["prefetch_hits"] == 1
    assert report["prefetches"] == 1
    assert report["prefetch_rows"] == 3


def test_a_wrong_prediction_is_ignored(tmp_path) -> None:
    geometry, payload = _table(tmp_path)
    state = _state(geometry)
    state.prefetch(np.array([1, 2], dtype=np.int64))
    # the forward asks for different rows, so the prefetched payload must not be used for them
    assert state.cached(np.array([3, 4], dtype=np.int64)) is None
    gathered = state.reader.gather(np.array([3, 4], dtype=np.int64))
    expected = np.frombuffer(payload, dtype=np.uint8).reshape(
        geometry.rows, _ROW_BYTES
    )[[3, 4]]
    assert np.array_equal(gathered.numpy(), expected)


class _IdRecorder(torch.nn.Module):
    """Stands in for the PLE embedding and keeps the row ids the layer computed for it."""

    def __init__(self, rows: int, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(rows, dim))
        self.seen: torch.Tensor | None = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.seen = input.detach().clone()
        return torch.zeros(*input.shape, self.weight.shape[-1], dtype=self.weight.dtype)


def _layer(
    heads_per_ngram: int = 4, ngram_size: int = 3, head_dim: int = 4, rows: int = 4096
):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextNGramEmbedding,
    )

    heads = (ngram_size - 1) * heads_per_ngram
    base = 128
    sizes = [base + 7 * head for head in range(heads)]
    config = SimpleNamespace(
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        vocab_size=256,
        ngram_vocab_size_base=base,
        make_ngram_vocab_size_divisible_by=32,
        seed=0,
        eos_token_id=3,
        ple_head_vocab_sizes=sizes,
        ple_head_offsets=[sum(sizes[:head]) for head in range(heads)],
        ple_layer_multipliers=[2**40 + 3, 2**33 + 5, 2**27 + 7],
        ple_vocab_size=rows,
    )
    layer = Qwen4ExpTextNGramEmbedding(
        cast(Any, config), embedding_dim=heads * head_dim, layer_idx=0
    )
    recorder = _IdRecorder(rows, head_dim)
    cast(Any, layer).ngram_embedding = recorder
    return layer, recorder


def test_ngram_ids_match_the_layer() -> None:
    layer, recorder = _layer()
    tokens = torch.randint(0, 256, (2, 64), dtype=torch.long)
    layer(tokens, None)
    assert recorder.seen is not None
    ids = NgramIdSource.from_layer(layer).ids_from_tokens(tokens[0].numpy())
    assert ids.shape == (64, (layer.ngram_size - 1) * layer.heads_per_ngram)
    assert np.array_equal(ids, recorder.seen[0].numpy())


def test_ngram_ids_follow_eos_boundaries() -> None:
    layer, recorder = _layer()
    tokens = torch.tensor([[5, 6, 3, 7, 8, 3, 9, 3]], dtype=torch.long)
    layer(tokens, None)
    assert recorder.seen is not None
    ids = NgramIdSource.from_layer(layer).ids_from_tokens(tokens[0].numpy())
    assert np.array_equal(ids, recorder.seen[0].numpy())


def test_prefetch_uses_the_layer_arithmetic(tmp_path, monkeypatch) -> None:
    geometry, payload = _table(tmp_path, rows=2048)
    state = _state(geometry)
    layer, _recorder = _layer(rows=2048)
    state.layer = layer
    state.id_source = NgramIdSource.from_layer(layer)
    monkeypatch.setattr(ple, "_STATE", state)
    tokens = torch.randint(0, 256, (32,), dtype=torch.long)
    assert ple.prefetch_ple_rows(torch.nn.Module(), tokens) > 0
    rows = np.unique(state.id_source.ids_from_tokens(tokens.numpy()).reshape(-1))
    cached = state.cached(rows)
    assert cached is not None
    table = np.frombuffer(payload, dtype=np.uint8).reshape(geometry.rows, _ROW_BYTES)
    assert np.array_equal(cached.numpy(), table[rows])


def test_prefetch_ignores_device_tokens(tmp_path, monkeypatch) -> None:
    geometry, _ = _table(tmp_path, rows=2048)
    state = _state(geometry)
    layer, _recorder = _layer(rows=2048)
    state.layer = layer
    state.id_source = NgramIdSource.from_layer(layer)
    monkeypatch.setattr(ple, "_STATE", state)
    with pytest.raises(PleDiskError, match="host token ids"):
        ple.prefetch_ple_rows(
            torch.nn.Module(), torch.zeros(4, dtype=torch.long, device="meta")
        )
