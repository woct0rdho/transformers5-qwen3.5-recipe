"""Keep the PLE n-gram table in the GGUF file and gather its rows per forward.

The PLE table is 26.82 GiB of the checkpoint's 61.85 GiB for a single embedding site, and a forward reads
sixteen 90-byte rows per token. The payload therefore never needs to be resident: what a forward consumes is
a few MiB of rows, which this module reads from the file, uploads, and dequantizes with the same kernel the
resident path uses, so the values are identical to it by construction.

What is kept where:
- the payload stays in the GGUF file. A hook on the GGUF quantizer's state dict replaces the tensor with an
  empty placeholder, so `from_pretrained` never materializes it, and a second hook reports zero bytes for it
  so the allocator warmup does not reserve the checkpoint's size either.
- the rows are gathered on the host per forward into a pinned buffer, and only those rows are uploaded.
- the dequantization runs on the device through `GgufQuantizedParameter.dequantize`, exactly as it does
  for a resident payload.
- the recomputation of the layer under gradient checkpointing gathers nothing: the payload of a lookup is
  cached and matched by its own row ids, so a repeated lookup is free and a wrong prediction is harmless.

The reader is the shape `llama.cpp` uses for its lazy PLE tensors (`llama-lazy-reader.cpp`): sort the row ids,
then let several threads issue one `pread` per row from a shared descriptor, with `POSIX_FADV_RANDOM` set
because a readahead window is bandwidth taken from rows that were not asked for. Sorting turns each worker's
reads into a forward sweep the block layer can reorder. This machine serves a 90-byte read in a whole 4 KiB
page, so what the gather costs is one random page read per distinct row, which is why the reader is worth
having and why the useful bytes are a small fraction of what the device moves.

`prefetch_ple_rows()` starts that gather for a batch whose tokens are already known on the host. A training
loop that reads its next batch while the current step's backward is still running hides the reads entirely.
`PleDiskState` keeps a few prefetched payloads so accumulating several batches does not evict them.

Every access fails closed: an unexpected tensor shape or type, a table that still holds a payload, a row id
outside the table, and a duplicate row id in one lookup all raise rather than serve the wrong bytes.
"""

import contextlib
import hashlib
import os
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from module_patching import ModulePatchSpec, patch_module_forwards

GGUF_TENSOR = "per_layer_token_embd.weight"
GGUF_QUANT_TYPE = 20  # IQ4_NL
BLOCK_ELEMENTS = 32
BLOCK_BYTES = 18
PATCH_MARKER = "_patched_ple_disk_residency"
HANDLED_KEY = "ple_disk_sites"
SKIPPED_KEY = "skipped_geometry"
SITE_SUFFIX = "ple_embedding.ngram_embedding"
WEIGHT_SUFFIX = f"{SITE_SUFFIX}.weight"
LAYER_TYPE_NAME = "Qwen4ExpTextNGramEmbedding"

# The work is I/O queue depth rather than compute, so the reader oversubscribes the cores.
DEFAULT_WORKERS = 64
# Rows per worker slice: a small lookup should not wake every thread.
ROWS_PER_TASK = 256
# The gather buffer is pinned so the transfer that follows is a direct DMA.
PINNED_STAGING_LIMIT_BYTES = 64 << 20
# Prefetched payloads kept for the forward that will ask for them.
PREFETCH_SLOTS = 4


class PleDiskError(RuntimeError):
    """Raised when the disk-backed path cannot be served exactly."""


@dataclass(frozen=True)
class PleTableGeometry:
    """Where the PLE payload lives in the GGUF file, and what its rows look like."""

    path: Path
    offset: int
    nbytes: int
    rows: int
    columns: int
    row_bytes: int
    ggml_type: int = GGUF_QUANT_TYPE

    @classmethod
    def from_header(
        cls, header: Any, *, tensor: str = GGUF_TENSOR
    ) -> "PleTableGeometry":
        for info in header.tensors:
            if info.name == tensor:
                if info.ggml_type != GGUF_QUANT_TYPE:
                    raise PleDiskError(
                        f"{tensor} is GGML type {info.ggml_type}, expected {GGUF_QUANT_TYPE} (IQ4_NL)"
                    )
                columns = int(info.shape[-1])
                if columns % BLOCK_ELEMENTS:
                    raise PleDiskError(
                        f"{tensor} has {columns} columns, not whole IQ4_NL blocks"
                    )
                return cls(
                    path=Path(header.path),
                    offset=int(header.data_start + info.offset),
                    nbytes=int(info.nbytes),
                    rows=int(info.shape[0]),
                    columns=columns,
                    row_bytes=columns // BLOCK_ELEMENTS * BLOCK_BYTES,
                )
        raise PleDiskError(f"{tensor} is not in {header.path}")

    @classmethod
    def from_file(
        cls, path: Path | str, *, tensor: str = GGUF_TENSOR
    ) -> "PleTableGeometry":
        from transformers.integrations.gguf.reader import GgufHeader

        return cls.from_header(GgufHeader.from_file(str(path)), tensor=tensor)


class PleRowReader:
    """Sorted multi-threaded `pread` over the payload, into a pinned staging tensor."""

    def __init__(self, geometry: PleTableGeometry, *, workers: int = DEFAULT_WORKERS):
        self.geometry = geometry
        self.workers = max(1, workers)
        self._fd: int = os.open(geometry.path, os.O_RDONLY)
        os.posix_fadvise(
            self._fd, geometry.offset, geometry.nbytes, os.POSIX_FADV_RANDOM
        )
        self._pool = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="ple-rows"
        )
        self.bytes_read = 0
        self.gathers = 0
        self.read_seconds = 0.0
        self.sort_seconds = 0.0

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __del__(self) -> None:
        # Closing is best effort here: an already-closed descriptor and a pool that is shutting down with
        # the interpreter both raise, and neither is worth reporting from a finalizer.
        with contextlib.suppress(OSError, RuntimeError):
            self.close()

    def _staging(self, rows: int) -> torch.Tensor:
        """A buffer for the gathered rows: pinned while they fit the staging limit."""
        pinned = rows * self.geometry.row_bytes <= PINNED_STAGING_LIMIT_BYTES
        return torch.empty(
            (rows, self.geometry.row_bytes), dtype=torch.uint8, pin_memory=pinned
        )

    def gather(self, rows: np.ndarray) -> torch.Tensor:
        """The packed payload of `rows` (int64, distinct, any order) as a pinned uint8 [n, row_bytes]."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1:
            raise PleDiskError(f"expected a 1-D row vector, got shape {rows.shape}")
        if rows.size == 0:
            return torch.empty((0, self.geometry.row_bytes), dtype=torch.uint8)
        lowest, highest = int(np.min(rows)), int(np.max(rows))
        if lowest < 0 or highest >= self.geometry.rows:
            raise PleDiskError(
                f"row id outside [0, {self.geometry.rows}): {lowest}..{highest}"
            )
        if np.unique(rows).size != rows.size:
            raise PleDiskError(
                "the row vector passed to the reader must already be deduplicated"
            )

        fd = self._fd
        if fd < 0:
            raise PleDiskError("the row reader is closed")
        out = self._staging(rows.size)
        view = memoryview(out.numpy().reshape(-1))
        started = time.perf_counter()
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        self.sort_seconds += time.perf_counter() - started
        base = self.geometry.offset
        row_bytes = self.geometry.row_bytes
        tasks = max(
            1, min(self.workers, (rows.size + ROWS_PER_TASK - 1) // ROWS_PER_TASK)
        )

        def slice_reads(worker: int, fd: int = fd) -> None:
            begin = rows.size * worker // tasks
            end = rows.size * (worker + 1) // tasks
            for position in range(begin, end):
                row = int(sorted_rows[position])
                destination = int(order[position]) * row_bytes
                done = os.preadv(
                    fd,
                    [view[destination : destination + row_bytes]],
                    base + row * row_bytes,
                )
                if done != row_bytes:
                    raise PleDiskError(
                        f"short read of row {row}: {done} of {row_bytes} bytes"
                    )

        started = time.perf_counter()
        if tasks == 1:
            slice_reads(0)
        else:
            list(self._pool.map(slice_reads, range(tasks)))
        self.read_seconds += time.perf_counter() - started
        self.bytes_read += int(rows.size) * row_bytes
        self.gathers += 1
        return out

    def read_payload_range(self, start: int, length: int) -> bytes:
        """Bytes straight out of the file, for identity checks that want the payload itself."""
        if start < 0 or length < 0 or start + length > self.geometry.nbytes:
            raise PleDiskError(
                f"range {start}..{start + length} is outside the {self.geometry.nbytes}-byte payload"
            )
        fd = self._fd
        if fd < 0:
            raise PleDiskError("the row reader is closed")
        chunks: list[bytes] = []
        done = 0
        while done < length:
            chunk = os.pread(fd, length - done, self.geometry.offset + start + done)
            if not chunk:
                raise PleDiskError(
                    f"payload ended after {done} of {length} bytes at {start}"
                )
            chunks.append(chunk)
            done += len(chunk)
        return b"".join(chunks)

    def sample_identity(self, *, chunk: int = 1024) -> dict[str, Any]:
        """The first, middle and last bytes of the payload, as the in-memory sampler would read them."""
        geometry = self.geometry
        chunk = min(chunk, geometry.nbytes)
        starts = (0, max((geometry.nbytes - chunk) // 2, 0), geometry.nbytes - chunk)
        sample = b"".join(self.read_payload_range(start, chunk) for start in starts)
        return {
            "name": GGUF_TENSOR,
            "sample_sha256": hashlib.sha256(sample).hexdigest(),
            "bytes": geometry.nbytes,
            "quant_type": geometry.ggml_type,
            "logical_shape": [geometry.rows, geometry.columns],
        }


@dataclass(frozen=True)
class NgramIdSource:
    """The layer's n-gram arithmetic, as plain arrays, so rows can be resolved from token ids alone."""

    multipliers: np.ndarray  # [ngram_size]
    head_vocab_sizes: np.ndarray  # [heads_per_ngram * (ngram_size - 1)]
    head_offsets: np.ndarray
    ngram_size: int
    heads_per_ngram: int
    context_len: int
    eos_token_id: int

    @classmethod
    def from_layer(cls, layer: torch.nn.Module) -> "NgramIdSource":
        source = cast(Any, layer)

        def array(name: str) -> np.ndarray:
            value = getattr(source, name, None)
            if value is None:
                raise PleDiskError(f"{type(layer).__name__} has no {name}")
            return np.asarray(
                value.detach().to("cpu", non_blocking=False), dtype=np.int64
            ).reshape(-1)

        return cls(
            multipliers=array("layer_multipliers"),
            head_vocab_sizes=array("ngram_heads_vocab_sizes"),
            head_offsets=array("ngram_heads_offsets"),
            ngram_size=int(source.ngram_size),
            heads_per_ngram=int(source.heads_per_ngram),
            context_len=int(source.context_len),
            eos_token_id=int(source.eos_token_id),
        )

    def head_count(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    def ids_from_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """The table's row ids for one sequence: [tokens, heads], the layer's own arithmetic in numpy.

        The layer's forward computes this on the device from the batch's token ids, so this only has to
        agree with it for the rows the forward will ask for. It is a pure function of the token ids, and the
        result is never trusted on its own: a prefetched payload is only used when the row ids the layer
        computes for the same lookup match it, so a disagreement costs a gather and nothing else.
        """
        tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        if tokens.size == 0:
            raise PleDiskError("expected a non-empty token sequence")
        if self.multipliers.size < self.ngram_size:
            raise PleDiskError(
                f"{self.multipliers.size} multipliers for an n-gram of {self.ngram_size}"
            )
        heads = self.head_count()
        if self.head_vocab_sizes.size != heads or self.head_offsets.size != heads:
            raise PleDiskError(
                f"expected {heads} heads, got {self.head_vocab_sizes.size} sizes and {self.head_offsets.size} offsets"
            )

        history = np.concatenate(
            [np.full(self.context_len, self.eos_token_id, dtype=np.int64), tokens]
        )
        positions = np.arange(history.size, dtype=np.int64)
        eos_positions = np.where(history == self.eos_token_id, positions, -1)
        previous_eos = np.concatenate(
            [np.full(1, -1, dtype=np.int64), np.maximum.accumulate(eos_positions)[:-1]]
        )
        position_in_segment = positions - (previous_eos + 1)
        shifted = []
        for shift in range(self.ngram_size):
            source = positions - shift
            shifted.append(
                np.where(
                    (position_in_segment >= shift) & (source >= 0),
                    history[np.clip(source, 0, None)],
                    self.eos_token_id,
                )[-tokens.size :]
            )

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            width = self.head_vocab_sizes[start : start + self.heads_per_ngram]
            mixed = shifted[0] * self.multipliers[0]
            for position in range(1, ngram):
                with np.errstate(over="ignore"):
                    mixed = np.bitwise_xor(
                        mixed, shifted[position] * self.multipliers[position]
                    )
            with np.errstate(over="ignore"):
                mixed = np.remainder(mixed[:, None], width[None, :])
            blocks.append(
                mixed + self.head_offsets[None, start : start + self.heads_per_ngram]
            )
        return np.concatenate(blocks, axis=1)


@dataclass
class PleDiskState:
    """The installed reader, the payload caches, and the counters the audit reports."""

    geometry: PleTableGeometry
    reader: PleRowReader
    identity: dict[str, Any] = field(default_factory=dict)
    layer: torch.nn.Module | None = None
    id_source: NgramIdSource | None = None
    cache_hits: int = 0
    cache_misses: int = 0
    prefetches: int = 0
    prefetch_hits: int = 0
    prefetch_rows: int = 0
    seconds: dict[str, float] = field(
        default_factory=lambda: {
            "unique": 0.0,
            "d2h": 0.0,
            "gather": 0.0,
            "h2d": 0.0,
            "dequantize": 0.0,
            "prefetch_ids": 0.0,
            "rows": 0.0,
        }
    )
    _cached_rows: np.ndarray | None = None
    _cached_payload: torch.Tensor | None = None
    _prefetched: list[tuple[np.ndarray, torch.Tensor]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def cached(self, rows: np.ndarray) -> torch.Tensor | None:
        """The payload of this exact lookup, if this step gathered it or a prefetch predicted it."""
        with self._lock:
            for index, (prefetched_rows, payload) in enumerate(self._prefetched):
                if prefetched_rows.size == rows.size and np.array_equal(
                    prefetched_rows, rows
                ):
                    self._prefetched.pop(index)
                    self.prefetch_hits += 1
                    self._cached_rows, self._cached_payload = rows, payload
                    return payload
            if (
                self._cached_rows is not None
                and self._cached_rows.size == rows.size
                and np.array_equal(self._cached_rows, rows)
            ):
                self.cache_hits += 1
                return self._cached_payload
            return None

    def remember(self, rows: np.ndarray, payload: torch.Tensor) -> None:
        with self._lock:
            self._cached_rows = rows
            self._cached_payload = payload

    def prefetch(self, rows: np.ndarray) -> int:
        """Gather `rows` (int64) and keep the payload for the forward that will ask for it."""
        rows = np.asarray(rows, dtype=np.int64)
        payload = self.reader.gather(rows)
        with self._lock:
            self._prefetched.append((rows, payload))
            del self._prefetched[:-PREFETCH_SLOTS]
            self.prefetches += 1
            self.prefetch_rows += int(rows.size)
        return int(rows.size)

    def report(self) -> dict[str, Any]:
        return {
            "disk_backed": True,
            "path": str(self.geometry.path),
            "offset": self.geometry.offset,
            "payload_bytes": self.geometry.nbytes,
            "rows": self.geometry.rows,
            "columns": self.geometry.columns,
            "row_bytes": self.geometry.row_bytes,
            "ggml_type": self.geometry.ggml_type,
            "workers": self.reader.workers,
            "gathers": self.reader.gathers,
            "bytes_read": self.reader.bytes_read,
            "read_seconds": round(self.reader.read_seconds, 4),
            "sort_seconds": round(self.reader.sort_seconds, 4),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "prefetches": self.prefetches,
            "prefetch_rows": self.prefetch_rows,
            "prefetch_hits": self.prefetch_hits,
            "seconds": {key: round(value, 4) for key, value in self.seconds.items()},
            "identity": dict(self.identity),
        }


_STATE: PleDiskState | None = None
_PENDING: dict[str, Any] = {}
_HOOKS_INSTALLED = False


def disk_state() -> PleDiskState | None:
    """The installed disk state, or None while the table is resident in memory."""
    return _STATE


def placeholder_for(geometry: PleTableGeometry) -> torch.nn.Parameter:
    """The parameter the loader installs: the table's geometry, no payload.

    A plain uint8 parameter of zero rows, deliberately not a `GgufQuantizedParameter`: the packed class
    validates its payload against `logical_shape` whenever it is rebuilt, and the loader rebuilds it with
    `.to(device=...)`, so an empty payload cannot carry the table's shape through the load. The loader
    compares `logical_shape` when it is present and nothing when it is not, so the placeholder stays plain
    and the geometry lives here, where the gather and the audit's identity check read it.
    """
    return torch.nn.Parameter(
        torch.empty((0, geometry.row_bytes), dtype=torch.uint8), requires_grad=False
    )


class _PleDiskSource:
    """Stands in for the lazy GGUF source of the PLE tensor, so its bytes are never read at load."""

    def __init__(self, geometry: PleTableGeometry, name: str):
        self.geometry = geometry
        self.name = name

    def __getitem__(self, key: Any) -> Any:
        if key is not Ellipsis:
            raise PleDiskError(
                f"{self.name}: the disk-backed PLE source is only readable as a whole tensor, got {key!r}"
            )
        return placeholder_for(self.geometry)

    def __repr__(self) -> str:
        return f"_PleDiskSource({self.name!r}, {self.geometry.rows} rows of {self.geometry.row_bytes} B)"


def _install_loader_hooks() -> None:
    """Hook the GGUF quantizer so the PLE tensor is a placeholder and is not reserved by the warmup."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    from transformers.quantizers.quantizer_gguf import GgufHfQuantizer

    original_state_dict = GgufHfQuantizer.get_state_dict
    original_element_size = GgufHfQuantizer.param_element_size

    def get_state_dict(self, checkpoint_file, model):
        state_dict = original_state_dict(self, checkpoint_file, model)
        header = getattr(self, "header", None)
        if header is None or GGUF_TENSOR not in state_dict:
            return state_dict
        geometry = PleTableGeometry.from_header(header)
        _PENDING["geometry"] = geometry
        state_dict[GGUF_TENSOR] = _PleDiskSource(geometry, GGUF_TENSOR)
        return state_dict

    def param_element_size(self, model, param_name, param):
        # `modeling_utils.caching_allocator_warmup` sizes one device allocation from these element sizes
        # before anything is materialized, and the pool keeps it for the rest of the run.
        if param_name.endswith(WEIGHT_SUFFIX):
            return 0.0
        return original_element_size(self, model, param_name, param)

    GgufHfQuantizer.get_state_dict = get_state_dict
    GgufHfQuantizer.param_element_size = param_element_size
    _HOOKS_INSTALLED = True


def _ple_embedding_forward(module: Any, input: torch.Tensor) -> torch.Tensor:
    """`GgufEmbedding.forward` with the packed rows gathered from the file instead of a resident payload."""
    state = _STATE
    if state is None:
        raise PleDiskError(
            "the PLE disk state is not installed. Call require_ple_disk_residency()"
        )
    if module.weight.numel():
        raise PleDiskError(
            "the PLE table still holds a payload. The loader hooks did not take effect"
        )

    started = time.perf_counter()
    rows, inverse = torch.unique(input.reshape(-1), sorted=False, return_inverse=True)
    state.seconds["unique"] += time.perf_counter() - started
    started = time.perf_counter()
    rows_cpu = (
        rows.detach().to("cpu", non_blocking=False).numpy().astype(np.int64, copy=False)
    )
    state.seconds["d2h"] += time.perf_counter() - started
    state.seconds["rows"] += rows_cpu.size
    payload = state.cached(rows_cpu)
    if payload is None:
        state.cache_misses += 1
        started = time.perf_counter()
        payload = state.reader.gather(rows_cpu)
        state.seconds["gather"] += time.perf_counter() - started
        state.remember(rows_cpu, payload)

    from transformers.integrations.gguf.gguf_quantized_parameter import (
        GgufQuantizedParameter,
    )

    started = time.perf_counter()
    payload = payload.to(input.device, non_blocking=True)
    state.seconds["h2d"] += time.perf_counter() - started
    packed = GgufQuantizedParameter(
        payload,
        quant_type=state.geometry.ggml_type,
        logical_shape=(rows.numel(), state.geometry.columns),
    )
    started = time.perf_counter()
    selected = packed.dequantize(dtype=module.compute_dtype, device=input.device)
    result = selected.index_select(0, inverse.to(selected.device)).reshape(
        *input.shape, module.embedding_dim
    )
    state.seconds["dequantize"] += time.perf_counter() - started
    return result


def configure_ple_disk_residency(
    *,
    checkpoint: Path | str | None = None,
    enabled: bool = True,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    """Make the next `from_pretrained` leave the PLE payload in the file. Call before the model is created."""
    if not enabled:
        return {"enabled": False}
    _install_loader_hooks()
    report: dict[str, Any] = {"enabled": True, "workers": workers}
    if checkpoint is not None:
        geometry = PleTableGeometry.from_file(checkpoint)
        _PENDING["geometry"] = geometry
        report["geometry"] = str(geometry)
    return report


def _find_layer(model: torch.nn.Module) -> torch.nn.Module:
    """The module that owns the table, which is where the n-gram id arithmetic lives."""
    for _name, module in model.named_modules():
        if (
            type(module).__name__ == LAYER_TYPE_NAME
            and getattr(module, "ngram_embedding", None) is not None
        ):
            return module
    raise PleDiskError(f"the model has no {LAYER_TYPE_NAME} owning a PLE table")


def require_ple_disk_residency(
    model: torch.nn.Module,
    *,
    expected_rows: int | None = None,
    expected_dim: int | None = None,
    expected_sites: int = 1,
) -> dict[str, Any]:
    """Install the disk-backed forward on the PLE table and fail closed on anything unexpected."""
    global _STATE
    geometry: PleTableGeometry | None = _PENDING.get("geometry")
    if geometry is None:
        raise PleDiskError(
            "configure_ple_disk_residency() was not called before the model was loaded"
        )
    if _STATE is None:
        reader = PleRowReader(geometry)
        _STATE = PleDiskState(
            geometry=geometry, reader=reader, identity=reader.sample_identity()
        )
    state = _STATE
    # The checkpoint is the authority on the table's shape. The caller only says it out loud when it wants
    # the load checked against a documented number.
    expected_rows = geometry.rows if expected_rows is None else expected_rows
    expected_dim = geometry.columns if expected_dim is None else expected_dim

    from transformers.integrations.gguf.modules import GgufEmbedding

    def matches(name: str, module: torch.nn.Module) -> bool:
        return name.endswith(SITE_SUFFIX)

    def validate(name: str, module: GgufEmbedding) -> None:
        weight = module.weight
        if not isinstance(weight, torch.nn.Parameter):
            raise PleDiskError(f"{name}: the PLE table is not a parameter")
        if weight.numel():
            raise PleDiskError(
                f"{name}: the PLE table still holds {weight.numel()} payload bytes. The loader hooks did not take effect"
            )
        if weight.dtype != torch.uint8 or tuple(weight.shape) != (
            0,
            geometry.row_bytes,
        ):
            raise PleDiskError(
                f"{name}: the PLE placeholder is {tuple(weight.shape)} {weight.dtype}, expected (0, {geometry.row_bytes}) uint8"
            )
        if (
            int(module.num_embeddings) != expected_rows
            or int(module.embedding_dim) != expected_dim
        ):
            raise PleDiskError(
                f"{name}: the PLE table is [{module.num_embeddings}, {module.embedding_dim}], "
                f"expected [{expected_rows}, {expected_dim}]"
            )

    report = patch_module_forwards(
        model,
        [
            ModulePatchSpec(
                module_type=GgufEmbedding,
                forward=_ple_embedding_forward,
                handled_key=HANDLED_KEY,
                matches=matches,
                validate=validate,
                marker=PATCH_MARKER,
                freeze_weight=False,
            )
        ],
        declared_keys=(SKIPPED_KEY,),
    )
    if report[HANDLED_KEY] != expected_sites:
        raise PleDiskError(
            f"expected {expected_sites} PLE table(s), patched {report[HANDLED_KEY]}"
        )
    if report[SKIPPED_KEY]:
        raise PleDiskError(f"{report[SKIPPED_KEY]} PLE table(s) were skipped")
    state.layer = _find_layer(model)
    state.id_source = NgramIdSource.from_layer(state.layer)
    report.update(state.report())
    return report


def prefetch_ple_rows(
    model: torch.nn.Module, input_ids: torch.Tensor | np.ndarray | Sequence[int]
) -> int:
    """Read the rows a batch of token ids will ask for, so the forward that follows finds them cached.

    The tokens have to be on the host: a training loop reads its next batch while the current step is still
    on the device, and copying them from the device would wait for work that is already queued. Returns the
    number of rows read, and zero when the table is resident and there is nothing to prefetch.
    """
    state = _STATE
    if state is None:
        return 0
    if state.id_source is None or state.layer is None:
        raise PleDiskError(
            "the PLE layer was not recorded. Call require_ple_disk_residency() first"
        )
    if isinstance(input_ids, torch.Tensor):
        if input_ids.device.type != "cpu":
            raise PleDiskError(
                "prefetch_ple_rows() needs host token ids, not a device tensor"
            )
        tokens = input_ids.detach().numpy()
    else:
        tokens = np.asarray(input_ids)
    started = time.perf_counter()
    ids = state.id_source.ids_from_tokens(tokens)
    with state._lock:
        state.seconds["prefetch_ids"] += time.perf_counter() - started
    if ids.min() < 0 or ids.max() >= state.geometry.rows:
        raise PleDiskError(
            f"the id arithmetic produced a row outside the table: {int(ids.min())}..{int(ids.max())}"
        )
    unique = np.unique(ids.reshape(-1))
    return state.prefetch(unique)
