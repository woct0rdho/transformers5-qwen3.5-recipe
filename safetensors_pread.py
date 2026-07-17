from contextlib import contextmanager
from functools import wraps

from transformers import modeling_utils


@contextmanager
def force_transformers_safetensors_pread():
    """Force tensor-wise ``pread`` only during a Transformers model load.

    Current Transformers explicitly requests safetensors' ``mmap`` backend for
    non-MPS checkpoints, overriding ``SAFETENSORS_BACKEND``. On this Strix Halo
    UMA machine, faulted mmap pages coexist with ROCm model allocations in the
    same physical-memory pool. The resulting near-double residency makes this
    checkpoint reclaim/thrash late in loading.

    Keep this override around only the blocking ``from_pretrained`` call so
    unrelated safetensors users retain their normal backend selection.
    """
    original_safe_open = modeling_utils.safe_open

    @wraps(original_safe_open)
    def safe_open_with_pread(*args, **kwargs):
        # Deliberately replace Transformers' explicit backend="mmap".
        kwargs["backend"] = "pread"
        return original_safe_open(*args, **kwargs)

    modeling_utils.safe_open = safe_open_with_pread
    yield
    modeling_utils.safe_open = original_safe_open
