"""Type-checking shim for `torch.nn.Module`.

PyTorch deliberately declares `Module.forward` as a writable
`Callable[..., Any]` instead of a method, to work around mypy's method-override
checking (pytorch/pytorch#42356). ty's `invalid-mutable-override` rule reports a
method that narrows such a writable attribute, because ty does not model an
arbitrary `Callable` as a bound-method descriptor (astral-sh/ty#491). Its
`liskov.md` tests pin this behavior.

Under type checking, redeclare `forward` as a method once here, which turns the
one torch-boundary mismatch into the single suppression below. Subclasses then
override a method. At runtime `TorchModule` is exactly `torch.nn.Module`, so
inheritance, MRO and behavior are unchanged. If ty starts treating callable
class attributes as method descriptors, this shim can be removed.
"""

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:

    class TorchModule(torch.nn.Module):
        def forward(self, *args: Any, **kwargs: Any) -> Any:  # ty: ignore[invalid-mutable-override]
            ...

else:
    TorchModule = torch.nn.Module
