"""Pytree dataclasses (SANDBOX.md §2).

State is a pure-function pytree: a snapshot is the state itself, a fork is a
copy, and replay is a rerun.  Frozen dataclasses with array fields are the only
container type used for world state.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence, TypeVar

import jax.tree_util as jtu

T = TypeVar("T")


def struct(cls: type[T], *, meta: Sequence[str] = ()) -> type[T]:
    """Register a dataclass as a JAX pytree node.

    The meta argument names fields treated as static metadata (Python values,
    never arrays); they must be hashable because JAX uses them in jit cache
    keys.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError("struct() needs a dataclass, got " + repr(cls))
    names = [f.name for f in dataclasses.fields(cls)]
    meta_names = list(meta)
    unknown = [name for name in meta_names if name not in names]
    if unknown:
        raise ValueError("unknown metadata fields on " + cls.__name__ + ": " + repr(unknown))
    data = [name for name in names if name not in meta_names]
    jtu.register_dataclass(cls, data_fields=data, meta_fields=meta_names)
    return cls


def update(obj: T, **updates: Any) -> T:
    """Return a copy with the given updates applied (dataclasses.replace)."""
    return dataclasses.replace(obj, **updates)
