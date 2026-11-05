"""Stable world-state hash (SANDBOX.md §7.2, §13).

The hash covers the pytree structure and every leaf's dtype, shape and bytes.
It is used by determinism, observer-noninterference and snapshot-resume tests.
"""

from __future__ import annotations

import hashlib
from typing import Any

import jax
import jax.tree_util as jtu
import numpy as np


def state_hash(state: Any) -> str:
    """Hex digest of a world state (16 bytes -> 32 hex chars)."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(repr(jtu.tree_structure(state)).encode())
    for leaf in jtu.tree_leaves(state):
        # PRNG keys cannot be converted to numpy directly.
        if jax.dtypes.issubdtype(getattr(leaf, "dtype", None), jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        array = np.asarray(leaf)
        digest.update(array.dtype.str.encode())
        digest.update(repr(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()
