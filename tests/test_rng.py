"""Key derivation (SANDBOX.md §2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from life_sandbox.core import rng


def _data(key):
    return np.asarray(jax.random.key_data(key))


def test_system_key_is_deterministic_and_tick_sensitive():
    key = rng.world_key(11)
    a = rng.system_key(key, jnp.asarray(7, jnp.int32), "think")
    b = rng.system_key(key, jnp.asarray(7, jnp.int32), "think")
    c = rng.system_key(key, jnp.asarray(8, jnp.int32), "think")
    assert np.array_equal(_data(a), _data(b))
    assert not np.array_equal(_data(a), _data(c))


def test_system_key_depends_on_system_name():
    key = rng.world_key(11)
    assert not np.array_equal(
        _data(rng.system_key(key, 0, "think")),
        _data(rng.system_key(key, 0, "learn")),
    )


def test_unknown_system_name_raises():
    with pytest.raises(KeyError):
        rng.system_key(rng.world_key(0), 0, "nope")


def test_system_ids_are_unique():
    assert len(set(rng.SYSTEM_IDS.values())) == len(rng.SYSTEM_IDS)
