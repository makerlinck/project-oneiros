"""Deterministic random keys (SANDBOX.md §2).

Every system derives its own key from the world key and the tick:

    system_key(world_key, tick, name) == fold_in(fold_in(world_key, tick), id)

Observers never receive a key, so opening or closing observation cannot change
the random stream (SANDBOX.md §7.1; tests/test_observer_noninterference.py).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Stable integer ids.  Appending is fine; reordering or reusing an id changes
# the recorded random stream of every run, so ids are part of the record format.
SYSTEM_IDS: dict[str, int] = {
    "init": 0,
    "fields": 1,
    "environment": 2,
    "sense": 3,
    "think": 4,
    "learn": 5,
    "motion": 6,
    "collide": 7,
    "interact": 8,
    "body": 9,
    "reproduce": 10,
    "die": 11,
    "events": 12,
}

if len(set(SYSTEM_IDS.values())) != len(SYSTEM_IDS):  # pragma: no cover - import guard
    raise RuntimeError("system ids must be unique")


def world_key(seed: int) -> jax.Array:
    """Root key of one world; every other key derives from it."""
    return jax.random.key(int(seed))


def system_key(key: jax.Array, tick, system: str) -> jax.Array:
    """Per-(system, tick) key, derived without consuming the world key."""
    if system not in SYSTEM_IDS:
        raise KeyError("unknown system " + repr(system) + "; register it in rng.SYSTEM_IDS")
    tick_u32 = jnp.asarray(tick, dtype=jnp.uint32)
    system_u32 = jnp.asarray(SYSTEM_IDS[system], dtype=jnp.uint32)
    return jax.random.fold_in(jax.random.fold_in(key, tick_u32), system_u32)


def subsplit(key: jax.Array, count: int) -> jax.Array:
    """Split a key into the given number of independent keys."""
    return jax.random.split(key, count)
