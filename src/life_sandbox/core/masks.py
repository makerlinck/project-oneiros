"""Slot allocation, uid assignment and reference validation (SANDBOX.md §3.1).

Slot indices are reused, uids are not.  A reference is (prototype id, slot, uid)
and is only valid while the slot still holds that uid and is alive.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from life_sandbox.core.components import (
    BUSH,
    CARCASS,
    CREATURE,
    NEUTRAL,
    NO_REF,
)


# int32 uid counters (SANDBOX.md §3.1, documented deviation from int64).  A run
# that approaches 2**31 must stop loudly instead of wrapping.  One million slots
# of headroom is far beyond any planned world lifetime.
UID_MAX: int = 2**31 - 1
UID_HEADROOM: int = 1_000_000


def ensure_uid_headroom(counter) -> None:
    """Host-side guard: stop before the int32 uid counter can wrap."""
    value = int(counter)
    if value > UID_MAX - UID_HEADROOM:
        raise RuntimeError(
            "uid counter "
            + str(value)
            + " is within "
            + str(UID_HEADROOM)
            + " of the int32 limit; stop the run and archive the world"
        )


def first_free_slot(alive: jax.Array) -> jax.Array:
    """Lowest free slot index, or -1 when the arena is full."""
    has_free = jnp.any(~alive)
    return jnp.where(has_free, jnp.argmax(~alive), -1).astype(jnp.int32)


def free_slots(alive: jax.Array, count: int) -> tuple[jax.Array, jax.Array]:
    """First count free slots in ascending order, plus an ok flag."""
    n = alive.shape[0]
    indices = jnp.nonzero(~alive, size=n, fill_value=n)[0]
    slots = indices[:count].astype(jnp.int32)
    ok = jnp.sum((~alive).astype(jnp.int32)) >= count
    return slots, ok


def allocate_uids(counter: jax.Array, count: int) -> tuple[jax.Array, jax.Array]:
    """Sequential uids for a batch of births; the counter never rewinds."""
    uids = counter + jnp.arange(count, dtype=jnp.int32)
    return uids, counter + jnp.asarray(count, dtype=jnp.int32)


def scatter_mask(slots: jax.Array, n: int) -> jax.Array:
    """Boolean mask of shape (count, n) selecting the given slots."""
    return jnp.arange(n, dtype=jnp.int32)[None, :] == slots[:, None]


def _gather(values: jax.Array, slot: jax.Array) -> jax.Array:
    return values[jnp.clip(slot, 0, values.shape[0] - 1)]


def uid_at(entities, kind: jax.Array, slot: jax.Array) -> jax.Array:
    """uid stored at (kind, slot), or -1 for an empty kind."""
    kind = jnp.asarray(kind, dtype=jnp.int32)
    slot = jnp.asarray(slot, dtype=jnp.int32)
    return jnp.select(
        [
            kind == CREATURE,
            kind == BUSH,
            kind == CARCASS,
            kind == NEUTRAL,
        ],
        [
            _gather(entities.creature.identity.uid, slot),
            _gather(entities.bush.uid, slot),
            _gather(entities.carcass.uid, slot),
            _gather(entities.neutral.uid, slot),
        ],
        default=jnp.asarray(NO_REF, dtype=jnp.int32),
    )


def alive_at(entities, kind: jax.Array, slot: jax.Array) -> jax.Array:
    """Whether (kind, slot) is an occupied slot."""
    kind = jnp.asarray(kind, dtype=jnp.int32)
    slot = jnp.asarray(slot, dtype=jnp.int32)
    return jnp.select(
        [kind == CREATURE, kind == BUSH, kind == CARCASS, kind == NEUTRAL],
        [
            _gather(entities.creature.alive, slot),
            _gather(entities.bush.alive, slot),
            _gather(entities.carcass.alive, slot),
            _gather(entities.neutral.alive, slot),
        ],
        default=jnp.asarray(False),
    )


def ref_valid(entities, kind: jax.Array, slot: jax.Array, uid: jax.Array) -> jax.Array:
    """A reference is valid while its uid still occupies the slot and is alive."""
    kind = jnp.asarray(kind, dtype=jnp.int32)
    return (
        (kind != NO_REF)
        & (jnp.asarray(slot, dtype=jnp.int32) >= 0)
        & alive_at(entities, kind, slot)
        & (uid_at(entities, kind, slot) == jnp.asarray(uid, dtype=jnp.int32))
    )


def held_mass(entities) -> jax.Array:
    """Mass of whatever each creature holds, 0 when empty or stale (§5.3)."""
    holding = entities.creature.grip.holding
    alive = entities.creature.alive
    valid = ref_valid(entities, holding.kind, holding.slot, holding.uid) & alive

    masses = []
    for kind, arena in (
        (CREATURE, entities.creature),
        (CARCASS, entities.carcass),
        (NEUTRAL, entities.neutral),
    ):
        mass = _gather(arena.shape.mass, holding.slot)
        masses.append(jnp.where(holding.kind == kind, mass, 0.0))
    total = masses[0] + masses[1] + masses[2]
    return jnp.where(valid, total, 0.0)
