"""Fixed-size masked event buffer (SANDBOX.md §3.3 step 12, §7.2).

Decision recorded here: events are a step output, not part of WorldState.
They therefore never enter the state hash or a snapshot; the host writes them
per tick, and a fork inherits no pending events.

Buffer contract: the first count entries are valid; dropped counts overflow
events that did not fit in the fixed capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from life_sandbox.core.components import NO_REF
from life_sandbox.core.struct import struct

EVENT_KINDS: tuple[str, ...] = (
    "birth",
    "birth_fail",
    "death",
    "bite",
    "grab",
    "grab_fail",
    "drop",
    "escape",
    "toxin_switch",
    "reseed",
    "sleep_start",
    "sleep_end",
    "recall",
    # Parent was mature and eligible but the per-tick birth budget was full;
    # it is deferred to a later tick (§5.5, C5).
    "birth_defer",
)

DEATH_CAUSES: tuple[str, ...] = (
    "none",
    "nutrition",
    "bite",
    "toxin",
    "heat",
    "cold",
    "age",
    "other",
)

REASONS: tuple[str, ...] = (
    "none",
    "occupied",
    "no_tissue",
    "too_heavy",
    "out_of_range",
    "escaped",
    "no_room",
    "not_edible",
)

_ARRAY_FIELDS = (
    "kind",
    "tick",
    "uid",
    "target_kind",
    "target_uid",
    "same_lineage",
    "x",
    "y",
    "dn",
    "absorbed",
    "dX",
    "damage",
    "cause",
    "success",
    "reason",
)


def kind_code(name: str) -> int:
    return EVENT_KINDS.index(name)


def cause_code(name: str) -> int:
    return DEATH_CAUSES.index(name)


def reason_code(name: str) -> int:
    return REASONS.index(name)


@struct
@dataclass(frozen=True)
class EventBuffer:
    count: jax.Array
    dropped: jax.Array
    kind: jax.Array
    tick: jax.Array
    uid: jax.Array
    target_kind: jax.Array
    target_uid: jax.Array
    same_lineage: jax.Array
    x: jax.Array
    y: jax.Array
    dn: jax.Array
    absorbed: jax.Array
    dX: jax.Array
    damage: jax.Array
    cause: jax.Array
    success: jax.Array
    reason: jax.Array


def empty(cfg) -> EventBuffer:
    capacity = cfg.world.event_capacity
    return EventBuffer(
        count=jnp.asarray(0, dtype=jnp.int32),
        dropped=jnp.asarray(0, dtype=jnp.int32),
        kind=jnp.zeros(capacity, dtype=jnp.int8),
        tick=jnp.zeros(capacity, dtype=jnp.int32),
        uid=jnp.zeros(capacity, dtype=jnp.int32),
        target_kind=jnp.full(capacity, NO_REF, dtype=jnp.int32),
        target_uid=jnp.zeros(capacity, dtype=jnp.int32),
        same_lineage=jnp.zeros(capacity, dtype=jnp.bool_),
        x=jnp.zeros(capacity, dtype=jnp.float32),
        y=jnp.zeros(capacity, dtype=jnp.float32),
        dn=jnp.zeros((capacity, 3), dtype=jnp.float32),
        absorbed=jnp.zeros((capacity, 3), dtype=jnp.float32),
        dX=jnp.zeros(capacity, dtype=jnp.float32),
        damage=jnp.zeros(capacity, dtype=jnp.float32),
        cause=jnp.zeros(capacity, dtype=jnp.int8),
        success=jnp.ones(capacity, dtype=jnp.bool_),
        reason=jnp.zeros(capacity, dtype=jnp.int8),
    )


def append(buffer: EventBuffer, **values) -> EventBuffer:
    """Write one event at the next free slot; overflow increments dropped."""
    capacity = buffer.kind.shape[0]
    idx = jnp.minimum(buffer.count, capacity - 1)
    accepted = buffer.count < capacity

    updates = {}
    for name in _ARRAY_FIELDS:
        array = getattr(buffer, name)
        value = values.get(name)
        if value is None:
            if name == "success":
                value = True
            elif array.ndim > 1:
                value = jnp.zeros(array.shape[1:], dtype=array.dtype)
            else:
                value = jnp.asarray(0, dtype=array.dtype)
        cast = jnp.asarray(value, dtype=array.dtype)
        current = array[idx]
        updates[name] = array.at[idx].set(jnp.where(accepted, cast, current))
    updates["count"] = buffer.count + accepted.astype(jnp.int32)
    updates["dropped"] = buffer.dropped + (~accepted).astype(jnp.int32)
    return EventBuffer(**updates)


def concat(a: EventBuffer, b: EventBuffer) -> EventBuffer:
    """Append all valid events of b after those of a, honouring capacity."""
    capacity = a.kind.shape[0]
    positions = jnp.arange(capacity, dtype=jnp.int32)
    total = a.count + b.count
    new_count = jnp.minimum(total, capacity)
    valid = (positions >= a.count) & (positions < new_count)
    src = jnp.clip(positions - a.count, 0, capacity - 1)

    merged = {}
    for name in _ARRAY_FIELDS:
        a_arr = getattr(a, name)
        b_arr = getattr(b, name)
        gathered = b_arr[src]
        mask = valid.reshape((-1,) + (1,) * (a_arr.ndim - 1))
        merged[name] = jnp.where(mask, gathered, a_arr)
    merged["count"] = new_count
    merged["dropped"] = a.dropped + b.dropped + jnp.maximum(total - capacity, 0)
    return EventBuffer(**merged)


def summary(buffer: EventBuffer) -> dict[str, int]:
    """Host-side count of events by kind (for CLI output and smoke tests)."""
    counts = jnp.bincount(buffer.kind[: buffer.count], minlength=len(EVENT_KINDS))
    return {name: int(counts[i]) for i, name in enumerate(EVENT_KINDS) if int(counts[i])}
