"""Fixed system order and the tick function (SANDBOX.md §3.3).

Every system is a pure function apply(state, cfg) -> (state, events).  The
system order is fixed by the specification and never by configuration; only
capability switches inside a system may change.

Events are a step output (documented decision): they are not stored in the
world state, so they never enter the state hash or a snapshot.
"""

from __future__ import annotations

from typing import Callable

import functools

import jax
import jax.numpy as jnp

from life_sandbox.config import Config
from life_sandbox.core import events
from life_sandbox.core.state import WorldState
from life_sandbox.core.struct import update
from life_sandbox.core.systems import (
    death,
    environment,
    interaction,
    learn,
    metabolism,
    motor,
    reproduction,
    sensing,
    think,
)
from life_sandbox.world import fields, physics

SYSTEM_ORDER: tuple[str, ...] = (
    "fields",
    "environment",
    "sense",
    "think",
    "learn",
    "motion",
    "collide",
    "interact",
    "body",
    "reproduce",
    "die",
    "events",
)

SystemFn = Callable[[WorldState, Config], tuple[WorldState, events.EventBuffer]]

SYSTEMS: tuple[tuple[str, SystemFn], ...] = (
    ("fields", fields.apply),
    ("environment", environment.apply),
    ("sense", sensing.apply),
    ("think", think.apply),
    ("learn", learn.apply),
    ("motion", motor.apply),
    ("collide", physics.apply),
    ("interact", interaction.apply),
    ("body", metabolism.apply),
    ("reproduce", reproduction.apply),
    ("die", death.apply),
)


def step(state: WorldState, cfg: Config) -> tuple[WorldState, events.EventBuffer]:
    """Advance one tick: run every system in the fixed order, then advance the clock."""
    buffer = events.empty(cfg)
    for _, system in SYSTEMS:
        state, produced = system(state, cfg)
        buffer = events.concat(buffer, produced)
    state = update(state, tick=(state.tick + jnp.asarray(1, dtype=jnp.int32)).astype(jnp.int32))
    return state, buffer


def _event_counts(buffer: events.EventBuffer) -> jax.Array:
    """Per-kind weights of the valid prefix, as an int32 vector."""
    capacity = buffer.kind.shape[0]
    valid = (jnp.arange(capacity, dtype=jnp.int32) < buffer.count).astype(jnp.float32)
    counts = jnp.bincount(buffer.kind.astype(jnp.int32), weights=valid, length=len(events.EVENT_KINDS))
    return counts.astype(jnp.int32)


def _scan_run(state: WorldState, cfg: Config, ticks: int) -> tuple[WorldState, jax.Array]:
    """Chunked run: K ticks compiled into one lax.scan segment (SANDBOX.md §3.3)."""

    def body(carry, _):
        current, totals = carry
        current, buffer = step(current, cfg)
        return (current, totals + _event_counts(buffer)), None

    initial = jnp.zeros(len(events.EVENT_KINDS), dtype=jnp.int32)
    (state, totals), _ = jax.lax.scan(body, (state, initial), None, length=ticks)
    return state, totals


# Production chunks ticks into segments so each segment compiles once and the
# host can write event buffers between segments (SANDBOX.md §3.3).
DEFAULT_SEGMENT_TICKS: int = 64


@functools.lru_cache(maxsize=64)
def _compiled_segment(cfg: Config, ticks: int):
    """Jitted segment runner, cached per configuration and segment length."""

    def _run(state: WorldState):
        return _scan_run(state, cfg, ticks)

    return jax.jit(_run)


def run(
    state: WorldState,
    cfg: Config,
    ticks: int,
    *,
    compile: bool = True,
    segment: int = DEFAULT_SEGMENT_TICKS,
) -> tuple[WorldState, dict[str, int]]:
    """Run ticks and return the final state plus event counts by kind.

    compile=True runs the production path: fixed-length lax.scan segments that
    compile once and are reused; compile=False is the eager reference.
    """
    ticks = int(ticks)
    if ticks <= 0:
        return state, {}
    if compile:
        totals = jnp.zeros(len(events.EVENT_KINDS), dtype=jnp.int32)
        remaining = ticks
        while remaining > 0:
            size = min(int(segment), remaining)
            state, part = _compiled_segment(cfg, size)(state)
            totals = totals + part
            remaining -= size
        values = {name: int(totals[i]) for i, name in enumerate(events.EVENT_KINDS)}
    else:
        values = {}
        for _ in range(ticks):
            state, buffer = step(state, cfg)
            for name, value in events.summary(buffer).items():
                values[name] = values.get(name, 0) + value
    return state, {name: value for name, value in values.items() if value}
