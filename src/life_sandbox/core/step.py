"""Fixed system order and the tick function (SANDBOX.md §3.3).

Every system is a pure function apply(state, cfg) -> (state, events).  The
system order is fixed by the specification and never by configuration; only
capability switches inside a system may change.

Events and the nutrient ledger are step outputs: the production scan emits both
per tick, so the host can write events and sum flows in float64 without ever
storing them in the world state (SANDBOX.md §3.3, §3.5).  The ledger draft area
inside WorldState is cleared at the start of every tick and is returned as a
scan output; nothing accumulates across ticks inside the state.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from life_sandbox.config import Config
from life_sandbox.core import components as C
from life_sandbox.core import events, masks
from life_sandbox.core.params import ScanParams
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

SystemFn = Callable[
    [WorldState, Config, ScanParams], tuple[WorldState, events.EventBuffer]
]

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

LEDGER_FIELDS: tuple[str, ...] = ("inflow", "outflow", "transfer_from", "transfer_to", "digested")


def step(
    state: WorldState, cfg: Config, params: ScanParams | None = None
) -> tuple[WorldState, events.EventBuffer]:
    """Advance one tick: clear the ledger, run every system in order, advance the clock.

    `params` supplies the runtime M1 scan knobs; when omitted they are derived
    from `cfg`, which keeps every existing single-configuration caller exactly
    as before.
    """
    if params is None:
        params = ScanParams.from_config(cfg)
    state = update(state, ledger=C.Ledger.zeros())
    buffer = events.empty(cfg)
    for _, system in SYSTEMS:
        state, produced = system(state, cfg, params)
        buffer = events.concat(buffer, produced)
    state = update(state, tick=(state.tick + jnp.asarray(1, dtype=jnp.int32)).astype(jnp.int32))
    return state, buffer


def _event_counts(buffer: events.EventBuffer) -> jax.Array:
    """Per-kind weights of the valid prefix, as an int32 vector."""
    capacity = buffer.kind.shape[0]
    valid = (jnp.arange(capacity, dtype=jnp.int32) < buffer.count).astype(jnp.float32)
    counts = jnp.bincount(buffer.kind.astype(jnp.int32), weights=valid, length=len(events.EVENT_KINDS))
    return counts.astype(jnp.int32)


def _scan_run(
    state: WorldState, cfg: Config, ticks: int, params: ScanParams | None = None
) -> tuple[WorldState, jax.Array, events.EventBuffer, C.Ledger]:
    """Chunked run: K ticks compiled into one lax.scan segment (SANDBOX.md §3.3)."""
    if params is None:
        params = ScanParams.from_config(cfg)

    def body(carry, _):
        current, totals = carry
        current, buffer = step(current, cfg, params)
        outputs = (buffer, current.ledger)
        return (current, totals + _event_counts(buffer)), outputs

    initial = jnp.zeros(len(events.EVENT_KINDS), dtype=jnp.int32)
    (state, totals), (buffers, ledgers) = jax.lax.scan(body, (state, initial), None, length=ticks)
    return state, totals, buffers, ledgers


# Production chunks ticks into segments so each segment compiles once and the
# host can write event buffers between segments (SANDBOX.md §3.3).
DEFAULT_SEGMENT_TICKS: int = 64


# Compilation accounting for the M1 throughput comparison: a cached segment
# factory miss means a new XLA compilation; hits reuse the one compiled graph
# for every configuration x world row of a batch.
_COMPILE_STATS: dict[str, int] = {"segment": 0, "batch_segment": 0}


def compile_stats() -> dict[str, int]:
    """Number of compiled segment builders created in this process."""
    return dict(_COMPILE_STATS)


def reset_compile_stats(*, clear_cache: bool = False) -> None:
    """Reset the counters (test hook); optionally drop the cached jitted runs."""
    for name in _COMPILE_STATS:
        _COMPILE_STATS[name] = 0
    if clear_cache:
        _compiled_segment.cache_clear()
        _compiled_batch_segment.cache_clear()


@functools.lru_cache(maxsize=64)
def _compiled_segment(cfg: Config, ticks: int):
    """Jitted segment runner, cached per configuration and segment length."""
    _COMPILE_STATS["segment"] += 1

    def _run(state: WorldState, params: ScanParams | None = None):
        return _scan_run(state, cfg, ticks, params)

    return jax.jit(_run)


def _zero_ledger_totals() -> dict[str, np.ndarray]:
    return {
        "inflow": np.zeros((C.N_INFLOW_CATEGORIES, C.N_NUTRIENTS), dtype=np.float64),
        "outflow": np.zeros((C.N_OUTFLOW_CATEGORIES, C.N_NUTRIENTS), dtype=np.float64),
        "transfer_from": np.zeros(C.N_NUTRIENTS, dtype=np.float64),
        "transfer_to": np.zeros(C.N_NUTRIENTS, dtype=np.float64),
        "digested": np.zeros(C.N_NUTRIENTS, dtype=np.float64),
    }


def _accumulate_ledger(totals: dict[str, np.ndarray], ledger, axis: int | None) -> None:
    for name in LEDGER_FIELDS:
        values = np.asarray(getattr(ledger, name), dtype=np.float64)
        totals[name] += values.sum(axis=0) if axis is not None else values


def _event_table(buffers: events.EventBuffer) -> dict[str, np.ndarray]:
    """Flatten a stacked per-tick buffer into numpy event columns."""
    counts = np.asarray(buffers.count, dtype=np.int64)
    capacity = buffers.kind.shape[1]
    mask = np.arange(capacity)[None, :] < counts[:, None]
    table: dict[str, np.ndarray] = {}
    for name in events._ARRAY_FIELDS:
        table[name] = np.asarray(getattr(buffers, name))[mask]
    table["dropped"] = np.asarray(buffers.dropped, dtype=np.int64).sum(keepdims=False)
    return table


def _stack_buffers(buffers: list[events.EventBuffer]) -> events.EventBuffer:
    stacked = {}
    for name in ("count", "dropped") + events._ARRAY_FIELDS:
        stacked[name] = np.stack([np.asarray(getattr(buffer, name)) for buffer in buffers])
    return events.EventBuffer(**stacked)


def _merge_event_tables(target: dict[str, np.ndarray] | None, table: dict[str, np.ndarray]) -> None:
    if target is None:
        return
    for name, values in table.items():
        if name == "dropped":
            continue
        if name in target:
            target[name] = np.concatenate([target[name], values], axis=0)
        else:
            target[name] = values


@functools.lru_cache(maxsize=64)
def _compiled_batch_segment(cfg: Config, ticks: int):
    """Jitted segment runner over a batch of worlds (leading axis = world)."""
    _COMPILE_STATS["batch_segment"] += 1

    def _run(states: WorldState, params: ScanParams | None = None):
        return _scan_run_batch(states, cfg, ticks, params)

    return jax.jit(_run)


def _batch_params(
    cfg: Config, batch: int, params: ScanParams | None
) -> ScanParams:
    """Normalise runtime params to one row per world (leading axis = batch)."""
    if params is None:
        return ScanParams.broadcast(ScanParams.from_config(cfg), batch)
    leaves = jax.tree.leaves(params)
    if leaves and leaves[0].shape == ():
        return ScanParams.broadcast(params, batch)
    if not leaves or leaves[0].shape[0] != batch:
        raise ValueError("batched ScanParams must have one row per state")
    return params


def _scan_run_batch(
    states: WorldState, cfg: Config, ticks: int, params: ScanParams | None = None
) -> tuple[WorldState, jax.Array, events.EventBuffer, C.Ledger]:
    """Chunked run over B worlds: one scan, vmap(step) per tick (SANDBOX.md §3.3)."""
    params = _batch_params(cfg, int(states.tick.shape[0]), params)
    batched_step = jax.vmap(lambda state, row: step(state, cfg, row))
    batched_counts = jax.vmap(_event_counts)

    def body(carry, _):
        current, totals = carry
        current, buffer = batched_step(current, params)
        return (current, totals + batched_counts(buffer)), (buffer, current.ledger)

    initial = jnp.zeros((len(events.EVENT_KINDS),), dtype=jnp.int32)
    initial = jnp.broadcast_to(initial, (states.tick.shape[0], len(events.EVENT_KINDS)))
    (states, totals), (buffers, ledgers) = jax.lax.scan(body, (states, initial), None, length=ticks)
    return states, totals, buffers, ledgers


@dataclass
class RunBatchResult:
    """Host-side result of a batched run (leading axis = world)."""

    states: WorldState
    counts: np.ndarray  # int64[B, KINDS]
    ticks: int
    inflow: np.ndarray  # float64[B, INFLOW, 3]
    outflow: np.ndarray  # float64[B, OUTFLOW, 3]
    transfer_from: np.ndarray  # float64[B, 3]
    transfer_to: np.ndarray
    digested: np.ndarray
    dropped_events: int = 0

    def counts_dict(self, world: int) -> dict[str, int]:
        row = self.counts[world]
        return {
            name: int(row[index])
            for index, name in enumerate(events.EVENT_KINDS)
            if int(row[index])
        }


def run_batch(
    states: WorldState,
    cfg: Config,
    ticks: int,
    *,
    compile: bool = True,
    segment: int | None = None,
    params: ScanParams | None = None,
) -> RunBatchResult:
    """Run B worlds together with vmap inside one lax.scan segment loop.

    'states' is a batched WorldState (leading axis = world); a single
    unbatched state is accepted and promoted to a batch of one.  `params` is a
    batched `ScanParams` (leading axis = world; a single row is broadcast);
    with params at runtime, rows may mix different M1 scan configurations in one
    compiled graph.  Per-world
    ledger sums are kept in float64 on the host; event tables are not collected
    here (M2 logging will stream them per segment).
    """
    ticks = int(ticks)
    segment = int(cfg.run.segment_ticks) if segment is None else int(segment)
    # A single (unbatched) world is accepted and treated as a batch of one, so
    # callers can use run_batch for any number of worlds.
    if jnp.ndim(states.tick) == 0:
        states = jax.tree.map(lambda leaf: jnp.expand_dims(leaf, 0), states)
    batch = int(states.tick.shape[0])
    batch_params = _batch_params(cfg, batch, params)
    inflow = np.zeros((batch, C.N_INFLOW_CATEGORIES, C.N_NUTRIENTS), dtype=np.float64)
    outflow = np.zeros((batch, C.N_OUTFLOW_CATEGORIES, C.N_NUTRIENTS), dtype=np.float64)
    transfer_from = np.zeros((batch, C.N_NUTRIENTS), dtype=np.float64)
    transfer_to = np.zeros((batch, C.N_NUTRIENTS), dtype=np.float64)
    digested = np.zeros((batch, C.N_NUTRIENTS), dtype=np.float64)
    counts_total = np.zeros((batch, len(events.EVENT_KINDS)), dtype=np.int64)
    dropped = 0

    def ledger_total(name, frame_axis):
        values = np.asarray(getattr(ledgers_holder["ledger"], name), dtype=np.float64)
        if frame_axis is not None:
            values = values.sum(axis=0)
        return values

    # Bind per call (Python closure assignment keeps mypy quiet about rebinding).
    ledgers_holder: dict[str, object] = {"ledger": None}

    def absorb(counts_vec, ledgers, frame_axis):
        ledgers_holder["ledger"] = ledgers
        inflow[:] += ledger_total("inflow", frame_axis)
        outflow[:] += ledger_total("outflow", frame_axis)
        transfer_from[:] += ledger_total("transfer_from", frame_axis)
        transfer_to[:] += ledger_total("transfer_to", frame_axis)
        digested[:] += ledger_total("digested", frame_axis)
        counts_total[:] += np.asarray(counts_vec, dtype=np.int64)

    if compile:
        remaining = ticks
        while remaining > 0:
            size = min(segment, remaining)
            states, counts_vec, buffers, ledgers = _compiled_batch_segment(cfg, size)(
                states, batch_params
            )
            absorb(counts_vec, ledgers, 0)
            dropped += int(np.asarray(buffers.dropped).sum())
            remaining -= size
            for world in range(batch):
                masks.ensure_uid_headroom(states.uid_counter[world])
    else:
        for _ in range(ticks):
            states, buffers = jax.vmap(lambda state, row: step(state, cfg, row))(
                states, batch_params
            )
            counts_vec = jax.vmap(_event_counts)(buffers)
            absorb(counts_vec, states.ledger, None)
            dropped += int(np.asarray(buffers.dropped).sum())
            for world in range(batch):
                masks.ensure_uid_headroom(states.uid_counter[world])

    return RunBatchResult(
        states=states,
        counts=counts_total,
        ticks=ticks,
        inflow=inflow,
        outflow=outflow,
        transfer_from=transfer_from,
        transfer_to=transfer_to,
        digested=digested,
        dropped_events=dropped,
    )


@dataclass
class RunResult:
    """Host-side result of one run segment loop.

    counts: events per kind; inflow/outflow/... are float64 sums of the per-tick
    ledger; events is the flattened event table when collect_events=True.
    """

    state: WorldState
    counts: dict[str, int]
    ticks: int
    inflow: np.ndarray
    outflow: np.ndarray
    transfer_from: np.ndarray
    transfer_to: np.ndarray
    digested: np.ndarray
    dropped_events: int = 0
    events: dict[str, np.ndarray] | None = field(default=None)

    def event_count(self) -> int:
        if self.events is None or "uid" not in self.events:
            return 0
        return int(self.events["uid"].shape[0])


def run(
    state: WorldState,
    cfg: Config,
    ticks: int,
    *,
    compile: bool = True,
    segment: int | None = None,
    collect_events: bool = False,
    params: ScanParams | None = None,
) -> RunResult:
    """Run ticks and return the final state, event counts, ledger totals and events.

    compile=True runs the production path: fixed-length lax.scan segments that
    compile once and are reused; compile=False is the eager reference.  Both
    paths return the per-tick events and ledger, so the production path no
    longer drops events.
    """
    ticks = int(ticks)
    segment = int(cfg.run.segment_ticks) if segment is None else int(segment)
    single_params = ScanParams.from_config(cfg) if params is None else params
    leaves = jax.tree.leaves(single_params)
    if leaves and leaves[0].shape != ():
        if leaves[0].shape != (1,):
            raise ValueError("run() takes one ScanParams row, not a batch")
        single_params = ScanParams.row(single_params, 0)
    totals = _zero_ledger_totals()
    counts: dict[str, int] = {}
    event_table: dict[str, np.ndarray] | None = {} if collect_events else None
    dropped = 0

    def add_counts(values) -> None:
        for index, name in enumerate(events.EVENT_KINDS):
            value = int(values[index])
            if value:
                counts[name] = counts.get(name, 0) + value

    if compile:
        remaining = ticks
        while remaining > 0:
            size = min(int(segment), remaining)
            state, counts_vec, buffers, ledgers = _compiled_segment(cfg, size)(
                state, single_params
            )
            add_counts(counts_vec)
            _accumulate_ledger(totals, ledgers, axis=0)
            dropped += int(np.asarray(buffers.dropped).sum())
            _merge_event_tables(event_table, _event_table(buffers))
            remaining -= size
            masks.ensure_uid_headroom(state.uid_counter)
    else:
        eager_buffers: list[events.EventBuffer] = []
        for _ in range(ticks):
            state, buffer = step(state, cfg, single_params)
            masks.ensure_uid_headroom(state.uid_counter)
            add_counts(
                np.asarray([events.summary(buffer).get(name, 0) for name in events.EVENT_KINDS])
            )
            _accumulate_ledger(totals, state.ledger, axis=None)
            dropped += int(np.asarray(buffer.dropped))
            if event_table is not None:
                eager_buffers.append(buffer)
        if event_table is not None and eager_buffers:
            _merge_event_tables(event_table, _event_table(_stack_buffers(eager_buffers)))
    if event_table is not None:
        event_table["dropped"] = np.asarray(dropped, dtype=np.int64)

    return RunResult(
        state=state,
        counts=counts,
        ticks=ticks,
        inflow=totals["inflow"],
        outflow=totals["outflow"],
        transfer_from=totals["transfer_from"],
        transfer_to=totals["transfer_to"],
        digested=totals["digested"],
        dropped_events=dropped,
        events=event_table,
    )
