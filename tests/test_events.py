"""Fixed-size masked event buffer (SANDBOX.md §3.3, §7.2)."""

from __future__ import annotations

import numpy as np

from life_sandbox.config import Config, WorldConfig
from life_sandbox.core import components as C
from life_sandbox.core import events
from life_sandbox.core.hashing import state_hash


def _small_cfg():
    return Config.default().replaced(world=WorldConfig(event_capacity=4))


def test_append_writes_prefix_and_count():
    cfg = _small_cfg()
    buffer = events.empty(cfg)
    buffer = events.append(buffer, kind=events.kind_code("birth"), tick=1, uid=5, x=1.5, y=2.5)
    assert int(buffer.count) == 1
    assert int(buffer.kind[0]) == events.kind_code("birth")
    assert int(buffer.uid[0]) == 5
    assert float(buffer.x[0]) == 1.5
    assert bool(buffer.success[0])


def test_overflow_increments_dropped_without_growing():
    cfg = _small_cfg()
    buffer = events.empty(cfg)
    for i in range(6):
        buffer = events.append(buffer, kind=events.kind_code("death"), tick=i, uid=i)
    assert int(buffer.count) == 4
    assert int(buffer.dropped) == 2


def test_concat_appends_b_before_capacity():
    cfg = _small_cfg()
    a = events.append(events.empty(cfg), kind=events.kind_code("birth"), tick=0, uid=0)
    b = events.append(events.empty(cfg), kind=events.kind_code("death"), tick=1, uid=1)
    merged = events.concat(a, b)
    assert int(merged.count) == 2
    assert int(merged.uid[0]) == 0
    assert int(merged.uid[1]) == 1


def test_scan_path_carries_events_not_only_counts(tiny_cfg):
    """The compiled production path must expose the same events as eager (problem 3)."""
    from life_sandbox.core.step import run
    from life_sandbox.world.init import initial_state

    state = initial_state(tiny_cfg)
    eager = run(state, tiny_cfg, 3, compile=False, collect_events=True)
    compiled = run(state, tiny_cfg, 3, compile=True, collect_events=True)
    assert eager.counts == compiled.counts
    assert eager.event_count() == compiled.event_count()
    assert eager.event_count() == sum(eager.counts.values())
    for name in ("kind", "uid", "x", "y"):
        assert eager.events[name].shape == compiled.events[name].shape
        assert np.array_equal(eager.events[name], compiled.events[name])
    assert eager.inflow.shape == (C.N_INFLOW_CATEGORIES, C.N_NUTRIENTS)
    assert eager.outflow.shape == (C.N_OUTFLOW_CATEGORIES, C.N_NUTRIENTS)
    # Events are exact; the float32 ledger can differ by ULPs between the eager
    # reference and the fused jit path (XLA fusion reorders float32 ops), so the
    # ledger comparison follows the §3.5 relative tolerance instead.
    assert np.allclose(eager.inflow, compiled.inflow, rtol=1e-6, atol=1e-9)
    assert np.allclose(eager.outflow, compiled.outflow, rtol=1e-6, atol=1e-9)
    assert np.allclose(eager.transfer_from, compiled.transfer_from, rtol=1e-6, atol=1e-9)
    assert np.allclose(eager.transfer_to, compiled.transfer_to, rtol=1e-6, atol=1e-9)
    # The compiled production path itself must be bit-reproducible.
    again = run(state, tiny_cfg, 3, compile=True, collect_events=True)
    assert state_hash(again.state) == state_hash(compiled.state)
    assert again.counts == compiled.counts


def test_summary_counts_by_kind():
    cfg = _small_cfg()
    buffer = events.empty(cfg)
    buffer = events.append(buffer, kind=events.kind_code("birth"), uid=0)
    buffer = events.append(buffer, kind=events.kind_code("birth"), uid=1)
    buffer = events.append(buffer, kind=events.kind_code("death"), uid=2)
    counts = events.summary(buffer)
    assert counts == {"birth": 2, "death": 1}
