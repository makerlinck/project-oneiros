"""Fixed-size masked event buffer (SANDBOX.md §3.3, §7.2)."""

from __future__ import annotations

from life_sandbox.config import Config, WorldConfig
from life_sandbox.core import events


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


def test_summary_counts_by_kind():
    cfg = _small_cfg()
    buffer = events.empty(cfg)
    buffer = events.append(buffer, kind=events.kind_code("birth"), uid=0)
    buffer = events.append(buffer, kind=events.kind_code("birth"), uid=1)
    buffer = events.append(buffer, kind=events.kind_code("death"), uid=2)
    counts = events.summary(buffer)
    assert counts == {"birth": 2, "death": 1}
