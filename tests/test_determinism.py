"""Determinism (SANDBOX.md §13 test_determinism)."""

from __future__ import annotations

from life_sandbox.core.hashing import state_hash
from life_sandbox.core.step import step
from life_sandbox.world.init import initial_state


def test_same_seed_runs_are_bit_identical(tiny_cfg):
    a = initial_state(tiny_cfg)
    b = initial_state(tiny_cfg)
    assert state_hash(a) == state_hash(b)


def test_step_is_deterministic(tiny_cfg):
    a = initial_state(tiny_cfg)
    b = initial_state(tiny_cfg)
    for _ in range(5):
        a, _ = step(a, tiny_cfg)
        b, _ = step(b, tiny_cfg)
    assert state_hash(a) == state_hash(b)


def test_tick_advances_and_hash_changes(tiny_cfg):
    state = initial_state(tiny_cfg)
    before = state_hash(state)
    state, _ = step(state, tiny_cfg)
    assert int(state.tick) == 1
    assert state_hash(state) != before


def test_different_seeds_diverge(tiny_cfg):
    assert state_hash(initial_state(tiny_cfg, seed=1)) != state_hash(initial_state(tiny_cfg, seed=2))
