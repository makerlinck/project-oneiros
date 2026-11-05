"""Observers are pure projections (SANDBOX.md §7.1, §13)."""

from __future__ import annotations

import inspect

import jax.numpy as jnp

from life_sandbox.config import Capacities, Config, RunConfig, WorldConfig
from life_sandbox.core.hashing import state_hash
from life_sandbox.core.step import step
from life_sandbox.world.init import initial_state


def _observe(state):
    """A read-only projection: population, mean reserves, mean integrity."""
    alive = state.entities.creature.alive
    count = jnp.maximum(jnp.sum(alive.astype(jnp.int32)), 1)
    return {
        "population": int(jnp.sum(alive.astype(jnp.int32))),
        "mean_reserve": float(jnp.sum(state.entities.creature.body.N) / count),
        "mean_integrity": float(jnp.sum(state.entities.creature.body.I) / count),
    }


def test_observing_does_not_change_a_state():
    cfg = Config.default().replaced(
        capacities=Capacities(creature=8, bush=6, carcass=2, neutral=1),
        world=WorldConfig(founders=4),
        run=RunConfig(seed=3, ticks=5),
    )
    state = initial_state(cfg)
    before = state_hash(state)
    _observe(state)
    assert state_hash(state) == before


def test_observer_on_off_keeps_the_trajectory(tiny_cfg):
    a = initial_state(tiny_cfg)
    b = initial_state(tiny_cfg)
    for _ in range(5):
        _observe(a)
        a, _ = step(a, tiny_cfg)
        b, _ = step(b, tiny_cfg)
    assert state_hash(a) == state_hash(b)


def test_observer_never_receives_a_key():
    parameters = list(inspect.signature(_observe).parameters)
    assert parameters == ["state"]
