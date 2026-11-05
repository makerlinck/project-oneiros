"""World state and initial placement (SANDBOX.md §3.1, §4.2, §5.1)."""

from __future__ import annotations

import jax.numpy as jnp

from life_sandbox.config import Capacities, Config, RunConfig, WorldConfig
from life_sandbox.core.hashing import state_hash
from life_sandbox.core.state import population
from life_sandbox.world.init import initial_state


def test_empty_state_shapes(tiny_cfg):
    from life_sandbox.core import rng
    from life_sandbox.core.state import empty

    state = empty(tiny_cfg, rng.world_key(0))
    assert int(population(state)) == 0
    assert state.slice.shape == (8, 70)
    assert state.fields.odor.shape == (64, 64, 4)
    assert state.comp.shape == (3, 3)


def test_initial_population_and_uids(tiny_cfg):
    state = initial_state(tiny_cfg)
    assert int(population(state)) == 4
    assert int(state.entities.bush.alive.sum()) == 6
    assert int(state.entities.neutral.alive.sum()) == 1
    assert int(state.uid_counter) == 11
    all_uids = jnp.concatenate(
        [
            state.entities.bush.uid,
            state.entities.creature.identity.uid[:4],
            state.entities.neutral.uid,
        ]
    )
    assert sorted(int(u) for u in all_uids) == list(range(11))


def test_initial_state_is_deterministic_and_seed_sensitive(tiny_cfg):
    assert state_hash(initial_state(tiny_cfg)) == state_hash(initial_state(tiny_cfg))
    assert state_hash(initial_state(tiny_cfg)) != state_hash(initial_state(tiny_cfg, seed=99))


def test_bush_positions_follow_temperature_preference():
    cfg = Config.default().replaced(
        capacities=Capacities(creature=4, bush=30, carcass=2, neutral=1),
        world=WorldConfig(founders=4),
        run=RunConfig(seed=5, ticks=0),
    )
    state = initial_state(cfg)
    kind = state.entities.bush.edible.kind
    y = state.entities.bush.transform.pos[:, 1]
    # T_geo increases with y; cold species prefer low y, warm species high y.
    assert float(y[kind == 0].mean()) < float(y[kind == 2].mean())
    assert state.entities.bush.transform.pos.shape == (30, 2)
    assert float(state.entities.bush.transform.pos.min()) >= 0.0
    assert float(state.entities.bush.transform.pos.max()) <= cfg.world.width


def test_focal_slots_start_empty(tiny_cfg):
    state = initial_state(tiny_cfg)
    # Focal.trace_slot uses -1 for every slot; observers assign real slots later.
    assert int(state.entities.creature.focal.trace_slot.min()) == -1
    assert int(state.entities.creature.focal.trace_slot.max()) == -1
