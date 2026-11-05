"""environment system tests (SANDBOX.md §4.2, §4.4, §4.5, §13 §12.5).

Covers berry regeneration, carcass rot and removal, the toxin schedule (period,
composition, conditional event) and the scripted wander of neutral objects.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from life_sandbox.config import Capacities, Config, RunConfig, WorldConfig
from life_sandbox.core import events, rng
from life_sandbox.core.state import empty
from life_sandbox.core.struct import update
from life_sandbox.core.systems import environment
from life_sandbox.world.constants import INITIAL_COMP

BASE = np.asarray(INITIAL_COMP, np.float64)


def make_cfg(**world_overrides) -> Config:
    world = dataclasses.replace(
        WorldConfig(width=16.0, height=16.0, cell=1.0, l_ticks=10.0, c_switch=1.0),
        **world_overrides,
    )
    return Config.default().replaced(
        world=world,
        capacities=Capacities(creature=2, bush=4, carcass=2, neutral=2),
        run=RunConfig(seed=13, ticks=0),
    )


def state_of(cfg: Config):
    return empty(cfg, rng.world_key(13))


def set_stock(state, stocks, kinds=None, alive=None):
    arena = state.entities.bush
    arena = update(
        arena,
        alive=arena.alive if alive is None else jnp.asarray(alive, jnp.bool_),
        edible=update(
            arena.edible,
            stock=jnp.asarray(stocks, jnp.float32),
            kind=arena.edible.kind if kinds is None else jnp.asarray(kinds, jnp.int32),
        ),
    )
    return update(state, entities=update(state.entities, bush=arena))


def set_carcass(state, slot, n, age, alive=True):
    arena = state.entities.carcass
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(alive),
        edible=update(
            arena.edible,
            n=arena.edible.n.at[slot].set(jnp.asarray(n, jnp.float32)),
            age_dead=arena.edible.age_dead.at[slot].set(age),
        ),
    )
    return update(state, entities=update(state.entities, carcass=arena))


def put_creature(state, slot, pos):
    arena = state.entities.creature
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(True),
        transform=update(
            arena.transform, pos=arena.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32))
        ),
        shape=update(arena.shape, radius=arena.shape.radius.at[slot].set(0.5)),
    )
    return update(state, entities=update(state.entities, creature=arena))


def put_neutral(state, slot, pos, heading=0.0, timer=50, held=False):
    arena = state.entities.neutral
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(True),
        transform=update(
            arena.transform,
            pos=arena.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32)),
            theta=arena.transform.theta.at[slot].set(heading),
        ),
        shape=update(arena.shape, radius=arena.shape.radius.at[slot].set(0.4)),
        scripted=update(
            arena.scripted,
            heading=arena.scripted.heading.at[slot].set(heading),
            timer=arena.scripted.timer.at[slot].set(timer),
        ),
    )
    if held:
        arena = update(
            arena,
            held_by=update(
                arena.held_by,
                kind=arena.held_by.kind.at[slot].set(0),
                slot=arena.held_by.slot.at[slot].set(0),
                uid=arena.held_by.uid.at[slot].set(0),
            ),
        )
    return update(state, entities=update(state.entities, neutral=arena))


def at_tick(state, tick):
    return update(state, tick=jnp.asarray(tick, jnp.int32))


def test_berry_regeneration_matches_the_logistic_step():
    cfg = make_cfg()
    state = set_stock(state_of(cfg), [0.3, 0.5, 1.0, 0.0], alive=[True, True, True, True])
    out, _ = environment.apply(state, cfg)
    stock = np.asarray(out.entities.bush.edible.stock)
    expected = [
        0.3 + 0.03 * 0.3 * 0.7 + 1e-4,
        0.5 + 0.03 * 0.5 * 0.5 + 1e-4,
        1.0,
        1e-4,
    ]
    assert stock.tolist() == pytest.approx(expected, abs=1e-7)


def test_dead_bushes_do_not_regenerate():
    cfg = make_cfg()
    state = set_stock(state_of(cfg), [0.3, 0.3], alive=[False, True])
    out, _ = environment.apply(state, cfg)
    stock = np.asarray(out.entities.bush.edible.stock)
    assert stock[0] == pytest.approx(0.3, abs=1e-7)
    assert stock[1] == pytest.approx(0.3 + 0.03 * 0.3 * 0.7 + 1e-4, abs=1e-7)


def test_carcass_rot_turns_plant_and_animal_matter_into_toxin():
    cfg = make_cfg()
    state = set_carcass(state_of(cfg), 0, [0.5, 0.4, 0.1], age=0)
    out, _ = environment.apply(state, cfg)
    carcass = out.entities.carcass
    assert np.asarray(carcass.edible.n[0]).tolist() == pytest.approx(
        [0.4975, 0.398, 0.1 + 0.005 * 0.9], abs=1e-7
    )
    assert int(carcass.edible.age_dead[0]) == 1
    assert bool(carcass.alive[0])


def test_carcass_is_removed_past_600_ticks():
    cfg = make_cfg()
    state = set_carcass(state_of(cfg), 0, [0.5, 0.0, 0.0], age=599)
    state = set_carcass(state, 1, [0.5, 0.0, 0.0], age=600)
    out, _ = environment.apply(state, cfg)
    alive = np.asarray(out.entities.carcass.alive)
    assert bool(alive[0]) is True
    assert bool(alive[1]) is False
    assert int(out.entities.carcass.edible.age_dead[0]) == 600


def test_carcass_is_removed_when_empty():
    cfg = make_cfg()
    state = set_carcass(state_of(cfg), 0, [0.0, 0.0, 0.0], age=5)
    out, _ = environment.apply(state, cfg)
    assert bool(out.entities.carcass.alive[0]) is False


def test_toxin_schedule_switches_every_period():
    cfg = make_cfg(l_ticks=10.0, c_switch=1.0)
    state = at_tick(state_of(cfg), 0)
    out, buffer = environment.apply(state, cfg)
    assert int(buffer.count) == 0
    assert np.allclose(np.asarray(out.comp), BASE, atol=1e-7)

    out, buffer = environment.apply(at_tick(out, 9), cfg)
    assert int(buffer.count) == 0
    assert np.allclose(np.asarray(out.comp), BASE, atol=1e-7)

    switched, buffer = environment.apply(at_tick(out, 10), cfg)
    assert int(buffer.count) == 1
    summary = events.summary(buffer)
    assert summary == {"toxin_switch": 1}
    chosen = int(buffer.uid[0])
    assert 0 <= chosen < BASE.shape[0]
    assert float(buffer.x[0]) == pytest.approx(float(chosen))
    assert int(buffer.tick[0]) == 10
    comp = np.asarray(switched.comp)
    toxic = 0.6 * BASE + 0.4 * np.asarray([0.0, 0.0, 1.0])
    for species in range(BASE.shape[0]):
        expected = toxic[species] if species == chosen else BASE[species]
        assert np.allclose(comp[species], expected, atol=1e-7)


def test_toxin_schedule_uses_the_c_switch_factor():
    cfg = make_cfg(l_ticks=10.0, c_switch=0.5)
    period = 5
    state = at_tick(state_of(cfg), period - 1)
    _, before = environment.apply(state, cfg)
    assert int(before.count) == 0
    _, on = environment.apply(at_tick(state, period), cfg)
    assert int(on.count) == 1


def test_toxin_schedule_is_deterministic():
    cfg = make_cfg()
    state = at_tick(state_of(cfg), 10)
    first, first_buffer = environment.apply(state, cfg)
    second, second_buffer = environment.apply(state, cfg)
    assert np.array_equal(np.asarray(first.comp), np.asarray(second.comp))
    assert int(first_buffer.uid[0]) == int(second_buffer.uid[0])


def test_neutral_objects_walk_at_the_scripted_speed():
    cfg = make_cfg()
    state = put_neutral(state_of(cfg), 0, [8.0, 8.0], heading=0.0, timer=50)
    out, _ = environment.apply(state, cfg)
    neutral = out.entities.neutral
    assert float(neutral.transform.pos[0, 0]) == pytest.approx(8.0 + environment.NEUTRAL_SPEED)
    assert float(neutral.transform.pos[0, 1]) == pytest.approx(8.0, abs=1e-6)
    assert int(neutral.scripted.timer[0]) == 49


def test_neutral_objects_bounce_off_the_walls():
    cfg = make_cfg()
    state = put_neutral(state_of(cfg), 0, [0.45, 8.0], heading=np.pi, timer=50)
    out, _ = environment.apply(state, cfg)
    neutral = out.entities.neutral
    assert float(neutral.transform.pos[0, 0]) > 0.45
    assert float(neutral.transform.pos[0, 0]) >= 0.4 - 1e-6
    # Heading was reflected from -x to +x.
    assert float(np.cos(float(neutral.scripted.heading[0]))) > 0.0


def test_held_neutral_objects_do_not_walk():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [8.0, 8.0])
    state = put_neutral(state, 0, [8.5, 8.0], heading=0.0, timer=50, held=True)
    out, _ = environment.apply(state, cfg)
    assert float(out.entities.neutral.transform.pos[0, 0]) == pytest.approx(8.5)
    assert float(out.entities.neutral.transform.pos[0, 1]) == pytest.approx(8.0)


def test_environment_is_deterministic():
    cfg = make_cfg()
    state = set_stock(state_of(cfg), [0.3, 0.4, 0.5, 0.6])
    state = set_carcass(state, 0, [0.5, 0.4, 0.1], age=3)
    state = put_neutral(state, 0, [8.0, 8.0], heading=1.0, timer=1)
    state = at_tick(state, 10)
    first, first_buffer = environment.apply(state, cfg)
    second, second_buffer = environment.apply(state, cfg)
    assert np.array_equal(
        np.asarray(first.entities.bush.edible.stock),
        np.asarray(second.entities.bush.edible.stock),
    )
    assert np.array_equal(
        np.asarray(first.entities.neutral.transform.pos),
        np.asarray(second.entities.neutral.transform.pos),
    )
    assert int(first_buffer.count) == int(second_buffer.count)


def test_apply_traces_under_jit():
    cfg = make_cfg()
    state = put_neutral(state_of(cfg), 0, [8.0, 8.0], heading=0.5, timer=3)
    state = at_tick(state, 10)
    compiled = jax.jit(environment.apply, static_argnums=(1,))
    out, buffer = compiled(state, cfg)
    out, buffer = compiled(out, cfg)
    assert out.entities.neutral.transform.pos.shape == (2, 2)
    assert int(buffer.count) == 1
