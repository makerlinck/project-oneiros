"""Collision tests (SANDBOX.md §5.3, §9, §13 test_collision).

Covers: residual overlap after resolution, mass-weighted push, contact quadrant
assignment (front / right / back / left), wall clipping with velocity halving,
wall contact depth, and the excluded pairs (held bodies, creature-vs-bush).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from life_sandbox.config import Capacities, Config, RunConfig, WorldConfig
from life_sandbox.core import rng
from life_sandbox.core.state import empty
from life_sandbox.core.struct import update
from life_sandbox.world import physics

FRONT, RIGHT, BACK, LEFT = 0, 1, 2, 3


def make_cfg() -> Config:
    return Config.default().replaced(
        world=WorldConfig(width=64.0, height=64.0, cell=1.0),
        capacities=Capacities(creature=4, bush=2, carcass=2, neutral=1),
        run=RunConfig(seed=5, ticks=0),
    )


def state_of(cfg: Config):
    return empty(cfg, rng.world_key(5))


def put_creature(state, slot, pos, theta=0.0, radius=0.5, mass=1.0, v=0.0):
    arena = state.entities.creature
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(True),
        transform=update(
            arena.transform,
            pos=arena.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32)),
            theta=arena.transform.theta.at[slot].set(jnp.asarray(theta, jnp.float32)),
            v=arena.transform.v.at[slot].set(v),
        ),
        shape=update(
            arena.shape,
            radius=arena.shape.radius.at[slot].set(radius),
            mass=arena.shape.mass.at[slot].set(mass),
        ),
    )
    return update(state, entities=update(state.entities, creature=arena))


def put_carcass(state, slot, pos, radius=0.5, mass=1.0, held_by_creature=None):
    arena = state.entities.carcass
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(True),
        transform=update(
            arena.transform, pos=arena.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32))
        ),
        shape=update(
            arena.shape,
            radius=arena.shape.radius.at[slot].set(radius),
            mass=arena.shape.mass.at[slot].set(mass),
        ),
    )
    if held_by_creature is not None:
        held_by = update(
            arena.held_by,
            kind=arena.held_by.kind.at[slot].set(0),
            slot=arena.held_by.slot.at[slot].set(held_by_creature),
            uid=arena.held_by.uid.at[slot].set(0),
        )
        arena = update(arena, held_by=held_by)
    return update(state, entities=update(state.entities, carcass=arena))


def put_bush(state, slot, pos, radius=1.0):
    arena = state.entities.bush
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(True),
        transform=update(
            arena.transform, pos=arena.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32))
        ),
        shape=update(arena.shape, radius=arena.shape.radius.at[slot].set(radius)),
    )
    return update(state, entities=update(state.entities, bush=arena))


def creature_pos(state):
    return np.asarray(state.entities.creature.transform.pos)


def creature_touch(state):
    return np.asarray(state.entities.creature.contact.touch)


def test_two_bodies_separate_to_exact_contact():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [10.0, 10.0])
    state = put_creature(state, 1, [10.8, 10.0])
    out, buffer = physics.apply(state, cfg)
    positions = creature_pos(out)
    distance = float(np.linalg.norm(positions[0] - positions[1]))
    assert distance == pytest.approx(1.0, abs=1e-3)
    assert float(physics.residual_overlap(out, cfg)) < 1e-3
    assert int(buffer.count) == 0
    assert positions[0][0] == pytest.approx(9.9, abs=1e-4)
    assert positions[1][0] == pytest.approx(10.9, abs=1e-4)


def test_push_is_mass_weighted():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [20.0, 10.0], mass=1.0)
    state = put_creature(state, 1, [20.8, 10.0], mass=0.3)
    out, _ = physics.apply(state, cfg)
    positions = creature_pos(out)
    # Overlap 0.2 is split in inverse proportion to mass: 0.2 * 0.3 / 1.3 and
    # 0.2 * 1.0 / 1.3.
    assert positions[0][0] - 20.0 == pytest.approx(-0.2 * 0.3 / 1.3, abs=1e-4)
    assert positions[1][0] - 20.8 == pytest.approx(0.2 * 1.0 / 1.3, abs=1e-4)
    assert 1.0 * (positions[0][0] - 20.0) + 0.3 * (positions[1][0] - 20.8) == pytest.approx(
        0.0, abs=1e-5
    )


def test_three_body_chain_resolves_within_tolerance():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [30.0, 10.0])
    state = put_creature(state, 1, [30.9, 10.0])
    state = put_creature(state, 2, [31.8, 10.0])
    out, _ = physics.apply(state, cfg)
    assert float(physics.residual_overlap(out, cfg)) < 1e-3


@pytest.mark.parametrize(
    ("target", "quadrant"),
    (
        ([10.8, 10.0], FRONT),
        ([10.0, 10.4], RIGHT),
        ([9.2, 10.0], BACK),
        ([10.0, 9.6], LEFT),
    ),
)
def test_contact_quadrant_assignment(target, quadrant):
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [10.0, 10.0], theta=0.0)
    state = put_creature(state, 1, target)
    out, _ = physics.apply(state, cfg)
    touch = creature_touch(out)[0]
    expected = [0.0, 0.0, 0.0, 0.0]
    expected[quadrant] = 1.0 - float(np.hypot(target[0] - 10.0, target[1] - 10.0))
    assert touch.tolist() == pytest.approx(expected, abs=1e-5)


def test_both_sides_of_a_contact_are_recorded():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [10.0, 10.0], theta=0.0)
    state = put_creature(state, 1, [10.8, 10.0], theta=0.0)
    out, _ = physics.apply(state, cfg)
    touch = creature_touch(out)
    assert touch[0][FRONT] == pytest.approx(0.2, abs=1e-5)
    assert touch[1][BACK] == pytest.approx(0.2, abs=1e-5)


def test_contact_is_replaced_next_tick():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [10.0, 10.0])
    state = put_creature(state, 1, [10.8, 10.0])
    first, _ = physics.apply(state, cfg)
    assert float(creature_touch(first)[0].sum()) > 0.0
    second, _ = physics.apply(first, cfg)
    assert np.allclose(creature_touch(second), 0.0, atol=1e-6)


def test_wall_clips_position_halves_velocity_and_records_contact():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [0.3, 10.0], theta=math.pi, v=0.4)
    out, _ = physics.apply(state, cfg)
    creature = out.entities.creature
    assert float(creature.transform.pos[0, 0]) == pytest.approx(0.5, abs=1e-6)
    assert float(creature.transform.v[0]) == pytest.approx(0.2, abs=1e-6)
    touch = creature_touch(out)[0]
    assert touch[FRONT] == pytest.approx(0.2, abs=1e-5)
    assert touch[RIGHT] == pytest.approx(0.0, abs=1e-6)
    assert touch[LEFT] == pytest.approx(0.0, abs=1e-6)


def test_free_body_keeps_velocity_and_has_no_contact():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [30.0, 30.0], theta=0.0, v=0.4)
    out, _ = physics.apply(state, cfg)
    creature = out.entities.creature
    assert float(creature.transform.pos[0, 0]) == pytest.approx(30.0, abs=1e-6)
    assert float(creature.transform.v[0]) == pytest.approx(0.4, abs=1e-6)
    assert np.allclose(creature_touch(out)[0], 0.0, atol=1e-6)


def test_wall_corner_records_two_quadrants():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [0.2, 63.8], theta=0.0)
    out, _ = physics.apply(state, cfg)
    touch = creature_touch(out)[0]
    # Facing +x: the left wall is behind (back), the top wall is to the right.
    assert touch[BACK] == pytest.approx(0.3, abs=1e-5)
    assert touch[RIGHT] == pytest.approx(0.3, abs=1e-5)
    assert touch[FRONT] == pytest.approx(0.0, abs=1e-6)
    assert touch[LEFT] == pytest.approx(0.0, abs=1e-6)


def test_held_bodies_do_not_collide():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [10.0, 10.0])
    state = put_carcass(state, 0, [10.5, 10.0], held_by_creature=0)
    out, _ = physics.apply(state, cfg)
    assert np.allclose(creature_pos(out)[0], [10.0, 10.0], atol=1e-6)
    assert np.allclose(
        np.asarray(out.entities.carcass.transform.pos[0]), [10.5, 10.0], atol=1e-6
    )
    assert np.allclose(creature_touch(out)[0], 0.0, atol=1e-6)


def test_creatures_pass_through_bushes():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [10.0, 10.0])
    state = put_bush(state, 0, [10.1, 10.0], radius=1.0)
    out, _ = physics.apply(state, cfg)
    assert np.allclose(creature_pos(out)[0], [10.0, 10.0], atol=1e-6)


def test_dead_bodies_are_ignored():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [10.0, 10.0])
    state = put_carcass(state, 0, [10.5, 10.0])
    carcass = state.entities.carcass
    state = update(
        state,
        entities=update(
            state.entities, carcass=update(carcass, alive=carcass.alive.at[0].set(False))
        ),
    )
    out, _ = physics.apply(state, cfg)
    assert np.allclose(creature_pos(out)[0], [10.0, 10.0], atol=1e-6)


def test_apply_traces_under_jit():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [10.0, 10.0])
    state = put_creature(state, 1, [10.6, 10.0])
    compiled = jax.jit(physics.apply, static_argnums=(1,))
    out, buffer = compiled(state, cfg)
    out, buffer = compiled(out, cfg)
    assert out.entities.creature.transform.pos.shape == (4, 2)
    assert int(buffer.count) == 0
