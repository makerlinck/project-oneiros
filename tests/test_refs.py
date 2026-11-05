"""References, slot validity and held mass (SANDBOX.md §3.1, §5.4)."""

from __future__ import annotations

import jax.numpy as jnp

from life_sandbox.core import components as C
from life_sandbox.core import masks
from life_sandbox.core.struct import update
from life_sandbox.world.init import initial_state


def test_ref_validates_uid_and_alive(tiny_cfg):
    state = initial_state(tiny_cfg)
    entities = state.entities
    uid = entities.bush.uid[0]
    assert bool(masks.ref_valid(entities, C.BUSH, jnp.asarray(0), uid))
    assert not bool(masks.ref_valid(entities, C.BUSH, jnp.asarray(0), uid + 1))
    assert not bool(masks.ref_valid(entities, C.CARCASS, jnp.asarray(0), jnp.asarray(0)))

    dead = update(entities.bush, alive=entities.bush.alive.at[0].set(False))
    entities = update(entities, bush=dead)
    assert not bool(masks.ref_valid(entities, C.BUSH, jnp.asarray(0), uid))


def test_uid_at_and_alive_at(tiny_cfg):
    state = initial_state(tiny_cfg)
    entities = state.entities
    kinds = jnp.asarray([C.BUSH, C.CREATURE, C.CARCASS], dtype=jnp.int32)
    slots = jnp.asarray([0, 0, 0], dtype=jnp.int32)
    assert int(masks.uid_at(entities, kinds, slots)[0]) == int(entities.bush.uid[0])
    assert bool(masks.alive_at(entities, kinds, slots)[0])
    assert not bool(masks.alive_at(entities, kinds, slots)[2])


def test_held_mass_uses_valid_reference(tiny_cfg):
    state = initial_state(tiny_cfg)
    entities = state.entities
    carcass = update(
        entities.carcass,
        alive=entities.carcass.alive.at[0].set(True),
        uid=entities.carcass.uid.at[0].set(123),
        shape=update(entities.carcass.shape, mass=entities.carcass.shape.mass.at[0].set(0.5)),
    )
    holding = C.Holding(
        kind=jnp.full(8, C.CARCASS, dtype=jnp.int32),
        slot=jnp.zeros(8, dtype=jnp.int32),
        uid=jnp.full(8, 123, dtype=jnp.int32),
    )
    creature = update(entities.creature, grip=update(entities.creature.grip, holding=holding))
    entities = update(entities, creature=creature, carcass=carcass)
    mass = masks.held_mass(entities)
    assert float(mass[0]) == 0.5
    assert float(mass[1]) == 0.5

    stale = update(holding, uid=jnp.full(8, 999, dtype=jnp.int32))
    creature = update(entities.creature, grip=update(entities.creature.grip, holding=stale))
    entities = update(entities, creature=creature)
    assert float(masks.held_mass(entities)[0]) == 0.0
