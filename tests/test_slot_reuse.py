"""Slot allocation and uid assignment (SANDBOX.md §3.1)."""

from __future__ import annotations

import jax.numpy as jnp

from life_sandbox.core import components as C
from life_sandbox.core import masks
from life_sandbox.core.struct import update


def test_first_free_slot():
    alive = jnp.asarray([True, True, False, True])
    assert int(masks.first_free_slot(alive)) == 2
    assert int(masks.first_free_slot(jnp.ones(3, dtype=jnp.bool_))) == -1


def test_free_slots_are_ascending_and_bounded():
    alive = jnp.asarray([True, False, True, False, False])
    slots, ok = masks.free_slots(alive, 2)
    assert bool(ok)
    assert [int(s) for s in slots] == [1, 3]
    slots, ok = masks.free_slots(alive, 4)
    assert not bool(ok)


def test_allocate_uids_is_sequential_and_monotonic():
    uids, counter = masks.allocate_uids(jnp.asarray(10, jnp.int32), 3)
    assert [int(u) for u in uids] == [10, 11, 12]
    assert int(counter) == 13
    uids, counter = masks.allocate_uids(counter, 2)
    assert [int(u) for u in uids] == [13, 14]
    assert int(counter) == 15


def test_slot_reuse_invalidates_old_reference(tiny_cfg):
    from life_sandbox.world.init import initial_state

    state = initial_state(tiny_cfg)
    entities = state.entities
    old_uid = entities.bush.uid[0]
    assert bool(masks.ref_valid(entities, C.BUSH, jnp.asarray(0), old_uid))

    # Reuse slot 0 for a new bush with a new uid; all components are overwritten.
    bush = update(
        entities.bush,
        uid=entities.bush.uid.at[0].set(999),
        alive=entities.bush.alive.at[0].set(True),
        edible=update(entities.bush.edible, stock=entities.bush.edible.stock.at[0].set(0.1)),
    )
    entities = update(entities, bush=bush)
    assert not bool(masks.ref_valid(entities, C.BUSH, jnp.asarray(0), old_uid))
    assert bool(masks.ref_valid(entities, C.BUSH, jnp.asarray(0), jnp.asarray(999)))
