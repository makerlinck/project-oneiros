"""World state (SANDBOX.md §2, §3).

WorldState is the complete pytree that systems map to a new pytree.  Everything
that evolves is a leaf here; configuration is passed separately and never
enters the state, so a snapshot is just the pytree plus the config hash.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from life_sandbox.core.components import Entities
from life_sandbox.core.struct import struct
from life_sandbox.world.constants import INITIAL_COMP


@struct
@dataclass(frozen=True)
class WorldFields:
    """Simulated fields (§4.3).  Temperature and light are analytic; odour is a grid."""

    odor: jax.Array


@struct
@dataclass(frozen=True)
class WorldState:
    key: jax.Array
    tick: jax.Array
    uid_counter: jax.Array
    entities: Entities
    fields: WorldFields
    comp: jax.Array
    pain_weights: jax.Array
    slice: jax.Array


def empty(cfg, key: jax.Array | None = None) -> WorldState:
    """Blank world: no entities alive, empty fields, default berry composition."""
    grid = cfg.world.grid_shape
    odor_channels = cfg.sensors.odor_channels
    return WorldState(
        key=jax.random.key(0) if key is None else key,
        tick=jnp.asarray(0, dtype=jnp.int32),
        uid_counter=jnp.asarray(0, dtype=jnp.int32),
        entities=Entities.empty(cfg),
        fields=WorldFields(odor=jnp.zeros((*grid, odor_channels), dtype=jnp.float32)),
        comp=jnp.asarray(INITIAL_COMP, dtype=jnp.float32),
        pain_weights=jnp.zeros((9, odor_channels), dtype=jnp.float32),
        slice=jnp.zeros(
            (cfg.capacities.creature, cfg.sensors.slice_dim + (5 if cfg.learning.readable_leak else 0)),
            dtype=jnp.float32,
        ),
    )


def population(state: WorldState) -> jax.Array:
    """Number of living creatures."""
    return jnp.sum(state.entities.creature.alive.astype(jnp.int32))
