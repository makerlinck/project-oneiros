"""Runtime calibration parameters (SANDBOX.md §9 M1).

The M1 scan varies a handful of numbers per run: base metabolism, berry
regeneration, temperature amplitude, founder reserves, startup-ramp length and
the number of bushes.  Keeping those in the static `Config` forces one XLA
compile per value.  `ScanParams` carries them as JAX arrays instead, so one
compiled segment can run a batch of "configuration x world" rows with vmap.

Scope: only host-observable/continuous knobs belong here.  Everything that
changes array shapes or the agent interface stays static in `Config`:
capacities (the bush arena is compiled at the maximum scanned size and
`bush_count` masks the alive slots), brain structure, sensor slices, world
geometry and event capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp

from life_sandbox.config import Config
from life_sandbox.core.struct import struct


@struct
@dataclass(frozen=True)
class ScanParams:
    """Per-world runtime values of the M1 scan knobs (JAX arrays).

    A single-world value has 0-d leaves; a batch has a leading axis.  Use
    `batched`/`broadcast`/`row` to move between the two layouts.
    """

    # BodyConfig.base_metabolism.
    base_metabolism: jax.Array
    # WorldConfig.bush_regen_rate / bush_regen_base.
    bush_regen_rate: jax.Array
    bush_regen_base: jax.Array
    # WorldConfig.temperature_scale.
    temperature_scale: jax.Array
    # WorldConfig.ramp_ticks (float32 so tick / ramp_ticks stays a traced op).
    ramp_ticks: jax.Array
    # BodyConfig.founder_reserves (founders and death reseeds).
    founder_reserves: jax.Array
    # Number of alive bushes; bounded by the static capacities.bush and
    # divisible by the three bush species.
    bush_count: jax.Array

    @classmethod
    def from_config(cls, cfg: Config) -> "ScanParams":
        """Single-world runtime parameters matching a static configuration."""
        return cls(
            base_metabolism=jnp.asarray(cfg.body.base_metabolism, jnp.float32),
            bush_regen_rate=jnp.asarray(cfg.world.bush_regen_rate, jnp.float32),
            bush_regen_base=jnp.asarray(cfg.world.bush_regen_base, jnp.float32),
            temperature_scale=jnp.asarray(cfg.world.temperature_scale, jnp.float32),
            ramp_ticks=jnp.asarray(cfg.world.ramp_ticks, jnp.float32),
            founder_reserves=jnp.asarray(cfg.body.founder_reserves, jnp.float32),
            bush_count=jnp.asarray(cfg.capacities.bush, jnp.int32),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, float], cfg: Config) -> "ScanParams":
        """Host-side constructor from a scan record (missing keys use `cfg`)."""
        defaults = cls.from_config(cfg)

        def pick(keys: tuple[str, ...], default, dtype):
            found = None
            for key in keys:
                candidate = values.get(key)
                if candidate is not None:
                    found = candidate
                    break
            return jnp.asarray(default if found is None else found, dtype)

        return cls(
            base_metabolism=pick(
                ("base_metabolism",), defaults.base_metabolism, jnp.float32
            ),
            bush_regen_rate=pick(
                ("bush_regen_rate",), defaults.bush_regen_rate, jnp.float32
            ),
            bush_regen_base=pick(
                ("bush_regen_base",), defaults.bush_regen_base, jnp.float32
            ),
            temperature_scale=pick(
                ("temperature_scale",), defaults.temperature_scale, jnp.float32
            ),
            ramp_ticks=pick(("ramp_ticks",), defaults.ramp_ticks, jnp.float32),
            founder_reserves=pick(
                ("founder_reserves",), defaults.founder_reserves, jnp.float32
            ),
            bush_count=pick(
                ("bush_capacity", "bush_count"), defaults.bush_count, jnp.int32
            ),
        )

    @staticmethod
    def batched(rows: list["ScanParams"]) -> "ScanParams":
        """Stack one row per (configuration, world) into a static batch pytree."""
        return jax.tree.map(lambda *leaves: jnp.stack(leaves), *rows)

    @staticmethod
    def broadcast(params: "ScanParams", size: int) -> "ScanParams":
        """Replicate one row across a batch axis."""
        return jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (size,) + jnp.shape(leaf)), params
        )

    @staticmethod
    def row(params: "ScanParams", index: int) -> "ScanParams":
        """Select one row from a batched params pytree."""
        return jax.tree.map(lambda leaf: leaf[index], params)
