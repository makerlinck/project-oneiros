"""Startup ramp (SANDBOX.md §9).

The first ticks of a world start from permissive environmental conditions
(more berries, lower metabolism, smaller temperature amplitude, weaker toxin)
and interpolate linearly to the target values at world.ramp_ticks.  Only
environmental conditions are ramped; behaviour is never designed.

The ramp is a pure function of the tick and static config, so it is jit-safe.

Toxin (rule fix from the 2026-11-06 review, problem 7): the ramp does not set
the berry toxin to zero.  Every individual with rho_T > 0 would then be
chronically short of the toxin nutrient from the first tick.  Instead the toxic
species carries alpha = 0.1 + 0.9*fraction of its target toxin share (0.4), so
the world starts with a weak but nonzero toxin that grows to the full share at
ramp_ticks.  The switch schedule itself runs normally during the ramp.
"""

from __future__ import annotations

import jax.numpy as jnp

# Toxin share of the toxic berry species at tick 0 of the ramp (§4.4, §9).
TOXIN_START_STRENGTH: float = 0.1


def fraction(tick, cfg, params=None) -> jnp.ndarray:
    """0 at the start of the ramp, 1 once the target conditions are reached.

    With runtime `params` the ramp length is a traced scalar, so every row of a
    batched scan can use its own ramp_ticks (including 0 = no ramp).
    """
    if params is None:
        ticks = cfg.world.ramp_ticks
        if ticks <= 0:
            return jnp.asarray(1.0, dtype=jnp.float32)
        return jnp.clip(jnp.asarray(tick, dtype=jnp.float32) / float(ticks), 0.0, 1.0)
    ticks = jnp.asarray(params.ramp_ticks, jnp.float32)
    ramped = jnp.clip(
        jnp.asarray(tick, dtype=jnp.float32) / jnp.maximum(ticks, 1.0), 0.0, 1.0
    )
    return jnp.where(ticks > 0.0, ramped, jnp.asarray(1.0, dtype=jnp.float32))


def blend(tick, cfg, start_multiplier: float, params=None) -> jnp.ndarray:
    """Linear blend from start_multiplier at tick 0 to 1.0 at ramp_ticks."""
    return jnp.asarray(start_multiplier, dtype=jnp.float32) + (
        1.0 - jnp.asarray(start_multiplier, dtype=jnp.float32)
    ) * fraction(tick, cfg, params)


def berry_multiplier(tick, cfg, params=None) -> jnp.ndarray:
    return blend(tick, cfg, cfg.world.ramp_berry_multiplier, params)


def metabolism_multiplier(tick, cfg, params=None) -> jnp.ndarray:
    return blend(tick, cfg, cfg.world.ramp_metabolism_multiplier, params)


def temperature_multiplier(tick, cfg, params=None) -> jnp.ndarray:
    return blend(tick, cfg, cfg.world.ramp_temperature_multiplier, params)


def toxin_strength(tick, cfg, params=None) -> jnp.ndarray:
    """Strength alpha of the toxic berry species: 0.1 at tick 0, 1 after the ramp.

    The toxic species composition is (1 - 0.4*alpha) * base + 0.4*alpha*toxin
    (SANDBOX.md §4.4).  alpha is never zero so that the toxin nutrient is
    available from the first tick; with ramp_ticks = 0 alpha is 1 (full share).
    """
    start = jnp.asarray(TOXIN_START_STRENGTH, dtype=jnp.float32)
    return start + (1.0 - start) * fraction(tick, cfg, params)
