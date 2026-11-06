"""Startup ramp (SANDBOX.md §9).

The first ticks of a world start from permissive environmental conditions
(more berries, lower metabolism, smaller temperature amplitude, no toxin) and
interpolate linearly to the target values at world.ramp_ticks.  Only
environmental conditions are ramped; behaviour is never designed.

The ramp is a pure function of the tick and static config, so it is jit-safe.
"""

from __future__ import annotations

import jax.numpy as jnp


def fraction(tick, cfg) -> jnp.ndarray:
    """0 at the start of the ramp, 1 once the target conditions are reached."""
    ticks = cfg.world.ramp_ticks
    if ticks <= 0:
        return jnp.asarray(1.0, dtype=jnp.float32)
    return jnp.clip(jnp.asarray(tick, dtype=jnp.float32) / float(ticks), 0.0, 1.0)


def blend(tick, cfg, start_multiplier: float) -> jnp.ndarray:
    """Linear blend from start_multiplier at tick 0 to 1.0 at ramp_ticks."""
    return jnp.asarray(start_multiplier, dtype=jnp.float32) + (
        1.0 - jnp.asarray(start_multiplier, dtype=jnp.float32)
    ) * fraction(tick, cfg)


def berry_multiplier(tick, cfg) -> jnp.ndarray:
    return blend(tick, cfg, cfg.world.ramp_berry_multiplier)


def metabolism_multiplier(tick, cfg) -> jnp.ndarray:
    return blend(tick, cfg, cfg.world.ramp_metabolism_multiplier)


def temperature_multiplier(tick, cfg) -> jnp.ndarray:
    return blend(tick, cfg, cfg.world.ramp_temperature_multiplier)


def toxin_active(tick, cfg) -> jnp.ndarray:
    """Toxin is zero during the ramp (§9)."""
    return fraction(tick, cfg) >= 1.0
