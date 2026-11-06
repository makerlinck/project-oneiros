"""Fields system (SANDBOX.md §3.3 step 1, §4.3).

Owner: world teammate.  Temperature and light stay analytic (no grid).  Odour is
a float32 grid [rows, cols, C_o] updated once per tick::

    c <- (1 - delta) * diffuse(c) + deposits

"diffuse" is the 3x3 five-point Laplacian stencil [[0, D, 0], [D, 1-4D, D],
[0, D, 0]] with edge-replicated boundaries, so it conserves total mass exactly
(delta = 0 keeps odour total; SANDBOX.md §9 M0 acceptance).  D =
world.odor_diffusion must stay below 0.25 for stability.

Deposits are rounded to int32 fixed point scaled by world.odor_scale and
accumulated with integer addition, so the result does not depend on the order in
which entities are visited (SANDBOX.md §2: fixed-point accumulation for
deposit/contact style multi-writer updates).

Odour sources (SANDBOX.md §4.3): bushes (species spectrum x stock), creatures
(genome spectrum), carcasses (rot spectrum x rot, rot = 1 - 0.995^age_dead),
neutral objects (fixed spectrum).  Concentration is a linear field; a genome
spectrum may be negative, so readers take log1p(max(c, 0)).

Grid convention: cell (row, col) covers world x in [col*cell, (col+1)*cell) and
y in [row*cell, (row+1)*cell).  A world point maps to grid coordinates
(x/cell - 0.5, y/cell - 0.5) so that cell centres sit on integers.

Temperature amplitude (SANDBOX.md §4.3, §9): the *whole* analytic temperature
(geographic gradient, day phase and the in-bush -0.5 offset) is multiplied by
world.temperature_scale (an M1 calibration factor) and by
core.ramp.temperature_multiplier(tick, cfg) (the startup ramp), so the world
starts with a small temperature difference and reaches world.temperature_scale
of the nominal amplitude at world.ramp_ticks.  Light is not scaled by either
factor (the ramp only scales the temperature difference).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from life_sandbox.config import Config
from life_sandbox.core import events, ramp
from life_sandbox.core.state import WorldState
from life_sandbox.core.struct import update
from life_sandbox.world.constants import BUSH_ODOR, ROT_ODOR

# Carcass rot strength per tick (§4.2): rot = 1 - (1 - ROT_RATE)^age_dead.
ROT_RATE: float = 0.005

# Nose offset in front of the body centre beyond its radius (§4.3, §5.6).
NOSE_OFFSET: float = 0.2


def temperature(cfg: Config, y, tick, in_bush=False):
    """Analytic temperature field T_geo(y) + day phase - bush shade (§4.3).

    T_geo(y) = -2 + 4*y/height; the day term is +0.5*cos(2*pi*t/T_day); a body
    inside a bush is 0.5 colder (§4.2).  The whole amplitude is scaled by the
    M1 calibration factor world.temperature_scale and by the startup ramp
    multiplier (§9); with ramp_ticks = 0 and temperature_scale = 1 the result is
    the nominal amplitude.
    """
    phase = jnp.cos(2.0 * jnp.pi * jnp.asarray(tick, jnp.float32) / cfg.world.day_ticks)
    bush = jnp.asarray(in_bush, jnp.float32)
    amplitude = (
        (-2.0 + 4.0 * jnp.asarray(y, jnp.float32) / cfg.world.height) + 0.5 * phase - 0.5 * bush
    )
    scale = ramp.temperature_multiplier(tick, cfg) * jnp.asarray(
        cfg.world.temperature_scale, jnp.float32
    )
    return scale * amplitude


def light(cfg: Config, tick, in_bush=False):
    """Analytic light field (0..1); a body inside a bush sees half of it (§4.3)."""
    phase = jnp.cos(2.0 * jnp.pi * jnp.asarray(tick, jnp.float32) / cfg.world.day_ticks)
    bush = jnp.asarray(in_bush, jnp.float32)
    return (0.5 + 0.5 * phase) * (1.0 - 0.5 * bush)


def diffusion_kernel(cfg: Config) -> jax.Array:
    """3x3 five-point Laplacian stencil for one diffusion step (§4.3).

    Center 1 - 4D and the four axis neighbours D; diagonals are zero.  The
    weights sum to one, so the stencil itself neither creates nor destroys
    odour.
    """
    d = jnp.asarray(cfg.world.odor_diffusion, jnp.float32)
    zero = jnp.asarray(0.0, jnp.float32)
    return jnp.asarray(
        [
            [zero, d, zero],
            [d, 1.0 - 4.0 * d, d],
            [zero, d, zero],
        ],
        jnp.float32,
    )


def diffuse(cfg: Config, grid: jax.Array) -> jax.Array:
    """One mass-conserving 3x3 diffusion step over the odour grid (§4.3).

    Edge replication keeps the boundary flux symmetric, which makes the total
    grid sum exactly invariant for any symmetric zero-diagonal stencil.
    """
    d = jnp.asarray(cfg.world.odor_diffusion, jnp.float32)
    padded = jnp.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="edge")
    center = (1.0 - 4.0 * d) * padded[1:-1, 1:-1, :]
    axes = padded[:-2, 1:-1, :] + padded[2:, 1:-1, :] + padded[1:-1, :-2, :] + padded[1:-1, 2:, :]
    return center + d * axes


def _spectrum_table(table, channels: int) -> jax.Array:
    """Pad or truncate a constant spectrum table to the configured C_o."""
    base = jnp.asarray(table, jnp.float32)
    if base.shape[1] == channels:
        return base
    if base.shape[1] > channels:
        return base[:, :channels]
    pad = jnp.zeros((base.shape[0], channels - base.shape[1]), jnp.float32)
    return jnp.concatenate([base, pad], axis=1)


def flat_index(cfg: Config, positions) -> jax.Array:
    """Flat row-major index of the grid cell containing each world point."""
    rows, cols = cfg.world.grid_shape
    positions = jnp.asarray(positions, jnp.float32)
    col = jnp.clip(jnp.floor(positions[..., 0] / cfg.world.cell), 0, cols - 1).astype(jnp.int32)
    row = jnp.clip(jnp.floor(positions[..., 1] / cfg.world.cell), 0, rows - 1).astype(jnp.int32)
    return row * cols + col


def _scatter(cfg: Config, grid, positions, spectrum, mask) -> jax.Array:
    """Add one entity type's deposit to the int32 accumulator (order-free)."""
    idx = flat_index(cfg, positions)
    scale = jnp.asarray(cfg.world.odor_scale, jnp.float32)
    amount = jnp.rint(jnp.asarray(spectrum, jnp.float32) * scale).astype(jnp.int32)
    amount = jnp.where(jnp.asarray(mask, jnp.bool_)[:, None], amount, jnp.int32(0))
    return grid.at[idx].add(amount)


def deposit_field(cfg: Config, state: WorldState) -> jax.Array:
    """Fixed-point deposits of every living entity, as a float32 grid (§4.3).

    The int32 accumulator is the determinism device: integer addition is exact
    and commutative, so entities writing the same cell in any slot order
    produce the same field.  Amounts are quantised once, on emission, at
    world.odor_scale units per concentration unit.
    """
    rows, cols = cfg.world.grid_shape
    channels = cfg.sensors.odor_channels
    grid = jnp.zeros((rows * cols, channels), jnp.int32)
    ent = state.entities

    bush_table = _spectrum_table(BUSH_ODOR, channels)
    kind = jnp.clip(ent.bush.edible.kind, 0, bush_table.shape[0] - 1)
    bush_spectrum = bush_table[kind] * ent.bush.edible.stock[:, None]
    grid = _scatter(cfg, grid, ent.bush.transform.pos, bush_spectrum, ent.bush.alive)

    grid = _scatter(
        cfg, grid, ent.creature.transform.pos, ent.creature.look.odor, ent.creature.alive
    )

    age = ent.carcass.edible.age_dead.astype(jnp.float32)
    rot = 1.0 - jnp.power(1.0 - ROT_RATE, age)
    rot_table = _spectrum_table((ROT_ODOR,), channels)[0]
    carcass_spectrum = rot_table[None, :] * rot[:, None]
    grid = _scatter(cfg, grid, ent.carcass.transform.pos, carcass_spectrum, ent.carcass.alive)

    grid = _scatter(cfg, grid, ent.neutral.transform.pos, ent.neutral.look.odor, ent.neutral.alive)

    scale = jnp.asarray(cfg.world.odor_scale, jnp.float32)
    return grid.reshape(rows, cols, channels).astype(jnp.float32) / scale


def sample_odor(cfg: Config, odor: jax.Array, points) -> jax.Array:
    """Bilinear sample of the odour grid at world points (§4.3, §5.6).

    points is [..., 2] in world units; the return has shape
    points.shape[:-1] + (C_o,).  Points outside the arena clamp to the border
    cells.
    """
    rows, cols = cfg.world.grid_shape
    points = jnp.asarray(points, jnp.float32)
    gx = jnp.clip(points[..., 0] / cfg.world.cell - 0.5, 0.0, cols - 1.0)
    gy = jnp.clip(points[..., 1] / cfg.world.cell - 0.5, 0.0, rows - 1.0)
    x0 = jnp.floor(gx).astype(jnp.int32)
    y0 = jnp.floor(gy).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, cols - 1)
    y1 = jnp.minimum(y0 + 1, rows - 1)
    fx = (gx - x0.astype(jnp.float32))[..., None]
    fy = (gy - y0.astype(jnp.float32))[..., None]
    c00 = odor[y0, x0]
    c10 = odor[y0, x1]
    c01 = odor[y1, x0]
    c11 = odor[y1, x1]
    return (
        (1.0 - fx) * (1.0 - fy) * c00
        + fx * (1.0 - fy) * c10
        + (1.0 - fx) * fy * c01
        + fx * fy * c11
    )


def in_bush(entities, points) -> jax.Array:
    """Whether each point lies inside a living bush circle (§4.2 shading)."""
    points = jnp.asarray(points, jnp.float32)
    lead = points.shape[:-1]
    flat = points.reshape(-1, 2)
    bush = entities.bush
    delta = flat[:, None, :] - bush.transform.pos[None, :, :]
    inside = jnp.sum(delta * delta, axis=-1) <= bush.shape.radius[None, :] ** 2
    inside = inside & bush.alive[None, :]
    return jnp.any(inside, axis=-1).reshape(lead)


def environment_at(state: WorldState, cfg: Config) -> tuple[jax.Array, jax.Array]:
    """Temperature and light at every creature centre (§4.3 sensor sampling)."""
    pos = state.entities.creature.transform.pos
    bush = in_bush(state.entities, pos)
    return temperature(cfg, pos[:, 1], state.tick, bush), light(cfg, state.tick, bush)


def apply(state: WorldState, cfg: Config) -> tuple[WorldState, events.EventBuffer]:
    """Step 1: decay + diffuse the odour grid, then add this tick's deposits."""
    decay = jnp.asarray(1.0 - cfg.world.odor_decay, jnp.float32)
    odor = decay * diffuse(cfg, state.fields.odor) + deposit_field(cfg, state)
    fields = update(state.fields, odor=odor.astype(jnp.float32))
    return update(state, fields=fields), events.empty(cfg)
