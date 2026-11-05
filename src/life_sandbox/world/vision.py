"""Ray-cast vision (SANDBOX.md §5.6).

Owner: world teammate.  One forward field of view of 60 degrees rendered as a
1x15 RGB strip::

    phi_i = theta - 30deg + (i + 0.5) * 4deg          (i = 0..14, left to right)
    cast a ray along phi_i up to R_VIEW = 16
    nearest living entity circle or wall -> colour c, distance d
    RGB_i = c * exp(-d / 6) * L + N(0, sigma_v^2)
    sigma_v = 0.02 + 0.08 * (1 - L)
    nothing hit -> RGB_i = 0 (+ noise)

Rays ignore the caster itself, whatever it holds, and every bush whose circle
contains the caster (§5.6 "the bush self is inside of").  Walls are grey and are
always drawn when within R_VIEW.  During sleep the field is closed: every pixel
is zero plus noise (§5.6, §5.2).

Determinism (SANDBOX.md §2): pairwise ray/circle tests are dense arrays reduced
along a fixed axis; the only randomness is the key passed in, which the sense
system derives from (world key, tick, "sense").
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from life_sandbox.config import Config
from life_sandbox.core import masks
from life_sandbox.core.components import BUSH, CARCASS, CREATURE, NEUTRAL
from life_sandbox.core.state import WorldState
from life_sandbox.world import fields

# Maximum ray length (§5.6).
R_VIEW: float = 16.0

# Brightness fall-off length in the attenuation exp(-d / ATTENUATION) (§5.6).
ATTENUATION: float = 6.0

# Wall colour: grey, value chosen locally (the section only says "grey").
WALL_RGB: tuple[float, float, float] = (0.5, 0.5, 0.5)

# Pixel noise: sigma_v = BASE + DARK * (1 - L) (§5.6).
VISION_BASE_NOISE: float = 0.02
VISION_DARK_NOISE: float = 0.08

_EPS = 1e-6


def pixel_angles(cfg: Config) -> jax.Array:
    """Pixel directions relative to heading, left to right (§5.6)."""
    count = cfg.sensors.vision_pixels
    fov = jnp.asarray(cfg.sensors.fov_deg, jnp.float32) * jnp.pi / 180.0
    index = jnp.arange(count, dtype=jnp.float32)
    return -0.5 * fov + (index + 0.5) * (fov / count)


def block_starts(cfg: Config) -> jax.Array:
    """Circle-array offset of each prototype block, indexed by prototype id."""
    n_creature = cfg.capacities.creature
    n_bush = cfg.capacities.bush
    n_carcass = cfg.capacities.carcass
    return jnp.asarray(
        [
            0,
            n_creature,
            n_creature + n_bush,
            n_creature + n_bush + n_carcass,
        ],
        jnp.int32,
    )


def ref_circle(cfg: Config, kind, slot) -> jax.Array:
    """Circle index of a (prototype, slot) reference; clip for stale refs."""
    start = jnp.take(block_starts(cfg), jnp.clip(jnp.asarray(kind, jnp.int32), 0, 3))
    total = (
        cfg.capacities.creature
        + cfg.capacities.bush
        + cfg.capacities.carcass
        + cfg.capacities.neutral
    )
    return jnp.clip(start + jnp.asarray(slot, jnp.int32), 0, total - 1)


def scene(state: WorldState, cfg: Config):
    """Every entity as a flat circle array.

    Returns (pos [M,2], radius [M], rgb [M,3], alive [M], proto [M], slot [M])
    with prototype blocks ordered creature, bush, carcass, neutral.
    """
    ent = state.entities
    arenas = (
        (CREATURE, ent.creature.transform.pos, ent.creature.shape.radius,
         ent.creature.look.rgb, ent.creature.alive),
        (BUSH, ent.bush.transform.pos, ent.bush.shape.radius,
         ent.bush.look.rgb, ent.bush.alive),
        (CARCASS, ent.carcass.transform.pos, ent.carcass.shape.radius,
         ent.carcass.look.rgb, ent.carcass.alive),
        (NEUTRAL, ent.neutral.transform.pos, ent.neutral.shape.radius,
         ent.neutral.look.rgb, ent.neutral.alive),
    )
    pos = jnp.concatenate([part[1] for part in arenas], axis=0)
    radius = jnp.concatenate([part[2] for part in arenas], axis=0)
    rgb = jnp.concatenate([part[3] for part in arenas], axis=0)
    alive = jnp.concatenate([part[4] for part in arenas], axis=0)
    proto = jnp.concatenate(
        [jnp.full(part[1].shape[0], part[0], jnp.int32) for part in arenas]
    )
    slot = jnp.concatenate(
        [jnp.arange(part[1].shape[0], dtype=jnp.int32) for part in arenas]
    )
    return pos, radius, rgb, alive, proto, slot


def _ignore_mask(state: WorldState, cfg: Config, total: int) -> jax.Array:
    """[N, M] bool: circles each caster's rays must skip (§5.6)."""
    ent = state.entities
    n = cfg.capacities.creature
    rows = jnp.arange(n, dtype=jnp.int32)
    ignore = jnp.zeros((n, total), jnp.bool_)

    # The caster itself: the creature block starts at index 0 and slot i == i.
    ignore = ignore.at[rows, rows].set(True)

    # Whatever the caster holds (a stale reference is clipped but not ignored).
    holding = ent.creature.grip.holding
    held_valid = masks.ref_valid(ent, holding.kind, holding.slot, holding.uid)
    held_circle = ref_circle(cfg, holding.kind, holding.slot)
    held_ignore = jnp.zeros((n, total), jnp.bool_).at[rows, held_circle].set(held_valid)
    ignore = ignore | held_ignore

    # Every bush whose circle contains the caster.
    n_bush = cfg.capacities.bush
    if n_bush:
        delta = ent.creature.transform.pos[:, None, :] - ent.bush.transform.pos[None, :, :]
        inside = jnp.sum(delta * delta, axis=-1) <= ent.bush.shape.radius[None, :] ** 2
        inside = inside & ent.bush.alive[None, :]
        start = cfg.capacities.creature
        block = ignore[:, start:start + n_bush] | inside
        ignore = ignore.at[:, start:start + n_bush].set(block)
    return ignore


def render(
    state: WorldState,
    cfg: Config,
    key: jax.Array,
    noise_base: float = VISION_BASE_NOISE,
    noise_dark: float = VISION_DARK_NOISE,
) -> jax.Array:
    """Render the visible strip of every creature slot: float32 [N, P, 3].

    The result is the sensor's raw reading (attenuated colour plus pixel noise);
    the slice assembly in the sense system flattens it into 45 dims.
    """
    ent = state.entities
    n = cfg.capacities.creature
    count = cfg.sensors.vision_pixels

    cpos, cradius, crgb, calive, _, _ = scene(state, cfg)
    total = cpos.shape[0]
    origin = ent.creature.transform.pos
    theta = ent.creature.transform.theta

    phi = theta[:, None] + pixel_angles(cfg)[None, :]
    dirs = jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=-1)  # [N, P, 2]

    dot_dc = jnp.einsum("npk,mk->npm", dirs, cpos)  # d . c
    dot_do = jnp.einsum("npk,nk->np", dirs, origin)  # d . o
    b = dot_dc - dot_do[:, :, None]  # d . (c - o)
    dot_oc = jnp.einsum("mk,nk->nm", cpos, origin)  # c . o
    centers = jnp.sum(cpos * cpos, axis=-1)  # [M]
    origins = jnp.sum(origin * origin, axis=-1)  # [N]
    # |o - c|^2 - r^2, the constant term of the ray/circle quadratic.
    quad = (
        centers[None, None, :]
        - 2.0 * dot_oc[:, None, :]
        + origins[:, None, None]
        - cradius[None, None, :] ** 2
    )
    disc = b * b - quad
    root = jnp.sqrt(jnp.maximum(disc, 0.0))
    t_near = b - root
    t_far = b + root
    # A ray starting inside a circle counts as touching it at distance zero.
    t_hit = jnp.where(quad <= 0.0, 0.0, jnp.maximum(t_near, 0.0))
    hit_ok = (disc >= 0.0) & (t_far >= 0.0)
    t_hit = jnp.where(hit_ok, t_hit, jnp.inf)
    t_hit = jnp.where(_ignore_mask(state, cfg, total)[:, None, :], jnp.inf, t_hit)
    t_hit = jnp.where(calive[None, None, :], t_hit, jnp.inf)

    entity_dist = jnp.min(t_hit, axis=-1)  # [N, P]
    entity_index = jnp.argmin(t_hit, axis=-1)  # [N, P]

    px = origin[:, 0][:, None]
    py = origin[:, 1][:, None]
    dx = dirs[..., 0]
    dy = dirs[..., 1]
    width = jnp.asarray(cfg.world.width, jnp.float32)
    height = jnp.asarray(cfg.world.height, jnp.float32)
    tx_num = jnp.where(dx > 0.0, width - px, -px)
    ty_num = jnp.where(dy > 0.0, height - py, -py)
    tx = jnp.where(jnp.abs(dx) > _EPS, tx_num / jnp.where(jnp.abs(dx) > _EPS, dx, 1.0), jnp.inf)
    ty = jnp.where(jnp.abs(dy) > _EPS, ty_num / jnp.where(jnp.abs(dy) > _EPS, dy, 1.0), jnp.inf)
    wall_dist = jnp.minimum(tx, ty)
    wall_dist = jnp.where(wall_dist < 0.0, jnp.inf, wall_dist)

    entity_hit = jnp.isfinite(entity_dist) & (entity_dist <= R_VIEW) & (entity_dist <= wall_dist)
    wall_hit = wall_dist < entity_dist

    distance = jnp.where(entity_hit, entity_dist, jnp.where(wall_hit, wall_dist, 0.0))
    colour = jnp.where(
        entity_hit[..., None],
        crgb[entity_index],
        jnp.asarray(WALL_RGB, jnp.float32)[None, None, :],
    )
    visible = entity_hit | wall_hit

    _, luminance = fields.environment_at(state, cfg)  # [N]
    shade = jnp.exp(-distance / ATTENUATION) * luminance[:, None]
    pixels = jnp.where(visible[..., None], colour * shade[..., None], 0.0)

    asleep = ent.creature.body.asleep[:, None, None]
    pixels = jnp.where(asleep, 0.0, pixels)

    sigma = noise_base + noise_dark * (1.0 - luminance)
    noise = jax.random.normal(key, (n, count, 3), dtype=jnp.float32) * sigma[:, None, None]
    return (pixels + noise).astype(jnp.float32)
