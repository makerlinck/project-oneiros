"""Collision system (SANDBOX.md §3.3 step 7, §5.3).

Owner: world teammate.  After motion integration, overlapping bodies are pushed
apart along their centre line, mass-weighted (the lighter body yields more).
Bodies that are held do not participate at all (they follow their holder), and
creatures never collide with bushes (§5.3).  Afterwards every body is clipped
back inside the walls and its velocity is halved when it was clipped.

Iteration count: §5.3 says "two iterations", while §9 (M0 acceptance) and §13
(test_collision) demand a residual overlap below 1e-3.  The two spec'd passes
are always executed; because a pair is resolved exactly by one pass but a chain
only halves its residual per pass, a bounded deterministic convergence loop then
continues up to MAX_RESOLVE_ITERATIONS until the largest pair overlap is below
RESIDUAL_TOLERANCE.  This is a documented implementation choice for the
specification conflict (spec lower bound + convergence cap), not a change of the
mass-weighted push rule itself.

Contact forces: each creature accumulates the pre-resolution penetration depth
of every contact, split into the four body quadrants front / right / back /
left, and stores it in Contact.touch for the next tick's touch sensor.  Wall
contacts contribute the distance the body had to be clipped.  Direction
convention follows §5.3 ("left is negative turn, right is positive"):
delta = atan2(cross(h, u), dot(h, u)) with h the heading and u the direction
toward the contact; front |delta| <= 45 deg, right (45, 135], back |delta| >
135, left [-135, -45).

Determinism (SANDBOX.md §2): body pairs are a dense array reduced along a fixed
axis; there is no randomness in this system.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from life_sandbox.config import Config
from life_sandbox.core import events, masks
from life_sandbox.core.state import WorldState
from life_sandbox.core.struct import update

# SANDBOX.md §5.3 specifies two resolution iterations, while §9 (M0 acceptance)
# and §13 (test_collision) require the residual overlap to end below 1e-3.
# Two passes are exact for an isolated pair but only halve a chain's residual per
# pass, so after the two spec'd passes the solver keeps iterating (bounded,
# deterministic, jit-safe) until the largest pair overlap is below the tolerance.
RESOLVE_ITERATIONS: int = 2
RESIDUAL_TOLERANCE: float = 1e-3
MAX_RESOLVE_ITERATIONS: int = 12

_EPS = 1e-6
_MASS_FLOOR = 1e-3

_FRONT, _RIGHT, _BACK, _LEFT = 0, 1, 2, 3


def held_masks(state: WorldState) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Whether each creature / carcass / neutral is currently held (§5.3)."""
    ent = state.entities
    creature = masks.ref_valid(
        ent,
        ent.creature.grip.held_by.kind,
        ent.creature.grip.held_by.slot,
        ent.creature.grip.held_by.uid,
    )
    carcass = masks.ref_valid(
        ent, ent.carcass.held_by.kind, ent.carcass.held_by.slot, ent.carcass.held_by.uid
    )
    neutral = masks.ref_valid(
        ent, ent.neutral.held_by.kind, ent.neutral.held_by.slot, ent.neutral.held_by.uid
    )
    return creature, carcass, neutral


def body_arrays(state: WorldState, cfg: Config):
    """Flat arrays of every collidable body: pos, radius, mass, active.

    Order is creature, carcass, neutral.  Bushes are excluded because
    creature-vs-bush pairs never collide (§5.3).  Held bodies are inactive.
    """
    ent = state.entities
    pos = jnp.concatenate(
        [ent.creature.transform.pos, ent.carcass.transform.pos, ent.neutral.transform.pos], axis=0
    )
    radius = jnp.concatenate(
        [ent.creature.shape.radius, ent.carcass.shape.radius, ent.neutral.shape.radius], axis=0
    )
    mass = jnp.concatenate(
        [ent.creature.shape.mass, ent.carcass.shape.mass, ent.neutral.shape.mass], axis=0
    )
    alive = jnp.concatenate([ent.creature.alive, ent.carcass.alive, ent.neutral.alive], axis=0)
    creature_held, carcass_held, neutral_held = held_masks(state)
    held = jnp.concatenate([creature_held, carcass_held, neutral_held], axis=0)
    active = alive & ~held
    return pos, radius, jnp.maximum(mass, _MASS_FLOOR), active


def _pair_geometry(pos, radius):
    """[T,T] distance, penetration depth and unit normal from i to j."""
    delta = pos[None, :, :] - pos[:, None, :]  # delta[i, j] = p_j - p_i
    distance = jnp.sqrt(jnp.sum(delta * delta, axis=-1))
    overlap = radius[:, None] + radius[None, :] - distance
    index = jnp.arange(pos.shape[0], dtype=jnp.int32)
    safe = jnp.where(distance > _EPS, distance, 1.0)
    normal = delta / safe[..., None]
    # Coincident centres need an antisymmetric fallback so the pair separates.
    sign = jnp.where(index[:, None] < index[None, :], 1.0, -1.0)
    fallback = jnp.stack([sign, jnp.zeros_like(sign)], axis=-1)
    normal = jnp.where((distance < _EPS)[..., None], fallback, normal)
    return distance, overlap, normal, index


def _max_overlap(pos, radius, active) -> jax.Array:
    """Largest penetration depth between two active bodies (§9 M0)."""
    _, overlap, _, index = _pair_geometry(pos, radius)
    pair = active[:, None] & active[None, :] & (index[:, None] != index[None, :])
    return jnp.max(jnp.where(pair, overlap, 0.0))


def residual_overlap(state: WorldState, cfg: Config) -> jax.Array:
    """Largest remaining penetration depth between two active bodies (§9 M0)."""
    pos, radius, _, active = body_arrays(state, cfg)
    return _max_overlap(pos, radius, active)


def _resolve(pos, radius, mass, active):
    """One mass-weighted separation pass over every active pair."""
    distance, overlap, normal, index = _pair_geometry(pos, radius)
    pair = active[:, None] & active[None, :] & (index[:, None] != index[None, :]) & (overlap > 0.0)
    share = mass[None, :] / (mass[:, None] + mass[None, :])  # mass_j / (m_i + m_j)
    push = -normal * (overlap * share)[..., None]
    displacement = jnp.sum(jnp.where(pair[..., None], push, 0.0), axis=1)
    return pos + displacement


def resolve_overlaps(pos, radius, mass, active):
    """Two spec'd passes, then bounded iteration until the residual is small.

    This is the implementation of the conflict between SANDBOX.md §5.3
    ("iterate twice") and §9/§13 ("residual overlap < 1e-3"): the two mandated
    passes always run, and a convergence loop continues up to
    MAX_RESOLVE_ITERATIONS while the largest pair overlap is still above
    RESIDUAL_TOLERANCE.  The loop is a traced scalar-condition while_loop, so it
    stays jit/scan compatible; with no contacts it exits immediately.
    """
    for _ in range(RESOLVE_ITERATIONS):
        pos = _resolve(pos, radius, mass, active)

    def condition(carry):
        current, step = carry
        return (step < MAX_RESOLVE_ITERATIONS) & (
            _max_overlap(current, radius, active) > RESIDUAL_TOLERANCE
        )

    def body(carry):
        current, step = carry
        return _resolve(current, radius, mass, active), step + jnp.int32(1)

    pos, _ = jax.lax.while_loop(
        condition, body, (pos, jnp.asarray(RESOLVE_ITERATIONS, jnp.int32))
    )
    return pos


def _quadrant(theta, direction):
    """Quadrant id of a direction relative to heading; right turn is positive."""
    heading_x = jnp.cos(theta)
    heading_y = jnp.sin(theta)
    cross = heading_x * direction[..., 1] - heading_y * direction[..., 0]
    dot = heading_x * direction[..., 0] + heading_y * direction[..., 1]
    angle = jnp.arctan2(cross, dot)
    quarter = jnp.pi / 4.0
    return jnp.where(
        jnp.abs(angle) <= quarter,
        _FRONT,
        jnp.where(
            (angle > quarter) & (angle <= 3.0 * quarter),
            _RIGHT,
            jnp.where(jnp.abs(angle) > 3.0 * quarter, _BACK, _LEFT),
        ),
    ).astype(jnp.int32)


def _body_contacts(state: WorldState, cfg: Config, pos, radius, active) -> jax.Array:
    """[n_creature, 4] pre-resolution penetration depth per quadrant (§5.3)."""
    n = cfg.capacities.creature
    total = pos.shape[0]
    theta = state.entities.creature.transform.theta
    delta = pos[None, :, :] - pos[:n, None, :]  # direction toward j, per creature i
    distance = jnp.sqrt(jnp.sum(delta * delta, axis=-1))
    overlap = radius[:n, None] + radius[None, :] - distance
    index = jnp.arange(total, dtype=jnp.int32)
    pair = (
        active[:n, None]
        & active[None, :]
        & (index[None, :] != jnp.arange(n, dtype=jnp.int32)[:, None])
        & (overlap > _EPS)
    )
    safe = jnp.where(distance > _EPS, distance, 1.0)
    direction = delta / safe[..., None]
    sign = jnp.where(jnp.arange(n, dtype=jnp.int32)[:, None] < index[None, :], 1.0, -1.0)
    fallback = jnp.stack([sign, jnp.zeros_like(sign)], axis=-1)
    direction = jnp.where((distance < _EPS)[..., None], fallback, direction)
    quadrant = _quadrant(theta[:, None], direction)
    depth = jnp.where(pair, overlap, 0.0)
    one_hot = quadrant[..., None] == jnp.arange(4, dtype=jnp.int32)
    return jnp.sum(depth[..., None] * one_hot.astype(jnp.float32), axis=1)


def _wall_contacts_and_clip(state: WorldState, cfg: Config, pos, radius, active):
    """Clip bodies inside the walls; return per-creature wall contact depths."""
    cfg_width = jnp.asarray(cfg.world.width, jnp.float32)
    cfg_height = jnp.asarray(cfg.world.height, jnp.float32)
    lower = radius[:, None]
    upper = jnp.stack([cfg_width, cfg_height], axis=0)[None, :] - radius[:, None]
    clipped_pos = jnp.clip(pos, lower, upper)
    was_clipped = jnp.any(pos != clipped_pos, axis=-1)
    unclipped = pos
    pos = jnp.where(active[:, None], clipped_pos, pos)

    n = cfg.capacities.creature
    theta = state.entities.creature.transform.theta
    # Penetration is measured before the clip, which is exactly the amount the
    # body stuck through the wall this tick.
    creature_pos = unclipped[:n]
    creature_radius = radius[:n]
    creature_active = active[:n]

    width = cfg_width
    height = cfg_height
    depths = jnp.stack(
        [
            jnp.maximum(0.0, creature_radius - creature_pos[:, 0]),  # left wall
            jnp.maximum(0.0, creature_pos[:, 0] - (width - creature_radius)),  # right wall
            jnp.maximum(0.0, creature_radius - creature_pos[:, 1]),  # bottom wall
            jnp.maximum(0.0, creature_pos[:, 1] - (height - creature_radius)),  # top wall
        ],
        axis=-1,
    )
    directions = jnp.asarray(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], jnp.float32
    )
    quadrant = _quadrant(theta[:, None], directions[None, :, :])  # [n, 4]
    one_hot = quadrant[..., None] == jnp.arange(4, dtype=jnp.int32)
    contacts = jnp.sum(depths[..., None] * one_hot.astype(jnp.float32), axis=1)
    contacts = jnp.where(creature_active[:, None], contacts, 0.0)
    return pos, was_clipped, contacts


def apply(
    state: WorldState, cfg: Config, params=None
) -> tuple[WorldState, events.EventBuffer]:
    """Step 7: resolve overlaps, clip against the walls, record contact depth."""
    ent = state.entities
    n_creature = cfg.capacities.creature
    pos, radius, mass, active = body_arrays(state, cfg)

    # Contact magnitudes are the pre-resolution penetration depths (§5.3); wall
    # contacts are measured on the resolved positions because walls are not part
    # of the pairwise solver.
    touch = _body_contacts(state, cfg, pos, radius, active)
    pos = resolve_overlaps(pos, radius, mass, active)

    pos, clipped, wall_touch = _wall_contacts_and_clip(state, cfg, pos, radius, active)
    touch = touch + wall_touch

    n_carcass = cfg.capacities.carcass
    creature_pos = pos[:n_creature]
    carcass_pos = pos[n_creature:n_creature + n_carcass]
    neutral_pos = pos[n_creature + n_carcass:]

    creature = ent.creature
    creature = update(
        creature,
        transform=update(
            creature.transform,
            pos=jnp.where(creature.alive[:, None], creature_pos, creature.transform.pos),
            v=jnp.where(
                creature.alive & clipped[:n_creature],
                0.5 * creature.transform.v,
                creature.transform.v,
            ),
        ),
        contact=update(creature.contact, touch=jnp.where(creature.alive[:, None], touch, 0.0)),
    )
    carcass = ent.carcass
    carcass_slice = slice(n_creature, n_creature + n_carcass)
    carcass = update(
        carcass,
        transform=update(
            carcass.transform,
            pos=jnp.where(carcass.alive[:, None], carcass_pos, carcass.transform.pos),
            v=jnp.where(
                carcass.alive & clipped[carcass_slice],
                0.5 * carcass.transform.v,
                carcass.transform.v,
            ),
        ),
    )
    neutral = ent.neutral
    neutral_slice = slice(n_creature + n_carcass, None)
    neutral = update(
        neutral,
        transform=update(
            neutral.transform,
            pos=jnp.where(neutral.alive[:, None], neutral_pos, neutral.transform.pos),
            v=jnp.where(
                neutral.alive & clipped[neutral_slice],
                0.5 * neutral.transform.v,
                neutral.transform.v,
            ),
        ),
    )

    entities = update(ent, creature=creature, carcass=carcass, neutral=neutral)
    return update(state, entities=entities), events.empty(cfg)
