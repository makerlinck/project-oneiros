"""environment system (§3.3 step 2, §4.2, §4.4, §4.5).

Owner: world teammate.  Contract: apply(state, cfg) -> (WorldState, EventBuffer).

What one tick does, in this fixed order:

1. Berry regeneration (§4.2): stock += (world.bush_regen_rate*stock*(1 - stock)
   + world.bush_regen_base) * ramp.berry_multiplier(tick, cfg), clipped to [0, 1]
   because stock is a fraction with that range.  The coefficients are M1
   calibration parameters read from the configuration, not module constants.
   The startup ramp (§9) starts at three times the regeneration and reaches the
   target at world.ramp_ticks.
2. Carcass rot (§4.2): 0.5% of plant and animal matter becomes toxin every tick.
   The conversion is an internal transfer in the nutrient book (§3.5), so it is
   recorded as ledger.transfer_from = [rotP, rotA, 0] and
   ledger.transfer_to = [0, 0, rotP + rotA].  A carcass is removed once it is
   older than 600 ticks or its nutrients are exhausted; its remaining n (this
   tick's post-rot value, kept unchanged in the dead slot) is booked as
   outflow[OUTFLOW_DECOMPOSITION], so the dead slot never silently drops mass.
3. Toxin schedule (§4.4): every T_switch = c_switch * L ticks (L = world.l_ticks,
   an M1 calibration output) the environment key picks one berry species; its
   composition becomes (1 - 0.4*alpha) * base + 0.4*alpha * (0, 0, 1) and every
   other species returns to the base composition, where
   alpha = ramp.toxin_strength(tick, cfg) (§9).  The schedule runs from tick 0
   and through the whole startup ramp; alpha only weakens the toxic share
   (0.1 at tick 0 -> 1 after ramp_ticks) instead of removing it, so the toxin
   nutrient is available from the first tick and no individual is chronically
   toxin-deficient.  The switch event carries the species index in the x field
   (the event buffer has no dedicated column).  Appearance and odour do not
   change: only the consequences differ (§4.3).
4. Neutral objects (§4.5): scripted wander.  The browser only specifies a random
   walk, so the parameters below are local choices: speed 0.08 units/tick and a
   new random heading every 30-90 ticks, reflecting off the walls.

Everything is a pure function of state and the environment system key; no Python
randomness and no observer writes (SANDBOX.md §2, §7.1).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from life_sandbox.config import Config
from life_sandbox.core import events, masks, ramp, rng
from life_sandbox.core.components import (
    OUTFLOW_DECOMPOSITION,
    BushArena,
    CarcassArena,
    NeutralArena,
)
from life_sandbox.core.events import kind_code
from life_sandbox.core.ledger import add_outflow, add_transfer
from life_sandbox.core.state import WorldState
from life_sandbox.core.struct import update
from life_sandbox.world.constants import BERRY_RGB, BUSH_GREEN, INITIAL_COMP

# Carcass decay (§4.2): 0.5% of plant and animal matter rots into toxin per tick.
CARCASS_ROT: float = 0.005
CARCASS_MAX_AGE: int = 600

# Toxic berry composition (§4.4): 0.6 * base + 0.4 * toxin.
TOXIN_SHARE: float = 0.4

# Neutral-object wander, local choices (§4.5 is silent on values).
NEUTRAL_SPEED: float = 0.08
NEUTRAL_TIMER_MIN: int = 30
NEUTRAL_TIMER_MAX: int = 90

_EMPTY_EPS = 1e-9


def berry_growth(arena: BushArena, tick, cfg: Config, params=None) -> BushArena:
    """One regeneration step for the bush stock field (§4.2, §9).

    The logistic increment (not the standing stock) uses the M1 calibration
    coefficients world.bush_regen_rate / world.bush_regen_base — taken from the
    runtime `params` when a batched scan supplies them — and is scaled by the
    startup ramp multiplier before clipping, so the ramp makes berries regrow
    faster without changing the [0, 1] range.
    """
    stock = arena.edible.stock
    if params is None:
        rate = jnp.asarray(cfg.world.bush_regen_rate, jnp.float32)
        base = jnp.asarray(cfg.world.bush_regen_base, jnp.float32)
    else:
        rate = params.bush_regen_rate
        base = params.bush_regen_base
    increment = (rate * stock * (1.0 - stock) + base) * ramp.berry_multiplier(
        tick, cfg, params
    )
    grown = jnp.clip(stock + increment, 0.0, 1.0)
    stock = jnp.where(arena.alive, grown, stock)
    return update(arena, edible=update(arena.edible, stock=stock))


def bush_appearance(arena: BushArena) -> BushArena:
    """Mix green with the species berry colour by stock (§4.2).

    rgb = BUSH_GREEN * (1 - stock) + BERRY_RGB[kind] * stock; dead slots keep
    their stored colour.  Odour is untouched here: the odour deposit already
    uses the species spectrum x stock in world/fields.py (§4.3).
    """
    kind = jnp.clip(arena.edible.kind, 0, len(BERRY_RGB) - 1)
    berry = jnp.asarray(BERRY_RGB, jnp.float32)[kind]
    green = jnp.asarray(BUSH_GREEN, jnp.float32)[None, :]
    stock = arena.edible.stock[:, None]
    rgb = green * (1.0 - stock) + berry * stock
    rgb = jnp.where(arena.alive[:, None], rgb, arena.look.rgb)
    return update(arena, look=update(arena.look, rgb=rgb))


def carcass_rot(arena: CarcassArena) -> tuple[CarcassArena, jax.Array, jax.Array]:
    """Age carcasses, move 0.5% of their plant/animal mass into toxin (§4.2).

    Returns (arena, rot_from, decomposition) where rot_from is the per-nutrient
    internal transfer [rotP, rotA, 0] of this tick and decomposition is the
    per-nutrient n of every carcass removed by this tick.  The removed slots
    keep their n (alive = False only): the ledger records the mass crossing the
    book boundary, not a second copy in the state (§3.5).
    """
    nun = arena.edible.n
    alive = arena.alive
    plant = nun[:, 0]
    animal = nun[:, 1]
    rot_plant = CARCASS_ROT * plant
    rot_animal = CARCASS_ROT * animal
    updated = jnp.stack(
        [
            plant - rot_plant,
            animal - rot_animal,
            nun[:, 2] + rot_plant + rot_animal,
        ],
        axis=-1,
    )
    updated = jnp.where(alive[:, None], updated, nun)
    age = jnp.where(alive, arena.edible.age_dead + 1, arena.edible.age_dead)
    empty = jnp.sum(updated, axis=-1) <= _EMPTY_EPS
    alive_after = alive & (age <= CARCASS_MAX_AGE) & ~empty
    removed = alive & ~alive_after

    rot_from = jnp.stack(
        [
            jnp.sum(jnp.where(alive, rot_plant, 0.0)),
            jnp.sum(jnp.where(alive, rot_animal, 0.0)),
            jnp.asarray(0.0, jnp.float32),
        ]
    ).astype(jnp.float32)
    decomposition = jnp.sum(jnp.where(removed[:, None], updated, 0.0), axis=0)
    arena = update(
        arena,
        alive=alive_after,
        edible=update(arena.edible, n=updated, age_dead=age),
    )
    return arena, rot_from, decomposition.astype(jnp.float32)


def toxin_composition(
    state: WorldState, cfg: Config, key: jax.Array, params=None
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reassigned berry composition, the chosen species and the switch flag.

    The switch schedule runs normally from tick 0 and throughout the startup
    ramp; only the toxic share is weakened by
    alpha = ramp.toxin_strength(tick, cfg) (§9, rule fix from the 2026-11-06
    review): the chosen species becomes
    (1 - 0.4*alpha) * base + 0.4*alpha * (0, 0, 1).  On non-switch ticks the
    previous composition is kept unchanged.  With ramp_ticks = 0 alpha is 1 and
    the composition is the full 0.6*base + 0.4*toxin of §4.4.
    """
    period = max(1, int(round(float(cfg.world.c_switch) * float(cfg.world.l_ticks))))
    tick = state.tick
    # Tick 0 is a period multiple: the first tick itself picks a toxic species.
    switch_now = jnp.mod(tick, jnp.asarray(period, jnp.int32)) == 0

    base = jnp.asarray(INITIAL_COMP, jnp.float32)
    n_species = base.shape[0]
    chosen = jax.random.randint(key, (), 0, n_species, dtype=jnp.int32)
    toxin = jnp.zeros((base.shape[1],), jnp.float32).at[2].set(1.0)
    toxic_share = TOXIN_SHARE * ramp.toxin_strength(tick, cfg, params)  # 0.4 * alpha
    toxic = (1.0 - toxic_share) * base + toxic_share * toxin[None, :]
    switched = jnp.where((jnp.arange(n_species, dtype=jnp.int32) == chosen)[:, None], toxic, base)
    return jnp.where(switch_now, switched, state.comp), chosen, switch_now


def neutral_wander(entities, cfg: Config, key: jax.Array) -> NeutralArena:
    """Scripted random walk for neutral objects (§4.5)."""
    arena = entities.neutral
    n = cfg.capacities.neutral
    pos = arena.transform.pos
    heading = arena.scripted.heading
    timer = arena.scripted.timer
    alive = arena.alive
    held = masks.ref_valid(entities, arena.held_by.kind, arena.held_by.slot, arena.held_by.uid)
    moving = alive & ~held

    heading_key, timer_key = rng.subsplit(key, 2)
    resample = alive & (timer <= 0)
    new_heading = jax.random.uniform(
        heading_key, (n,), minval=0.0, maxval=2.0 * jnp.pi, dtype=jnp.float32
    )
    new_timer = jax.random.randint(
        timer_key, (n,), NEUTRAL_TIMER_MIN, NEUTRAL_TIMER_MAX + 1, dtype=jnp.int32
    )
    heading = jnp.where(resample, new_heading, heading)
    timer = jnp.where(resample, new_timer, jnp.maximum(timer - 1, 0))

    step = jnp.where(moving, NEUTRAL_SPEED, 0.0)
    cos_h = jnp.cos(heading)
    sin_h = jnp.sin(heading)
    proposed = pos + step[:, None] * jnp.stack([cos_h, sin_h], axis=-1)

    radius = arena.shape.radius
    upper_x = cfg.world.width - radius
    upper_y = cfg.world.height - radius
    flip_x = (proposed[:, 0] < radius) | (proposed[:, 0] > upper_x)
    flip_y = (proposed[:, 1] < radius) | (proposed[:, 1] > upper_y)
    cos_h = jnp.where(flip_x, -cos_h, cos_h)
    sin_h = jnp.where(flip_y, -sin_h, sin_h)
    heading = jnp.where(moving, jnp.arctan2(sin_h, cos_h), heading)

    advanced = pos + step[:, None] * jnp.stack([jnp.cos(heading), jnp.sin(heading)], axis=-1)
    upper = jnp.stack([upper_x, upper_y], axis=-1)
    advanced = jnp.clip(advanced, radius[:, None], upper)
    pos = jnp.where(moving[:, None], advanced, pos)

    return update(
        arena,
        transform=update(
            arena.transform,
            pos=pos,
            theta=jnp.where(alive, heading, arena.transform.theta),
            v=jnp.where(moving, jnp.asarray(NEUTRAL_SPEED, jnp.float32), arena.transform.v),
        ),
        scripted=update(arena.scripted, heading=heading, timer=timer),
    )


def apply(
    state: WorldState, cfg: Config, params=None
) -> tuple[WorldState, events.EventBuffer]:
    """Step 2: berry growth, carcass rot, toxin schedule, neutral wander."""
    key = rng.system_key(state.key, state.tick, "environment")
    switch_key, wander_key = rng.subsplit(key, 2)

    bush = bush_appearance(berry_growth(state.entities.bush, state.tick, cfg, params))
    carcass, rot_from, decomposition = carcass_rot(state.entities.carcass)
    entities = update(state.entities, bush=bush, carcass=carcass)
    entities = update(entities, neutral=neutral_wander(entities, cfg, wander_key))
    comp, chosen, switch_now = toxin_composition(state, cfg, switch_key, params)

    # Internal transfer (plant/animal -> toxin) and the mass leaving the book
    # when a carcass is removed (§3.5).
    rot_to = jnp.asarray([0.0, 0.0, rot_from[0] + rot_from[1]], jnp.float32)
    ledger = add_transfer(state.ledger, from_amount=rot_from, to_amount=rot_to)
    ledger = add_outflow(ledger, OUTFLOW_DECOMPOSITION, decomposition)

    event = events.append(
        events.empty(cfg),
        kind=kind_code("toxin_switch"),
        tick=state.tick,
        uid=chosen,
        x=chosen.astype(jnp.float32),
    )
    # events.append always writes; zeroing the count is how a conditional event
    # is represented without a data-dependent branch.
    event = update(event, count=jnp.where(switch_now, jnp.int32(1), jnp.int32(0)))
    buffer = events.concat(events.empty(cfg), event)

    return update(state, entities=entities, comp=comp, ledger=ledger), buffer
