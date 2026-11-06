"""body system (§3.3 step 9, §5.1, §5.2, §3.5).

Contract: `apply(state, cfg) -> (WorldState, EventBuffer)` — a pure function.

Implements, in this fixed order (SANDBOX.md §3.3 step 9):

1. metabolic knob: k = clip(u_k, 0, 1), forced to 0 while asleep (§5.2);
2. knob lag p and heartbeat phase phi_hb (§5.2);
3. digestion and absorption (§5.1): delta = r_dig*G, N += eps*delta with
   eps_j = 0.9*rho_j^gamma and N_cap,j = rho_j; overflow leaves the book;
   unabsorbed plant/animal matter is excreted, unabsorbed toxin becomes X with
   kappa_tox = 3;
4. metabolic demand (§5.3) split by rho, then deficiency damage I -= 5*sum(d_j);
5. toxin damage I -= 0.01*X, then X *= 0.95 (§5.1);
6. thermogenesis b += 0.05*(T - b) + 0.06*k^2; |b| > 1 damages I (§5.2);
7. growth toward mu = 1: at most 0.7/a_mature mass per tick, paying
   0.1*delta_m*c_flesh from N, only when the full amount is available (§5.2);
8. ageing: a += 1; the metabolic bonus starts after a_sen (§5.2);
9. acute pain: a_inj = 0.7*a_inj + 10*injury, then injury is cleared (§5.6);
   bite injury is booked by the interact system into Damage.

Sleep pressure S, asleep, consolidation and the memory/recall surcharge are
phase 2b (ARCHITECTURES.md §3): M0 keeps S = 0, asleep = False and no recall
cost.  Death causes are booked into the Damage component (§5.5); the die system
reads them.

Documented decisions (SANDBOX.md is silent):
- Toxin absorbed above N_cap is overflow (an accounting outflow), not a return
  to X; only the unabsorbed fraction (1 - eps_T)*delta_T reaches X (§5.1
  wording: "unabsorbed T becomes X", overflow is recorded separately).
- Integrity is repaired only while I < 1 and sum(N) > 0.5; the repaired amount
  is min(0.002, 1-I) and exactly that amount enters c_tot, split by rho
  (§5.2, §5.3 "cost of repair if there was repair this tick").
- The motion part of c_tot uses the effective controls Motor.u_eff = u' (the
  noise-added, clipped outputs the motion system applied, §5.3), not the raw
  intended outputs; the bite surcharge uses the intended u_bite.
- The startup ramp (§9) multiplies only the base upkeep term by
  ramp.metabolism_multiplier(tick, cfg); movement, bite and repair costs are
  physical work and are not ramped.
- The base-demand coefficient is cfg.body.base_metabolism (M1 scan); the same
  coefficient scales the |u_acc'| movement term (§5.3 uses one 0.002 there).
- Growth is all-or-nothing per tick: if any component of 0.1*delta_m*c_flesh
  is unavailable, the tick adds no mass.
- "inside a bush" for shading is the centre within the bush radius; temperature
  uses world.fields.temperature (§4.3).
- I is clamped to [0, 1]; causes are overwritten only by a source that actually
  deals damage this tick, so the recorded cause is the last damage source.
"""

from __future__ import annotations

import jax.numpy as jnp

from life_sandbox.config import Config
from life_sandbox.core import components as C
from life_sandbox.core import events, masks, ramp
from life_sandbox.core.ledger import add_digested, add_outflow
from life_sandbox.core.state import WorldState
from life_sandbox.core.struct import update
from life_sandbox.core.systems import motor
from life_sandbox.world import fields as world_fields
from life_sandbox.world import init as world_init

# SANDBOX.md §5.1 / §5.2 initial values.  Module constants until the M1
# calibration moves body values into the configuration record.
R_DIG: float = 0.05
GAMMA: float = 0.5
CONVERSION: float = 0.9
KAPPA_TOX: float = 3.0
A_SEN: int = 400
A_MATURE: int = 100
GROWTH_PER_TICK: float = 0.7 / A_MATURE
TISSUE_PER_MASS: float = 0.1
REPAIR_RATE: float = 0.002
REPAIR_MIN_RESERVES: float = 0.5
# Base-demand coefficient: cfg.body.base_metabolism (M1 calibration knob).
# Knob term (1 + 3*k^2) (§5.3); the 3 is the knob energy coefficient, not an
# ageing rate; the ageing bonus is derived from a_sen.
KNOB_ENERGY_COEFF: float = 3.0
INJURY_DECAY: float = 0.7
INJURY_GAIN: float = 10.0


def in_bush(entities: C.Entities) -> jax.Array:
    """Whether each creature centre lies inside a living bush (§4.2 shading)."""
    delta = (
        entities.bush.transform.pos[None, :, :]
        - entities.creature.transform.pos[:, None, :]
    )
    distance = jnp.linalg.norm(delta, axis=-1)
    inside = distance <= entities.bush.shape.radius[None, :]
    inside = inside & entities.bush.alive[None, :]
    return jnp.any(inside, axis=1)


def _book_damage(damage: C.Damage, amount: jax.Array, cause: int) -> C.Damage:
    """Record `cause` for rows where `amount` > 0 (last source wins)."""
    hit = amount > 0.0
    return C.Damage(
        cause=jnp.where(hit, jnp.asarray(cause, jnp.int8), damage.cause).astype(jnp.int8),
        source_uid=jnp.where(hit, jnp.asarray(C.NO_REF, jnp.int32), damage.source_uid).astype(
            jnp.int32
        ),
        same_lineage=jnp.where(hit, False, damage.same_lineage),
        nutrient=jnp.where(hit, jnp.asarray(C.NO_REF, jnp.int8), damage.nutrient).astype(jnp.int8),
        injury=damage.injury,
    )


def apply(state: WorldState, cfg: Config) -> tuple[WorldState, events.EventBuffer]:
    """Advance one tick of body physiology (SANDBOX.md §3.3 step 9)."""
    entities = state.entities
    creature = entities.creature
    alive = creature.alive
    body = creature.body
    last = creature.motor.last_out

    knob = motor.intended_knob(state)
    asleep = body.asleep
    awake = (~asleep).astype(jnp.float32)

    # 1-2. knob, lag and heartbeat phase (§5.2).
    p = 0.9 * body.p + 0.1 * knob
    phi_hb = body.phi_hb + 2.0 * jnp.pi * (1.0 + 2.0 * p) / 20.0
    age = body.age + alive.astype(jnp.int32)
    aging_bonus = jnp.maximum(
        jnp.asarray(0.0, jnp.float32),
        (age.astype(jnp.float32) - A_SEN) / jnp.asarray(A_SEN, jnp.float32),
    )

    # 3. digestion and absorption (§5.1).
    diet = world_init.diet_from_genome(creature.genome.g, cfg)
    conversion = CONVERSION * jnp.power(diet, GAMMA)
    delta = R_DIG * body.G
    stomach = body.G - delta
    absorbed = conversion * delta
    reserves = body.N + absorbed
    overflow = jnp.maximum(reserves - diet, 0.0)
    reserves = jnp.clip(reserves, 0.0, diet)
    toxin_intake = KAPPA_TOX * (1.0 - conversion[:, C.TOXIN]) * delta[:, C.TOXIN]
    toxin = body.X + toxin_intake

    # 4. metabolic demand (§5.3).
    m_held = masks.held_mass(entities)
    mass = creature.shape.mass
    # Motion cost uses the effective controls u' the motion system applied
    # (§5.3: "0.002*|u'_acc| ..."), not the raw intended outputs.
    u_eff = creature.motor.u_eff
    base_metabolism = cfg.body.base_metabolism
    movement = (
        base_metabolism
        * jnp.abs(u_eff[:, 0])
        * (1.0 + knob)
        * (mass + m_held)
        + 0.0005 * jnp.abs(u_eff[:, 1])
        + 0.002 * jnp.maximum(last[:, C.OUT_BITE], 0.0)
    ) * awake
    # Startup ramp (§9): only the base demand is scaled, never the movement,
    # bite or repair surcharges (those are physical work, not upkeep).
    base = (
        base_metabolism
        * (1.0 + KNOB_ENERGY_COEFF * knob * knob)
        * (1.0 + aging_bonus)
        * jnp.where(asleep, 0.5, 1.0)
        * ramp.metabolism_multiplier(state.tick, cfg)
    )
    # Repair only happens while integrity is below 1 and reserves can pay for
    # it; the demand charges exactly the amount repaired this tick (§5.3).
    repair = jnp.where(
        alive
        & (body.I < 1.0)
        & (jnp.sum(reserves, axis=1) > REPAIR_MIN_RESERVES),
        jnp.minimum(REPAIR_RATE, 1.0 - body.I),
        0.0,
    )
    demand = base + movement + repair
    required = diet * demand[:, None]
    deficit = jnp.maximum(required - reserves, 0.0)
    reserves = jnp.maximum(reserves - required, 0.0)
    deficiency = 5.0 * jnp.sum(deficit, axis=1)
    nutrient = jnp.argmax(deficit, axis=1).astype(jnp.int8)
    has_deficit = jnp.sum(deficit, axis=1) > 0.0

    # 5. toxin damage (§5.1).
    toxin_damage = 0.01 * toxin
    toxin = toxin * 0.95

    # 6. thermogenesis and temperature damage (§5.2, §4.3).
    bush = in_bush(entities)
    ambient = world_fields.temperature(cfg, creature.transform.pos[:, 1], state.tick, bush)
    temperature = body.b + 0.05 * (ambient - body.b) + 0.06 * knob * knob
    over = jnp.maximum(jnp.abs(temperature) - 1.0, 0.0) * 0.002
    heat = jnp.where(temperature > 1.0, over, 0.0)
    cold = jnp.where(temperature < -1.0, over, 0.0)

    # 7. growth (§5.2).
    delta_mass = jnp.minimum(GROWTH_PER_TICK, jnp.maximum(1.0 - mass, 0.0))
    tissue_cost = TISSUE_PER_MASS * delta_mass
    flesh = jnp.asarray(C.FLESH_COMP, dtype=jnp.float32)
    cost = tissue_cost[:, None] * flesh[None, :]
    can_grow = alive & jnp.all(reserves >= cost, axis=1)
    grow = jnp.where(can_grow, delta_mass, 0.0)
    reserves = jnp.where(can_grow[:, None], reserves - cost, reserves)
    tissue = body.B + jnp.where(can_grow[:, None], cost, 0.0)
    mass = mass + grow
    radius = 0.5 * jnp.sqrt(mass)
    maturity = jnp.clip((mass - 0.3) / 0.7, 0.0, 1.0)

    # Assemble integrity in the documented order: deficiency, toxin, thermal.
    damage = creature.damage
    damage = _book_damage(damage, deficiency, events.cause_code("nutrition"))
    damage = C.Damage(
        cause=jnp.where(has_deficit, jnp.asarray(events.cause_code("nutrition"), jnp.int8), damage.cause).astype(jnp.int8),
        source_uid=damage.source_uid,
        same_lineage=damage.same_lineage,
        nutrient=jnp.where(has_deficit, nutrient, damage.nutrient).astype(jnp.int8),
        injury=damage.injury,
    )
    damage = _book_damage(damage, toxin_damage, events.cause_code("toxin"))
    damage = _book_damage(damage, heat, events.cause_code("heat"))
    damage = _book_damage(damage, cold, events.cause_code("cold"))
    integrity = jnp.clip(
        body.I
        + repair
        - deficiency
        - toxin_damage
        - heat
        - cold,
        0.0,
        1.0,
    )

    # 9. acute pain from this tick's bite injury (§5.6); injury is per tick.
    a_inj = INJURY_DECAY * creature.pain.a_inj + INJURY_GAIN * damage.injury
    pain = update(creature.pain, a_inj=jnp.where(alive, a_inj, 0.0))
    damage = C.Damage(
        cause=damage.cause,
        source_uid=damage.source_uid,
        same_lineage=damage.same_lineage,
        nutrient=damage.nutrient,
        injury=jnp.zeros_like(damage.injury),
    )

    body = C.Body(
        G=stomach,
        N=reserves,
        B=tissue,
        I=integrity,
        X=toxin,
        b=temperature,
        k=knob,
        p=p,
        phi_hb=jnp.mod(phi_hb, 2.0 * jnp.pi),
        S=body.S,
        asleep=asleep,
        age=age,
        mu=maturity,
    )
    shape = update(creature.shape, mass=mass, radius=radius)
    creature = update(creature, body=body, shape=shape, damage=damage, pain=pain)

    # ---------------------------------------------------------- §3.5 ledger
    # Digestion G->N is an internal transfer (diagnostic only).  Only alive
    # slots contribute: dead slots keep stale components until reuse.
    alive_mask = alive[:, None]
    digested = jnp.sum(jnp.where(alive_mask, delta, 0.0), axis=0)
    consumed = jnp.sum(jnp.where(alive_mask, required - deficit, 0.0), axis=0)
    unabsorbed = (1.0 - conversion) * delta
    excretion = jnp.sum(jnp.where(alive_mask, unabsorbed, 0.0), axis=0) * jnp.asarray(
        [1.0, 1.0, 0.0], jnp.float32
    )
    toxin_load = jnp.zeros(C.N_NUTRIENTS, jnp.float32).at[C.TOXIN].set(
        jnp.sum(jnp.where(alive, unabsorbed[:, C.TOXIN], 0.0))
    )
    overflow_flow = jnp.sum(jnp.where(alive_mask, overflow, 0.0), axis=0)
    ledger = state.ledger
    ledger = add_digested(ledger, digested)
    ledger = add_outflow(ledger, C.OUTFLOW_METABOLISM, consumed)
    ledger = add_outflow(ledger, C.OUTFLOW_EXCRETION, excretion)
    ledger = add_outflow(ledger, C.OUTFLOW_OVERFLOW, overflow_flow)
    ledger = add_outflow(ledger, C.OUTFLOW_TOXIN_LOAD, toxin_load)
    return (
        update(state, entities=update(entities, creature=creature), ledger=ledger),
        events.empty(cfg),
    )
