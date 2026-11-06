"""reproduce system (§3.3 step 10, §5.5, §6.3).

Contract: `apply(state, cfg) -> (WorldState, EventBuffer)` — a pure function.
All randomness (child genomes, headings, uids) comes from the reproduce key.
The system is written for the jitted `lax.scan` path: every array has a
fixed runtime shape (arena capacity), traced values are never converted to
Python scalars, and the expensive branch is entered through `jax.lax.cond`.

A creature splits automatically when SANDBOX.md §5.5 holds:
sum(N) >= theta_repro (genome, clipped to [0.5, 1]) and mu = 1.

Parent payment, in the documented order:
1. B_child = 0.03*c_flesh paid from the parent's tissue; if any component is
   missing the birth fails with `no_tissue`;
2. 0.3*rho_child reserves, component-wise "as much as the parent has";
3. a birth loss of 0.02*rho_parent (whatever is left is taken).

Child: mass 0.3, radius 0.5*sqrt(0.3) (§4.2), empty stomach, parent's core
temperature, integrity 1, random heading, genome mutated by
g' = g + sigma_block*N(0, 1) (§6.3), zero hidden state, fast weights, pain,
damage and grip.  Position: behind the parent at r_parent + r_child, trying
+-45, +-90 and +-135 degrees; a candidate is occupied when it overlaps a living
creature, a carcass or a neutral object (creatures pass through bushes, §4.2),
or leaves the world rectangle.  Free slots are assigned in creature-slot order.

Documented decisions (SANDBOX.md is silent):
- Occupancy is checked against entities alive at the start of reproduction;
  siblings born in the same tick are not checked against each other, so two
  simultaneous births may overlap for one tick until the collision pass.
- A parent whose tissue is insufficient fails with `no_tissue`; a parent with
  no free candidate position, or with no free creature slot left in slot
  order, fails with `no_room`.  Failed births consume no slot, uid or payment.
- Mutation scales come from brain/genome.mutation_sigmas (§6.3), and genome /
  fast-weight widths come from brain/genome.layout(cfg): they are derived from
  H, D, M and N_OUT, never configured by hand.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from life_sandbox.brain import genome as brain_genome
from life_sandbox.config import Config
from life_sandbox.core import components as C
from life_sandbox.core import events, genome as core_genome, rng
from life_sandbox.core.ledger import add_outflow
from life_sandbox.core.state import WorldState
from life_sandbox.core.struct import update
from life_sandbox.core.systems.interaction import append_events_batch
from life_sandbox.world import init as world_init

# SANDBOX.md §5.5 / §6.3 initial values.
CHILD_MASS: float = 0.3
CHILD_TISSUE: float = 0.03
CHILD_RESERVE_FRACTION: float = 0.3
BIRTH_LOSS: float = 0.02
THETA_REPRO_MIN: float = 0.5
THETA_REPRO_MAX: float = 1.0
MATURE_EPS: float = 1e-6

_CHILD_ANGLES = jnp.asarray(
    [0.0, jnp.pi / 4, -jnp.pi / 4, jnp.pi / 2, -jnp.pi / 2, 3 * jnp.pi / 4, -3 * jnp.pi / 4],
    dtype=jnp.float32,
)


def theta_repro(creature: C.CreatureArena, cfg: Config) -> jax.Array:
    """Reproduction threshold from the genome, clipped to [0.5, 1] (§5.5)."""
    index = core_genome.theta_repro_index(cfg.sensors.odor_channels)
    raw = creature.genome.g[:, index]
    return jnp.clip(raw, THETA_REPRO_MIN, THETA_REPRO_MAX)


def eligible(creature: C.CreatureArena, cfg: Config) -> jax.Array:
    """Alive, mature and above the reproduction threshold (§5.5)."""
    return (
        creature.alive
        & (creature.body.mu >= 1.0 - MATURE_EPS)
        & (jnp.sum(creature.body.N, axis=1) >= theta_repro(creature, cfg))
    )


def _scatter_rows(array: jax.Array, slots: jax.Array, mask: jax.Array, values: jax.Array):
    """Write `values` at `slots` where `mask`; other rows are untouched.

    Rows with `mask=False` get an out-of-range index, which JAX drops for
    scatter updates, so no host-side compaction is needed.
    """
    capacity = array.shape[0]
    index = jnp.where(mask, slots, jnp.asarray(capacity, jnp.int32))
    return array.at[index].set(values)


def _candidate_positions(
    state: WorldState, cfg: Config, parent_slots: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Child position candidates (capacity, 7, 2) and their free/inside masks."""
    ents = state.entities
    creature = ents.creature
    parent_pos = creature.transform.pos[parent_slots]
    parent_theta = creature.transform.theta[parent_slots]
    parent_radius = creature.shape.radius[parent_slots]
    child_radius = 0.5 * jnp.sqrt(CHILD_MASS)
    distance = parent_radius + child_radius
    angles = parent_theta[:, None] + jnp.pi + _CHILD_ANGLES[None, :]
    direction = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    positions = parent_pos[:, None, :] + direction * distance[:, None, None]
    inside = (
        (positions[..., 0] >= 0.0)
        & (positions[..., 0] <= cfg.world.width)
        & (positions[..., 1] >= 0.0)
        & (positions[..., 1] <= cfg.world.height)
    )
    occupied = jnp.zeros(positions.shape[:2], dtype=jnp.bool_)
    for arena_name in ("creature", "carcass", "neutral"):
        arena = getattr(ents, arena_name)
        delta = positions[:, :, None, :] - arena.transform.pos[None, None, :, :]
        overlap = jnp.linalg.norm(delta, axis=-1) < (
            arena.shape.radius[None, None, :] + child_radius
        )
        overlap = overlap & arena.alive[None, None, :]
        if arena_name == "creature":
            # The parent sits exactly r_parent + r_child away; floating point
            # must not turn the intended contact into an overlap.
            parent_of = (
                jnp.arange(arena.alive.shape[0], dtype=jnp.int32)[None, None, :]
                == parent_slots[:, None, None]
            )
            overlap = overlap & ~parent_of
        occupied = occupied | jnp.any(overlap, axis=-1)
    return positions, inside & ~occupied


def apply(state: WorldState, cfg: Config) -> tuple[WorldState, events.EventBuffer]:
    """Split mature, well-fed creatures (§3.3 step 10, §5.5)."""
    creature = state.entities.creature
    return jax.lax.cond(
        jnp.any(eligible(creature, cfg)),
        lambda s: _split(s, cfg),
        lambda s: (s, events.empty(cfg)),
        state,
    )


def _split(state: WorldState, cfg: Config) -> tuple[WorldState, events.EventBuffer]:
    """Fixed-shape birth settlement (the `lax.cond` true branch)."""
    ents = state.entities
    creature = ents.creature
    capacity = creature.alive.shape[0]
    tick = state.tick
    buffer = events.empty(cfg)
    slots = jnp.arange(capacity, dtype=jnp.int32)
    eligible_mask = eligible(creature, cfg)

    key_noise, key_theta = jax.random.split(rng.system_key(state.key, tick, "reproduce"), 2)

    # Parent payment 1: tissue.
    flesh = jnp.asarray(C.FLESH_COMP, dtype=jnp.float32)
    tissue_cost = CHILD_TISSUE * flesh
    parent_body = creature.body
    tissue_ok = jnp.all(parent_body.B >= tissue_cost[None, :], axis=1)

    # Child position: first free candidate in the documented order.
    positions, free_position = _candidate_positions(state, cfg, slots)
    free_position_any = jnp.any(free_position, axis=1)
    first_free = jnp.argmax(free_position.astype(jnp.int32), axis=1)
    child_position = jnp.take_along_axis(positions, first_free[:, None, None], axis=1)[:, 0, :]

    # Free slots are consumed in slot order by the successful parents.
    free_slots = jnp.nonzero(
        ~creature.alive, size=capacity, fill_value=capacity
    )[0].astype(jnp.int32)
    free_count = jnp.sum(~creature.alive).astype(jnp.int32)
    tentative = eligible_mask & tissue_ok & free_position_any
    tentative_rank = jnp.cumsum(tentative.astype(jnp.int32)) - 1
    success = tentative & (tentative_rank < free_count)
    success_rank = jnp.cumsum(success.astype(jnp.int32)) - 1
    child_slots = free_slots[jnp.clip(success_rank, 0, capacity - 1)]

    # Child genome mutation (§6.3) and heading.  The genome width is the brain
    # layout's derived total, not a configured number.
    lay = brain_genome.layout(cfg)
    sigma = brain_genome.mutation_sigmas(cfg)
    noise = jax.random.normal(key_noise, (capacity, lay.total), dtype=jnp.float32)
    child_genome = creature.genome.g + noise * sigma[None, :]
    child_heading = jax.random.uniform(
        key_theta, (capacity,), dtype=jnp.float32, minval=-jnp.pi, maxval=jnp.pi
    )

    # Parent payment 2-3: reserves and birth loss.
    child_diet = world_init.diet_from_genome(child_genome, cfg)
    child_reserves = jnp.minimum(CHILD_RESERVE_FRACTION * child_diet, parent_body.N)
    parent_rho = world_init.diet_from_genome(creature.genome.g, cfg)
    remaining_n = jnp.maximum(parent_body.N - child_reserves, 0.0)
    birth_loss = jnp.minimum(BIRTH_LOSS * parent_rho, remaining_n)
    parent_n = parent_body.N - jnp.where(
        success[:, None], child_reserves + birth_loss, 0.0
    )
    parent_b = parent_body.B - jnp.where(success[:, None], tissue_cost[None, :], 0.0)

    # Child records.
    child_uid = state.uid_counter + success_rank
    zero_like_uids = jnp.zeros((capacity,), dtype=jnp.int32)
    identity = C.Identity(
        uid=_scatter_rows(creature.identity.uid, child_slots, success, child_uid),
        parent_uid=_scatter_rows(
            creature.identity.parent_uid, child_slots, success,
            jnp.where(success, creature.identity.uid, zero_like_uids),
        ),
        lineage_id=_scatter_rows(
            creature.identity.lineage_id, child_slots, success,
            jnp.where(success, creature.identity.lineage_id, zero_like_uids),
        ),
        generation=_scatter_rows(
            creature.identity.generation, child_slots, success,
            jnp.where(success, creature.identity.generation + 1, zero_like_uids),
        ),
        birth_tick=_scatter_rows(
            creature.identity.birth_tick, child_slots, success,
            jnp.where(success, jnp.full((capacity,), tick, jnp.int32), zero_like_uids),
        ),
    )
    child_radius = 0.5 * jnp.sqrt(CHILD_MASS)
    transform = C.Transform(
        pos=_scatter_rows(creature.transform.pos, child_slots, success, child_position),
        theta=_scatter_rows(creature.transform.theta, child_slots, success, child_heading),
        v=_scatter_rows(
            creature.transform.v, child_slots, success, jnp.zeros((capacity,), jnp.float32)
        ),
        omega=_scatter_rows(
            creature.transform.omega, child_slots, success, jnp.zeros((capacity,), jnp.float32)
        ),
    )
    shape = C.Shape(
        radius=_scatter_rows(
            creature.shape.radius, child_slots, success,
            jnp.full((capacity,), child_radius, jnp.float32),
        ),
        mass=_scatter_rows(
            creature.shape.mass, child_slots, success,
            jnp.full((capacity,), CHILD_MASS, jnp.float32),
        ),
    )
    child_look = world_init.look_from_genome(child_genome, cfg)
    look = C.Look(
        rgb=_scatter_rows(creature.look.rgb, child_slots, success, child_look.rgb),
        odor=_scatter_rows(creature.look.odor, child_slots, success, child_look.odor),
    )
    body = C.Body(
        G=_scatter_rows(
            creature.body.G, child_slots, success,
            jnp.zeros((capacity, C.N_NUTRIENTS), jnp.float32),
        ),
        N=_scatter_rows(parent_n, child_slots, success, child_reserves),
        B=_scatter_rows(
            parent_b, child_slots, success,
            jnp.broadcast_to(tissue_cost[None, :], (capacity, C.N_NUTRIENTS)),
        ),
        I=_scatter_rows(
            creature.body.I, child_slots, success, jnp.ones((capacity,), jnp.float32)
        ),
        X=_scatter_rows(
            creature.body.X, child_slots, success, jnp.zeros((capacity,), jnp.float32)
        ),
        b=_scatter_rows(creature.body.b, child_slots, success, creature.body.b),
        k=_scatter_rows(
            creature.body.k, child_slots, success, jnp.zeros((capacity,), jnp.float32)
        ),
        p=_scatter_rows(
            creature.body.p, child_slots, success, jnp.zeros((capacity,), jnp.float32)
        ),
        phi_hb=_scatter_rows(
            creature.body.phi_hb, child_slots, success, jnp.zeros((capacity,), jnp.float32)
        ),
        S=_scatter_rows(
            creature.body.S, child_slots, success, jnp.zeros((capacity,), jnp.float32)
        ),
        asleep=_scatter_rows(
            creature.body.asleep, child_slots, success, jnp.zeros((capacity,), jnp.bool_)
        ),
        age=_scatter_rows(
            creature.body.age, child_slots, success, jnp.zeros((capacity,), jnp.int32)
        ),
        mu=_scatter_rows(
            creature.body.mu, child_slots, success, jnp.zeros((capacity,), jnp.float32)
        ),
    )
    genome = C.Genome(g=_scatter_rows(creature.genome.g, child_slots, success, child_genome))
    zero_brain = C.BrainState.zeros(
        capacity, cfg.brain.h, state.slice.shape[1], lay.plastic_dim
    )
    brain = C.BrainState(
        h=_scatter_rows(creature.brain.h, child_slots, success, zero_brain.h),
        s_hat=_scatter_rows(creature.brain.s_hat, child_slots, success, zero_brain.s_hat),
        w_fast=_scatter_rows(creature.brain.w_fast, child_slots, success, zero_brain.w_fast),
        e=_scatter_rows(creature.brain.e, child_slots, success, zero_brain.e),
        h_prev=_scatter_rows(creature.brain.h_prev, child_slots, success, zero_brain.h_prev),
    )
    zero_grip = C.Grip.zeros(capacity)
    grip = C.Grip(
        holding=C.Holding(
            kind=_scatter_rows(
                creature.grip.holding.kind, child_slots, success, zero_grip.holding.kind
            ),
            slot=_scatter_rows(
                creature.grip.holding.slot, child_slots, success, zero_grip.holding.slot
            ),
            uid=_scatter_rows(
                creature.grip.holding.uid, child_slots, success, zero_grip.holding.uid
            ),
        ),
        held_by=C.HeldBy(
            kind=_scatter_rows(
                creature.grip.held_by.kind, child_slots, success, zero_grip.held_by.kind
            ),
            slot=_scatter_rows(
                creature.grip.held_by.slot, child_slots, success, zero_grip.held_by.slot
            ),
            uid=_scatter_rows(
                creature.grip.held_by.uid, child_slots, success, zero_grip.held_by.uid
            ),
        ),
    )
    zero_contact = C.Contact.zeros(capacity)
    contact = C.Contact(
        touch=_scatter_rows(creature.contact.touch, child_slots, success, zero_contact.touch)
    )
    zero_pain = C.Pain.zeros(capacity)
    pain = C.Pain(
        a_inj=_scatter_rows(creature.pain.a_inj, child_slots, success, zero_pain.a_inj),
        x=_scatter_rows(creature.pain.x, child_slots, success, zero_pain.x),
    )
    zero_damage = C.Damage.zeros(capacity)
    damage = C.Damage(
        cause=_scatter_rows(creature.damage.cause, child_slots, success, zero_damage.cause),
        source_uid=_scatter_rows(
            creature.damage.source_uid, child_slots, success, zero_damage.source_uid
        ),
        same_lineage=_scatter_rows(
            creature.damage.same_lineage, child_slots, success, zero_damage.same_lineage
        ),
        nutrient=_scatter_rows(
            creature.damage.nutrient, child_slots, success, zero_damage.nutrient
        ),
        injury=_scatter_rows(creature.damage.injury, child_slots, success, zero_damage.injury),
    )
    zero_motor = C.Motor.zeros(capacity)
    motor = C.Motor(
        last_out=_scatter_rows(
            creature.motor.last_out, child_slots, success, zero_motor.last_out
        ),
        u_eff=_scatter_rows(creature.motor.u_eff, child_slots, success, zero_motor.u_eff),
    )
    zero_focal = C.Focal.zeros(capacity)
    focal = C.Focal(
        trace_slot=_scatter_rows(
            creature.focal.trace_slot, child_slots, success, zero_focal.trace_slot
        )
    )
    alive = _scatter_rows(
        creature.alive, child_slots, success, jnp.ones((capacity,), jnp.bool_)
    )
    creature = update(
        creature,
        alive=alive,
        identity=identity,
        transform=transform,
        shape=shape,
        look=look,
        body=body,
        pain=pain,
        damage=damage,
        grip=grip,
        contact=contact,
        genome=genome,
        brain=brain,
        motor=motor,
        focal=focal,
    )
    # §3.5: the 0.02*rho_parent birth loss leaves the nutrient book; the tissue
    # and reserve payments to the child are internal transfers.
    birth_outflow = jnp.sum(jnp.where(success[:, None], birth_loss, 0.0), axis=0)
    state = update(
        state,
        entities=update(ents, creature=creature),
        uid_counter=state.uid_counter + jnp.sum(success).astype(jnp.int32),
        ledger=add_outflow(state.ledger, C.OUTFLOW_BIRTH_LOSS, birth_outflow),
    )

    # Every eligible parent reports either a birth or a failure (§5.5).
    reason = jnp.where(
        ~tissue_ok,
        jnp.asarray(events.reason_code("no_tissue"), jnp.int8),
        jnp.asarray(events.reason_code("no_room"), jnp.int8),
    )
    event_uid = jnp.where(success, child_uid, creature.identity.uid)
    event_kind = jnp.where(
        success,
        jnp.asarray(events.kind_code("birth"), jnp.int8),
        jnp.asarray(events.kind_code("birth_fail"), jnp.int8),
    )
    event_x = jnp.where(success, child_position[:, 0], creature.transform.pos[:, 0])
    event_y = jnp.where(success, child_position[:, 1], creature.transform.pos[:, 1])
    buffer = append_events_batch(
        buffer,
        eligible_mask,
        kind=event_kind,
        tick=tick,
        uid=event_uid,
        target_kind=jnp.full((capacity,), C.CREATURE, jnp.int32),
        target_uid=creature.identity.uid,
        x=event_x,
        y=event_y,
        success=success,
        reason=jnp.where(success, jnp.asarray(0, jnp.int8), reason),
    )
    return state, buffer
