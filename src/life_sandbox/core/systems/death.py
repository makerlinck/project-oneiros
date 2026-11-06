"""die system (§3.3 step 11, §5.5).

Contract: `apply(state, cfg) -> (WorldState, EventBuffer)` — a pure function.
All randomness for a reseed comes from the die key.  The system is written for
the jitted `lax.scan` path: fixed array shapes, no traced-to-Python
conversions, and the reseed branch is entered through `jax.lax.cond`.

Implements SANDBOX.md §5.5:
- integrity zero (or below) means death;
- the recorded cause is the largest element of the decaying per-source damage
  ledger `Damage.loss` (nutrition / toxin / heat / cold / bite; all-zero ->
  other), with the fallback age cap 5*a_sen overriding it.  A nutrition cause
  also reports the missing nutrient index; a bite cause also reports the
  biter's uid and the same-lineage flag;
- a carcass is spawned with n = B + 0.5*N + G; the other half of N is a loss;
- whatever the dead creature held is put down in place, and a living holder of
  the dead creature releases it;
- when fewer than 8 creatures remain, up to 32 random genomes are reseeded;
  each seeded individual emits one `reseed` event and one `birth` event with
  reason `reseed` (so lifespan grouping sees the birth).

Documented decisions (SANDBOX.md is silent):
- Deaths are processed in creature-slot order; carcass slots are assigned in
  the same order.  If the carcass arena is full the remaining death's nutrients
  are lost as decomposition (recorded in accounting, not in a separate event).
- A reseed seeds min(32, free slots, capacity) individuals at the founder
  convention (world/init.py helpers), so capacity limits are never exceeded.
- Death events carry the killer's uid in target_uid and CREATURE as
  target_kind only for cause `bite`; other causes use NO_REF.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from life_sandbox.brain import genome as brain_genome
from life_sandbox.config import Config
from life_sandbox.core import components as C
from life_sandbox.core import events, rng
from life_sandbox.core.ledger import add_inflow, add_outflow
from life_sandbox.core.state import WorldState
from life_sandbox.core.struct import update
from life_sandbox.core.systems.interaction import append_events_batch
from life_sandbox.world import constants as K
from life_sandbox.world import init as world_init

# SANDBOX.md §5.2 / §5.5 initial values.
A_SEN: int = 400
AGE_CAP_FACTOR: int = 5
RESEED_THRESHOLD: int = 8
RESEED_COUNT: int = 32
RESEED_MARGIN: float = 2.0
CARCASS_RGB_SCALE: float = 0.4


def dead_mask(creature: C.CreatureArena) -> jax.Array:
    """Integrity zero/below, or the fallback age cap 5*a_sen (§5.5)."""
    integrity_zero = creature.body.I <= 0.0
    age_cap = creature.body.age >= AGE_CAP_FACTOR * A_SEN
    return creature.alive & (integrity_zero | age_cap)


def apply(state: WorldState, cfg: Config) -> tuple[WorldState, events.EventBuffer]:
    """Judge deaths, spawn carcasses and release references (§3.3 step 11)."""
    ents = state.entities
    creature = ents.creature
    tick = state.tick
    capacity = creature.alive.shape[0]
    dead = dead_mask(creature)
    buffer = events.empty(cfg)

    # ------------------------------------------------------ reference release
    # 1) Items held by a dead creature are put down in place (velocity 0).
    creature_arena = ents.creature
    carcass_arena = ents.carcass
    neutral_arena = ents.neutral
    for arena_name in ("creature", "carcass", "neutral"):
        if arena_name == "creature":
            held_by = creature_arena.grip.held_by
        else:
            held_by = getattr(ents, arena_name).held_by
        slot_ok = (held_by.slot >= 0) & (held_by.slot < capacity)
        owner_slot = jnp.clip(held_by.slot, 0, capacity - 1)
        released = (
            (held_by.kind == C.CREATURE)
            & slot_ok
            & dead[owner_slot]
            & (held_by.uid == creature.identity.uid[owner_slot])
        )
        if arena_name == "creature":
            creature_arena = update(
                creature_arena,
                grip=update(creature_arena.grip, held_by=_clear_refs(held_by, released)),
                transform=_zero_velocity(creature_arena.transform, released),
            )
        elif arena_name == "carcass":
            carcass_arena = update(
                carcass_arena,
                held_by=_clear_refs(held_by, released),
                transform=_zero_velocity(carcass_arena.transform, released),
            )
        else:
            neutral_arena = update(
                neutral_arena,
                held_by=_clear_refs(held_by, released),
                transform=_zero_velocity(neutral_arena.transform, released),
            )

    # 2) A living holder releases the dead creature.
    held_by_self = creature_arena.grip.held_by
    holder_slot = jnp.clip(held_by_self.slot, 0, capacity - 1)
    holder_alive = creature_arena.alive[holder_slot]
    held_valid = (
        dead
        & (held_by_self.kind == C.CREATURE)
        & holder_alive
        & (held_by_self.uid == creature.identity.uid[holder_slot])
    )
    holder_release = (
        jnp.zeros(capacity, dtype=jnp.int32)
        .at[holder_slot]
        .max(jnp.where(held_valid, 1, 0))
        > 0
    )
    holding = creature_arena.grip.holding
    holding = C.Holding(
        kind=jnp.where(holder_release, C.NO_REF, holding.kind).astype(jnp.int32),
        slot=jnp.where(holder_release, jnp.asarray(0, jnp.int32), holding.slot).astype(jnp.int32),
        uid=jnp.where(holder_release, jnp.asarray(0, jnp.int32), holding.uid).astype(jnp.int32),
    )
    creature_arena = update(creature_arena, grip=update(creature_arena.grip, holding=holding))
    buffer = append_events_batch(
        buffer,
        held_valid,
        kind=events.kind_code("drop"),
        tick=tick,
        uid=creature.identity.uid[holder_slot],
        target_kind=jnp.full((capacity,), C.CREATURE, jnp.int32),
        target_uid=creature.identity.uid,
        x=creature.transform.pos[holder_slot, 0],
        y=creature.transform.pos[holder_slot, 1],
    )

    # 3) Clear the dead creatures' own references (both directions).
    holding = creature_arena.grip.holding
    holding = C.Holding(
        kind=jnp.where(dead, C.NO_REF, holding.kind).astype(jnp.int32),
        slot=jnp.where(dead, jnp.asarray(0, jnp.int32), holding.slot).astype(jnp.int32),
        uid=jnp.where(dead, jnp.asarray(0, jnp.int32), holding.uid).astype(jnp.int32),
    )
    creature_arena = update(
        creature_arena,
        grip=C.Grip(holding=holding, held_by=_clear_refs(creature_arena.grip.held_by, dead)),
    )

    # ---------------------------------------------------------------- carcass
    counter = state.uid_counter
    nca = carcass_arena.alive.shape[0]
    nutrition = creature.body.B + 0.5 * creature.body.N + creature.body.G
    assigned = jnp.zeros(capacity, dtype=jnp.bool_)
    if nca > 0:
        free_order = jnp.nonzero(
            ~carcass_arena.alive, size=nca, fill_value=nca
        )[0].astype(jnp.int32)
        free_count = jnp.sum(~carcass_arena.alive).astype(jnp.int32)
        dead_rank = jnp.cumsum(dead.astype(jnp.int32)) - 1
        assigned = dead & (dead_rank < free_count)
        carcass_slot = free_order[jnp.clip(dead_rank, 0, nca - 1)]
        index = jnp.where(assigned, carcass_slot, jnp.asarray(nca, jnp.int32))
        carcass_uids = counter + jnp.clip(dead_rank, 0, capacity - 1)
        carcass_arena = update(
            carcass_arena,
            alive=carcass_arena.alive.at[index].set(True),
            uid=carcass_arena.uid.at[index].set(carcass_uids),
            transform=C.Transform(
                pos=carcass_arena.transform.pos.at[index].set(creature.transform.pos),
                theta=carcass_arena.transform.theta.at[index].set(creature.transform.theta),
                v=carcass_arena.transform.v.at[index].set(0.0),
                omega=carcass_arena.transform.omega.at[index].set(0.0),
            ),
            shape=C.Shape(
                radius=carcass_arena.shape.radius.at[index].set(creature.shape.radius),
                mass=carcass_arena.shape.mass.at[index].set(creature.shape.mass),
            ),
            look=C.Look(
                rgb=carcass_arena.look.rgb.at[index].set(
                    creature.look.rgb * CARCASS_RGB_SCALE
                ),
                odor=carcass_arena.look.odor.at[index].set(0.0),
            ),
            held_by=C.HeldBy(
                kind=carcass_arena.held_by.kind.at[index].set(C.NO_REF),
                slot=carcass_arena.held_by.slot.at[index].set(0),
                uid=carcass_arena.held_by.uid.at[index].set(0),
            ),
            edible=C.CarcassEdible(
                n=carcass_arena.edible.n.at[index].set(nutrition),
                age_dead=carcass_arena.edible.age_dead.at[index].set(0),
            ),
        )
        counter = counter + jnp.sum(assigned).astype(jnp.int32)

    # ------------------------------------------------------------ death event
    age_dead = creature.body.age >= AGE_CAP_FACTOR * A_SEN
    loss = creature.damage.loss
    source_index = jnp.argmax(loss, axis=1).astype(jnp.int32)
    # LOSS_* order is (nutrition, toxin, heat, cold, bite); map it to the
    # DEATH_CAUSES event codes (§5.5).
    source_cause = jnp.asarray(
        [
            events.cause_code("nutrition"),
            events.cause_code("toxin"),
            events.cause_code("heat"),
            events.cause_code("cold"),
            events.cause_code("bite"),
        ],
        dtype=jnp.int8,
    )[source_index]
    has_loss = jnp.max(loss, axis=1) > 0.0
    cause = jnp.where(
        has_loss, source_cause, jnp.asarray(events.cause_code("other"), jnp.int8)
    )
    cause = jnp.where(age_dead, jnp.asarray(events.cause_code("age"), jnp.int8), cause)
    bite = cause == events.cause_code("bite")
    nutrition_death = cause == events.cause_code("nutrition")
    nutrient = jnp.where(
        nutrition_death,
        creature.damage.nutrient,
        jnp.asarray(C.NO_REF, jnp.int8),
    ).astype(jnp.int8)
    buffer = append_events_batch(
        buffer,
        dead,
        kind=events.kind_code("death"),
        tick=tick,
        uid=creature.identity.uid,
        target_kind=jnp.where(bite, C.CREATURE, C.NO_REF).astype(jnp.int32),
        target_uid=jnp.where(bite, creature.damage.source_uid, C.NO_REF).astype(jnp.int32),
        same_lineage=jnp.where(bite, creature.damage.same_lineage, False),
        x=creature.transform.pos[:, 0],
        y=creature.transform.pos[:, 1],
        cause=cause,
        nutrient=nutrient,
        loss=loss,
        success=jnp.ones((capacity,), dtype=jnp.bool_),
    )

    creature_arena = update(creature_arena, alive=creature_arena.alive & ~dead)
    # §3.5: the other half of N never enters the carcass (death loss); when the
    # carcass arena is full the whole edible n decomposes instead.
    death_loss = jnp.sum(jnp.where(dead[:, None], 0.5 * creature.body.N, 0.0), axis=0)
    decomposition = jnp.sum(
        jnp.where((dead & ~assigned)[:, None], nutrition, 0.0), axis=0
    )
    ledger = state.ledger
    ledger = add_outflow(ledger, C.OUTFLOW_DEATH_LOSS, death_loss)
    ledger = add_outflow(ledger, C.OUTFLOW_DECOMPOSITION, decomposition)
    state = update(
        state,
        entities=update(
            state.entities,
            creature=creature_arena,
            carcass=carcass_arena,
            neutral=neutral_arena,
        ),
        uid_counter=counter,
        ledger=ledger,
    )

    # ---------------------------------------------------------------- reseed
    population = jnp.sum(creature_arena.alive)
    return jax.lax.cond(
        population < RESEED_THRESHOLD,
        lambda s, b: _reseed(s, cfg, b),
        lambda s, b: (s, b),
        state,
        buffer,
    )


def _clear_refs(held_by: C.HeldBy, mask: jax.Array) -> C.HeldBy:
    return C.HeldBy(
        kind=jnp.where(mask, C.NO_REF, held_by.kind).astype(jnp.int32),
        slot=jnp.where(mask, jnp.asarray(0, jnp.int32), held_by.slot).astype(jnp.int32),
        uid=jnp.where(mask, jnp.asarray(0, jnp.int32), held_by.uid).astype(jnp.int32),
    )


def _zero_velocity(transform: C.Transform, mask: jax.Array) -> C.Transform:
    return update(transform, v=jnp.where(mask, 0.0, transform.v))


def _reseed(
    state: WorldState, cfg: Config, buffer: events.EventBuffer
) -> tuple[WorldState, events.EventBuffer]:
    """Reseed random genomes when fewer than RESEED_THRESHOLD creatures remain.

    Fixed-shape branch: `n_seed = min(32, capacity)` candidates are always
    built; slots beyond the free count are dropped by the scatter mask.
    Diagnostics can disable the emergency rescue with world.reseed_enabled.
    """
    if not cfg.world.reseed_enabled:
        return state, buffer
    creature = state.entities.creature
    capacity = creature.alive.shape[0]
    n_seed = min(RESEED_COUNT, capacity)
    free_order = jnp.nonzero(
        ~creature.alive, size=capacity, fill_value=capacity
    )[0].astype(jnp.int32)
    free_count = jnp.sum(~creature.alive).astype(jnp.int32)
    seed_mask = jnp.arange(n_seed, dtype=jnp.int32) < free_count
    slots = free_order[:n_seed]

    key = rng.system_key(state.key, state.tick, "die")
    key_genome, key_pos, key_theta = jax.random.split(key, 3)
    genome = world_init.random_genome(key_genome, n_seed, cfg)
    look = world_init.look_from_genome(genome, cfg)
    body = world_init.body_from_genome(genome, cfg)
    position = jax.random.uniform(
        key_pos,
        (n_seed, 2),
        dtype=jnp.float32,
        minval=RESEED_MARGIN,
        maxval=jnp.asarray([cfg.world.width - RESEED_MARGIN, cfg.world.height - RESEED_MARGIN]),
    )
    heading = jax.random.uniform(
        key_theta, (n_seed,), dtype=jnp.float32, minval=-jnp.pi, maxval=jnp.pi
    )
    uids = state.uid_counter + jnp.arange(n_seed, dtype=jnp.int32)
    index = jnp.where(seed_mask, slots, jnp.asarray(capacity, jnp.int32))
    lay = brain_genome.layout(cfg)
    zero_brain = C.BrainState.zeros(
        n_seed, cfg.brain.h, state.slice.shape[1], lay.plastic_dim
    )
    zero_grip = C.Grip.zeros(n_seed)
    zero_pain = C.Pain.zeros(n_seed)
    zero_damage = C.Damage.zeros(n_seed)
    zero_contact = C.Contact.zeros(n_seed)
    zero_motor = C.Motor.zeros(n_seed)
    zero_focal = C.Focal.zeros(n_seed)
    creature = update(
        creature,
        alive=creature.alive.at[index].set(True),
        identity=C.Identity(
            uid=creature.identity.uid.at[index].set(uids),
            parent_uid=creature.identity.parent_uid.at[index].set(C.NO_REF),
            lineage_id=creature.identity.lineage_id.at[index].set(uids),
            generation=creature.identity.generation.at[index].set(0),
            birth_tick=creature.identity.birth_tick.at[index].set(state.tick),
        ),
        transform=C.Transform(
            pos=creature.transform.pos.at[index].set(position),
            theta=creature.transform.theta.at[index].set(heading),
            v=creature.transform.v.at[index].set(0.0),
            omega=creature.transform.omega.at[index].set(0.0),
        ),
        shape=C.Shape(
            radius=creature.shape.radius.at[index].set(K.FOUNDER_RADIUS),
            mass=creature.shape.mass.at[index].set(K.ADULT_MASS),
        ),
        look=C.Look(
            rgb=creature.look.rgb.at[index].set(look.rgb),
            odor=creature.look.odor.at[index].set(look.odor),
        ),
        body=C.Body(
            G=creature.body.G.at[index].set(body.G),
            N=creature.body.N.at[index].set(body.N),
            B=creature.body.B.at[index].set(body.B),
            I=creature.body.I.at[index].set(body.I),
            X=creature.body.X.at[index].set(body.X),
            b=creature.body.b.at[index].set(body.b),
            k=creature.body.k.at[index].set(body.k),
            p=creature.body.p.at[index].set(body.p),
            phi_hb=creature.body.phi_hb.at[index].set(body.phi_hb),
            S=creature.body.S.at[index].set(body.S),
            asleep=creature.body.asleep.at[index].set(body.asleep),
            age=creature.body.age.at[index].set(body.age),
            mu=creature.body.mu.at[index].set(body.mu),
        ),
        pain=C.Pain(
            a_inj=creature.pain.a_inj.at[index].set(zero_pain.a_inj),
            x=creature.pain.x.at[index].set(zero_pain.x),
        ),
        damage=C.Damage(
            source_uid=creature.damage.source_uid.at[index].set(zero_damage.source_uid),
            same_lineage=creature.damage.same_lineage.at[index].set(zero_damage.same_lineage),
            nutrient=creature.damage.nutrient.at[index].set(zero_damage.nutrient),
            injury=creature.damage.injury.at[index].set(zero_damage.injury),
            loss=creature.damage.loss.at[index].set(zero_damage.loss),
        ),
        grip=C.Grip(
            holding=C.Holding(
                kind=creature.grip.holding.kind.at[index].set(zero_grip.holding.kind),
                slot=creature.grip.holding.slot.at[index].set(zero_grip.holding.slot),
                uid=creature.grip.holding.uid.at[index].set(zero_grip.holding.uid),
            ),
            held_by=C.HeldBy(
                kind=creature.grip.held_by.kind.at[index].set(zero_grip.held_by.kind),
                slot=creature.grip.held_by.slot.at[index].set(zero_grip.held_by.slot),
                uid=creature.grip.held_by.uid.at[index].set(zero_grip.held_by.uid),
            ),
        ),
        contact=C.Contact(touch=creature.contact.touch.at[index].set(zero_contact.touch)),
        genome=C.Genome(g=creature.genome.g.at[index].set(genome)),
        brain=C.BrainState(
            h=creature.brain.h.at[index].set(zero_brain.h),
            s_hat=creature.brain.s_hat.at[index].set(zero_brain.s_hat),
            w_fast=creature.brain.w_fast.at[index].set(zero_brain.w_fast),
            e=creature.brain.e.at[index].set(zero_brain.e),
            h_prev=creature.brain.h_prev.at[index].set(zero_brain.h_prev),
        ),
        motor=C.Motor(
            last_out=creature.motor.last_out.at[index].set(zero_motor.last_out),
            u_eff=creature.motor.u_eff.at[index].set(zero_motor.u_eff),
        ),
        focal=C.Focal(trace_slot=creature.focal.trace_slot.at[index].set(zero_focal.trace_slot)),
    )
    # §3.5: a reseed injects the seeded bodies' B + N from outside the book.
    seeded_mass = jnp.sum(jnp.where(seed_mask[:, None], body.N + body.B, 0.0), axis=0)
    state = update(
        state,
        entities=update(state.entities, creature=creature),
        uid_counter=state.uid_counter + jnp.sum(seed_mask).astype(jnp.int32),
        ledger=add_inflow(state.ledger, C.INFLOW_RESEED, seeded_mass),
    )
    buffer = append_events_batch(
        buffer,
        seed_mask,
        kind=events.kind_code("reseed"),
        tick=state.tick,
        uid=uids,
        x=position[:, 0],
        y=position[:, 1],
    )
    # Reseeded individuals also get a birth event (reason = reseed) so the
    # lives/lifespan tables see them as births like any other individual.
    buffer = append_events_batch(
        buffer,
        seed_mask,
        kind=events.kind_code("birth"),
        tick=state.tick,
        uid=uids,
        x=position[:, 0],
        y=position[:, 1],
        reason=jnp.asarray(events.reason_code("reseed"), jnp.int8),
        success=jnp.ones((n_seed,), dtype=jnp.bool_),
    )
    return state, buffer
