"""interact system (§3.3 step 8, §5.4, §5.1).

Contract: `apply(state, cfg) -> (WorldState, EventBuffer)` — a pure function.
The per-tick priority permutation comes from the interact key, so no Python
randomness is involved and observers do not perturb the stream.

Implements the judgement table of SANDBOX.md §5.4 with the eating rules of
§5.1:

- Range: target centre within +-30 degrees of the heading and
  d <= r_self + r_target + 0.2.  While inside a bush, that bush is also a valid
  bite target; only its direction constraint is waived (the distance condition
  is already satisfied).  "Inside" means the centre lies within the bush
  radius (documented decision, the document does not define it).
- Per tick and creature the internal order is drop, then grab, then bite.
  Grabs and bites are settled in the random priority order of §3.4: only the
  highest-priority successful grab holds a target, and several biters of one
  target take from what is left in priority order.
- Every q is limited by the biter's remaining stomach capacity; bite damage to
  a living target is not.  The table fixes physical consequences only: this
  system never distinguishes eating from attacking, kin from non-kin, or diet
  class (SANDBOX.md §5.4 last bullet, §12.4).

Documented decisions where SANDBOX.md is silent:
- Grab takes the nearest target of a pickable prototype (bushes are excluded
  by prototype, not by their infinite mass); a too-heavy target fails with
  `too_heavy`, a target already held by another creature fails with
  `occupied`, and a living target may resist with the escape rule
  (`escaped`).
- Escape rule (§5.4): |u'_acc| * a_max of the creature > 0.03 *
  (1 + k_holder) * g_holder.  u'_acc is read from Motor.u_eff, the effective
  control the motion system applied this tick; the noise is never re-derived
  from the motion key here.
- Held living creatures try to escape once at the start of the tick; both
  references are cleared and an `escape` event is emitted.  The same condition
  is what makes a free living target resist a grab this tick.
- Dropping/escaping keeps the current position and zeroes the released
  entity's velocity; overlap is resolved by the next tick's collision pass.
- Only bites on bushes cross the nutrient book boundary (§3.5).  The event
  `absorbed` field stays zero because absorption is delayed by digestion
  (§5.1); per-bite attribution would need a stomach FIFO, out of M0 scope.
- No target in range => no grab/bite event.  A bite on a neutral object emits
  `bite` with success=False, reason `not_edible`.  For the other kinds
  success := (q > 0 or damage > 0).
- Living targets lose B, N and G proportionally (factor 1 - Q/|B+N+G| of the
  tick-start content); biters add their intake to the stomach afterwards.
  This fixed order keeps the update linear and deterministic.
- Bite damage is booked in the `Damage` component: the actually applied amount
  enters `loss[LOSS_BITE]`, `injury` keeps it for acute pain (§5.6), and the
  source uid / same-lineage flag are kept for the death event.  The body system
  decays and accumulates the loss vector, and the die system reports the largest
  decaying source (§5.5).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from life_sandbox.config import Config
from life_sandbox.core import components as C
from life_sandbox.core import events, masks, rng
from life_sandbox.core.ledger import add_inflow
from life_sandbox.core.state import WorldState
from life_sandbox.core.struct import update
from life_sandbox.core.systems import motor

# SANDBOX.md §5.4 / §5.1 initial values.  Module constants until M1
# calibration moves body values into the configuration record.
_ANGLE_LIMIT = jnp.pi / 6.0
_REACH = 0.2
_GRAB_MASS_FRACTION = 0.5
_ESCAPE_SCALE = 0.03
_STOMACH_CAP = 0.3
_BUSH_BITE_SCALE = 0.05
_CARCASS_BITE_SCALE = 0.05
_CREATURE_BITE_SCALE = 0.03
_CREATURE_DAMAGE_SCALE = 0.1

_EVENT_FIELDS = (
    "kind",
    "tick",
    "uid",
    "parent_uid",
    "target_kind",
    "target_uid",
    "same_lineage",
    "x",
    "y",
    "dn",
    "absorbed",
    "dX",
    "damage",
    "cause",
    "nutrient",
    "loss",
    "success",
    "reason",
)


def append_events_batch(buffer: events.EventBuffer, valid: jax.Array, **values) -> events.EventBuffer:
    """Vectorised `events.append`: write a batch of events after the valid prefix.

    Shared with the reproduce and die systems (same body contract).  Entries
    with `valid=False` are skipped; overflow is counted in `dropped` exactly
    like `events.append`.  Skipped entries use an out-of-range scatter index,
    which JAX drops, so they can never overwrite an accepted row.
    """
    capacity = buffer.kind.shape[0]
    batch = valid.shape[0]
    # Valid rows must be compacted into a contiguous suffix of the buffer
    # prefix: consumers read only the first `count` entries, so a sparse batch
    # (e.g. only creature slots 3, 6, 7 are eligible) must not leave holes.
    batch_rank = jnp.cumsum(valid.astype(jnp.int32)) - 1
    positions = buffer.count + batch_rank
    inside = valid & (positions < capacity)
    over = valid & (positions >= capacity)
    index = jnp.where(inside, positions, jnp.asarray(capacity, jnp.int32))

    merged = {}
    for name in _EVENT_FIELDS:
        array = getattr(buffer, name)
        value = values.get(name)
        if value is None:
            # Mirror events.append's defaults for the fields where zero is not
            # the neutral value (death-only nutrient, no target, success).
            if name == "nutrient":
                fill = -1
            elif name in ("target_kind", "parent_uid"):
                fill = C.NO_REF
            elif name == "success":
                fill = True
            else:
                fill = 0
            cast = jnp.full((batch,) + array.shape[1:], fill, dtype=array.dtype)
        else:
            cast = jnp.asarray(value, dtype=array.dtype)
            if cast.ndim < array.ndim:
                cast = jnp.broadcast_to(cast, (batch,) + array.shape[1:])
        merged[name] = array.at[index].set(cast)
    merged["count"] = buffer.count + jnp.sum(inside).astype(jnp.int32)
    merged["dropped"] = buffer.dropped + jnp.sum(over).astype(jnp.int32)
    return events.EventBuffer(**merged)


def _geometry(
    self_pos: jax.Array,
    self_theta: jax.Array,
    self_radius: jax.Array,
    target_pos: jax.Array,
    target_radius: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Distance matrix and the +-30 degree / reach mask of §5.4."""
    delta = target_pos[None, :, :] - self_pos[:, None, :]
    distance = jnp.linalg.norm(delta, axis=-1)
    bearing = jnp.arctan2(delta[..., 1], delta[..., 0])
    offset = bearing - self_theta[:, None]
    angle_ok = jnp.abs(jnp.arctan2(jnp.sin(offset), jnp.cos(offset))) <= _ANGLE_LIMIT
    reach_ok = distance <= self_radius[:, None] + target_radius[None, :] + _REACH
    return distance, angle_ok & reach_ok


def _nearest(distance: jax.Array, valid: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Nearest valid target per row: distance, slot, found."""
    masked = jnp.where(valid, distance, jnp.inf)
    best = jnp.min(masked, axis=1)
    slot = jnp.argmin(masked, axis=1).astype(jnp.int32)
    return best, slot, jnp.isfinite(best)


def _pick_best(
    best_distance: jax.Array,
    best_slot: jax.Array,
    best_found: jax.Array,
    best_kind: jax.Array,
    distance: jax.Array,
    slot: jax.Array,
    found: jax.Array,
    kind: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Replace the current best when the candidate is strictly nearer.

    Ties keep the prototype order passed by the caller; `argmin` resolves ties
    inside one prototype by the lowest slot.
    """
    take = found & (~best_found | (distance < best_distance))
    return (
        jnp.where(take, distance, best_distance),
        jnp.where(take, slot, best_slot),
        best_found | found,
        jnp.where(take, jnp.asarray(kind, jnp.int32), best_kind),
    )


def _target_mask(row_mask: jax.Array, row_slots: jax.Array, capacity: int) -> jax.Array:
    """Scatter an actor-row mask onto a target arena (out-of-range fills drop)."""
    index = jnp.where(row_mask, row_slots, jnp.asarray(capacity, jnp.int32))
    return jnp.zeros(capacity, dtype=jnp.bool_).at[index].set(True)


def _clear_held_by(held_by: C.HeldBy, mask: jax.Array) -> C.HeldBy:
    """Clear the held_by reference where `mask` (arena-shaped) is set."""
    return C.HeldBy(
        kind=jnp.where(mask, C.NO_REF, held_by.kind).astype(jnp.int32),
        slot=jnp.where(mask, jnp.asarray(0, jnp.int32), held_by.slot).astype(jnp.int32),
        uid=jnp.where(mask, jnp.asarray(0, jnp.int32), held_by.uid).astype(jnp.int32),
    )


def _assign_held_by(
    held_by: C.HeldBy,
    row_mask: jax.Array,
    row_slots: jax.Array,
    holder_slots: jax.Array,
    holder_uids: jax.Array,
    capacity: int,
) -> C.HeldBy:
    """Set held_by = (CREATURE, holder) for the winning actor of each target."""
    index = jnp.where(row_mask, row_slots, jnp.asarray(capacity, jnp.int32))
    assigned = jnp.zeros(capacity, dtype=jnp.bool_).at[index].set(True)
    target_holder = jnp.zeros(capacity, dtype=jnp.int32).at[index].set(holder_slots)
    target_uid = jnp.zeros(capacity, dtype=jnp.int32).at[index].set(holder_uids)
    return C.HeldBy(
        kind=jnp.where(assigned, C.CREATURE, held_by.kind).astype(jnp.int32),
        slot=jnp.where(assigned, target_holder, held_by.slot).astype(jnp.int32),
        uid=jnp.where(assigned, target_uid, held_by.uid).astype(jnp.int32),
    )


def _zero_velocity(transform: C.Transform, mask: jax.Array) -> C.Transform:
    return update(transform, v=jnp.where(mask, jnp.asarray(0.0, jnp.float32), transform.v))


def _take_1d(values: jax.Array, slot: jax.Array) -> jax.Array:
    return values[jnp.clip(slot, 0, values.shape[0] - 1)]


def _take_rows(values: jax.Array, slot: jax.Array) -> jax.Array:
    """Per-row pick from a (rows, columns) matrix."""
    idx = jnp.clip(slot, 0, values.shape[1] - 1)[:, None]
    return jnp.take_along_axis(values, idx, axis=1)[:, 0]


def _uid_of(entities: C.Entities, kind: jax.Array, slot: jax.Array) -> jax.Array:
    """uid of (kind, slot) per row, with a safe fallback for empty entries."""
    return jnp.where(
        kind == C.CREATURE,
        _take_1d(entities.creature.identity.uid, slot),
        jnp.where(
            kind == C.BUSH,
            _take_1d(entities.bush.uid, slot),
            jnp.where(
                kind == C.CARCASS,
                _take_1d(entities.carcass.uid, slot),
                _take_1d(entities.neutral.uid, slot),
            ),
        ),
    ).astype(jnp.int32)


def apply(state: WorldState, cfg: Config) -> tuple[WorldState, events.EventBuffer]:
    """Resolve one tick of drop / grab / bite interactions (§3.3 step 8, §5.4)."""
    entities = state.entities
    tick = state.tick
    n = entities.creature.alive.shape[0]
    nb = entities.bush.alive.shape[0]
    nca = entities.carcass.alive.shape[0]
    nn = entities.neutral.alive.shape[0]
    total = n + nb + nca + nn
    offset_creature = 0
    offset_bush = n
    offset_carcass = n + nb
    offset_neutral = n + nb + nca

    creature = entities.creature
    last = creature.motor.last_out
    # Effective controls u' written by the motion system (§5.3).  Escape uses
    # |u_acc'|; the noise is never re-derived from the motion key here.
    u_eff = creature.motor.u_eff
    a_max = motor.max_acceleration(state)
    knob = motor.intended_knob(state)
    efficiency = motor.body_efficiency(creature.body)
    grip_power = (1.0 + knob) * efficiency

    drop_flag = last[:, C.OUT_DROP] > 0.0
    grab_flag = last[:, C.OUT_GRAB] > 0.0
    bite_flag = last[:, C.OUT_BITE] > 0.0
    bite_strength = jnp.maximum(last[:, C.OUT_BITE], 0.0)

    key = rng.system_key(state.key, tick, "interact")
    perm = jax.random.permutation(key, n).astype(jnp.int32)
    rank = jnp.argsort(perm).astype(jnp.int32)

    buffer = events.empty(cfg)
    alive = creature.alive
    actor_uid = creature.identity.uid
    actor_pos = creature.transform.pos

    # ---------------------------------------------------------------- escape
    held_by = creature.grip.held_by
    holder_ok = (
        masks.ref_valid(entities, held_by.kind, held_by.slot, held_by.uid)
        & (held_by.kind == C.CREATURE)
        & alive
    )
    holder_slot = jnp.clip(held_by.slot, 0, n - 1)
    holder_ok = holder_ok & alive[holder_slot]
    escape = holder_ok & (
        jnp.abs(u_eff[:, 0]) * a_max
        > _ESCAPE_SCALE * (1.0 + knob[holder_slot]) * efficiency[holder_slot]
    )
    buffer = append_events_batch(
        buffer,
        escape,
        kind=events.kind_code("escape"),
        tick=tick,
        uid=actor_uid,
        target_kind=C.CREATURE,
        target_uid=actor_uid[holder_slot],
        x=actor_pos[:, 0],
        y=actor_pos[:, 1],
    )
    held_by = C.HeldBy(
        kind=jnp.where(escape, C.NO_REF, held_by.kind).astype(jnp.int32),
        slot=jnp.where(escape, jnp.asarray(0, jnp.int32), held_by.slot).astype(jnp.int32),
        uid=jnp.where(escape, jnp.asarray(0, jnp.int32), held_by.uid).astype(jnp.int32),
    )
    holding = creature.grip.holding
    holder_escape = (
        jnp.zeros(n, dtype=jnp.int32)
        .at[holder_slot]
        .max(jnp.where(holder_ok, escape.astype(jnp.int32), 0))
    ) > 0
    holding = C.Holding(
        kind=jnp.where(holder_escape, C.NO_REF, holding.kind).astype(jnp.int32),
        slot=jnp.where(holder_escape, jnp.asarray(0, jnp.int32), holding.slot).astype(jnp.int32),
        uid=jnp.where(holder_escape, jnp.asarray(0, jnp.int32), holding.uid).astype(jnp.int32),
    )
    creature = update(
        creature,
        grip=C.Grip(holding=holding, held_by=held_by),
        transform=_zero_velocity(creature.transform, escape),
    )
    entities = update(entities, creature=creature)
    creature = entities.creature

    # ------------------------------------------------------------------ drop
    holding = creature.grip.holding
    holding_valid = masks.ref_valid(
        entities, holding.kind, holding.slot, holding.uid
    ) & creature.alive
    drop = drop_flag & holding_valid
    dropped_uid = _uid_of(entities, holding.kind, holding.slot)
    buffer = append_events_batch(
        buffer,
        drop,
        kind=events.kind_code("drop"),
        tick=tick,
        uid=actor_uid,
        target_kind=holding.kind,
        target_uid=dropped_uid,
        x=actor_pos[:, 0],
        y=actor_pos[:, 1],
    )
    drop_slot = holding.slot
    creature_mask = _target_mask(drop & (holding.kind == C.CREATURE), drop_slot, n)
    carcass_mask = _target_mask(drop & (holding.kind == C.CARCASS), drop_slot, nca)
    neutral_mask = _target_mask(drop & (holding.kind == C.NEUTRAL), drop_slot, nn)
    creature_arena = entities.creature
    creature_arena = update(
        creature_arena,
        grip=update(
            creature_arena.grip,
            held_by=_clear_held_by(creature_arena.grip.held_by, creature_mask),
        ),
        transform=_zero_velocity(creature_arena.transform, creature_mask),
    )
    carcass_arena = entities.carcass
    carcass_arena = update(
        carcass_arena,
        held_by=_clear_held_by(carcass_arena.held_by, carcass_mask),
        transform=_zero_velocity(carcass_arena.transform, carcass_mask),
    )
    neutral_arena = entities.neutral
    neutral_arena = update(
        neutral_arena,
        held_by=_clear_held_by(neutral_arena.held_by, neutral_mask),
        transform=_zero_velocity(neutral_arena.transform, neutral_mask),
    )
    creature_arena = update(
        creature_arena,
        grip=update(
            creature_arena.grip,
            holding=C.Holding(
                kind=jnp.where(drop, C.NO_REF, holding.kind).astype(jnp.int32),
                slot=jnp.where(drop, jnp.asarray(0, jnp.int32), holding.slot).astype(jnp.int32),
                uid=jnp.where(drop, jnp.asarray(0, jnp.int32), holding.uid).astype(jnp.int32),
            ),
        ),
    )
    entities = update(
        entities,
        creature=creature_arena,
        carcass=carcass_arena,
        neutral=neutral_arena,
    )
    creature = entities.creature
    holding = creature.grip.holding
    holding_valid = masks.ref_valid(
        entities, holding.kind, holding.slot, holding.uid
    ) & creature.alive

    # ------------------------------------------------------------ candidates
    self_pos = creature.transform.pos
    self_theta = creature.transform.theta
    self_radius = creature.shape.radius
    self_mass = creature.shape.mass
    actor_alive = creature.alive[:, None]

    d_creature, reach_creature = _geometry(
        self_pos, self_theta, self_radius,
        entities.creature.transform.pos, entities.creature.shape.radius,
    )
    valid_creature = (
        reach_creature
        & entities.creature.alive[None, :]
        & ~jnp.eye(n, dtype=jnp.bool_)
        & actor_alive
    )
    d_bush, reach_bush = _geometry(
        self_pos, self_theta, self_radius,
        entities.bush.transform.pos, entities.bush.shape.radius,
    )
    inside_bush = d_bush <= entities.bush.shape.radius[None, :]
    valid_bush = (reach_bush | inside_bush) & entities.bush.alive[None, :] & actor_alive
    d_carcass, reach_carcass = _geometry(
        self_pos, self_theta, self_radius,
        entities.carcass.transform.pos, entities.carcass.shape.radius,
    )
    valid_carcass = reach_carcass & entities.carcass.alive[None, :] & actor_alive
    d_neutral, reach_neutral = _geometry(
        self_pos, self_theta, self_radius,
        entities.neutral.transform.pos, entities.neutral.shape.radius,
    )
    valid_neutral = reach_neutral & entities.neutral.alive[None, :] & actor_alive

    cd, cs, cf = _nearest(d_creature, valid_creature)
    bd, bs, bf = _nearest(d_bush, valid_bush)
    kad, kas, kaf = _nearest(d_carcass, valid_carcass)
    nd, nsl, nf = _nearest(d_neutral, valid_neutral)

    # Bite: nearest among all four prototypes (§5.4).
    bite_distance, bite_slot, bite_found, bite_kind = _pick_best(
        cd, cs, cf, jnp.full(n, C.CREATURE, jnp.int32), bd, bs, bf, C.BUSH
    )
    bite_distance, bite_slot, bite_found, bite_kind = _pick_best(
        bite_distance, bite_slot, bite_found, bite_kind, kad, kas, kaf, C.CARCASS
    )
    bite_distance, bite_slot, bite_found, bite_kind = _pick_best(
        bite_distance, bite_slot, bite_found, bite_kind, nd, nsl, nf, C.NEUTRAL
    )
    bite_emit = bite_found & bite_flag & creature.alive
    bite_kind = jnp.where(bite_emit, bite_kind, C.NO_REF)

    # Grab: nearest pickable target; bushes are excluded by prototype.
    grab_distance, grab_slot, grab_found, grab_kind = _pick_best(
        cd, cs, cf, jnp.full(n, C.CREATURE, jnp.int32), kad, kas, kaf, C.CARCASS
    )
    grab_distance, grab_slot, grab_found, grab_kind = _pick_best(
        grab_distance, grab_slot, grab_found, grab_kind, nd, nsl, nf, C.NEUTRAL
    )
    grab_emit = grab_found & grab_flag & creature.alive

    # ---------------------------------------------- grab conflict settlement
    creature_mass = entities.creature.shape.mass
    carcass_mass = entities.carcass.shape.mass
    neutral_mass = entities.neutral.shape.mass
    target_mass = jnp.where(
        grab_kind == C.CREATURE,
        _take_1d(creature_mass, grab_slot),
        jnp.where(
            grab_kind == C.CARCASS,
            _take_1d(carcass_mass, grab_slot),
            _take_1d(neutral_mass, grab_slot),
        ),
    )
    held_by_creature = masks.ref_valid(
        entities,
        entities.creature.grip.held_by.kind,
        entities.creature.grip.held_by.slot,
        entities.creature.grip.held_by.uid,
    )
    held_by_carcass = masks.ref_valid(
        entities, entities.carcass.held_by.kind, entities.carcass.held_by.slot,
        entities.carcass.held_by.uid,
    )
    held_by_neutral = masks.ref_valid(
        entities, entities.neutral.held_by.kind, entities.neutral.held_by.slot,
        entities.neutral.held_by.uid,
    )
    target_held = jnp.where(
        grab_kind == C.CREATURE,
        _take_1d(held_by_creature, grab_slot),
        jnp.where(
            grab_kind == C.CARCASS,
            _take_1d(held_by_carcass, grab_slot),
            _take_1d(held_by_neutral, grab_slot),
        ),
    )
    resistance = (jnp.abs(u_eff[:, 0])[None, :] * a_max[None, :]) > (
        _ESCAPE_SCALE * (1.0 + knob[:, None]) * efficiency[:, None]
    )
    resist = jnp.where(
        grab_kind == C.CREATURE, _take_rows(resistance, grab_slot), False
    )
    too_heavy = grab_found & (target_mass > _GRAB_MASS_FRACTION * self_mass)
    eligible = grab_emit & ~holding_valid & ~target_held & ~too_heavy & ~resist
    grab_id = jnp.where(
        grab_found,
        jnp.where(
            grab_kind == C.CREATURE,
            offset_creature + grab_slot,
            jnp.where(
                grab_kind == C.CARCASS,
                offset_carcass + grab_slot,
                offset_neutral + grab_slot,
            ),
        ),
        jnp.asarray(-1, jnp.int32),
    )
    same_target = (grab_id[:, None] == grab_id[None, :]) & eligible[:, None]
    no_rank = jnp.asarray(n + 1, jnp.int32)
    best_rank = jnp.min(jnp.where(same_target, rank[None, :], no_rank), axis=1)
    grab_win = eligible & (rank == best_rank)
    grab_reason = jnp.full(n, events.reason_code("none"), dtype=jnp.int8)
    grab_reason = jnp.where(
        holding_valid | target_held, events.reason_code("occupied"), grab_reason
    )
    grab_reason = jnp.where(too_heavy, events.reason_code("too_heavy"), grab_reason)
    grab_reason = jnp.where(resist, events.reason_code("escaped"), grab_reason)
    grab_reason = jnp.where(eligible & ~grab_win, events.reason_code("occupied"), grab_reason)
    grab_reason = jnp.where(grab_win, events.reason_code("none"), grab_reason)
    grab_target_uid = _uid_of(entities, grab_kind, grab_slot)
    buffer = append_events_batch(
        buffer,
        grab_emit,
        kind=events.kind_code("grab"),
        tick=tick,
        uid=actor_uid,
        target_kind=grab_kind,
        target_uid=grab_target_uid,
        x=actor_pos[:, 0],
        y=actor_pos[:, 1],
        success=grab_win,
        reason=grab_reason,
    )

    # Apply successful grabs (one winner per target, so no double holding).
    holding = C.Holding(
        kind=jnp.where(grab_win, grab_kind, holding.kind).astype(jnp.int32),
        slot=jnp.where(grab_win, grab_slot, holding.slot).astype(jnp.int32),
        uid=jnp.where(grab_win, grab_target_uid, holding.uid).astype(jnp.int32),
    )
    holder_slots = jnp.arange(n, dtype=jnp.int32)
    creature_arena = entities.creature
    creature_grab = grab_win & (grab_kind == C.CREATURE)
    creature_arena = update(
        creature_arena,
        grip=C.Grip(
            holding=holding,
            held_by=_assign_held_by(
                creature_arena.grip.held_by, creature_grab, grab_slot, holder_slots, actor_uid, n
            ),
        ),
    )
    carcass_arena = entities.carcass
    carcass_arena = update(
        carcass_arena,
        held_by=_assign_held_by(
            carcass_arena.held_by, grab_win & (grab_kind == C.CARCASS), grab_slot,
            holder_slots, actor_uid, nca,
        ),
    )
    neutral_arena = entities.neutral
    neutral_arena = update(
        neutral_arena,
        held_by=_assign_held_by(
            neutral_arena.held_by, grab_win & (grab_kind == C.NEUTRAL), grab_slot,
            holder_slots, actor_uid, nn,
        ),
    )
    entities = update(
        entities,
        creature=creature_arena,
        carcass=carcass_arena,
        neutral=neutral_arena,
    )
    creature = entities.creature

    # --------------------------------------------------------------- eating
    stomach_remaining = jnp.maximum(
        jnp.asarray(_STOMACH_CAP, jnp.float32) - jnp.sum(creature.body.G, axis=1),
        jnp.asarray(0.0, jnp.float32),
    )
    bite_scale = jnp.where(
        bite_kind == C.CREATURE,
        jnp.asarray(_CREATURE_BITE_SCALE, jnp.float32),
        jnp.where(
            bite_kind == C.BUSH,
            jnp.asarray(_BUSH_BITE_SCALE, jnp.float32),
            jnp.asarray(_CARCASS_BITE_SCALE, jnp.float32),
        ),
    )
    demand = jnp.minimum(bite_scale * bite_strength * grip_power, stomach_remaining)
    alpha = jnp.where(
        bite_kind == C.CREATURE, jnp.asarray(0.5, jnp.float32), jnp.asarray(1.0, jnp.float32)
    )
    content_active = bite_emit & (bite_kind != C.NEUTRAL)

    creature_content = creature.body.B + creature.body.N + creature.body.G
    creature_norm = jnp.sum(creature_content, axis=1)
    bush_kind = _take_1d(entities.bush.edible.kind, bite_slot)
    bush_content = state.comp[jnp.clip(bush_kind, 0, state.comp.shape[0] - 1)]
    carcass_content = _take_1d(entities.carcass.edible.n, bite_slot)
    target_content = jnp.where(
        (bite_kind == C.CREATURE)[:, None],
        _take_1d(creature_content, bite_slot),
        jnp.where(
            (bite_kind == C.BUSH)[:, None],
            bush_content,
            jnp.where((bite_kind == C.CARCASS)[:, None], carcass_content, 0.0),
        ),
    )
    target_norm = jnp.sum(target_content, axis=1)

    remaining = jnp.concatenate(
        [
            creature_norm,
            entities.bush.edible.stock,
            jnp.sum(entities.carcass.edible.n, axis=1),
            jnp.zeros(nn, dtype=jnp.float32),
        ]
    )
    bite_id = jnp.where(
        content_active,
        jnp.where(
            bite_kind == C.CREATURE,
            offset_creature + bite_slot,
            jnp.where(
                bite_kind == C.BUSH,
                offset_bush + bite_slot,
                offset_carcass + bite_slot,
            ),
        ),
        jnp.asarray(0, jnp.int32),
    )
    alpha_safe = jnp.where(content_active, alpha, jnp.asarray(0.0, jnp.float32))
    demand_safe = jnp.where(content_active, demand, jnp.asarray(0.0, jnp.float32))

    def _settle_step(carry, priority_slot):
        remaining_inner = carry
        target = bite_id[priority_slot]
        rem = remaining_inner[target]
        take_inner = jnp.minimum(demand_safe[priority_slot], alpha_safe[priority_slot] * rem)
        take_inner = jnp.maximum(take_inner, 0.0)
        remaining_inner = remaining_inner.at[target].set(rem - take_inner)
        return remaining_inner, take_inner

    _, take_by_priority = jax.lax.scan(_settle_step, remaining, perm)
    take = jnp.zeros(n, dtype=jnp.float32).at[perm].set(take_by_priority)

    taken_total = jnp.zeros(total, dtype=jnp.float32).at[bite_id].add(take)
    creature_take = taken_total[offset_creature : offset_creature + n]
    bush_take = taken_total[offset_bush : offset_bush + nb]
    carcass_take = taken_total[offset_carcass : offset_carcass + nca]

    dn = jnp.where(
        content_active[:, None],
        take[:, None] * target_content / jnp.maximum(target_norm[:, None], 1e-9),
        0.0,
    )

    # Reduce targets proportionally, then add the biter intake to the stomach.
    creature_factor = jnp.clip(
        1.0 - creature_take / jnp.maximum(creature_norm, 1e-9), 0.0, 1.0
    )
    body = creature.body
    body = update(
        body,
        B=body.B * creature_factor[:, None],
        N=body.N * creature_factor[:, None],
        G=body.G * creature_factor[:, None],
    )
    body = update(body, G=body.G + dn)
    creature = update(creature, body=body)

    carcass_factor = jnp.clip(
        1.0 - carcass_take / jnp.maximum(jnp.sum(entities.carcass.edible.n, axis=1), 1e-9),
        0.0,
        1.0,
    )
    carcass_arena = entities.carcass
    carcass_arena = update(
        carcass_arena,
        edible=update(carcass_arena.edible, n=carcass_arena.edible.n * carcass_factor[:, None]),
    )
    bush_arena = entities.bush
    bush_arena = update(
        bush_arena,
        edible=update(
            bush_arena.edible,
            stock=jnp.maximum(bush_arena.edible.stock - bush_take, 0.0),
        ),
    )
    entities = update(entities, creature=creature, carcass=carcass_arena, bush=bush_arena)

    # Bite damage: living targets only, never limited by the biter's stomach.
    damage = jnp.where(
        content_active & (bite_kind == C.CREATURE),
        _CREATURE_DAMAGE_SCALE * bite_strength * grip_power,
        jnp.asarray(0.0, jnp.float32),
    )
    damage_target = jnp.where(
        content_active & (bite_kind == C.CREATURE), bite_slot, jnp.asarray(0, jnp.int32)
    )
    actor_index = jnp.arange(n, dtype=jnp.int32)
    damage_by_target = (
        jnp.zeros((n, n), dtype=jnp.float32)
        .at[damage_target, actor_index]
        .set(jnp.where(damage > 0.0, damage, 0.0))
    )
    total_damage = jnp.sum(damage_by_target, axis=1)
    applied = jnp.minimum(total_damage, creature.body.I)
    body = update(creature.body, I=jnp.maximum(creature.body.I - total_damage, 0.0))
    creature = update(creature, body=body)

    rank_by_target = (
        jnp.full((n, n), -1, dtype=jnp.int32)
        .at[damage_target, actor_index]
        .set(jnp.where(damage > 0.0, rank, -1))
    )
    last_rank = jnp.max(rank_by_target, axis=1)
    has_damage = (last_rank >= 0) & (applied > 0.0) & creature.alive
    source_slot = perm[jnp.clip(last_rank, 0, n - 1)]
    damage_state = creature.damage
    # §5.5 attribution: only the damage actually applied (min(total, I)) is
    # booked to LOSS_BITE; injury keeps this tick's bite damage for acute pain.
    loss = damage_state.loss.at[:, C.LOSS_BITE].add(
        jnp.where(creature.alive, applied, 0.0)
    )
    damage_state = C.Damage(
        source_uid=jnp.where(
            has_damage, creature.identity.uid[source_slot], damage_state.source_uid
        ).astype(jnp.int32),
        same_lineage=jnp.where(
            has_damage,
            creature.identity.lineage_id[source_slot] == creature.identity.lineage_id,
            damage_state.same_lineage,
        ),
        nutrient=damage_state.nutrient,
        injury=jnp.where(
            creature.alive, damage_state.injury + applied, damage_state.injury
        ).astype(jnp.float32),
        loss=loss.astype(jnp.float32),
    )
    creature = update(creature, damage=damage_state)

    same_lineage = jnp.where(
        bite_kind == C.CREATURE,
        _take_rows(
            creature.identity.lineage_id[None, :] == creature.identity.lineage_id[:, None],
            bite_slot,
        ),
        False,
    )
    bite_reason = jnp.where(
        bite_kind == C.NEUTRAL,
        jnp.asarray(events.reason_code("not_edible"), jnp.int8),
        jnp.asarray(events.reason_code("none"), jnp.int8),
    )
    buffer = append_events_batch(
        buffer,
        bite_emit,
        kind=events.kind_code("bite"),
        tick=tick,
        uid=actor_uid,
        target_kind=bite_kind,
        target_uid=_uid_of(entities, bite_kind, bite_slot),
        same_lineage=same_lineage,
        x=actor_pos[:, 0],
        y=actor_pos[:, 1],
        dn=dn,
        damage=damage,
        success=(take > 0.0) | (damage > 0.0),
        reason=bite_reason,
    )

    # §3.5 nutrient book: only bites on bushes cross the boundary (the mass is
    # taken from the berry book outside it).  Bites on carcasses and living
    # creatures move nutrients inside the book and are not recorded.
    bush_dn = jnp.where((content_active & (bite_kind == C.BUSH))[:, None], dn, 0.0)
    ledger = add_inflow(state.ledger, C.INFLOW_BITES, jnp.sum(bush_dn, axis=0))
    entities = update(entities, creature=creature)
    return update(state, entities=entities, ledger=ledger), buffer
