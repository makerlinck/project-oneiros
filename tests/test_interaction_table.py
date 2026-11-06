"""Judgement-table coverage (SANDBOX.md §5.4, §5.1, §13 test_interaction_table).

Every cell of the §5.4 table is exercised, plus the grab failure reasons
(too heavy, occupied, escaped), the priority settlement of grab conflicts, the
stomach-cap rule (q limited, bite damage not), the escape rule and the
drop-before-grab ordering.  All tests run with noise_scale = 0 so the values
are the closed-form table values.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from life_sandbox.config import Capacities, Config, MotionConfig
from life_sandbox.core import components as C, events
from life_sandbox.core.state import empty
from life_sandbox.core.struct import update
from life_sandbox.core.systems import death, interaction, metabolism

N = 4
NB = 4
NCA = 2
NN = 2

FLESH = jnp.asarray([0.15, 0.35, 0.0], jnp.float32)


def small_cfg() -> Config:
    return Config.default().replaced(
        capacities=Capacities(creature=N, bush=NB, carcass=NCA, neutral=NN),
        motion=MotionConfig(noise_scale=0.0),
    )


def output_row(*, acc=0.0, turn=0.0, k=0.0, bite=0.0, grab=0.0, drop=0.0) -> jnp.ndarray:
    row = jnp.zeros(C.N_OUTPUTS, jnp.float32)
    row = row.at[C.OUT_ACC].set(acc)
    row = row.at[C.OUT_TURN].set(turn)
    row = row.at[C.OUT_K].set(k)
    row = row.at[C.OUT_BITE].set(bite)
    row = row.at[C.OUT_GRAB].set(grab)
    row = row.at[C.OUT_DROP].set(drop)
    return row


def add_creature(
    state,
    slot: int,
    *,
    pos=(10.0, 10.0),
    theta=0.0,
    uid=None,
    lineage=None,
    mass=1.0,
    integrity=1.0,
    maturity=1.0,
    stomach=(0.0, 0.0, 0.0),
    reserves=(0.0, 0.0, 0.0),
    out=None,
    eff=None,
):
    cfg = small_cfg()
    cre = state.entities.creature
    uid = slot + 1 if uid is None else uid
    lineage = uid if lineage is None else lineage
    transform = update(
        cre.transform,
        pos=cre.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32)),
        theta=cre.transform.theta.at[slot].set(theta),
    )
    body = update(
        cre.body,
        G=cre.body.G.at[slot].set(jnp.asarray(stomach, jnp.float32)),
        N=cre.body.N.at[slot].set(jnp.asarray(reserves, jnp.float32)),
        B=cre.body.B.at[slot].set(FLESH),
        I=cre.body.I.at[slot].set(integrity),
        mu=cre.body.mu.at[slot].set(maturity),
    )
    identity = update(
        cre.identity,
        uid=cre.identity.uid.at[slot].set(uid),
        lineage_id=cre.identity.lineage_id.at[slot].set(lineage),
    )
    motor = cre.motor
    if out is not None:
        motor = update(motor, last_out=motor.last_out.at[slot].set(out))
    if eff is not None:
        motor = update(
            motor, u_eff=motor.u_eff.at[slot].set(jnp.asarray(eff, jnp.float32))
        )
    cre = update(
        cre,
        alive=cre.alive.at[slot].set(True),
        identity=identity,
        transform=transform,
        body=body,
        shape=update(
            cre.shape,
            radius=cre.shape.radius.at[slot].set(0.5),
            mass=cre.shape.mass.at[slot].set(mass),
        ),
        motor=motor,
    )
    return update(state, entities=update(state.entities, creature=cre))


def add_bush(state, slot, *, pos=(11.0, 10.0), stock=0.5, kind=0):
    bush = state.entities.bush
    bush = update(
        bush,
        alive=bush.alive.at[slot].set(True),
        uid=bush.uid.at[slot].set(100 + slot),
        transform=update(
            bush.transform, pos=bush.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32))
        ),
        shape=update(
            bush.shape,
            radius=bush.shape.radius.at[slot].set(1.0),
            mass=bush.shape.mass.at[slot].set(jnp.inf),
        ),
        edible=update(
            bush.edible,
            stock=bush.edible.stock.at[slot].set(stock),
            kind=bush.edible.kind.at[slot].set(kind),
        ),
    )
    return update(state, entities=update(state.entities, bush=bush))


def add_carcass(state, slot, *, pos=(11.0, 10.0), n=(0.2, 0.2, 0.2), mass=0.5, uid=None):
    carcass = state.entities.carcass
    uid = 200 + slot if uid is None else uid
    carcass = update(
        carcass,
        alive=carcass.alive.at[slot].set(True),
        uid=carcass.uid.at[slot].set(uid),
        transform=update(
            carcass.transform,
            pos=carcass.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32)),
        ),
        shape=update(
            carcass.shape,
            radius=carcass.shape.radius.at[slot].set(0.5),
            mass=carcass.shape.mass.at[slot].set(mass),
        ),
        edible=update(carcass.edible, n=carcass.edible.n.at[slot].set(jnp.asarray(n, jnp.float32))),
    )
    return update(state, entities=update(state.entities, carcass=carcass))


def add_neutral(state, slot, *, pos=(11.0, 10.0), mass=0.2, uid=None):
    neutral = state.entities.neutral
    uid = 300 + slot if uid is None else uid
    neutral = update(
        neutral,
        alive=neutral.alive.at[slot].set(True),
        uid=neutral.uid.at[slot].set(uid),
        transform=update(
            neutral.transform,
            pos=neutral.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32)),
        ),
        shape=update(
            neutral.shape,
            radius=neutral.shape.radius.at[slot].set(0.4),
            mass=neutral.shape.mass.at[slot].set(mass),
        ),
    )
    return update(state, entities=update(state.entities, neutral=neutral))


def set_holding(state, holder_slot, kind, slot, uid):
    cre = state.entities.creature
    holding = C.Holding(
        kind=cre.grip.holding.kind.at[holder_slot].set(kind),
        slot=cre.grip.holding.slot.at[holder_slot].set(slot),
        uid=cre.grip.holding.uid.at[holder_slot].set(uid),
    )
    cre = update(cre, grip=update(cre.grip, holding=holding))
    return update(state, entities=update(state.entities, creature=cre))


def set_held_by(state, arena_name, slot, holder_slot, holder_uid):
    arena = getattr(state.entities, arena_name)
    current = arena.grip.held_by if arena_name == "creature" else arena.held_by
    held_by = C.HeldBy(
        kind=current.kind.at[slot].set(C.CREATURE),
        slot=current.slot.at[slot].set(holder_slot),
        uid=current.uid.at[slot].set(holder_uid),
    )
    if arena_name == "creature":
        arena = update(arena, grip=update(arena.grip, held_by=held_by))
    else:
        arena = update(arena, held_by=held_by)
    return update(state, entities=update(state.entities, **{arena_name: arena}))


def records(buffer) -> list[dict]:
    out = []
    for i in range(int(buffer.count)):
        out.append(
            {
                "kind": int(buffer.kind[i]),
                "uid": int(buffer.uid[i]),
                "target_kind": int(buffer.target_kind[i]),
                "target_uid": int(buffer.target_uid[i]),
                "dn": [float(v) for v in buffer.dn[i]],
                "damage": float(buffer.damage[i]),
                "success": bool(buffer.success[i]),
                "reason": int(buffer.reason[i]),
                "same_lineage": bool(buffer.same_lineage[i]),
                "cause": int(buffer.cause[i]),
                "nutrient": int(buffer.nutrient[i]),
                "loss": [float(v) for v in buffer.loss[i]],
            }
        )
    return out


def of_kind(buffer, name):
    code = events.kind_code(name)
    return [r for r in records(buffer) if r["kind"] == code]


# ---------------------------------------------------------------- bite: bush
def test_bite_bush_takes_berry_mass():
    """Row bush, bite: q = min(stock, 0.05*s*(1+k)*g) (§5.4 table)."""
    cfg = small_cfg()
    state = add_bush(empty(cfg), 0, stock=0.5, kind=0)
    state = add_creature(state, 0, out=output_row(bite=1.0))
    out, buffer = interaction.apply(state, cfg)
    assert float(out.entities.bush.edible.stock[0]) == pytest.approx(0.45, abs=1e-6)
    expected = 0.05 * jnp.asarray([0.85, 0.15, 0.0])
    assert jnp.allclose(out.entities.creature.body.G[0], expected, atol=1e-6)
    bites = of_kind(buffer, "bite")
    assert len(bites) == 1 and bites[0]["target_kind"] == C.BUSH
    assert bites[0]["success"] and bites[0]["target_uid"] == 100
    assert jnp.allclose(jnp.asarray(bites[0]["dn"]), expected, atol=1e-6)
    # §3.5: the bush bite is the nutrient-book inflow.
    assert jnp.allclose(out.ledger.inflow[C.INFLOW_BITES], expected, atol=1e-6)


def test_bite_inside_bush_ignores_angle():
    """Inside a bush, that bush is a valid bite target regardless of heading."""
    cfg = small_cfg()
    state = add_bush(empty(cfg), 0, pos=(10.0, 10.0), stock=0.5)
    state = add_creature(state, 0, theta=jnp.pi, out=output_row(bite=1.0))
    out, buffer = interaction.apply(state, cfg)
    assert float(out.entities.bush.edible.stock[0]) == pytest.approx(0.45, abs=1e-6)
    assert len(of_kind(buffer, "bite")) == 1


def test_bite_without_target_emits_nothing():
    cfg = small_cfg()
    state = add_creature(empty(cfg), 0, out=output_row(bite=1.0))
    out, buffer = interaction.apply(state, cfg)
    assert int(buffer.count) == 0
    assert float(jnp.sum(out.entities.creature.body.G[0])) == 0.0


# ------------------------------------------------------------- bite: carcass
def test_bite_carcass_takes_proportional_nutrients():
    """Row carcass, bite: q = min(|n|, 0.05*s*(1+k)*g), dn = q*n/|n| (§5.1)."""
    cfg = small_cfg()
    state = add_carcass(empty(cfg), 0, n=(0.2, 0.2, 0.2))
    state = add_creature(state, 0, out=output_row(bite=1.0))
    out, buffer = interaction.apply(state, cfg)
    q = 0.05
    assert jnp.allclose(out.entities.carcass.edible.n[0], 0.2 - q / 3.0, atol=1e-6)
    assert jnp.allclose(out.entities.creature.body.G[0], q / 3.0, atol=1e-6)
    bites = of_kind(buffer, "bite")
    assert len(bites) == 1 and bites[0]["target_kind"] == C.CARCASS
    assert bites[0]["damage"] == 0.0
    # Bites on carcasses are internal transfers, not inflow (§3.5).
    assert float(jnp.sum(out.ledger.inflow[C.INFLOW_BITES])) == 0.0


# -------------------------------------------------------- bite: living target
def test_bite_living_target_damages_and_feeds():
    """Row living: I -= 0.1*s*(1+k)*g; q = min(0.5*|B+N+G|, 0.03*s*(1+k)*g) (§5.4)."""
    cfg = small_cfg()
    state = add_creature(empty(cfg), 1, pos=(11.0, 10.0), reserves=(0.3, 0.1, 0.0))
    state = add_creature(state, 0, out=output_row(bite=1.0))
    out, buffer = interaction.apply(state, cfg)
    target = out.entities.creature
    assert float(target.body.I[1]) == pytest.approx(0.9, abs=1e-6)
    q = 0.03
    factor = 1.0 - q / 0.9
    assert jnp.allclose(target.body.B[1], FLESH * factor, atol=1e-6)
    assert jnp.allclose(target.body.N[1], jnp.asarray([0.3, 0.1, 0.0]) * factor, atol=1e-6)
    assert jnp.allclose(out.entities.creature.body.G[0], jnp.asarray([0.015, 0.015, 0.0]), atol=1e-6)
    bites = of_kind(buffer, "bite")
    assert len(bites) == 1 and bites[0]["target_kind"] == C.CREATURE
    assert bites[0]["damage"] == pytest.approx(0.1, abs=1e-6)
    assert float(jnp.sum(out.ledger.inflow[C.INFLOW_BITES])) == 0.0
    # §5.5 attribution: the applied bite damage is booked to LOSS_BITE and the
    # acute-pain injury only keeps this tick's bite damage.
    assert float(target.damage.loss[1, C.LOSS_BITE]) == pytest.approx(0.1, abs=1e-6)
    assert float(target.damage.injury[1]) == pytest.approx(0.1, abs=1e-6)


def test_bite_neutral_has_no_effect():
    """Row neutral, bite: no content; event success=False reason not_edible."""
    cfg = small_cfg()
    state = add_neutral(empty(cfg), 0)
    state = add_creature(state, 0, out=output_row(bite=1.0))
    out, buffer = interaction.apply(state, cfg)
    bites = of_kind(buffer, "bite")
    assert len(bites) == 1
    assert not bites[0]["success"]
    assert bites[0]["reason"] == events.reason_code("not_edible")
    assert bites[0]["dn"] == pytest.approx([0.0, 0.0, 0.0])
    assert float(jnp.sum(out.entities.creature.body.G[0])) == 0.0


def test_stomach_capacity_limits_q():
    """Every q is limited by the biter's remaining stomach capacity (§5.4)."""
    cfg = small_cfg()
    state = add_bush(empty(cfg), 0, stock=0.5)
    state = add_creature(state, 0, stomach=(0.28, 0.0, 0.0), out=output_row(bite=1.0))
    out, buffer = interaction.apply(state, cfg)
    assert float(out.entities.bush.edible.stock[0]) == pytest.approx(0.48, abs=1e-6)
    assert float(jnp.sum(out.entities.creature.body.G[0])) == pytest.approx(0.3, abs=1e-6)


def test_stomach_capacity_does_not_limit_bite_damage():
    """Bite damage to a living target is not limited by stomach capacity (§5.4)."""
    cfg = small_cfg()
    state = add_creature(empty(cfg), 1, pos=(11.0, 10.0))
    state = add_creature(state, 0, stomach=(0.3, 0.0, 0.0), out=output_row(bite=1.0))
    out, buffer = interaction.apply(state, cfg)
    assert float(out.entities.creature.body.I[1]) == pytest.approx(0.9, abs=1e-6)
    bites = of_kind(buffer, "bite")
    assert len(bites) == 1
    assert bites[0]["dn"] == pytest.approx([0.0, 0.0, 0.0])
    assert bites[0]["damage"] == pytest.approx(0.1, abs=1e-6)


def test_bite_same_lineage_flag():
    """The bite event reports whether biter and target share a lineage (§7.2)."""
    cfg = small_cfg()
    state = add_creature(empty(cfg), 1, pos=(11.0, 10.0), uid=7, lineage=42)
    state = add_creature(state, 0, uid=8, lineage=42, out=output_row(bite=1.0))
    _, buffer = interaction.apply(state, cfg)
    bites = of_kind(buffer, "bite")
    assert len(bites) == 1 and bites[0]["same_lineage"]


def test_bite_conflict_settles_in_priority_order():
    """Several biters of one target take from what is left (§3.4)."""
    cfg = small_cfg()
    state = add_carcass(empty(cfg), 0, pos=(11.0, 10.0), n=(0.02, 0.02, 0.02))
    state = add_creature(state, 0, out=output_row(bite=1.0))
    state = add_creature(state, 1, pos=(10.5, 10.3), theta=-jnp.pi / 4, out=output_row(bite=1.0))
    out, buffer = interaction.apply(state, cfg)
    bites = of_kind(buffer, "bite")
    assert len(bites) == 2
    total = jnp.sum(jnp.asarray([b["dn"] for b in bites]), axis=0)
    assert jnp.allclose(total, jnp.asarray([0.02, 0.02, 0.02]), atol=1e-6)
    assert jnp.allclose(out.entities.carcass.edible.n[0], 0.0, atol=1e-6)


# ---------------------------------------------------------------- grab cells
def test_grab_neutral_succeeds():
    """Row neutral, grab: success; one item at a time (§5.4)."""
    cfg = small_cfg()
    state = add_neutral(empty(cfg), 0)
    state = add_creature(state, 0, out=output_row(grab=1.0))
    out, buffer = interaction.apply(state, cfg)
    grabs = of_kind(buffer, "grab")
    assert len(grabs) == 1 and grabs[0]["success"]
    assert int(out.entities.creature.grip.holding.kind[0]) == C.NEUTRAL
    assert int(out.entities.creature.grip.holding.slot[0]) == 0
    assert int(out.entities.neutral.held_by.kind[0]) == C.CREATURE
    assert int(out.entities.neutral.held_by.slot[0]) == 0
    assert int(out.entities.neutral.held_by.uid[0]) == 1


def test_grab_carcass_succeeds_when_light_enough():
    """Row carcass, grab: mass <= 0.5*m succeeds (§5.4)."""
    cfg = small_cfg()
    state = add_carcass(empty(cfg), 0, mass=0.4)
    state = add_creature(state, 0, out=output_row(grab=1.0))
    out, buffer = interaction.apply(state, cfg)
    grabs = of_kind(buffer, "grab")
    assert len(grabs) == 1 and grabs[0]["success"]
    assert int(out.entities.creature.grip.holding.kind[0]) == C.CARCASS


def test_grab_fails_when_too_heavy():
    cfg = small_cfg()
    state = add_carcass(empty(cfg), 0, mass=1.0)
    state = add_creature(state, 0, out=output_row(grab=1.0))
    out, buffer = interaction.apply(state, cfg)
    grabs = of_kind(buffer, "grab")
    assert len(grabs) == 1 and not grabs[0]["success"]
    assert grabs[0]["reason"] == events.reason_code("too_heavy")
    assert int(out.entities.creature.grip.holding.kind[0]) == C.NO_REF


def test_grab_living_succeeds_when_not_escaping():
    """Row living, grab: target mass <= 0.5*m and no escape this tick (§5.4)."""
    cfg = small_cfg()
    state = add_creature(empty(cfg), 1, pos=(11.0, 10.0), mass=0.4)
    state = add_creature(state, 0, out=output_row(grab=1.0))
    out, buffer = interaction.apply(state, cfg)
    grabs = of_kind(buffer, "grab")
    assert len(grabs) == 1 and grabs[0]["success"]
    assert int(out.entities.creature.grip.held_by.kind[1]) == C.CREATURE
    assert int(out.entities.creature.grip.held_by.uid[1]) == 1


def test_grab_living_fails_when_target_escapes():
    """The grab target resists with the escape rule (§5.4)."""
    cfg = small_cfg()
    state = add_creature(empty(cfg), 1, pos=(11.0, 10.0), mass=1.0, eff=(1.0, 0.0))
    state = add_creature(state, 0, out=output_row(grab=1.0))
    out, buffer = interaction.apply(state, cfg)
    grabs = of_kind(buffer, "grab")
    assert len(grabs) == 1 and not grabs[0]["success"]
    assert grabs[0]["reason"] == events.reason_code("escaped")
    assert int(out.entities.creature.grip.holding.kind[0]) == C.NO_REF


def test_grab_fails_when_target_already_held():
    cfg = small_cfg()
    state = add_neutral(empty(cfg), 0)
    state = add_creature(state, 2, pos=(9.0, 10.0))
    state = set_held_by(state, "neutral", 0, holder_slot=2, holder_uid=3)
    state = add_creature(state, 0, out=output_row(grab=1.0))
    out, buffer = interaction.apply(state, cfg)
    grabs = of_kind(buffer, "grab")
    assert len(grabs) == 1 and not grabs[0]["success"]
    assert grabs[0]["reason"] == events.reason_code("occupied")


def test_grab_conflict_one_winner():
    """Highest priority wins a grab conflict, the other reports occupied (§3.4)."""
    cfg = small_cfg()
    state = add_neutral(empty(cfg), 0, pos=(10.8, 10.0))
    state = add_creature(state, 0, out=output_row(grab=1.0))
    state = add_creature(state, 1, pos=(10.8, 10.6), theta=-jnp.pi / 2, out=output_row(grab=1.0))
    out, buffer = interaction.apply(state, cfg)
    grabs = of_kind(buffer, "grab")
    assert len(grabs) == 2
    wins = [g for g in grabs if g["success"]]
    losses = [g for g in grabs if not g["success"]]
    assert len(wins) == 1 and len(losses) == 1
    assert losses[0]["reason"] == events.reason_code("occupied")
    winner_slot = 0 if wins[0]["uid"] == 1 else 1
    assert int(out.entities.creature.grip.holding.slot[winner_slot]) == 0


def test_grab_does_not_target_bushes():
    cfg = small_cfg()
    state = add_bush(empty(cfg), 0)
    state = add_creature(state, 0, out=output_row(grab=1.0))
    out, buffer = interaction.apply(state, cfg)
    assert int(buffer.count) == 0
    assert int(out.entities.creature.grip.holding.kind[0]) == C.NO_REF


# ------------------------------------------------------------ escape and drop
def test_held_creature_escapes():
    """Escape rule: |u'_acc|*a_max > 0.03*(1+k_holder)*g_holder (§5.4)."""
    cfg = small_cfg()
    state = add_creature(empty(cfg), 0, eff=(1.0, 0.0))
    state = add_creature(state, 1, pos=(9.0, 10.0))
    state = set_holding(state, 1, C.CREATURE, 0, 1)
    state = set_held_by(state, "creature", 0, holder_slot=1, holder_uid=2)
    out, buffer = interaction.apply(state, cfg)
    escapes = of_kind(buffer, "escape")
    assert len(escapes) == 1
    assert int(out.entities.creature.grip.held_by.kind[0]) == C.NO_REF
    assert int(out.entities.creature.grip.holding.kind[1]) == C.NO_REF


def test_held_creature_cannot_escape_without_thrust():
    cfg = small_cfg()
    # Intended output is high but the effective control is zero: no escape.
    state = add_creature(empty(cfg), 0, out=output_row(acc=1.0), eff=(0.0, 0.0))
    state = add_creature(state, 1, pos=(9.0, 10.0))
    state = set_holding(state, 1, C.CREATURE, 0, 1)
    state = set_held_by(state, "creature", 0, holder_slot=1, holder_uid=2)
    out, buffer = interaction.apply(state, cfg)
    assert len(of_kind(buffer, "escape")) == 0
    assert int(out.entities.creature.grip.held_by.kind[0]) == C.CREATURE


def test_escape_reads_effective_control_not_intended_output():
    """The escape rule uses Motor.u_eff[0], never the raw intended u_acc (§5.4)."""
    cfg = small_cfg()
    # High intended output, zero effective control -> remains held.
    kept = add_creature(empty(cfg), 0, out=output_row(acc=1.0), eff=(0.0, 0.0))
    kept = add_creature(kept, 1, pos=(9.0, 10.0))
    kept = set_holding(kept, 1, C.CREATURE, 0, 1)
    kept = set_held_by(kept, "creature", 0, holder_slot=1, holder_uid=2)
    _, kept_buffer = interaction.apply(kept, cfg)
    assert len(of_kind(kept_buffer, "escape")) == 0

    # Zero intended output, high effective control -> escapes.
    freed = add_creature(empty(cfg), 0, out=output_row(acc=0.0), eff=(1.0, 0.0))
    freed = add_creature(freed, 1, pos=(9.0, 10.0))
    freed = set_holding(freed, 1, C.CREATURE, 0, 1)
    freed = set_held_by(freed, "creature", 0, holder_slot=1, holder_uid=2)
    _, freed_buffer = interaction.apply(freed, cfg)
    assert len(of_kind(freed_buffer, "escape")) == 1


def test_drop_releases_held_entity():
    """Row drop: place the entity where it is, clear both references (§5.4)."""
    cfg = small_cfg()
    state = add_neutral(empty(cfg), 0, pos=(10.9, 10.0))
    state = add_creature(state, 0, out=output_row(drop=1.0))
    state = set_holding(state, 0, C.NEUTRAL, 0, 300)
    state = set_held_by(state, "neutral", 0, holder_slot=0, holder_uid=1)
    out, buffer = interaction.apply(state, cfg)
    drops = of_kind(buffer, "drop")
    assert len(drops) == 1
    assert int(out.entities.creature.grip.holding.kind[0]) == C.NO_REF
    assert int(out.entities.neutral.held_by.kind[0]) == C.NO_REF
    pos = out.entities.neutral.transform.pos[0]
    assert float(pos[0]) == pytest.approx(10.9, abs=1e-6)


def test_drop_then_grab_same_tick():
    """Per creature the order is drop, then grab, then bite (§3.4)."""
    cfg = small_cfg()
    state = add_neutral(empty(cfg), 0, pos=(10.9, 10.0))
    state = add_neutral(state, 1, pos=(10.5, 10.0))
    state = add_creature(state, 0, out=output_row(drop=1.0, grab=1.0))
    state = set_holding(state, 0, C.NEUTRAL, 0, 300)
    state = set_held_by(state, "neutral", 0, holder_slot=0, holder_uid=1)
    out, buffer = interaction.apply(state, cfg)
    assert len(of_kind(buffer, "drop")) == 1
    grabs = of_kind(buffer, "grab")
    assert len(grabs) == 1 and grabs[0]["success"]
    assert int(out.entities.creature.grip.holding.slot[0]) == 1
    assert int(out.entities.neutral.held_by.slot[1]) == 0


def test_bite_kill_is_not_rescued_by_body():
    """A bite that takes I to 0 kills in the same tick; body may not repair it."""
    cfg = small_cfg()
    state = add_creature(
        empty(cfg),
        1,
        pos=(11.0, 10.0),
        reserves=(0.3, 0.3, 0.3),
        integrity=0.05,
    )
    state = add_creature(state, 0, out=output_row(bite=1.0))
    # Keep reseed uids disjoint from the hand-assigned victim/actor uids.
    state = update(state, uid_counter=jnp.asarray(100, jnp.int32))
    mid, _ = interaction.apply(state, cfg)
    victim = mid.entities.creature
    assert float(victim.body.I[1]) == 0.0
    # applied = min(total damage 0.1, I 0.05) = 0.05
    assert float(victim.damage.loss[1, C.LOSS_BITE]) == pytest.approx(0.05, abs=1e-6)

    after_body, _ = metabolism.apply(mid, cfg)
    victim = after_body.entities.creature
    assert float(victim.body.I[1]) == 0.0  # I <= 0 at entry -> no repair
    assert float(victim.damage.loss[1, C.LOSS_BITE]) > 0.0

    out, buffer = death.apply(after_body, cfg)
    # The slot may be reused by the reseed rescue; the victim's uid is gone.
    alive_uids = {
        int(uid)
        for uid in out.entities.creature.identity.uid[out.entities.creature.alive]
    }
    assert 2 not in alive_uids
    deaths = [row for row in of_kind(buffer, "death") if row["uid"] == 2]
    assert len(deaths) == 1
    death_event = deaths[0]
    assert death_event["cause"] == events.cause_code("bite")
    assert death_event["target_kind"] == C.CREATURE
    assert death_event["target_uid"] == 1
    assert not death_event["same_lineage"]
    assert death_event["loss"][C.LOSS_BITE] > 0.0
    assert death_event["nutrient"] == -1
