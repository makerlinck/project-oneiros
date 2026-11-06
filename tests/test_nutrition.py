"""Nutrition and body physiology against the closed-form rules (SANDBOX.md
§5.1, §5.2, §5.3, §13 test_nutrition).

Covers: first-order digestion and absorption, conversion eps_j = 0.9*rho_j^0.5,
reserve overflow, unabsorbed plant/animal matter vs unabsorbed toxin -> X,
deficiency damage, toxin damage with lag, repair, growth payment, aging bonus,
thermogenesis and thermal damage, knob lag / heartbeat phase, acute pain from
bite injury, and the birth payment.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from life_sandbox.brain.genome import layout as brain_layout
from life_sandbox.config import Capacities, Config, MotionConfig, WorldConfig
from life_sandbox.core import components as C, events
from life_sandbox.core.state import empty
from life_sandbox.core.struct import update
from life_sandbox.core.systems import metabolism, reproduction

N = 4
FLESH = jnp.asarray([0.3, 0.7, 0.0], jnp.float32)


def small_cfg() -> Config:
    return Config.default().replaced(
        capacities=Capacities(creature=N, bush=6, carcass=2, neutral=1),
        motion=MotionConfig(noise_scale=0.0),
    )


def ramp_cfg() -> Config:
    """Tiny config with the startup ramp on (half upkeep at tick 0)."""
    return Config.default().replaced(
        capacities=Capacities(creature=N, bush=6, carcass=2, neutral=1),
        world=WorldConfig(founders=2, ramp_ticks=10, ramp_metabolism_multiplier=0.5),
        motion=MotionConfig(noise_scale=0.0),
    )


def diet_genome(cfg: Config, rho) -> jnp.ndarray:
    """Genome whose diet logits give softmax(logits) = rho."""
    genome = jnp.zeros((N, brain_layout(cfg).total), jnp.float32)
    logits = jnp.log(jnp.asarray(rho, jnp.float32))
    return genome.at[:, 0:3].set(jnp.broadcast_to(logits, (N, 3)))


def body_state(
    cfg: Config,
    *,
    rho=(0.2, 0.5, 0.3),
    pos=(10.0, 10.0),
    stomach=(0.0, 0.0, 0.0),
    reserves=(0.0, 0.0, 0.0),
    tissue=None,
    integrity=1.0,
    toxin=0.0,
    temperature=0.0,
    age=0,
    mass=1.0,
    maturity=1.0,
    last_out=None,
    u_eff=None,
    injury=0.0,
    a_inj=0.0,
):
    state = empty(cfg)
    cre = state.entities.creature
    last = cre.motor.last_out if last_out is None else last_out
    tissue = FLESH * 0.5 if tissue is None else jnp.asarray(tissue, jnp.float32)
    cre = update(
        cre,
        alive=cre.alive.at[0].set(True),
        identity=update(
            cre.identity,
            uid=cre.identity.uid.at[0].set(1),
            lineage_id=cre.identity.lineage_id.at[0].set(1),
        ),
        transform=update(
            cre.transform,
            pos=cre.transform.pos.at[0].set(jnp.asarray(pos, jnp.float32)),
        ),
        shape=update(
            cre.shape,
            radius=cre.shape.radius.at[0].set(0.5 * jnp.sqrt(mass)),
            mass=cre.shape.mass.at[0].set(mass),
        ),
        body=update(
            cre.body,
            G=cre.body.G.at[0].set(jnp.asarray(stomach, jnp.float32)),
            N=cre.body.N.at[0].set(jnp.asarray(reserves, jnp.float32)),
            B=cre.body.B.at[0].set(tissue),
            I=cre.body.I.at[0].set(integrity),
            X=cre.body.X.at[0].set(toxin),
            b=cre.body.b.at[0].set(temperature),
            age=cre.body.age.at[0].set(age),
            mu=cre.body.mu.at[0].set(maturity),
        ),
        pain=update(cre.pain, a_inj=cre.pain.a_inj.at[0].set(a_inj)),
        damage=update(cre.damage, injury=cre.damage.injury.at[0].set(injury)),
        genome=C.Genome(g=diet_genome(cfg, rho)),
        motor=update(
            cre.motor,
            last_out=last,
            u_eff=(
                jnp.broadcast_to(jnp.asarray([0.0, 0.0], jnp.float32), (N, 2))
                if u_eff is None
                else cre.motor.u_eff.at[0].set(jnp.asarray(u_eff, jnp.float32))
            ),
        ),
    )
    return update(state, entities=update(state.entities, creature=cre))


def conversions(rho):
    rho = jnp.asarray(rho, jnp.float32)
    return 0.9 * jnp.sqrt(rho), rho


def test_first_order_digestion_and_absorption():
    """delta = r_dig*G, N += eps*delta with eps_j = 0.9*rho_j^0.5 (§5.1)."""
    cfg = small_cfg()
    rho = (0.2, 0.5, 0.3)
    stomach = (0.1, 0.2, 0.05)
    reserves = (0.05, 0.05, 0.05)
    state = body_state(cfg, rho=rho, stomach=stomach, reserves=reserves)
    out, _ = metabolism.apply(state, cfg)
    eps, cap = conversions(rho)
    delta = 0.05 * jnp.asarray(stomach)
    absorbed = eps * delta
    required = cap * (0.002)  # awake, k=0, no repair (sum N < 0.5)
    expected_n = jnp.asarray(reserves) + absorbed - required
    body = out.entities.creature.body
    assert jnp.allclose(body.G[0], 0.95 * jnp.asarray(stomach), atol=1e-7)
    assert jnp.allclose(body.N[0], expected_n, atol=1e-6)
    expected_x = 3.0 * (1.0 - eps[C.TOXIN]) * delta[C.TOXIN]
    assert float(body.X[0]) == pytest.approx(float(expected_x * 0.95), abs=1e-7)

    # §3.5 ledger classification for this tick.
    ledger = out.ledger
    required = cap * 0.002  # awake, k = 0, I = 1 -> no repair
    unabsorbed = (1.0 - eps) * delta
    assert jnp.allclose(ledger.digested, delta, atol=1e-7)
    assert jnp.allclose(ledger.outflow[C.OUTFLOW_METABOLISM], required, atol=1e-7)
    assert jnp.allclose(
        ledger.outflow[C.OUTFLOW_EXCRETION],
        unabsorbed * jnp.asarray([1.0, 1.0, 0.0]),
        atol=1e-7,
    )
    expected_toxin_load = jnp.zeros(3).at[C.TOXIN].set(unabsorbed[C.TOXIN])
    assert jnp.allclose(ledger.outflow[C.OUTFLOW_TOXIN_LOAD], expected_toxin_load, atol=1e-7)
    assert jnp.allclose(ledger.outflow[C.OUTFLOW_OVERFLOW], 0.0, atol=1e-7)


def test_conversion_rates_follow_diet_genes():
    """The same stomach gives different absorption for different rho (§5.1)."""
    cfg = small_cfg()
    stomach = (0.1, 0.1, 0.0)
    first = body_state(cfg, rho=(0.8, 0.1, 0.1), stomach=stomach, reserves=(0.0, 0.0, 0.0))
    second = body_state(cfg, rho=(0.1, 0.8, 0.1), stomach=stomach, reserves=(0.0, 0.0, 0.0))
    out_a, _ = metabolism.apply(first, cfg)
    out_b, _ = metabolism.apply(second, cfg)
    absorbed_a = out_a.entities.creature.body.N[0] + 0.002 * jnp.asarray([0.8, 0.1, 0.1])
    absorbed_b = out_b.entities.creature.body.N[0] + 0.002 * jnp.asarray([0.1, 0.8, 0.1])
    assert absorbed_a[0] > absorbed_b[0]
    assert absorbed_a[1] < absorbed_b[1]


def test_absorption_overflow_clamps_at_capacity():
    """Overflow above N_cap,j = rho_j leaves the book (§5.1)."""
    cfg = small_cfg()
    rho = (0.2, 0.5, 0.3)
    reserves = (0.199, 0.499, 0.299)
    state = body_state(cfg, rho=rho, stomach=(0.3, 0.3, 0.3), reserves=reserves, maturity=0.0)
    out, _ = metabolism.apply(state, cfg)
    body = out.entities.creature.body
    rho_arr = jnp.asarray(rho)
    eps, _ = conversions(rho)
    delta = 0.05 * jnp.asarray([0.3, 0.3, 0.3])
    reserves_after = jnp.asarray(reserves) + eps * delta
    assert bool(jnp.all(reserves_after > rho_arr))  # overflow really happened
    demand = 0.002  # base only: I = 1 means no repair this tick
    expected = rho_arr - rho_arr * demand
    assert jnp.allclose(body.N[0], expected, atol=1e-6)
    # Overflow above N_cap,j is booked in its own outflow category (§3.5).
    assert jnp.allclose(
        out.ledger.outflow[C.OUTFLOW_OVERFLOW],
        reserves_after - rho_arr,
        atol=1e-6,
    )
    assert jnp.allclose(
        out.ledger.outflow[C.OUTFLOW_METABOLISM], rho_arr * demand, atol=1e-6
    )


def test_unabsorbed_toxin_becomes_toxin_load():
    """X += kappa_tox*(1-eps_T)*delta_T, kappa_tox = 3 (§5.1)."""
    cfg = small_cfg()
    rho = (0.2, 0.3, 0.5)
    eps, _ = conversions(rho)
    state = body_state(cfg, rho=rho, stomach=(0.0, 0.0, 0.2), toxin=0.1)
    out, _ = metabolism.apply(state, cfg)
    expected = (0.1 + 3.0 * (1.0 - eps[C.TOXIN]) * (0.05 * 0.2)) * 0.95
    assert float(out.entities.creature.body.X[0]) == pytest.approx(float(expected), abs=1e-7)


def test_unabsorbed_plant_and_animal_do_not_become_toxin():
    cfg = small_cfg()
    state = body_state(cfg, rho=(0.4, 0.4, 0.2), stomach=(0.2, 0.2, 0.0), toxin=0.0)
    out, _ = metabolism.apply(state, cfg)
    assert float(out.entities.creature.body.X[0]) == 0.0


def test_deficiency_damage_and_cause():
    """Missing reserves: I -= 5*sum(d_j), cause nutrition + nutrient (§5.1, §5.5)."""
    cfg = small_cfg()
    rho = (0.2, 0.5, 0.3)
    state = body_state(cfg, rho=rho, reserves=(0.0, 0.0, 0.0), integrity=1.0)
    out, _ = metabolism.apply(state, cfg)
    body = out.entities.creature.body
    required = jnp.asarray(rho) * 0.002  # no repair: sum N = 0
    assert float(body.I[0]) == pytest.approx(1.0 - 5.0 * 0.002, abs=1e-6)
    damage = out.entities.creature.damage
    assert int(damage.cause[0]) == events.cause_code("nutrition")
    assert int(damage.nutrient[0]) == int(jnp.argmax(required))
    # Actual consumption is required - deficit: nothing could be paid here.
    assert jnp.allclose(out.ledger.outflow[C.OUTFLOW_METABOLISM], 0.0, atol=1e-7)


def test_repair_restores_integrity_and_costs_reserves():
    """sum(N) > 0.5: I += 0.002 and the same amount enters c_tot (§5.2)."""
    cfg = small_cfg()
    rho = (0.2, 0.5, 0.3)
    reserves = (0.2, 0.2, 0.2)
    state = body_state(cfg, rho=rho, reserves=reserves, integrity=0.5)
    out, _ = metabolism.apply(state, cfg)
    body = out.entities.creature.body
    demand = 0.002 + 0.002  # base + repair
    assert float(body.I[0]) == pytest.approx(0.502, abs=1e-6)
    assert jnp.allclose(
        body.N[0], jnp.asarray(reserves) - jnp.asarray(rho) * demand, atol=1e-6
    )
    assert jnp.allclose(
        out.ledger.outflow[C.OUTFLOW_METABOLISM], jnp.asarray(rho) * demand, atol=1e-6
    )


def test_repair_only_when_integrity_below_one():
    """I = 1 -> no repair, so no repair cost in c_tot even if sum(N) > 0.5."""
    cfg = small_cfg()
    rho = (0.2, 0.5, 0.3)
    reserves = (0.2, 0.2, 0.2)  # sum > 0.5, so reserves would allow repair
    state = body_state(cfg, rho=rho, reserves=reserves, integrity=1.0)
    out, _ = metabolism.apply(state, cfg)
    body = out.entities.creature.body
    demand = metabolism.BASE_METABOLISM  # base only
    assert float(body.I[0]) == pytest.approx(1.0, abs=1e-7)
    assert jnp.allclose(body.N[0], jnp.asarray(reserves) - jnp.asarray(rho) * demand, atol=1e-6)
    assert jnp.allclose(
        out.ledger.outflow[C.OUTFLOW_METABOLISM], jnp.asarray(rho) * demand, atol=1e-6
    )


def test_repair_amount_capped_by_missing_integrity():
    """repair = min(0.002, 1-I); c_tot charges exactly that amount (§5.3)."""
    cfg = small_cfg()
    rho = (0.2, 0.5, 0.3)
    reserves = (0.2, 0.2, 0.2)
    state = body_state(cfg, rho=rho, reserves=reserves, integrity=0.999)
    out, _ = metabolism.apply(state, cfg)
    body = out.entities.creature.body
    repaired = 1.0 - 0.999
    demand = metabolism.BASE_METABOLISM + repaired
    assert float(body.I[0]) == pytest.approx(1.0, abs=1e-7)
    assert jnp.allclose(body.N[0], jnp.asarray(reserves) - jnp.asarray(rho) * demand, atol=1e-6)
    assert jnp.allclose(
        out.ledger.outflow[C.OUTFLOW_METABOLISM], jnp.asarray(rho) * demand, atol=1e-6
    )


def test_motion_cost_uses_effective_controls():
    """Motion demand uses u' = Motor.u_eff, not the intended last_out (§5.3)."""
    cfg = small_cfg()
    rho = (0.2, 0.5, 0.3)
    reserves = (0.1, 0.1, 0.1)  # below the repair threshold
    intended = jnp.zeros((N, C.N_OUTPUTS), jnp.float32).at[0, C.OUT_ACC].set(1.0)
    intended = intended.at[0, C.OUT_TURN].set(1.0)
    high_intent = body_state(cfg, rho=rho, reserves=reserves, last_out=intended)
    high_intent = update(
        high_intent,
        entities=update(
            high_intent.entities,
            creature=update(
                high_intent.entities.creature,
                motor=update(
                    high_intent.entities.creature.motor,
                    u_eff=jnp.zeros((N, 2), jnp.float32),
                ),
            ),
        ),
    )
    out_high_intent, _ = metabolism.apply(high_intent, cfg)
    drop_high_intent = jnp.asarray(reserves) - out_high_intent.entities.creature.body.N[0]

    high_eff = body_state(cfg, rho=rho, reserves=reserves, u_eff=(1.0, 1.0))
    out_high_eff, _ = metabolism.apply(high_eff, cfg)
    drop_high_eff = jnp.asarray(reserves) - out_high_eff.entities.creature.body.N[0]

    # Intended outputs alone cost nothing; the effective controls do.
    assert jnp.allclose(
        drop_high_intent,
        jnp.asarray(rho) * metabolism.BASE_METABOLISM,
        atol=1e-7,
    )
    movement = 0.002 * 1.0 * 1.0 + 0.0005 * 1.0  # |u_acc'|*(1+k)*m, |u_turn'|
    assert jnp.allclose(
        drop_high_eff,
        jnp.asarray(rho) * (metabolism.BASE_METABOLISM + movement),
        atol=1e-6,
    )


def test_base_demand_knob_and_aging_terms():
    """Base demand = 0.002*(1+3k^2)*(1+ageing bonus), ageing after a_sen (§5.3)."""
    cfg = small_cfg()
    rho = (0.2, 0.5, 0.3)
    reserves = (0.1, 0.1, 0.1)
    knob_on = jnp.zeros((N, C.N_OUTPUTS), jnp.float32).at[0, C.OUT_K].set(1.0)
    young = body_state(cfg, rho=rho, reserves=reserves, last_out=knob_on, age=0)
    old = body_state(cfg, rho=rho, reserves=reserves, last_out=knob_on, age=799)
    out_young, _ = metabolism.apply(young, cfg)
    out_old, _ = metabolism.apply(old, cfg)
    drop_young = jnp.asarray(reserves) - out_young.entities.creature.body.N[0]
    drop_old = jnp.asarray(reserves) - out_old.entities.creature.body.N[0]
    young_base = metabolism.BASE_METABOLISM * (1.0 + metabolism.KNOB_ENERGY_COEFF)
    assert jnp.allclose(drop_young, jnp.asarray(rho) * young_base, atol=1e-6)
    # age 799 -> 800 in this tick, so the bonus is (800-400)/400 = 1.
    assert jnp.allclose(drop_old, jnp.asarray(rho) * young_base * 2.0, atol=1e-6)


def test_toxin_damage_is_delayed_by_one_tick():
    """I -= 0.01*X, then X *= 0.95 (§5.1)."""
    cfg = small_cfg()
    state = body_state(cfg, rho=(0.2, 0.4, 0.4), reserves=(0.1, 0.1, 0.1), toxin=0.5)
    out, _ = metabolism.apply(state, cfg)
    body = out.entities.creature.body
    assert float(body.X[0]) == pytest.approx(0.475, abs=1e-7)
    assert float(body.I[0]) == pytest.approx(1.0 - 0.01 * 0.5, abs=1e-6)
    assert int(out.entities.creature.damage.cause[0]) == events.cause_code("toxin")


def test_thermogenesis_and_thermal_damage():
    """b += 0.05*(T-b) + 0.06*k^2; |b| > 1 damages I (§5.2)."""
    cfg = small_cfg()
    reserves = (0.1, 0.1, 0.1)  # below the repair threshold, above the deficit
    # At y=10, tick=0: T = -2 + 4*10/64 + 0.5 = -0.875, outside bushes.
    cold = body_state(cfg, temperature=0.0, reserves=reserves)
    out_cold, _ = metabolism.apply(cold, cfg)
    assert float(out_cold.entities.creature.body.b[0]) == pytest.approx(-0.04375, abs=1e-6)
    assert float(out_cold.entities.creature.body.I[0]) == pytest.approx(1.0, abs=1e-6)

    hot = body_state(cfg, temperature=2.0, reserves=reserves)
    out_hot, _ = metabolism.apply(hot, cfg)
    body = out_hot.entities.creature.body
    expected_b = 2.0 + 0.05 * (-0.875 - 2.0)
    assert float(body.b[0]) == pytest.approx(expected_b, abs=1e-6)
    expected_damage = 0.002 * (abs(expected_b) - 1.0)
    assert float(body.I[0]) == pytest.approx(1.0 - expected_damage, abs=1e-6)
    assert int(out_hot.entities.creature.damage.cause[0]) == events.cause_code("heat")

    freezing = body_state(cfg, temperature=-2.0, reserves=reserves)
    out_freezing, _ = metabolism.apply(freezing, cfg)
    body = out_freezing.entities.creature.body
    assert int(out_freezing.entities.creature.damage.cause[0]) == events.cause_code("cold")
    assert float(body.I[0]) < 1.0


def test_growth_payment():
    """Juvenile: dm <= 0.7/a_mature, N pays 0.1*dm*c_flesh into B (§5.2)."""
    cfg = small_cfg()
    rho = (0.2, 0.5, 0.3)
    reserves = (0.5, 0.5, 0.5)
    state = body_state(cfg, rho=rho, reserves=reserves, mass=0.3, maturity=0.0)
    out, _ = metabolism.apply(state, cfg)
    body = out.entities.creature.body
    shape = out.entities.creature.shape
    delta_mass = 0.7 / 100.0
    demand = 0.002  # base only: I = 1 means no repair this tick
    cost = 0.1 * delta_mass * FLESH
    capped = jnp.minimum(jnp.asarray(reserves), jnp.asarray(rho))
    expected_n = capped - jnp.asarray(rho) * demand - cost
    assert float(shape.mass[0]) == pytest.approx(0.3 + delta_mass, abs=1e-7)
    assert float(shape.radius[0]) == pytest.approx(
        float(0.5 * jnp.sqrt(0.3 + delta_mass)), abs=1e-7
    )
    assert float(body.mu[0]) == pytest.approx((0.3 + delta_mass - 0.3) / 0.7, abs=1e-6)
    assert jnp.allclose(body.B[0], FLESH * 0.5 + cost, atol=1e-6)
    assert jnp.allclose(body.N[0], expected_n, atol=1e-6)


def test_growth_blocked_without_nutrients():
    """No growth this tick when the tissue payment is not available (§5.2)."""
    cfg = small_cfg()
    state = body_state(
        cfg, rho=(0.2, 0.5, 0.3), reserves=(0.00001, 0.00001, 0.0),
        mass=0.3, maturity=0.0,
    )
    out, _ = metabolism.apply(state, cfg)
    assert float(out.entities.creature.shape.mass[0]) == pytest.approx(0.3, abs=1e-7)
    assert jnp.allclose(out.entities.creature.body.B[0], FLESH * 0.5, atol=1e-7)


def test_aging_metabolic_bonus():
    """After a_sen = 400 the base demand gains (a - a_sen)/a_sen (§5.2)."""
    cfg = small_cfg()
    rho = (0.2, 0.5, 0.3)
    reserves = (0.1, 0.1, 0.1)
    young = body_state(cfg, rho=rho, reserves=reserves, age=0)
    old = body_state(cfg, rho=rho, reserves=reserves, age=799)
    young_out, _ = metabolism.apply(young, cfg)
    old_out, _ = metabolism.apply(old, cfg)
    young_drop = jnp.asarray(reserves) - young_out.entities.creature.body.N[0]
    old_drop = jnp.asarray(reserves) - old_out.entities.creature.body.N[0]
    assert jnp.allclose(young_drop, jnp.asarray(rho) * 0.002, atol=1e-6)
    assert jnp.allclose(old_drop, jnp.asarray(rho) * 0.002 * 2.0, atol=1e-6)
    assert int(old_out.entities.creature.body.age[0]) == 800


def test_knob_lag_and_heartbeat_phase():
    """p <- 0.9p + 0.1k; phi_hb += 2*pi*(1+2p)/20 (§5.2)."""
    cfg = small_cfg()
    last = jnp.zeros((N, C.N_OUTPUTS), jnp.float32).at[0, C.OUT_K].set(1.0)
    state = body_state(cfg, last_out=last)
    out, _ = metabolism.apply(state, cfg)
    body = out.entities.creature.body
    assert float(body.k[0]) == pytest.approx(1.0, abs=1e-7)
    assert float(body.p[0]) == pytest.approx(0.1, abs=1e-7)
    expected_phi = jnp.mod(2.0 * jnp.pi * (1.0 + 0.2) / 20.0, 2.0 * jnp.pi)
    assert float(body.phi_hb[0]) == pytest.approx(float(expected_phi), abs=1e-6)


def test_acute_pain_from_bite_injury():
    """a_inj = 0.7*a_inj + 10*injury; injury is cleared each tick (§5.6)."""
    cfg = small_cfg()
    state = body_state(cfg, a_inj=0.5, injury=0.2)
    out, _ = metabolism.apply(state, cfg)
    assert float(out.entities.creature.pain.a_inj[0]) == pytest.approx(0.7 * 0.5 + 2.0, abs=1e-6)
    assert float(out.entities.creature.damage.injury[0]) == 0.0


def test_birth_payment_and_child_setup():
    """Parent pays B_child, reserves and the birth loss (§5.5)."""
    cfg = small_cfg()
    state = body_state(
        cfg, rho=(0.2, 0.5, 0.3), reserves=(0.5, 0.5, 0.5), tissue=(0.15, 0.35, 0.0)
    )
    out, buffer = reproduction.apply(state, cfg)
    cre = out.entities.creature
    assert int(buffer.count) == 1
    assert bool(buffer.success[0]) and int(buffer.kind[0]) == events.kind_code("birth")
    assert bool(cre.alive[1])
    assert float(cre.shape.mass[1]) == pytest.approx(0.3, abs=1e-7)
    assert jnp.allclose(cre.body.B[1], 0.03 * FLESH, atol=1e-7)
    assert jnp.allclose(cre.body.G[1], 0.0, atol=1e-7)
    child_rho = jnp.asarray([0.2, 0.5, 0.3])  # child genome ~ parent genome
    expected_child_n = jnp.minimum(0.3 * child_rho, jnp.asarray([0.5, 0.5, 0.5]))
    assert jnp.allclose(cre.body.N[1], expected_child_n, atol=1e-2)
    expected_parent_n = (
        jnp.asarray([0.5, 0.5, 0.5])
        - expected_child_n
        - 0.02 * jnp.asarray([0.2, 0.5, 0.3])
    )
    assert jnp.allclose(cre.body.N[0], expected_parent_n, atol=1e-2)
    assert jnp.allclose(cre.body.B[0], jnp.asarray([0.15, 0.35, 0.0]) - 0.03 * FLESH, atol=1e-6)


def test_birth_fails_without_tissue():
    """B_child cannot be paid -> birth_fail with reason no_tissue (§5.5)."""
    cfg = small_cfg()
    state = body_state(
        cfg, rho=(0.2, 0.5, 0.3), reserves=(0.5, 0.5, 0.5), tissue=(0.001, 0.001, 0.0)
    )
    out, buffer = reproduction.apply(state, cfg)
    assert int(buffer.count) == 1
    assert not bool(buffer.success[0])
    assert int(buffer.kind[0]) == events.kind_code("birth_fail")
    assert int(buffer.reason[0]) == events.reason_code("no_tissue")
    assert not bool(out.entities.creature.alive[1])


def test_metabolism_ramp_scales_only_base_demand():
    """§9 ramp: base upkeep = 0.5x at tick 0, 1x at ramp_ticks; work is unscaled."""
    cfg = ramp_cfg()
    rho = (0.2, 0.5, 0.3)
    reserves = (0.1, 0.1, 0.1)  # below the repair threshold
    movement = 0.002 * 1.0 * 1.0 * 1.0  # |u_acc'|=1, k=0, mass=1, m_held=0
    for tick, multiplier in ((0, 0.5), (5, 0.75), (10, 1.0), (50, 1.0)):
        state = body_state(cfg, rho=rho, reserves=reserves, u_eff=(1.0, 0.0))
        state = update(state, tick=jnp.asarray(tick, jnp.int32))
        out, _ = metabolism.apply(state, cfg)
        demand = metabolism.BASE_METABOLISM * multiplier + movement
        drop = jnp.asarray(reserves) - out.entities.creature.body.N[0]
        assert jnp.allclose(drop, jnp.asarray(rho) * demand, atol=1e-6), tick
        assert jnp.allclose(
            out.ledger.outflow[C.OUTFLOW_METABOLISM], jnp.asarray(rho) * demand, atol=1e-6
        ), tick


def test_metabolism_ramp_does_not_scale_repair():
    """Repair is physical work: its 0.002 is not multiplied by the ramp (§9)."""
    cfg = ramp_cfg()
    rho = (0.2, 0.5, 0.3)
    reserves = (0.2, 0.2, 0.2)  # repair condition holds
    state = body_state(cfg, rho=rho, reserves=reserves, integrity=0.5)
    state = update(state, tick=jnp.asarray(0, jnp.int32))
    out, _ = metabolism.apply(state, cfg)
    demand = metabolism.BASE_METABOLISM * 0.5 + metabolism.REPAIR_RATE
    drop = jnp.asarray(reserves) - out.entities.creature.body.N[0]
    assert jnp.allclose(drop, jnp.asarray(rho) * demand, atol=1e-6)
    assert float(out.entities.creature.body.I[0]) == pytest.approx(0.502, abs=1e-6)
