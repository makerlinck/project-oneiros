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
from life_sandbox.config import (
    BodyConfig,
    Capacities,
    Config,
    MotionConfig,
    ReproductionConfig,
    WorldConfig,
)
from life_sandbox.core import components as C, events
from life_sandbox.core.state import empty
from life_sandbox.core.struct import update
from life_sandbox.core.systems import death, metabolism, reproduction
from life_sandbox.world import init as world_init

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
    loss=None,
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
        damage=update(
            cre.damage,
            injury=cre.damage.injury.at[0].set(injury),
            loss=(
                cre.damage.loss
                if loss is None
                else cre.damage.loss.at[0].set(jnp.asarray(loss, jnp.float32))
            ),
        ),
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
    required = cap * cfg.body.base_metabolism  # awake, k=0, no repair
    expected_n = jnp.asarray(reserves) + absorbed - required
    body = out.entities.creature.body
    assert jnp.allclose(body.G[0], 0.95 * jnp.asarray(stomach), atol=1e-7)
    assert jnp.allclose(body.N[0], expected_n, atol=1e-6)
    expected_x = 3.0 * (1.0 - eps[C.TOXIN]) * delta[C.TOXIN]
    assert float(body.X[0]) == pytest.approx(float(expected_x * 0.95), abs=1e-7)

    # §3.5 ledger classification for this tick.
    ledger = out.ledger
    required = cap * cfg.body.base_metabolism  # awake, k = 0, I = 1 -> no repair
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
    demand = cfg.body.base_metabolism
    absorbed_a = out_a.entities.creature.body.N[0] + demand * jnp.asarray([0.8, 0.1, 0.1])
    absorbed_b = out_b.entities.creature.body.N[0] + demand * jnp.asarray([0.1, 0.8, 0.1])
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
    demand = cfg.body.base_metabolism  # base only: I = 1 -> no repair
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
    """Missing reserves: I -= 5*sum(d_j), attributed to LOSS_NUTRITION (§5.1, §5.5)."""
    cfg = small_cfg()
    rho = (0.2, 0.5, 0.3)
    state = body_state(cfg, rho=rho, reserves=(0.0, 0.0, 0.0), integrity=1.0)
    out, _ = metabolism.apply(state, cfg)
    body = out.entities.creature.body
    required = jnp.asarray(rho) * cfg.body.base_metabolism  # no repair: sum N = 0
    deficiency = 5.0 * jnp.sum(required)
    assert float(body.I[0]) == pytest.approx(1.0 - float(deficiency), abs=1e-6)
    damage = out.entities.creature.damage
    assert int(jnp.argmax(damage.loss[0])) == C.LOSS_NUTRITION
    assert float(damage.loss[0, C.LOSS_NUTRITION]) == pytest.approx(
        float(deficiency), abs=1e-6
    )
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
    demand = cfg.body.base_metabolism + metabolism.REPAIR_RATE  # base + repair
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
    demand = cfg.body.base_metabolism  # base only
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
    demand = cfg.body.base_metabolism + repaired
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
        jnp.asarray(rho) * cfg.body.base_metabolism,
        atol=1e-7,
    )
    movement = (
        cfg.body.base_metabolism * 1.0 * 1.0 + 0.0005 * 1.0
    )  # |u_acc'|*(1+k)*m, |u_turn'|
    assert jnp.allclose(
        drop_high_eff,
        jnp.asarray(rho) * (cfg.body.base_metabolism + movement),
        atol=1e-6,
    )


def test_base_demand_knob_and_aging_terms():
    """Base demand = cfg.body.base_metabolism*(1+3k^2)*(1+ageing), after a_sen."""
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
    young_base = cfg.body.base_metabolism * (1.0 + metabolism.KNOB_ENERGY_COEFF)
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
    damage = out.entities.creature.damage
    assert int(jnp.argmax(damage.loss[0])) == C.LOSS_TOXIN
    assert float(damage.loss[0, C.LOSS_TOXIN]) == pytest.approx(0.01 * 0.5, abs=1e-7)


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
    hot_damage = out_hot.entities.creature.damage
    assert int(jnp.argmax(hot_damage.loss[0])) == C.LOSS_HEAT
    assert float(hot_damage.loss[0, C.LOSS_HEAT]) == pytest.approx(
        expected_damage, abs=1e-7
    )

    freezing = body_state(cfg, temperature=-2.0, reserves=reserves)
    out_freezing, _ = metabolism.apply(freezing, cfg)
    body = out_freezing.entities.creature.body
    cold_damage = out_freezing.entities.creature.damage
    assert int(jnp.argmax(cold_damage.loss[0])) == C.LOSS_COLD
    assert float(cold_damage.loss[0, C.LOSS_COLD]) > 0.0
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
    # Base upkeep now scales with the body mass: m^0.75 at m = 0.3.
    demand = cfg.body.base_metabolism * (0.3 ** 0.75)  # I = 1 -> no repair
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
    assert jnp.allclose(
        young_drop, jnp.asarray(rho) * cfg.body.base_metabolism, atol=1e-6
    )
    assert jnp.allclose(
        old_drop, jnp.asarray(rho) * cfg.body.base_metabolism * 2.0, atol=1e-6
    )
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
    """Default arm pays B_child and reserves from N; adult B is untouched (§5.5)."""
    cfg = small_cfg()
    parent_n = jnp.asarray([0.5, 0.5, 0.5])
    parent_b = jnp.asarray([0.15, 0.35, 0.0])
    state = body_state(
        cfg, rho=(0.2, 0.5, 0.3), reserves=tuple(parent_n), tissue=tuple(parent_b)
    )
    out, buffer = reproduction.apply(state, cfg)
    cre = out.entities.creature
    assert int(buffer.count) == 1
    assert bool(buffer.success[0]) and int(buffer.kind[0]) == events.kind_code("birth")
    assert bool(cre.alive[1])
    assert float(cre.shape.mass[1]) == pytest.approx(0.3, abs=1e-7)
    assert jnp.allclose(cre.body.B[1], 0.03 * FLESH, atol=1e-7)
    assert jnp.allclose(cre.body.G[1], 0.0, atol=1e-7)
    # B_child comes out of N first, then the child reserves from what is left.
    tissue_cost = 0.03 * FLESH
    child_rho = world_init.diet_from_genome(cre.genome.g[1:2], cfg)[0]
    expected_child_n = jnp.minimum(0.3 * child_rho, parent_n - tissue_cost)
    assert jnp.allclose(cre.body.N[1], expected_child_n, atol=1e-6)
    remaining = parent_n - tissue_cost - expected_child_n
    parent_rho = jnp.asarray([0.2, 0.5, 0.3])  # parent genome diet genes
    loss = jnp.minimum(0.02 * parent_rho, remaining)
    expected_parent_n = parent_n - tissue_cost - expected_child_n - loss
    assert jnp.allclose(cre.body.N[0], expected_parent_n, atol=1e-6)
    # The adult's tissue is not consumed by reproduction.
    assert jnp.array_equal(cre.body.B[0], parent_b)


def test_birth_fails_without_tissue_in_control_arm():
    """Control arm (tissue_from_reserves=false) keeps the old no_tissue failure."""
    cfg = small_cfg().replaced(
        reproduction=ReproductionConfig(tissue_from_reserves=False)
    )
    state = body_state(
        cfg, rho=(0.2, 0.5, 0.3), reserves=(0.5, 0.5, 0.5), tissue=(0.001, 0.001, 0.0)
    )
    out, buffer = reproduction.apply(state, cfg)
    assert int(buffer.count) == 1
    assert not bool(buffer.success[0])
    assert int(buffer.kind[0]) == events.kind_code("birth_fail")
    assert int(buffer.reason[0]) == events.reason_code("no_tissue")
    assert not bool(out.entities.creature.alive[1])


def test_birth_fails_without_reserve():
    """Default arm: one N component below B_child fails with no_reserve (§5.5)."""
    cfg = small_cfg()
    reserves = (0.001, 0.5, 0.5)  # first component < 0.03*0.3 = 0.009
    tissue = (0.15, 0.35, 0.0)  # enough tissue in the control arm, irrelevant here
    state = body_state(cfg, rho=(0.2, 0.5, 0.3), reserves=reserves, tissue=tissue)
    out, buffer = reproduction.apply(state, cfg)
    assert int(buffer.count) == 1
    assert int(buffer.kind[0]) == events.kind_code("birth_fail")
    assert int(buffer.reason[0]) == events.reason_code("no_reserve")
    cre = out.entities.creature
    assert not bool(cre.alive[1])
    # A failed birth consumes no payment.
    assert jnp.allclose(cre.body.N[0], jnp.asarray(reserves), atol=1e-7)
    assert jnp.allclose(cre.body.B[0], jnp.asarray(tissue), atol=1e-7)


def test_base_metabolism_reads_body_config():
    """The base-demand coefficient is cfg.body.base_metabolism (M1 scan)."""
    from life_sandbox.config import BodyConfig

    cfg = small_cfg().replaced(body=BodyConfig(base_metabolism=0.01))
    rho = (0.2, 0.5, 0.3)
    reserves = (0.1, 0.1, 0.1)  # below the repair threshold
    state = body_state(cfg, rho=rho, reserves=reserves)  # I = 1, no motion
    out, _ = metabolism.apply(state, cfg)
    drop = jnp.asarray(reserves) - out.entities.creature.body.N[0]
    assert jnp.allclose(drop, jnp.asarray(rho) * cfg.body.base_metabolism, atol=1e-6)
    assert cfg.body.base_metabolism == 0.01  # the default is 0.002

    # The same coefficient scales the |u_acc'| work term (§5.3).
    moving = body_state(
        cfg, rho=rho, reserves=(0.05, 0.05, 0.05), u_eff=(1.0, 0.0)
    )
    out_moving, _ = metabolism.apply(moving, cfg)
    movement = cfg.body.base_metabolism * 1.0 * (1.0 + 0.0) * (1.0 + 0.0)
    drop_moving = jnp.asarray([0.05, 0.05, 0.05]) - out_moving.entities.creature.body.N[0]
    assert jnp.allclose(
        drop_moving, jnp.asarray(rho) * (cfg.body.base_metabolism + movement), atol=1e-6
    )


def test_metabolism_ramp_scales_whole_energy():
    """§9 ramp: the whole energy scale is 0.5x at tick 0 and 1x at ramp_ticks."""
    cfg = ramp_cfg()
    rho = (0.2, 0.5, 0.3)
    reserves = (0.1, 0.1, 0.1)  # below the repair threshold
    movement = cfg.body.base_metabolism  # |u_acc'|=1, k=0, mass=1, m_held=0
    for tick, multiplier in ((0, 0.5), (5, 0.75), (10, 1.0), (50, 1.0)):
        state = body_state(cfg, rho=rho, reserves=reserves, u_eff=(1.0, 0.0))
        state = update(state, tick=jnp.asarray(tick, jnp.int32))
        out, _ = metabolism.apply(state, cfg)
        demand = (cfg.body.base_metabolism + movement) * multiplier
        drop = jnp.asarray(reserves) - out.entities.creature.body.N[0]
        assert jnp.allclose(drop, jnp.asarray(rho) * demand, atol=1e-6), tick
        assert jnp.allclose(
            out.ledger.outflow[C.OUTFLOW_METABOLISM], jnp.asarray(rho) * demand, atol=1e-6
        ), tick


def test_loss_decays_with_tau_and_keeps_the_larger_source():
    """loss <- loss*exp(-1/50) before this tick's damage is added (§5.5)."""
    cfg = small_cfg()
    rho = (0.2, 0.5, 0.3)
    reserves = (0.1, 0.1, 0.1)
    decay = jnp.exp(-1.0 / C.DAMAGE_DECAY_TAU)

    # Pure decay: no new source this tick.
    state = body_state(
        cfg, rho=rho, reserves=reserves, loss=(0.0, 1.0, 0.0, 0.0, 0.0)
    )
    out, _ = metabolism.apply(state, cfg)
    damage = out.entities.creature.damage
    assert float(damage.loss[0, C.LOSS_TOXIN]) == pytest.approx(float(decay), abs=1e-7)
    assert int(jnp.argmax(damage.loss[0])) == C.LOSS_TOXIN

    # A small source written later in the same tick does not override a larger
    # decayed source: heat 1.0 * decay stays above the tiny cold damage.
    state = body_state(
        cfg,
        rho=rho,
        reserves=reserves,
        temperature=-2.0,
        loss=(0.0, 0.0, 1.0, 0.0, 0.0),
    )
    out, _ = metabolism.apply(state, cfg)
    damage = out.entities.creature.damage
    assert float(damage.loss[0, C.LOSS_HEAT]) == pytest.approx(float(decay), abs=1e-7)
    assert float(damage.loss[0, C.LOSS_COLD]) > 0.0
    assert int(jnp.argmax(damage.loss[0])) == C.LOSS_HEAT


def test_decay_lets_new_damage_win_across_ticks():
    """Previous toxin 1.0 decays to 0.98, so 0.99 nutrition damage wins (§5.5)."""
    cfg = small_cfg().replaced(body=BodyConfig(base_metabolism=0.198))
    rho = (0.2, 0.5, 0.3)
    state = body_state(
        cfg, rho=rho, reserves=(0.0, 0.0, 0.0), loss=(0.0, 1.0, 0.0, 0.0, 0.0)
    )
    out, _ = metabolism.apply(state, cfg)
    damage = out.entities.creature.damage
    decay = float(jnp.exp(-1.0 / C.DAMAGE_DECAY_TAU))
    deficiency = 5.0 * cfg.body.base_metabolism  # 0.99
    assert float(damage.loss[0, C.LOSS_TOXIN]) == pytest.approx(decay, abs=1e-6)
    assert float(damage.loss[0, C.LOSS_NUTRITION]) == pytest.approx(
        deficiency, abs=1e-6
    )
    assert int(jnp.argmax(damage.loss[0])) == C.LOSS_NUTRITION


def test_chronic_toxin_does_not_win_over_larger_nutrition():
    """A persistent tiny toxin damage never out-attributes larger starvation."""
    cfg = small_cfg()
    rho = (0.2, 0.5, 0.3)
    state = body_state(cfg, rho=rho, reserves=(0.0, 0.0, 0.0))
    for _ in range(150):
        out, _ = metabolism.apply(state, cfg)
        body = out.entities.creature.body
        cre = update(
            out.entities.creature,
            body=update(
                body,
                I=body.I.at[0].set(1.0),
                X=body.X.at[0].set(0.01),  # persistent trickle: 1e-4 damage/tick
                N=body.N.at[0].set(jnp.zeros(3)),
                G=body.G.at[0].set(jnp.zeros(3)),
                b=body.b.at[0].set(0.0),
                age=body.age.at[0].set(0),
            ),
        )
        state = update(out, entities=update(out.entities, creature=cre))
    damage = state.entities.creature.damage
    nutrition = float(damage.loss[0, C.LOSS_NUTRITION])
    toxin = float(damage.loss[0, C.LOSS_TOXIN])
    assert nutrition > 0.0 and toxin > 0.0
    assert int(jnp.argmax(damage.loss[0])) == C.LOSS_NUTRITION
    assert nutrition > 50.0 * toxin

    # Death from this state is attributed to nutrition.
    victim = update(
        state,
        entities=update(
            state.entities,
            creature=update(
                state.entities.creature,
                body=update(state.entities.creature.body, I=jnp.zeros((N,), jnp.float32)),
            ),
        ),
    )
    _, buffer = death.apply(victim, cfg)
    deaths = [
        i for i in range(int(buffer.count)) if int(buffer.kind[i]) == events.kind_code("death")
    ]
    assert len(deaths) == 1
    assert int(buffer.cause[deaths[0]]) == events.cause_code("nutrition")
    assert int(buffer.nutrient[deaths[0]]) == int(jnp.argmax(jnp.asarray(rho)))


def test_large_toxin_load_is_attributed_to_toxin():
    cfg = small_cfg()
    state = body_state(
        cfg, rho=(0.2, 0.5, 0.3), reserves=(0.0, 0.0, 0.0), toxin=100.0
    )
    out, _ = metabolism.apply(state, cfg)
    damage = out.entities.creature.damage
    assert int(jnp.argmax(damage.loss[0])) == C.LOSS_TOXIN
    _, buffer = death.apply(out, cfg)
    deaths = [
        i for i in range(int(buffer.count)) if int(buffer.kind[i]) == events.kind_code("death")
    ]
    assert len(deaths) == 1
    assert int(buffer.cause[deaths[0]]) == events.cause_code("toxin")


def test_integrity_zero_is_not_repaired():
    """I <= 0 at body entry is dead already and must not be repaired (§5.2)."""
    cfg = small_cfg()
    rho = (0.2, 0.5, 0.3)
    reserves = (0.2, 0.2, 0.2)  # repair resources exist
    state = body_state(cfg, rho=rho, reserves=reserves, integrity=0.0)
    out, _ = metabolism.apply(state, cfg)
    body = out.entities.creature.body
    assert float(body.I[0]) == 0.0
    # The unrepaired tick charges only the base demand.
    drop = jnp.asarray(reserves) - body.N[0]
    assert jnp.allclose(drop, jnp.asarray(rho) * cfg.body.base_metabolism, atol=1e-6)
    _, buffer = death.apply(out, cfg)
    deaths = [
        i for i in range(int(buffer.count)) if int(buffer.kind[i]) == events.kind_code("death")
    ]
    assert len(deaths) == 1
    assert int(buffer.uid[deaths[0]]) == 1


def test_metabolism_ramp_does_not_scale_repair():
    """Repair is physical work: its 0.002 is not multiplied by the ramp (§9)."""
    cfg = ramp_cfg()
    rho = (0.2, 0.5, 0.3)
    reserves = (0.2, 0.2, 0.2)  # repair condition holds
    state = body_state(cfg, rho=rho, reserves=reserves, integrity=0.5)
    state = update(state, tick=jnp.asarray(0, jnp.int32))
    out, _ = metabolism.apply(state, cfg)
    demand = cfg.body.base_metabolism * 0.5 + metabolism.REPAIR_RATE
    drop = jnp.asarray(reserves) - out.entities.creature.body.N[0]
    assert jnp.allclose(drop, jnp.asarray(rho) * demand, atol=1e-6)
    assert float(out.entities.creature.body.I[0]) == pytest.approx(0.502, abs=1e-6)


def _demand_of(cfg, *, rho=(0.2, 0.5, 0.3), mass=1.0, maturity=1.0, u_eff=None,
                bite=0.0, reserves=(0.05, 0.05, 0.05), integrity=1.0):
    """Total c_tot this tick, read back from the reserve drop (split by rho)."""
    last = jnp.zeros((N, C.N_OUTPUTS), jnp.float32)
    if bite:
        last = last.at[0, C.OUT_BITE].set(bite)
    state = body_state(
        cfg,
        rho=rho,
        reserves=reserves,
        mass=mass,
        maturity=maturity,
        integrity=integrity,
        last_out=last,
        u_eff=u_eff,
    )
    out, _ = metabolism.apply(state, cfg)
    drop = jnp.asarray(reserves) - out.entities.creature.body.N[0]
    # A juvenile also pays growth tissue from N; remove that known cost so the
    # measured drop reflects c_tot only (demand is split by rho).
    growth_cost = jnp.zeros(3, jnp.float32)
    if mass < 1.0:
        delta_mass = min(0.7 / 100.0, 1.0 - mass)
        growth_cost = 0.1 * delta_mass * FLESH
    demand = (drop - growth_cost) / jnp.asarray(rho)
    assert jnp.allclose(demand, demand[0], atol=1e-7)
    ledger = out.ledger.outflow[C.OUTFLOW_METABOLISM]
    assert jnp.allclose(ledger, jnp.asarray(rho) * demand[0], atol=1e-6)
    return float(demand[0])


def test_demand_terms_match_the_old_formula_at_adult_default():
    """Default cfg + adult: base 0.002, turn 0.25*base, bite = base (§5.3)."""
    cfg = small_cfg()
    base = cfg.body.base_metabolism
    assert _demand_of(cfg) == pytest.approx(float(base), abs=1e-7)
    assert _demand_of(cfg, u_eff=(1.0, 0.0)) == pytest.approx(float(2.0 * base), abs=1e-7)
    assert _demand_of(cfg, u_eff=(0.0, 1.0)) == pytest.approx(
        float(base + 0.25 * base), abs=1e-7
    )
    assert _demand_of(cfg, bite=1.0) == pytest.approx(float(2.0 * base), abs=1e-7)
    assert _demand_of(cfg, u_eff=(1.0, 1.0), bite=1.0) == pytest.approx(
        float(base * (1.0 + 1.0 + 0.25 + 1.0)), abs=1e-7
    )


def test_base_metabolism_scales_all_terms_five_x():
    """A 5x cfg.body.base_metabolism scales all four demand terms together."""
    cfg = small_cfg().replaced(body=BodyConfig(base_metabolism=0.01))
    base = cfg.body.base_metabolism
    assert base == pytest.approx(5.0 * 0.002)
    assert _demand_of(cfg) == pytest.approx(float(base), abs=1e-7)
    assert _demand_of(cfg, u_eff=(1.0, 0.0)) == pytest.approx(float(2.0 * base), abs=1e-7)
    assert _demand_of(cfg, u_eff=(0.0, 1.0)) == pytest.approx(
        float(base + 0.25 * base), abs=1e-7
    )
    assert _demand_of(cfg, bite=1.0) == pytest.approx(float(2.0 * base), abs=1e-7)
    assert _demand_of(cfg, u_eff=(1.0, 1.0), bite=1.0) == pytest.approx(
        float(base * 3.25), abs=1e-7
    )
    # Repair still charges the actual repaired amount (REPAIR_RATE), not 5x.
    repair_demand = _demand_of(cfg, reserves=(0.2, 0.2, 0.2), integrity=0.5)
    assert repair_demand == pytest.approx(
        float(base + metabolism.REPAIR_RATE), abs=1e-7
    )


def test_juvenile_demand_uses_mass_power_075():
    """Base upkeep scales as m^0.75; the acceleration term uses (m + m_held)."""
    cfg = small_cfg()
    base = cfg.body.base_metabolism
    mass = 0.3
    mass_scale = mass ** 0.75
    assert mass_scale == pytest.approx(0.405, abs=1e-3)

    # Base term alone: exactly m^0.75 of the adult value (1e-6 tolerance).
    base_only = _demand_of(cfg, mass=mass, maturity=0.0)
    assert base_only == pytest.approx(float(base * mass_scale), abs=1e-6)

    # Acceleration term carries (m + m_held) = 0.3, on top of the m^0.75 base.
    moving = _demand_of(cfg, mass=mass, maturity=0.0, u_eff=(1.0, 0.0))
    assert moving == pytest.approx(float(base * (mass_scale + mass)), abs=1e-6)

    everything = _demand_of(
        cfg, mass=mass, maturity=0.0, u_eff=(1.0, 1.0), bite=1.0
    )
    assert everything == pytest.approx(
        float(base * (mass_scale + mass + 0.25 + 1.0)), abs=1e-6
    )
