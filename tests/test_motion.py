"""Motion integration and analytic values (SANDBOX.md §5.3, §13 test_motion).

Every expectation below is the closed-form §5.3 update with noise_scale = 0, so
the test checks the integration (acceleration -> drag -> speed cap -> turn ->
position) rather than re-deriving it from the code.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from life_sandbox.config import Capacities, Config, MotionConfig
from life_sandbox.core import components as C
from life_sandbox.core.state import empty
from life_sandbox.core.struct import update
from life_sandbox.core.systems import death, interaction, metabolism, motor, reproduction

N = 4


def small_cfg(noise: float = 0.0) -> Config:
    return Config.default().replaced(
        capacities=Capacities(creature=N, bush=6, carcass=2, neutral=1),
        motion=MotionConfig(noise_scale=noise),
    )


def outputs(acc=0.0, turn=0.0, k=0.0, bite=0.0, grab=0.0, drop=0.0) -> jax.Array:
    out = jnp.zeros((N, C.N_OUTPUTS), jnp.float32)
    out = out.at[:, C.OUT_ACC].set(acc)
    out = out.at[:, C.OUT_TURN].set(turn)
    out = out.at[:, C.OUT_K].set(k)
    out = out.at[:, C.OUT_BITE].set(bite)
    out = out.at[:, C.OUT_GRAB].set(grab)
    out = out.at[:, C.OUT_DROP].set(drop)
    return out


def creature_state(
    cfg: Config,
    *,
    pos=(10.0, 10.0),
    theta=0.0,
    v=0.0,
    integrity=1.0,
    maturity=1.0,
    mass=1.0,
    asleep=False,
    last=None,
    holding=None,
):
    state = empty(cfg)
    cre = state.entities.creature
    last = outputs() if last is None else last
    grip = cre.grip
    if holding is not None:
        grip = C.Grip(
            holding=C.Holding(
                kind=jnp.full((N,), holding[0], jnp.int32),
                slot=jnp.full((N,), holding[1], jnp.int32),
                uid=jnp.full((N,), holding[2], jnp.int32),
            ),
            held_by=cre.grip.held_by,
        )
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
            theta=cre.transform.theta.at[0].set(theta),
            v=cre.transform.v.at[0].set(v),
        ),
        shape=update(
            cre.shape,
            radius=cre.shape.radius.at[0].set(0.5),
            mass=cre.shape.mass.at[0].set(mass),
        ),
        body=update(
            cre.body,
            I=cre.body.I.at[0].set(integrity),
            mu=cre.body.mu.at[0].set(maturity),
            asleep=cre.body.asleep.at[0].set(asleep),
        ),
        motor=update(cre.motor, last_out=last),
        grip=grip,
    )
    return update(state, entities=update(state.entities, creature=cre))


def test_acceleration_drag_analytic():
    """Adult, intact, k=0, u_acc=1: v = (0 + 0.05*1*1)*0.9 = 0.045."""
    cfg = small_cfg()
    state = creature_state(cfg, last=outputs(acc=1.0))
    out, buffer = motor.apply(state, cfg)
    assert int(buffer.count) == 0
    assert float(out.entities.creature.transform.v[0]) == pytest.approx(0.045, abs=1e-6)
    pos = out.entities.creature.transform.pos[0]
    assert float(pos[0]) == pytest.approx(10.0 + 0.045, abs=1e-6)
    assert float(pos[1]) == pytest.approx(10.0, abs=1e-6)


def test_drag_without_thrust():
    """u_acc=0: v <- v*(1 - 0.1) (§5.3)."""
    cfg = small_cfg()
    state = creature_state(cfg, v=0.2)
    out, _ = motor.apply(state, cfg)
    assert float(out.entities.creature.transform.v[0]) == pytest.approx(0.18, abs=1e-6)


def test_speed_cap_at_full_knob():
    """k=1: a_max=0.1, cap = 0.5*(1+0.5) = 0.75 (§5.3)."""
    cfg = small_cfg()
    state = creature_state(cfg, v=0.9, last=outputs(acc=1.0, k=1.0))
    out, _ = motor.apply(state, cfg)
    assert float(out.entities.creature.transform.v[0]) == pytest.approx(0.75, abs=1e-6)


def test_turn_and_wrap():
    """theta <- theta + 0.3*c_I*u_turn, wrapped to [-pi, pi) (§5.3)."""
    cfg = small_cfg()
    state = creature_state(cfg, integrity=0.5, last=outputs(turn=1.0))
    out, _ = motor.apply(state, cfg)
    assert float(out.entities.creature.transform.omega[0]) == pytest.approx(
        0.3 * (0.3 + 0.7 * 0.5), abs=1e-6
    )
    assert float(out.entities.creature.transform.theta[0]) == pytest.approx(0.195, abs=1e-6)
    assert float(out.entities.creature.transform.pos[0, 0]) == pytest.approx(10.0, abs=1e-7)

    state = creature_state(cfg, theta=3.1, last=outputs(turn=1.0))
    out, _ = motor.apply(state, cfg)
    theta = float(out.entities.creature.transform.theta[0])
    assert -jnp.pi <= theta < jnp.pi
    assert theta == pytest.approx(3.1 + 0.3 - 2.0 * jnp.pi, abs=1e-6)


def test_injury_weakens_acceleration():
    """g = (0.3 + 0.7*mu)*c_I; I=0.5 -> g = 0.65 (§5.3)."""
    cfg = small_cfg()
    state = creature_state(cfg, integrity=0.5, last=outputs(acc=1.0))
    out, _ = motor.apply(state, cfg)
    assert float(out.entities.creature.transform.v[0]) == pytest.approx(
        0.05 * 0.65 * 0.9, abs=1e-6
    )


def test_held_mass_reduces_acceleration():
    """a_max = 0.05*(1+k)*g / (1 + m_held/m); m_held/m = 1 -> half (§5.3)."""
    cfg = small_cfg()
    state = creature_state(cfg, last=outputs(acc=1.0), holding=(C.CARCASS, 0, 123))
    carcass = state.entities.carcass
    carcass = update(
        carcass,
        alive=carcass.alive.at[0].set(True),
        uid=carcass.uid.at[0].set(123),
        shape=update(carcass.shape, mass=carcass.shape.mass.at[0].set(1.0)),
    )
    state = update(state, entities=update(state.entities, carcass=carcass))
    out, _ = motor.apply(state, cfg)
    assert float(out.entities.creature.transform.v[0]) == pytest.approx(
        0.025 * 0.9, abs=1e-6
    )


def test_held_entity_follows_holder():
    """pos_target = pos_holder + (r_h + r_t)*(cos theta, sin theta) (§5.4)."""
    cfg = small_cfg()
    state = creature_state(cfg, holding=(C.NEUTRAL, 0, 55))
    neutral = state.entities.neutral
    neutral = update(
        neutral,
        alive=neutral.alive.at[0].set(True),
        uid=neutral.uid.at[0].set(55),
        shape=update(
            neutral.shape,
            radius=neutral.shape.radius.at[0].set(0.4),
            mass=neutral.shape.mass.at[0].set(0.2),
        ),
        held_by=C.HeldBy(
            kind=jnp.full((1,), C.CREATURE, jnp.int32),
            slot=jnp.zeros((1,), jnp.int32),
            uid=jnp.full((1,), 1, jnp.int32),
        ),
    )
    state = update(state, entities=update(state.entities, neutral=neutral))
    out, _ = motor.apply(state, cfg)
    pos = out.entities.neutral.transform.pos[0]
    assert float(pos[0]) == pytest.approx(10.0 + 0.9, abs=1e-6)
    assert float(pos[1]) == pytest.approx(10.0, abs=1e-6)
    assert float(out.entities.neutral.transform.theta[0]) == pytest.approx(0.0, abs=1e-6)


def test_asleep_decays_speed_and_ignores_outputs():
    """Sleep: outputs invalid, v <- 0.9*v, position still integrates (§5.3)."""
    cfg = small_cfg()
    state = creature_state(cfg, v=0.4, asleep=True, last=outputs(acc=1.0, turn=1.0))
    out, _ = motor.apply(state, cfg)
    assert float(out.entities.creature.transform.v[0]) == pytest.approx(0.36, abs=1e-6)
    assert float(out.entities.creature.transform.omega[0]) == pytest.approx(0.0, abs=1e-7)
    pos = out.entities.creature.transform.pos[0]
    assert float(pos[0]) == pytest.approx(10.0 + 0.36, abs=1e-6)


def test_noise_is_deterministic_and_bounded():
    cfg = small_cfg(noise=1.0)
    state = creature_state(cfg, integrity=0.0, last=outputs(acc=1.0, turn=1.0))
    first, _ = motor.apply(state, cfg)
    second, _ = motor.apply(state, cfg)
    for field in ("pos", "theta", "v", "omega"):
        assert jnp.array_equal(
            getattr(first.entities.creature.transform, field),
            getattr(second.entities.creature.transform, field),
        )
    assert bool(jnp.all(jnp.abs(first.entities.creature.transform.v) <= 0.5))


@pytest.mark.parametrize(
    "system", [motor.apply, interaction.apply, metabolism.apply, reproduction.apply, death.apply]
)
def test_body_systems_trace_under_jit(system):
    """The production path is jitted (SANDBOX.md §3.3); two ticks must trace."""
    cfg = small_cfg()
    state = creature_state(cfg)
    step = jax.jit(lambda s: system(s, cfg))
    first, _ = step(state)
    second, _ = step(first)
    assert second.entities.creature.alive.shape == state.entities.creature.alive.shape
