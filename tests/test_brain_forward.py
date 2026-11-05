"""B0 forward pass, genome decode and think wiring (SANDBOX.md 6.1, 6.3).

The forward pass is checked against a hand-written loop over every term of the
SANDBOX.md 6.1 equations on a tiny network (H=2, D=3, M=2).  The genome tests
pin the documented block layout, its contiguity from core.genome.brain_base,
the slow/plastic decode and the w_fast split/join round-trip.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from life_sandbox.brain import base
from life_sandbox.brain import genome as brain_genome
from life_sandbox.config import (
    BrainConfig,
    Capacities,
    Config,
    LearningConfig,
    RunConfig,
    WorldConfig,
)
from life_sandbox.core import components as C
from life_sandbox.core import genome as core_genome
from life_sandbox.core import rng
from life_sandbox.core.hashing import state_hash
from life_sandbox.core.systems import learn, think
from life_sandbox.world.init import initial_state

NH = 2  # hand-computed hidden size
ND = 3  # hand-computed slice size
NM = 2  # hand-computed modulator count
NN = 1  # one creature in the hand-computed forward pass
NPOST = NH + C.N_OUTPUTS


def _grid(*shape: int, scale: float, offset: float) -> jax.Array:
    count = 1
    for dim in shape:
        count *= int(dim)
    flat = jnp.arange(count, dtype=jnp.float32) * jnp.float32(scale) + jnp.float32(offset)
    return flat.reshape(shape).astype(jnp.float32)


def _hand_weights() -> tuple[base.SlowWeights, base.FastWeights]:
    slow = base.SlowWeights(
        w_in=_grid(NN, NH, ND, scale=0.1, offset=-0.2),
        w_err=_grid(NN, NH, ND, scale=0.05, offset=0.1),
        w_rec=_grid(NN, NH, NH, scale=0.2, offset=-0.1),
        w_out=_grid(NN, C.N_OUTPUTS, NH, scale=0.15, offset=-0.25),
        w_mod=_grid(NN, NM, NH, scale=0.3, offset=-0.1),
        g=_grid(NN, NPOST, NM, scale=0.2, offset=-0.2),
        w_pred=_grid(NN, ND, NH, scale=0.25, offset=-0.15),
        b=_grid(NN, NH, scale=0.1, offset=0.05),
        b_out=_grid(NN, C.N_OUTPUTS, scale=0.2, offset=-0.05),
    )
    fast = base.FastWeights(
        w_in=_grid(NN, NH, ND, scale=0.02, offset=0.01),
        w_rec=_grid(NN, NH, NH, scale=0.03, offset=-0.02),
        w_out=_grid(NN, C.N_OUTPUTS, NH, scale=0.05, offset=0.02),
        w_pred=_grid(NN, ND, NH, scale=0.04, offset=-0.01),
    )
    return slow, fast


def _hand_forward(slow, fast, s, e, h_prev, sleep, noise, prediction_enabled=True):
    """Loop over the SANDBOX.md 6.1 equations, one term at a time."""
    h = jnp.zeros((NH,), dtype=jnp.float32)
    for j in range(NH):
        drive = slow.b[0, j] + base.SIGMA_N * sleep[0] * noise[0, j]
        for i in range(ND):
            drive = drive + (slow.w_in[0, j, i] + fast.w_in[0, j, i]) * s[0, i]
            if prediction_enabled:
                drive = drive + slow.w_err[0, j, i] * e[0, i]
        for k in range(NH):
            drive = drive + (slow.w_rec[0, j, k] + fast.w_rec[0, j, k]) * h_prev[0, k]
        h = h.at[j].set(jnp.tanh(drive))

    raw = jnp.zeros((C.N_OUTPUTS,), dtype=jnp.float32)
    for o in range(C.N_OUTPUTS):
        drive = slow.b_out[0, o]
        for j in range(NH):
            drive = drive + (slow.w_out[0, o, j] + fast.w_out[0, o, j]) * h[j]
        raw = raw.at[o].set(drive)

    movement = jnp.tanh(raw[: C.N_OUTPUTS - 1])
    knob = jax.nn.sigmoid(raw[C.N_OUTPUTS - 1])
    out = jnp.stack(
        [
            movement[C.OUT_ACC],
            movement[C.OUT_TURN],
            knob,
            movement[C.OUT_BITE - 1],
            movement[C.OUT_GRAB - 1],
            movement[C.OUT_DROP - 1],
        ]
    )

    m = jnp.zeros((NM,), dtype=jnp.float32)
    for k in range(NM):
        drive = jnp.float32(0.0)
        for j in range(NH):
            drive = drive + slow.w_mod[0, k, j] * h[j]
        m = m.at[k].set(jnp.tanh(drive))
    m_tilde = jnp.zeros((NPOST,), dtype=jnp.float32)
    for p in range(NPOST):
        value = jnp.float32(0.0)
        for k in range(NM):
            value = value + slow.g[0, p, k] * m[k]
        m_tilde = m_tilde.at[p].set(value)

    if prediction_enabled:
        pred = jnp.zeros((ND,), dtype=jnp.float32)
        for i in range(ND):
            value = jnp.float32(0.0)
            for j in range(NH):
                value = value + (slow.w_pred[0, i, j] + fast.w_pred[0, i, j]) * h[j]
            pred = pred.at[i].set(value)
    else:
        pred = jnp.zeros((ND,), dtype=jnp.float32)
    return h, out, m, m_tilde, pred


def test_forward_matches_hand_computation():
    slow, fast = _hand_weights()
    key = jax.random.key(0)
    s = jax.random.normal(key, (NN, ND), dtype=jnp.float32)
    e = jax.random.normal(jax.random.key(1), (NN, ND), dtype=jnp.float32)
    h_prev = jax.random.normal(jax.random.key(2), (NN, NH), dtype=jnp.float32)
    sleep = jnp.asarray([0.4], dtype=jnp.float32)
    noise = jax.random.normal(jax.random.key(3), (NN, NH), dtype=jnp.float32)

    result = base.forward(
        slow, fast, s, e, h_prev, sleep, noise, prediction_enabled=True
    )
    h, out, m, m_tilde, pred = _hand_forward(
        slow, fast, s, e, h_prev, sleep, noise, prediction_enabled=True
    )
    assert jnp.allclose(result.h[0], h, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(result.out[0], out, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(result.m[0], m, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(result.m_tilde[0], m_tilde, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(result.prediction[0], pred, rtol=1e-5, atol=1e-6)


def test_outputs_use_motor_order():
    slow, fast = _hand_weights()
    s = jnp.zeros((NN, ND), dtype=jnp.float32)
    e = jnp.zeros((NN, ND), dtype=jnp.float32)
    h_prev = jnp.zeros((NN, NH), dtype=jnp.float32)
    sleep = jnp.zeros((NN,), dtype=jnp.float32)
    noise = jnp.zeros((NN, NH), dtype=jnp.float32)
    result = base.forward(slow, fast, s, e, h_prev, sleep, noise, prediction_enabled=True)
    _, expected, _, _, _ = _hand_forward(
        slow, fast, s, e, h_prev, sleep, noise, prediction_enabled=True
    )
    # OUT_K (=2) is the sigmoid knob; the four remaining slots are tanh movements
    # in the order acc, turn, bite, grab, drop with k inserted at index 2.
    assert float(result.out[0, C.OUT_K]) == float(expected[C.OUT_K])
    assert 0.0 <= float(result.out[0, C.OUT_K]) <= 1.0
    for column in (C.OUT_ACC, C.OUT_TURN, C.OUT_BITE, C.OUT_GRAB, C.OUT_DROP):
        assert -1.0 <= float(result.out[0, column]) <= 1.0


def test_prediction_switch_silences_error_units_and_prediction():
    slow, fast = _hand_weights()
    s = jnp.full((NN, ND), 0.3, dtype=jnp.float32)
    e = jnp.full((NN, ND), 0.9, dtype=jnp.float32)
    h_prev = jnp.full((NN, NH), -0.2, dtype=jnp.float32)
    sleep = jnp.zeros((NN,), dtype=jnp.float32)
    noise = jnp.zeros((NN, NH), dtype=jnp.float32)

    on = base.forward(slow, fast, s, e, h_prev, sleep, noise, prediction_enabled=True)
    off = base.forward(slow, fast, s, e, h_prev, sleep, noise, prediction_enabled=False)
    assert not jnp.allclose(on.h, off.h)
    assert bool(jnp.all(off.prediction == 0))
    _, _, _, _, pred = _hand_forward(
        slow, fast, s, e, h_prev, sleep, noise, prediction_enabled=False
    )
    assert jnp.allclose(off.prediction[0], pred, rtol=1e-6, atol=1e-7)


# --- genome layout -----------------------------------------------------------

def _plastic_cfg(h: int = 2, modulators: int = 2, learning: LearningConfig | None = None) -> Config:
    """Small B0 config whose genome fits exactly with a little slack."""
    d = 70
    plastic_dim = h * d + h * h + C.N_OUTPUTS * h + d * h
    total = (
        core_genome.brain_base(4)
        + 2 * h * d
        + h * h
        + C.N_OUTPUTS * h
        + modulators * h
        + (h + C.N_OUTPUTS) * modulators
        + d * h
        + h
        + C.N_OUTPUTS
        + 2 * (h + C.N_OUTPUTS)
        + 12
        + 1
    )
    cfg = Config.default().replaced(
        capacities=Capacities(creature=4, bush=6, carcass=2, neutral=1),
        world=WorldConfig(founders=2, event_capacity=16),
        run=RunConfig(seed=3, ticks=2),
        brain=BrainConfig(
            kind="plastic_rnn",
            h=h,
            modulators=modulators,
            genome_dim=total + 64,
            plastic_dim=plastic_dim,
            motion_noise=0.05,
        ),
        learning=LearningConfig() if learning is None else learning,
    )
    assert brain_genome.layout(cfg).total <= cfg.brain.genome_dim
    return cfg


def test_layout_is_contiguous_and_fits():
    cfg = _plastic_cfg()
    lay = brain_genome.layout(cfg)
    assert lay.base == core_genome.brain_base(cfg.sensors.odor_channels)
    assert lay.total <= cfg.brain.genome_dim
    order = (
        "w_in",
        "w_err",
        "w_rec",
        "w_out",
        "w_mod",
        "g",
        "w_pred",
        "b",
        "b_out",
        "log_eta",
        "log_lambda",
        "hebb",
        "log_eta_p",
    )
    cursor = lay.base
    for name in order:
        block = getattr(lay, name)
        assert block.start == cursor
        cursor = block.stop
    assert cursor == lay.total
    sizes = lay.block_sizes()
    assert sizes["w_in"] == lay.h * lay.d
    assert sizes["w_err"] == lay.h * lay.d
    assert sizes["w_rec"] == lay.h * lay.h
    assert sizes["w_out"] == lay.n_out * lay.h
    assert sizes["w_mod"] == lay.m * lay.h
    assert sizes["g"] == lay.post * lay.m
    assert sizes["w_pred"] == lay.d * lay.h
    assert sizes["log_eta"] == lay.post
    assert sizes["log_lambda"] == lay.post
    assert sizes["hebb"] == 12
    assert sizes["log_eta_p"] == 1
    assert lay.plastic_dim == cfg.brain.plastic_dim


def test_decode_slow_reads_each_documented_block():
    cfg = _plastic_cfg()
    lay = brain_genome.layout(cfg)
    n = 2
    filled = jnp.zeros((n, cfg.brain.genome_dim), dtype=jnp.float32)
    values = {
        "w_in": 0.5,
        "w_err": -0.25,
        "w_rec": 0.75,
        "w_out": 0.125,
        "w_mod": -0.5,
        "g": 0.375,
        "w_pred": 0.625,
        "b": -0.2,
        "b_out": 0.3,
    }
    for name, value in values.items():
        filled = filled.at[:, getattr(lay, name)].set(value)
    slow = brain_genome.decode_slow(cfg, filled)
    assert slow.w_in.shape == (n, lay.h, lay.d)
    assert slow.w_err.shape == (n, lay.h, lay.d)
    assert slow.w_rec.shape == (n, lay.h, lay.h)
    assert slow.w_out.shape == (n, lay.n_out, lay.h)
    assert slow.w_mod.shape == (n, lay.m, lay.h)
    assert slow.g.shape == (n, lay.post, lay.m)
    assert slow.w_pred.shape == (n, lay.d, lay.h)
    assert slow.b.shape == (n, lay.h)
    assert slow.b_out.shape == (n, lay.n_out)
    for name, value in values.items():
        assert bool(jnp.all(getattr(slow, name) == value))


def test_decode_plastic_uses_logs():
    cfg = _plastic_cfg()
    lay = brain_genome.layout(cfg)
    n = 2
    genome = jnp.zeros((n, cfg.brain.genome_dim), dtype=jnp.float32)
    genome = genome.at[:, lay.log_eta].set(-1.0)
    genome = genome.at[:, lay.log_lambda].set(-2.0)
    genome = genome.at[:, lay.log_eta_p].set(-3.0)
    coefs = _grid(n, 3, 4, scale=0.1, offset=-0.2)
    genome = genome.at[:, lay.hebb].set(coefs.reshape(n, 12))
    plastic = brain_genome.decode_plastic(cfg, genome)
    assert plastic.eta.shape == (n, lay.post)
    assert plastic.lam.shape == (n, lay.post)
    assert bool(jnp.allclose(plastic.eta, jnp.exp(jnp.full((n, lay.post), -1.0))))
    assert bool(jnp.allclose(plastic.lam, jnp.exp(jnp.full((n, lay.post), -2.0))))
    assert bool(jnp.allclose(plastic.eta_p, jnp.exp(jnp.full((n,), -3.0))))
    assert bool(jnp.allclose(plastic.hebb, coefs))


def test_fast_weight_split_join_roundtrip():
    cfg = _plastic_cfg()
    lay = brain_genome.layout(cfg)
    n = 3
    w_fast = jax.random.normal(jax.random.key(0), (n, cfg.brain.plastic_dim), dtype=jnp.float32)
    fast = brain_genome.split_fast(w_fast, lay)
    assert fast.w_in.shape == (n, lay.h, lay.d)
    assert fast.w_rec.shape == (n, lay.h, lay.h)
    assert fast.w_out.shape == (n, lay.n_out, lay.h)
    assert fast.w_pred.shape == (n, lay.d, lay.h)
    assert bool(jnp.array_equal(brain_genome.join_fast(fast, lay), w_fast))


def test_mutation_sigmas_per_block():
    cfg = _plastic_cfg()
    lay = brain_genome.layout(cfg)
    sigma = brain_genome.mutation_sigmas(cfg)
    assert sigma.shape == (cfg.brain.genome_dim,)
    assert sigma.dtype == jnp.float32
    odor = int(cfg.sensors.odor_channels)
    assert bool(jnp.all(sigma[core_genome.diet_logits_slice()] == 0.05))
    assert bool(jnp.all(sigma[core_genome.color_slice()] == 0.05))
    assert bool(jnp.all(sigma[core_genome.odor_slice(odor)] == 0.05))
    assert bool(jnp.isclose(sigma[core_genome.theta_repro_index(odor)], 0.02))
    # Weight-like brain blocks mutate at the weight scale (SANDBOX 6.3).
    for name in ("w_in", "w_err", "w_rec", "w_out", "w_mod", "g", "w_pred", "b", "b_out", "hebb"):
        assert bool(jnp.all(sigma[getattr(lay, name)] == 0.02))
    # Log rates mutate at 0.1.
    for name in ("log_eta", "log_lambda", "log_eta_p"):
        assert bool(jnp.all(sigma[getattr(lay, name)] == 0.1))
    # Slots beyond the layout keep the default weight scale.
    if lay.total < cfg.brain.genome_dim:
        assert bool(jnp.all(sigma[lay.total :] == 0.02))


# --- think wiring ------------------------------------------------------------

def test_think_writes_b0_forward():
    cfg = _plastic_cfg()
    state = initial_state(cfg)
    creature = state.entities.creature
    lay = brain_genome.layout(cfg, slice_dim=int(state.slice.shape[1]))
    slow = brain_genome.decode_slow(cfg, creature.genome.g, lay)
    fast = brain_genome.split_fast(creature.brain.w_fast, lay)
    e = state.slice - creature.brain.s_hat
    noise = jax.random.normal(
        rng.system_key(state.key, state.tick, "think"),
        creature.brain.h.shape,
        dtype=jnp.float32,
    )
    expected = base.forward(
        slow,
        fast,
        state.slice,
        e,
        creature.brain.h,
        creature.body.S,
        noise,
        prediction_enabled=cfg.learning.prediction,
    )

    after, buffer = think.apply(state, cfg)
    assert int(buffer.count) == 0
    out_creature = after.entities.creature
    assert bool(jnp.array_equal(out_creature.brain.h, expected.h))
    assert bool(jnp.array_equal(out_creature.brain.s_hat, expected.prediction))
    assert bool(jnp.array_equal(out_creature.brain.last_pre, e))
    assert bool(jnp.array_equal(out_creature.brain.last_post, creature.brain.h))
    assert bool(jnp.array_equal(out_creature.motor.last_out, expected.out))


def test_think_prediction_off_keeps_hat_zero():
    cfg = _plastic_cfg(learning=LearningConfig(prediction=False, plasticity=True))
    state = initial_state(cfg)
    after, _ = think.apply(state, cfg)
    assert bool(jnp.all(after.entities.creature.brain.s_hat == 0))


def test_default_size_think_learn_traces_under_jit():
    # Production runner shape: default H=32 / D=70 / genome_dim=8400 in a scan
    # over a small arena; only the capability switch differs from the M0 arm.
    cfg = Config.default().replaced(
        capacities=Capacities(creature=4, bush=6, carcass=2, neutral=1),
        world=WorldConfig(founders=2, event_capacity=16),
        run=RunConfig(seed=3, ticks=2),
        brain=BrainConfig(kind="plastic_rnn"),
    )
    state = initial_state(cfg)

    def two_ticks(st):
        st, _ = think.apply(st, cfg)
        st, _ = learn.apply(st, cfg)
        return st

    out = jax.jit(two_ticks)(state)
    assert out.entities.creature.brain.h.shape == state.entities.creature.brain.h.shape
    assert bool(jnp.all(jnp.isfinite(out.entities.creature.brain.h)))
    assert bool(jnp.all(jnp.isfinite(out.entities.creature.brain.w_fast)))
    assert bool(jnp.all(jnp.isfinite(out.entities.creature.motor.last_out)))


def test_think_and_learn_compile_under_jit_and_scan():
    cfg = _plastic_cfg()
    state = initial_state(cfg)

    def two_ticks(st):
        st, _ = think.apply(st, cfg)
        st, _ = learn.apply(st, cfg)
        return st

    jitted = jax.jit(two_ticks)
    first = jitted(state)
    second = jitted(state)
    assert state_hash(first) == state_hash(second)
    assert bool(jnp.all(jnp.isfinite(first.entities.creature.brain.h)))
    assert int(first.tick) == int(state.tick)

    def body(st, _):
        return think.apply(st, cfg)[0], None

    scanned, _ = jax.lax.scan(body, state, None, length=2)
    assert scanned.entities.creature.brain.h.shape == state.entities.creature.brain.h.shape
    assert bool(jnp.all(jnp.isfinite(scanned.entities.creature.brain.h)))
