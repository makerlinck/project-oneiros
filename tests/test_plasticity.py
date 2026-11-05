"""Modulated Hebbian rule, prediction delta rule and learn wiring (SANDBOX 6.1).

The two rules are checked term by term against a hand-written loop, then learn
is run end to end on a state whose genome blocks are constants, so the new fast
weights can be predicted from the same formulas without decoding the genome.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

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
from life_sandbox.core.struct import update
from life_sandbox.core.systems import learn, plasticity, prediction
from life_sandbox.world.init import initial_state


def _grid(*shape: int, scale: float, offset: float) -> jax.Array:
    count = 1
    for dim in shape:
        count *= int(dim)
    flat = jnp.arange(count, dtype=jnp.float32) * jnp.float32(scale) + jnp.float32(offset)
    return flat.reshape(shape).astype(jnp.float32)


def test_hebbian_update_matches_formula():
    n, q, p = 2, 3, 4
    weight = _grid(n, q, p, scale=0.1, offset=-0.2)
    pre = _grid(n, p, scale=0.1, offset=-0.3)
    post = _grid(n, q, scale=0.05, offset=0.1)
    eta = _grid(n, q, scale=0.1, offset=0.2)
    lam = _grid(n, q, scale=0.01, offset=0.05)
    modulation = _grid(n, q, scale=0.2, offset=-0.4)
    a = _grid(n, scale=0.1, offset=0.5)
    b = _grid(n, scale=0.1, offset=-0.3)
    c = _grid(n, scale=0.1, offset=0.2)
    d = _grid(n, scale=0.1, offset=0.05)

    got = plasticity.hebbian_update(weight, pre, post, eta, lam, modulation, a, b, c, d)

    expected = weight
    for i in range(n):
        for j in range(q):
            for k in range(p):
                rule = (
                    a[i] * pre[i, k] * post[i, j]
                    + b[i] * pre[i, k]
                    + c[i] * post[i, j]
                    + d[i]
                )
                delta = eta[i, j] * modulation[i, j] * rule - lam[i, j] * weight[i, j, k]
                expected = expected.at[i, j, k].set(jnp.clip(weight[i, j, k] + delta, -1.0, 1.0))
    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_hebbian_update_clips_to_fast_weight_box():
    n, q, p = 1, 2, 2
    weight = jnp.full((n, q, p), 0.9, dtype=jnp.float32)
    pre = jnp.ones((n, p), dtype=jnp.float32)
    post = jnp.ones((n, q), dtype=jnp.float32)
    eta = jnp.full((n, q), 10.0, dtype=jnp.float32)
    lam = jnp.zeros((n, q), dtype=jnp.float32)
    modulation = jnp.ones((n, q), dtype=jnp.float32)
    one = jnp.ones((n,), dtype=jnp.float32)
    up = plasticity.hebbian_update(weight, pre, post, eta, lam, modulation, one, one, one, one)
    assert bool(jnp.all(up == 1.0))

    minus_one = -one
    zero = jnp.zeros((n,), dtype=jnp.float32)
    down = plasticity.hebbian_update(
        -weight, pre, post, eta, lam, modulation, minus_one, zero, zero, zero
    )
    assert bool(jnp.all(down == -1.0))


def test_zero_eta_reduces_to_decay_only():
    n, q, p = 2, 3, 2
    weight = _grid(n, q, p, scale=0.2, offset=-0.5)
    pre = _grid(n, p, scale=0.3, offset=0.1)
    post = _grid(n, q, scale=0.2, offset=-0.1)
    modulation = _grid(n, q, scale=0.5, offset=0.5)
    lam = _grid(n, q, scale=0.1, offset=0.1)
    zero = jnp.zeros((n, q), dtype=jnp.float32)
    one = jnp.ones((n,), dtype=jnp.float32)
    got = plasticity.hebbian_update(weight, pre, post, zero, lam, modulation, one, one, one, one)
    expected = jnp.clip(weight - lam[:, :, None] * weight, -1.0, 1.0)
    assert jnp.allclose(got, expected, rtol=1e-6, atol=1e-7)


def test_aggregate_rates_per_unit_identity():
    eta = _grid(2, 8, scale=0.1, offset=0.2)
    lam = _grid(2, 8, scale=0.01, offset=0.05)
    eta_out, lam_out = plasticity.aggregate_rates(eta, lam, 6, "per_unit")
    assert bool(jnp.array_equal(eta_out, eta))
    assert bool(jnp.array_equal(lam_out, lam))


def test_aggregate_rates_per_group_means_blocks():
    eta = _grid(2, 8, scale=0.1, offset=0.2)
    lam = _grid(2, 8, scale=0.01, offset=0.05)
    eta_out, lam_out = plasticity.aggregate_rates(eta, lam, 6, "per_group")
    for i in range(2):
        for column in range(6):
            assert float(eta_out[i, column]) == pytest.approx(float(jnp.mean(eta[i, :6])), rel=1e-6)
            assert float(lam_out[i, column]) == pytest.approx(float(jnp.mean(lam[i, :6])), rel=1e-6)
        for column in range(6, 8):
            assert float(eta_out[i, column]) == pytest.approx(float(jnp.mean(eta[i, 6:])), rel=1e-6)
            assert float(lam_out[i, column]) == pytest.approx(float(jnp.mean(lam[i, 6:])), rel=1e-6)


def test_aggregate_rates_shared_is_one_value():
    eta = _grid(2, 8, scale=0.1, offset=0.2)
    lam = _grid(2, 8, scale=0.01, offset=0.05)
    eta_out, lam_out = plasticity.aggregate_rates(eta, lam, 6, "shared")
    for i in range(2):
        assert bool(jnp.all(eta_out[i] == jnp.mean(eta[i])))
        assert bool(jnp.all(lam_out[i] == jnp.mean(lam[i])))


def test_aggregate_rates_rejects_unknown_granularity():
    eta = jnp.ones((2, 8), dtype=jnp.float32)
    lam = jnp.ones((2, 8), dtype=jnp.float32)
    with pytest.raises(ValueError):
        plasticity.aggregate_rates(eta, lam, 6, "per_synapse")


def test_prediction_error_and_delta_rule():
    n, d, h = 2, 3, 2
    s = _grid(n, d, scale=0.2, offset=-0.2)
    s_hat = _grid(n, d, scale=0.05, offset=0.1)
    e = prediction.error(s, s_hat)
    assert bool(jnp.array_equal(e, s - s_hat))

    weight = _grid(n, d, h, scale=0.1, offset=-0.1)
    h_prev = _grid(n, h, scale=0.2, offset=0.3)
    eta_p = jnp.asarray([0.5, 0.25], dtype=jnp.float32)
    got = prediction.delta_update(weight, e, h_prev, eta_p)
    expected = weight
    for i in range(n):
        for k in range(d):
            for j in range(h):
                value = weight[i, k, j] + eta_p[i] * e[i, k] * h_prev[i, j]
                expected = expected.at[i, k, j].set(jnp.clip(value, -1.0, 1.0))
    assert jnp.allclose(got, expected, rtol=1e-6, atol=1e-7)


def test_prediction_delta_rule_clips():
    n, d, h = 1, 2, 1
    weight = jnp.full((n, d, h), 0.5, dtype=jnp.float32)
    e = jnp.full((n, d), 10.0, dtype=jnp.float32)
    h_prev = jnp.full((n, h), 10.0, dtype=jnp.float32)
    eta_p = jnp.ones((n,), dtype=jnp.float32)
    got = prediction.delta_update(weight, e, h_prev, eta_p)
    assert bool(jnp.all(got == 1.0))


# --- learn system end to end -------------------------------------------------

def _plastic_cfg(h: int = 2, modulators: int = 2, learning: LearningConfig | None = None) -> Config:
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


def test_learn_step_matches_rule_with_constant_genome():
    cfg = _plastic_cfg(learning=LearningConfig(plasticity=True, prediction=True))
    state = initial_state(cfg)
    lay = brain_genome.layout(cfg, slice_dim=int(state.slice.shape[1]))
    n = cfg.capacities.creature
    d = lay.d
    h_units = lay.h

    # Constants: W_mod = 0.5, G = 0.25, eta = 0.5, lambda = 0.1, eta_p = 0.7,
    # and one distinct A-D row per Hebbian group.
    coefs = jnp.asarray(
        [
            [0.5, -0.25, 0.3, 0.1],
            [-0.4, 0.2, 0.15, -0.05],
            [0.25, 0.1, -0.2, 0.05],
        ],
        dtype=jnp.float32,
    )
    genome = jnp.zeros((n, cfg.brain.genome_dim), dtype=jnp.float32)
    genome = genome.at[:, lay.w_mod].set(0.5)
    genome = genome.at[:, lay.g].set(0.25)
    genome = genome.at[:, lay.log_eta].set(jnp.log(0.5))
    genome = genome.at[:, lay.log_lambda].set(jnp.log(0.1))
    genome = genome.at[:, lay.log_eta_p].set(jnp.log(0.7))
    genome = genome.at[:, lay.hebb].set(
        jnp.broadcast_to(coefs.reshape(1, 12), (n, 12))
    )

    h = jnp.tile(jnp.asarray([[0.3, -0.2]], dtype=jnp.float32), (n, 1))
    h_prev = jnp.tile(jnp.asarray([[0.1, 0.4]], dtype=jnp.float32), (n, 1))
    e = jnp.tile(_grid(1, d, scale=0.001, offset=0.002), (n, 1))
    s = jnp.tile(_grid(1, d, scale=0.01, offset=-0.3), (n, 1))
    out = jnp.tile(
        jnp.asarray([[0.2, -0.3, 0.5, -0.1, 0.15, 0.05]], dtype=jnp.float32), (n, 1)
    )

    creature = state.entities.creature
    creature = update(creature, genome=update(creature.genome, g=genome))
    creature = update(
        creature,
        brain=update(creature.brain, h=h, last_pre=e, last_post=h_prev),
        motor=update(creature.motor, last_out=out),
    )
    state = update(state, entities=update(state.entities, creature=creature), slice=s)

    after, buffer = learn.apply(state, cfg)
    assert int(buffer.count) == 0
    fast = brain_genome.split_fast(after.entities.creature.brain.w_fast, lay)

    m = jnp.tanh(0.5 * jnp.sum(h, axis=1))
    # G = 0.25 for both modulators, so m~_j = sum_k 0.25 * m_k = 0.5 * m.
    mod_tilde = jnp.broadcast_to((0.5 * m)[:, None], (n, lay.post)).astype(jnp.float32)
    mod_hidden = mod_tilde[:, :h_units]
    mod_out = mod_tilde[:, h_units:]
    eta = 0.5
    lam = 0.1

    def hebb(pre, post, mod, coef_row):
        rule = (
            coef_row[0] * pre[:, None, :] * post[:, :, None]
            + coef_row[1] * pre[:, None, :]
            + coef_row[2] * post[:, :, None]
            + coef_row[3]
        )
        return jnp.clip(eta * mod[:, :, None] * rule, -1.0, 1.0)

    expected_in = hebb(s, h, mod_hidden, coefs[0])
    expected_rec = hebb(h_prev, h, mod_hidden, coefs[1])
    expected_out = hebb(h, out, mod_out, coefs[2])
    expected_pred = jnp.clip(0.7 * e[:, :, None] * h_prev[:, None, :], -1.0, 1.0)

    assert jnp.allclose(fast.w_in, expected_in, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(fast.w_rec, expected_rec, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(fast.w_out, expected_out, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(fast.w_pred, expected_pred, rtol=1e-5, atol=1e-6)


def test_learn_disabled_leaves_fast_weights_untouched():
    cfg = _plastic_cfg(learning=LearningConfig(plasticity=False, prediction=False))
    state = initial_state(cfg)
    after, _ = learn.apply(state, cfg)
    assert bool(jnp.array_equal(after.entities.creature.brain.w_fast, state.entities.creature.brain.w_fast))
