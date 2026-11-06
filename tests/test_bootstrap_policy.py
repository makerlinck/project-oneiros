"""Bootstrap random policy for M0 (SANDBOX.md 9).

The bootstrap policy is NOT a candidate brain: it only drives bodies with random
outputs so the engine can run in M0.  These tests pin down its contract (six
outputs in Motor.last_out order, correct ranges, deterministic key use) and the
fact that the think system writes only Motor.last_out on that path.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from life_sandbox.brain import base
from life_sandbox.core import components as C, rng
from life_sandbox.core.hashing import state_hash
from life_sandbox.core.systems import learn, think
from life_sandbox.core.systems.think import KIND_BOOTSTRAP
from life_sandbox.world.init import initial_state

MOVEMENT_COLUMNS = (C.OUT_ACC, C.OUT_TURN, C.OUT_BITE, C.OUT_GRAB, C.OUT_DROP)


def _bootstrap_cfg(tiny_cfg):
    """tiny_cfg with the architecture switch set explicitly (M0 random drive)."""
    return tiny_cfg.replaced(brain=dataclasses.replace(tiny_cfg.brain, kind=KIND_BOOTSTRAP))


def test_shape_dtype_and_order():
    out = base.bootstrap_outputs(jax.random.key(7), 5)
    assert out.shape == (5, C.N_OUTPUTS)
    assert out.dtype == jnp.float32
    # k is the third column of Motor.last_out (components.OUT_K == 2).
    assert C.OUT_K == 2


def test_ranges_and_coverage():
    out = base.bootstrap_outputs(jax.random.key(11), 4096)
    movement = out[:, list(MOVEMENT_COLUMNS)]
    knob = out[:, C.OUT_K]
    # Range contract: movement outputs in [-1, 1), k in [0, 1).
    assert float(movement.min()) >= -1.0
    assert float(movement.max()) < 1.0
    assert float(knob.min()) >= 0.0
    assert float(knob.max()) < 1.0
    # Coverage: with 4096 draws every movement column spans most of the range.
    assert float(movement.min()) < -0.99
    assert float(movement.max()) > 0.99
    assert float(knob.min()) < 0.01
    assert float(knob.max()) > 0.99


def test_key_determinism_and_stream_independence():
    key = jax.random.key(3)
    a = base.bootstrap_outputs(key, 16)
    b = base.bootstrap_outputs(key, 16)
    assert bool(jnp.array_equal(a, b))
    c = base.bootstrap_outputs(jax.random.key(4), 16)
    assert not bool(jnp.array_equal(a, c))
    # The knob column is drawn from its own sub-key, so it is not a shift of a
    # movement column.
    for column in MOVEMENT_COLUMNS:
        assert not bool(jnp.array_equal(a[:, column], a[:, C.OUT_K]))


def test_think_bootstrap_writes_only_motor(tiny_cfg):
    cfg = _bootstrap_cfg(tiny_cfg)
    state = initial_state(cfg)
    after, buffer = think.apply(state, cfg)
    assert int(buffer.count) == 0

    expected = base.bootstrap_outputs(
        rng.system_key(state.key, state.tick, "think"),
        state.entities.creature.motor.last_out.shape[0],
    )
    assert bool(jnp.array_equal(after.entities.creature.motor.last_out, expected))
    # Brain state is untouched: h, s_hat, w_fast stay at their initialized zeros.
    assert bool(jnp.all(after.entities.creature.brain.h == 0))
    assert bool(jnp.all(after.entities.creature.brain.s_hat == 0))
    assert bool(jnp.all(after.entities.creature.brain.w_fast == 0))
    assert bool(jnp.all(after.entities.creature.brain.e == 0))
    assert bool(jnp.all(after.entities.creature.brain.h_prev == 0))


def test_bootstrap_does_not_learn(tiny_cfg):
    cfg = _bootstrap_cfg(tiny_cfg)
    state = initial_state(cfg)
    after_think, _ = think.apply(state, cfg)
    after_learn, buffer = learn.apply(after_think, cfg)
    assert int(buffer.count) == 0
    assert state_hash(after_learn) == state_hash(after_think)


def test_bootstrap_two_ticks_are_deterministic(tiny_cfg):
    cfg = _bootstrap_cfg(tiny_cfg)

    def two_ticks():
        state = initial_state(cfg)
        for _ in range(2):
            state, _ = think.apply(state, cfg)
            state, _ = learn.apply(state, cfg)
        return state

    assert state_hash(two_ticks()) == state_hash(two_ticks())
