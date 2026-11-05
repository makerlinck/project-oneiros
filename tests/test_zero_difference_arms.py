"""Zero-difference checks for the learning capability switches (SANDBOX 6.2, 9; PRINCIPLES 8.3).

SANDBOX 6.2: "turning on the Hebbian code path but forcing eta to 0 must be
bit-identical to turning plasticity off".  Both comparisons run the real think+learn
systems on the same initial state and compare the full state hash, so any stray
state write (fast weights, last_pre/last_post, s_hat) shows up.

Forcing eta to 0: decode_plastic() is monkeypatched to return eta zeros; that is
the only difference between the two runs besides cfg.learning.plasticity.  The
prediction error itself (e_t) is identical by construction: think stores it and
learn reuses the same array.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

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
from life_sandbox.core.hashing import state_hash
from life_sandbox.core.systems import learn, think
from life_sandbox.world.init import initial_state


def _plastic_cfg(learning: LearningConfig) -> Config:
    h, modulators, d = 2, 2, 70
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
        learning=learning,
    )
    assert brain_genome.layout(cfg).total <= cfg.brain.genome_dim
    return cfg


def _ticks(state, cfg, ticks=1):
    for _ in range(ticks):
        state, _ = think.apply(state, cfg)
        state, _ = learn.apply(state, cfg)
        state = dataclasses.replace(
            state, tick=(state.tick + jnp.asarray(1, dtype=jnp.int32)).astype(jnp.int32)
        )
    return state


def _force_zero_eta(monkeypatch):
    original = brain_genome.decode_plastic

    def forced(cfg, genome, lay=None):
        params = original(cfg, genome, lay)
        return dataclasses.replace(params, eta=jnp.zeros_like(params.eta))

    monkeypatch.setattr(brain_genome, "decode_plastic", forced)


def test_same_config_is_bit_identical():
    cfg = _plastic_cfg(LearningConfig(plasticity=True, prediction=True))
    assert state_hash(_ticks(initial_state(cfg), cfg, ticks=2)) == state_hash(
        _ticks(initial_state(cfg), cfg, ticks=2)
    )


def test_zero_eta_with_prediction_matches_plasticity_off(monkeypatch):
    cfg_on = _plastic_cfg(LearningConfig(plasticity=True, prediction=True))
    cfg_off = _plastic_cfg(LearningConfig(plasticity=False, prediction=True))
    # Arms differ only in the capability switch: the initial states are the same.
    assert state_hash(initial_state(cfg_on)) == state_hash(initial_state(cfg_off))

    _force_zero_eta(monkeypatch)

    on_state = _ticks(initial_state(cfg_on), cfg_on, ticks=2)
    off_state = _ticks(initial_state(cfg_off), cfg_off, ticks=2)
    assert state_hash(on_state) == state_hash(off_state)
    # The prediction delta rule still runs (prediction is on), but the Hebbian
    # matrices must stay at +0.0: no eta, no drift.
    lay = brain_genome.layout(cfg_on, slice_dim=int(on_state.slice.shape[1]))
    on_fast = brain_genome.split_fast(on_state.entities.creature.brain.w_fast, lay)
    assert bool(jnp.all(on_fast.w_in == 0))
    assert bool(jnp.all(on_fast.w_rec == 0))
    assert bool(jnp.all(on_fast.w_out == 0))


def test_zero_eta_without_prediction_matches_plasticity_off(monkeypatch):
    cfg_on = _plastic_cfg(LearningConfig(plasticity=True, prediction=False))
    cfg_off = _plastic_cfg(LearningConfig(plasticity=False, prediction=False))
    assert state_hash(initial_state(cfg_on)) == state_hash(initial_state(cfg_off))

    _force_zero_eta(monkeypatch)

    on_state = _ticks(initial_state(cfg_on), cfg_on, ticks=2)
    off_state = _ticks(initial_state(cfg_off), cfg_off, ticks=2)
    assert state_hash(on_state) == state_hash(off_state)
    # Prediction switch off: s_hat stays zero in both arms.
    assert bool(jnp.all(on_state.entities.creature.brain.s_hat == 0))


def test_prediction_switch_alone_changes_the_state():
    """Control: the zero-difference tests are not vacuous."""
    cfg_on = _plastic_cfg(LearningConfig(plasticity=True, prediction=True))
    cfg_off = _plastic_cfg(LearningConfig(plasticity=True, prediction=False))
    assert state_hash(initial_state(cfg_on)) == state_hash(initial_state(cfg_off))
    assert state_hash(_ticks(initial_state(cfg_on), cfg_on)) != state_hash(
        _ticks(initial_state(cfg_off), cfg_off)
    )
