"""Initial genome scales (C4 weight-init table, moved before M1; SANDBOX 6.3).

initial_sigmas(cfg) is the per-slot standard deviation world/init draws founder
genomes with.  These tests pin the fan-in formulas per block, the derived array
width, the dependence on H/D/M, and the M1 symptom the table fixes: a
fixed-weight B0 on a real initial_state no longer produces near-constant body
outputs.
"""

from __future__ import annotations

import math

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
    Sensors,
    WorldConfig,
)
from life_sandbox.core import components as C
from life_sandbox.core.struct import update
from life_sandbox.core.systems import think
from life_sandbox.world.init import initial_state


def _small_cfg(h: int = 8, modulators: int = 4) -> Config:
    """Small arena with a B0-sized genome but tiny capacities (fast sampling)."""
    return Config.default().replaced(
        capacities=Capacities(creature=4, bush=6, carcass=2, neutral=1),
        world=WorldConfig(founders=2, event_capacity=16),
        run=RunConfig(seed=3, ticks=2),
        brain=BrainConfig(kind="plastic_rnn", h=h, modulators=modulators, motion_noise=0.05),
        learning=LearningConfig(plasticity=False, prediction=False),
    )


def _formulas(lay) -> dict[str, float]:
    """Expected standard deviation per genome block (C4, SANDBOX 6.3)."""
    d_std = 1.0 / math.sqrt(lay.d)
    h_std = 1.0 / math.sqrt(lay.h)
    return {
        "w_in": d_std,
        "w_err": d_std,
        "w_rec": h_std,
        "w_out": h_std,
        "w_mod": h_std,
        "w_pred": h_std,
        "g": 1.0 / math.sqrt(lay.m),
        "b": 0.1,
        "b_out": 1.0,
        "log_eta": 0.1,
        "log_lambda": 0.1,
        "log_eta_p": 0.1,
        "hebb": 0.1,
    }


def test_shape_dtype_and_width_are_derived():
    cfg = Config.default()
    lay = brain_genome.layout(cfg)
    sigma = brain_genome.initial_sigmas(cfg)
    assert sigma.shape == (lay.total,)
    assert sigma.shape == (lay.genome_dim,)
    assert sigma.dtype == jnp.float32
    # Shared header slots are left at 0.1; world/init fills them.
    assert bool(jnp.all(sigma[: lay.base] == 0.1))


def test_block_values_match_fan_in_formulas():
    cfg = Config.default()
    lay = brain_genome.layout(cfg)
    sigma = brain_genome.initial_sigmas(cfg)
    for name, expected in _formulas(lay).items():
        block = getattr(lay, name)
        assert int(block.stop - block.start) > 0
        assert bool(jnp.allclose(sigma[block], expected, rtol=1e-6, atol=0.0)), name


def test_sampled_block_std_matches_formula():
    cfg = _small_cfg()
    lay = brain_genome.layout(cfg)
    sigma = brain_genome.initial_sigmas(cfg)
    n = 1024
    sample = jax.random.normal(jax.random.key(0), (n, lay.total), dtype=jnp.float32)
    sample = sample * sigma[None, :]
    for name, expected in _formulas(lay).items():
        values = sample[:, getattr(lay, name)]
        measured = float(jnp.std(values))
        assert measured == pytest.approx(expected, rel=0.05), name


def test_scales_follow_h_d_and_m():
    base_cfg = Config.default()
    base_lay = brain_genome.layout(base_cfg)
    base_sigma = brain_genome.initial_sigmas(base_cfg)

    wide_cfg = base_cfg.replaced(sensors=Sensors(sound_channels=2))  # D = 74
    assert wide_cfg.sensors.slice_dim == 74
    wide_lay = brain_genome.layout(wide_cfg)
    wide_sigma = brain_genome.initial_sigmas(wide_cfg)

    narrow_cfg = base_cfg.replaced(brain=BrainConfig(h=16, modulators=8))
    narrow_lay = brain_genome.layout(narrow_cfg)
    narrow_sigma = brain_genome.initial_sigmas(narrow_cfg)

    assert wide_sigma.shape == (wide_lay.total,)
    assert narrow_sigma.shape == (narrow_lay.total,)
    assert wide_lay.total != base_lay.total
    assert narrow_lay.total != base_lay.total

    # W_in/W_err scale with D, W_rec/W_out/W_mod/W_pred with H, G with M.
    assert float(base_sigma[base_lay.w_in][0]) == pytest.approx(1.0 / math.sqrt(70), rel=1e-6)
    assert float(wide_sigma[wide_lay.w_in][0]) == pytest.approx(1.0 / math.sqrt(74), rel=1e-6)
    assert float(wide_sigma[wide_lay.w_rec][0]) == pytest.approx(1.0 / math.sqrt(32), rel=1e-6)
    assert float(narrow_sigma[narrow_lay.w_in][0]) == pytest.approx(1.0 / math.sqrt(70), rel=1e-6)
    assert float(narrow_sigma[narrow_lay.w_rec][0]) == pytest.approx(1.0 / math.sqrt(16), rel=1e-6)
    assert float(narrow_sigma[narrow_lay.g][0]) == pytest.approx(1.0 / math.sqrt(8), rel=1e-6)
    assert float(narrow_sigma[narrow_lay.b][0]) == pytest.approx(0.1, rel=1e-6)
    assert float(narrow_sigma[narrow_lay.b_out][0]) == pytest.approx(1.0, rel=1e-6)


def test_fixed_arm_outputs_are_not_constant():
    # M1 symptom: with the old 0.1-everywhere genome, a fixed-weight B0 barely
    # moves its outputs.  The C4 table must give a real spread of body outputs
    # across founders on a real initial_state.
    cfg = Config.default().replaced(
        brain=BrainConfig(kind="plastic_rnn"),
        learning=LearningConfig(plasticity=False, prediction=False),
    )
    state = initial_state(cfg)
    after, buffer = think.apply(state, cfg)
    assert int(buffer.count) == 0

    alive = state.entities.creature.alive
    outputs = after.entities.creature.motor.last_out[alive]
    assert int(alive.sum()) == cfg.world.founders
    acc = outputs[:, C.OUT_ACC]
    bite = outputs[:, C.OUT_BITE]
    assert bool(jnp.all(jnp.isfinite(outputs)))
    # Cross-individual variance: different genomes -> different behaviour.
    assert float(jnp.std(acc)) > 1e-3
    assert float(jnp.std(bite)) > 1e-3
    # Not all clustered at zero: the outputs actually drive a body.
    assert float(jnp.mean(jnp.abs(acc))) > 1e-2
    assert float(jnp.mean(jnp.abs(bite))) > 1e-2
    assert float(jnp.max(jnp.abs(acc))) > 0.05

    # Also exercise the actual C4 draw explicitly (world/init's wiring is the
    # Lead's task; this keeps the table verified end to end either way).
    lay = brain_genome.layout(cfg)
    sigma = brain_genome.initial_sigmas(cfg)
    c4_genome = (
        jax.random.normal(
            jax.random.key(123),
            (state.entities.creature.genome.g.shape[0], lay.genome_dim),
            dtype=jnp.float32,
        )
        * sigma[None, :]
    )
    creature = state.entities.creature
    c4_creature = update(creature, genome=update(creature.genome, g=c4_genome))
    c4_state = update(state, entities=update(state.entities, creature=c4_creature))
    c4_after, _ = think.apply(c4_state, cfg)
    c4_outputs = c4_after.entities.creature.motor.last_out[alive]
    assert float(jnp.std(c4_outputs[:, C.OUT_ACC])) > 1e-3
    assert float(jnp.mean(jnp.abs(c4_outputs[:, C.OUT_ACC]))) > 1e-2
    assert float(jnp.std(c4_outputs[:, C.OUT_BITE])) > 1e-3
