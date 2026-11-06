"""Odour field tests (SANDBOX.md §4.3, §9, §13 test_field_odor).

Covers: mass conservation without decay, the analytic decay factor, diffusion
kernel behaviour, fixed-point deposits (per source, order independence,
inactive entities), bilinear sampling and the analytic temperature/light
helpers.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from life_sandbox.config import Capacities, Config, RunConfig, WorldConfig
from life_sandbox.core import rng
from life_sandbox.core.state import empty
from life_sandbox.core.struct import update
from life_sandbox.world import fields

CHANNELS = 4


def make_cfg(**world_overrides) -> Config:
    world = dataclasses.replace(
        WorldConfig(width=8.0, height=8.0, cell=1.0), **world_overrides
    )
    return Config.default().replaced(
        world=world,
        capacities=Capacities(creature=2, bush=3, carcass=2, neutral=1),
        run=RunConfig(seed=3, ticks=0),
    )


def state_of(cfg: Config):
    return empty(cfg, rng.world_key(3))


def _put_creature(state, slot, pos, *, radius=0.5, odor=None):
    arena = state.entities.creature
    if odor is not None:
        arena = update(
            arena,
            look=update(
                arena.look,
                odor=arena.look.odor.at[slot].set(jnp.asarray(odor, jnp.float32)),
            ),
        )
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(True),
        transform=update(
            arena.transform, pos=arena.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32))
        ),
        shape=update(arena.shape, radius=arena.shape.radius.at[slot].set(radius)),
    )
    return update(state, entities=update(state.entities, creature=arena))


def _put_bush(state, slot, pos, *, stock=0.0, kind=0, radius=1.0):
    arena = state.entities.bush
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(True),
        transform=update(
            arena.transform, pos=arena.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32))
        ),
        shape=update(arena.shape, radius=arena.shape.radius.at[slot].set(radius)),
        edible=update(
            arena.edible,
            stock=arena.edible.stock.at[slot].set(stock),
            kind=arena.edible.kind.at[slot].set(kind),
        ),
    )
    return update(state, entities=update(state.entities, bush=arena))


def _put_carcass(state, slot, pos, *, age_dead=0):
    arena = state.entities.carcass
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(True),
        transform=update(
            arena.transform, pos=arena.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32))
        ),
        shape=update(arena.shape, radius=arena.shape.radius.at[slot].set(0.5)),
        edible=update(
            arena.edible, age_dead=arena.edible.age_dead.at[slot].set(age_dead)
        ),
    )
    return update(state, entities=update(state.entities, carcass=arena))


def _put_neutral(state, slot, pos, *, odor=None):
    arena = state.entities.neutral
    if odor is not None:
        arena = update(
            arena,
            look=update(
                arena.look, odor=arena.look.odor.at[slot].set(jnp.asarray(odor, jnp.float32))
            ),
        )
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(True),
        transform=update(
            arena.transform, pos=arena.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32))
        ),
        shape=update(arena.shape, radius=arena.shape.radius.at[slot].set(0.4)),
    )
    return update(state, entities=update(state.entities, neutral=arena))


def test_diffusion_kernel_is_a_conserving_stencil():
    cfg = make_cfg(odor_diffusion=0.2)
    kernel = np.asarray(fields.diffusion_kernel(cfg))
    assert kernel.shape == (3, 3)
    assert float(kernel.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(kernel[1, 1]) == pytest.approx(1.0 - 4.0 * 0.2, abs=1e-6)
    assert float(kernel[0, 1]) == pytest.approx(0.2, abs=1e-6)
    assert float(kernel[0, 0]) == pytest.approx(0.0)


def test_total_odour_is_conserved_without_decay():
    cfg = make_cfg(odor_decay=0.0)
    state = state_of(cfg)
    pattern = (jnp.arange(8 * 8 * CHANNELS, dtype=jnp.float32) % 7.0).reshape(8, 8, CHANNELS)
    state = update(state, fields=update(state.fields, odor=pattern))
    before = float(jnp.sum(state.fields.odor))
    out, buffer = fields.apply(state, cfg)
    after = float(jnp.sum(out.fields.odor))
    assert after == pytest.approx(before, rel=1e-6)
    assert int(buffer.count) == 0


def test_diffusion_moves_into_neighbours_and_keeps_mass():
    cfg = make_cfg()
    grid = jnp.zeros((8, 8, CHANNELS), jnp.float32).at[4, 4, :].set(1.0)
    out = fields.diffuse(cfg, grid)
    assert float(out[4, 4, 0]) == pytest.approx(1.0 - 4.0 * cfg.world.odor_diffusion, abs=1e-6)
    assert float(out[3, 4, 0]) == pytest.approx(cfg.world.odor_diffusion, abs=1e-6)
    assert float(out[4, 3, 0]) == pytest.approx(cfg.world.odor_diffusion, abs=1e-6)
    assert float(out[0, 0, 0]) == pytest.approx(0.0)
    assert float(jnp.sum(out)) == pytest.approx(float(jnp.sum(grid)), rel=1e-6)


def test_decay_multiplies_the_total_by_one_minus_delta():
    cfg = make_cfg(odor_decay=0.02)
    state = state_of(cfg)
    pattern = (jnp.arange(8 * 8 * CHANNELS, dtype=jnp.float32) % 5.0 + 1.0).reshape(
        8, 8, CHANNELS
    )
    state = update(state, fields=update(state.fields, odor=pattern))
    before = float(jnp.sum(state.fields.odor))
    out, _ = fields.apply(state, cfg)
    assert float(jnp.sum(out.fields.odor)) == pytest.approx(0.98 * before, rel=1e-6)


def test_bush_deposit_is_species_spectrum_times_stock():
    cfg = make_cfg()
    state = _put_bush(state_of(cfg), 0, [3.5, 2.5], stock=0.5, kind=0)
    deposit = np.asarray(fields.deposit_field(cfg, state))
    assert deposit[2, 3, 0] == pytest.approx(0.5, abs=1e-6)
    assert deposit[2, 3, 1:].tolist() == [0.0, 0.0, 0.0]


def test_every_source_deposits_its_own_spectrum():
    cfg = make_cfg()
    state = state_of(cfg)
    state = _put_bush(state, 0, [1.5, 1.5], stock=0.5, kind=1)
    state = _put_carcass(state, 0, [3.5, 1.5], age_dead=100)
    state = _put_neutral(state, 0, [5.5, 1.5], odor=(0.4, 0.3, 0.2, 0.1))
    state = _put_creature(state, 0, [7.5, 1.5], odor=(0.1, 0.2, 0.3, 0.4))
    deposit = np.asarray(fields.deposit_field(cfg, state))

    assert deposit[1, 1].tolist() == pytest.approx([0.0, 0.5, 0.0, 0.0], abs=1e-6)
    rot = 1.0 - 0.995**100
    assert deposit[1, 3].tolist() == pytest.approx([0.0, 0.0, 0.0, rot], abs=1e-6)
    assert deposit[1, 5].tolist() == pytest.approx([0.4, 0.3, 0.2, 0.1], abs=1e-6)
    assert deposit[1, 7].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4], abs=1e-6)


def test_dead_entities_deposit_nothing():
    cfg = make_cfg()
    state = state_of(cfg)
    deposit = np.asarray(fields.deposit_field(cfg, state))
    assert np.allclose(deposit, 0.0)


def test_deposits_are_order_independent():
    cfg = make_cfg()
    first = _put_bush(state_of(cfg), 0, [2.5, 2.5], stock=0.3, kind=0)
    first = _put_bush(first, 1, [2.5, 2.5], stock=0.7, kind=1)
    second = _put_bush(state_of(cfg), 0, [2.5, 2.5], stock=0.7, kind=1)
    second = _put_bush(second, 1, [2.5, 2.5], stock=0.3, kind=0)
    deposit_first = np.asarray(fields.deposit_field(cfg, first))
    deposit_second = np.asarray(fields.deposit_field(cfg, second))
    assert np.array_equal(deposit_first, deposit_second)
    assert deposit_first[2, 2].tolist() == pytest.approx([0.3, 0.7, 0.0, 0.0], abs=1e-6)


def test_bilinear_sampling_reads_the_grid():
    cfg = make_cfg()
    index = jnp.arange(64, dtype=jnp.float32).reshape(8, 8)
    odor = jnp.repeat(index[..., None], CHANNELS, axis=-1)

    centers = jnp.stack(
        [jnp.tile(jnp.arange(8, dtype=jnp.float32) + 0.5, 8),
         jnp.repeat(jnp.arange(8, dtype=jnp.float32) + 0.5, 8)],
        axis=-1,
    )
    sampled = np.asarray(fields.sample_odor(cfg, odor, centers))
    assert sampled.shape == (64, CHANNELS)
    assert sampled[:, 0].tolist() == pytest.approx(index.reshape(-1).tolist(), abs=1e-6)

    midpoint = jnp.asarray([[4.0, 2.5]], jnp.float32)  # halfway between columns 3 and 4
    assert float(fields.sample_odor(cfg, odor, midpoint)[0, 0]) == pytest.approx(
        (2 * 8 + 3 + 2 * 8 + 4) / 2.0, abs=1e-6
    )


def test_sampling_clamps_outside_the_grid():
    cfg = make_cfg()
    index = jnp.arange(64, dtype=jnp.float32).reshape(8, 8)
    odor = jnp.repeat(index[..., None], CHANNELS, axis=-1)
    points = jnp.asarray([[-5.0, -5.0], [100.0, 100.0]], jnp.float32)
    sampled = np.asarray(fields.sample_odor(cfg, odor, points))
    assert sampled[0, 0] == pytest.approx(0.0)
    assert sampled[1, 0] == pytest.approx(index[7, 7])


def test_in_bush_counts_living_bushes_only():
    cfg = make_cfg()
    state = _put_bush(state_of(cfg), 0, [4.0, 4.0], radius=1.0)
    points = jnp.asarray([[4.0, 4.0], [4.9, 4.0], [6.0, 4.0]], jnp.float32)
    inside = np.asarray(fields.in_bush(state.entities, points))
    assert inside.tolist() == [True, True, False]

    dead = state.entities.bush
    dead = update(dead, alive=dead.alive.at[0].set(False))
    state = update(state, entities=update(state.entities, bush=dead))
    inside = np.asarray(fields.in_bush(state.entities, points))
    assert inside.tolist() == [False, False, False]


def test_environment_at_follows_the_analytic_fields():
    cfg = make_cfg()
    state = _put_creature(state_of(cfg), 0, [4.0, 2.0])
    temperature, luminance = fields.environment_at(state, cfg)
    assert float(temperature[0]) == pytest.approx(-2.0 + 4.0 * 2.0 / 8.0 + 0.5, abs=1e-6)
    assert float(luminance[0]) == pytest.approx(1.0, abs=1e-6)

    # Inside a bush the body is 0.5 colder and sees half the light (§4.2).
    shaded = _put_bush(state_of(cfg), 0, [4.0, 2.0], radius=1.0)
    shaded = _put_creature(shaded, 0, [4.0, 2.0])
    temperature, luminance = fields.environment_at(shaded, cfg)
    assert float(temperature[0]) == pytest.approx(-2.0 + 4.0 * 2.0 / 8.0 + 0.5 - 0.5, abs=1e-6)
    assert float(luminance[0]) == pytest.approx(0.5, abs=1e-6)


def test_light_and_temperature_day_night_cycle():
    cfg = make_cfg(width=8.0)
    noon = float(fields.light(cfg, 0))
    night = float(fields.light(cfg, int(cfg.world.day_ticks) // 2))
    assert noon == pytest.approx(1.0, abs=1e-6)
    assert night == pytest.approx(0.0, abs=1e-6)
    cold = float(fields.temperature(cfg, 0.0, 0))
    warm = float(fields.temperature(cfg, cfg.world.height, 0))
    assert warm - cold == pytest.approx(4.0, abs=1e-6)


def test_apply_traces_under_jit():
    cfg = make_cfg()
    state = _put_bush(state_of(cfg), 0, [3.5, 2.5], stock=0.5, kind=0)
    compiled = jax.jit(fields.apply, static_argnums=(1,))
    out, buffer = compiled(state, cfg)
    out, buffer = compiled(out, cfg)
    assert out.fields.odor.shape == cfg.world.grid_shape + (CHANNELS,)
    assert int(buffer.count) == 0

def test_temperature_ramp_scales_the_whole_amplitude():
    plain = make_cfg()
    ramped = make_cfg(ramp_ticks=10)  # ramp_temperature_multiplier defaults to 0.2
    # tick 0: 0.2x; halfway through the ramp: 0.2 + 0.8 * 0.5 = 0.6x.
    for tick, multiplier in ((0, 0.2), (5, 0.6)):
        for y in (0.0, 4.0, 8.0):
            base = float(fields.temperature(plain, y, tick))
            assert float(fields.temperature(ramped, y, tick)) == pytest.approx(
                multiplier * base, rel=1e-6, abs=1e-7
            ), (tick, y)
    # The in-bush -0.5 offset is part of the temperature difference, so it scales.
    plain_inside = float(fields.temperature(plain, 4.0, 0, True))
    plain_outside = float(fields.temperature(plain, 4.0, 0, False))
    ramp_inside = float(fields.temperature(ramped, 4.0, 0, True))
    ramp_outside = float(fields.temperature(ramped, 4.0, 0, False))
    assert ramp_inside - ramp_outside == pytest.approx(
        0.2 * (plain_inside - plain_outside), rel=1e-6, abs=1e-7
    )
    # Once the ramp is over the full amplitude is back.
    assert float(fields.temperature(ramped, 4.0, 10)) == pytest.approx(
        float(fields.temperature(plain, 4.0, 10)), rel=1e-6
    )
    # Light is not ramped (§9 only scales the temperature difference).
    for tick in (0, 5, 10):
        assert float(fields.light(ramped, tick)) == pytest.approx(
            float(fields.light(plain, tick)), rel=1e-6
        )


def test_environment_at_uses_the_ramped_temperature():
    plain = make_cfg()
    ramped = make_cfg(ramp_ticks=10)
    base_state = _put_creature(state_of(plain), 0, [4.0, 2.0])
    ramp_state = _put_creature(state_of(ramped), 0, [4.0, 2.0])
    base_temperature, base_light = fields.environment_at(base_state, plain)
    ramp_temperature, ramp_light = fields.environment_at(ramp_state, ramped)
    assert float(ramp_temperature[0]) == pytest.approx(
        0.2 * float(base_temperature[0]), rel=1e-6, abs=1e-7
    )
    assert float(ramp_light[0]) == pytest.approx(float(base_light[0]), rel=1e-6)


def test_ramped_fields_trace_under_jit():
    cfg = make_cfg(ramp_ticks=10, temperature_scale=0.5)
    state = _put_creature(state_of(cfg), 0, [4.0, 2.0])
    state = _put_bush(state, 0, [4.0, 2.0], stock=0.5, kind=0)
    compiled = jax.jit(fields.apply, static_argnums=(1,))
    out, buffer = compiled(state, cfg)
    out, buffer = compiled(out, cfg)
    assert out.fields.odor.shape == cfg.world.grid_shape + (CHANNELS,)
    assert int(buffer.count) == 0

    # Inside a jitted function the ramp multiplier (0.2 at tick 0) and the
    # calibrated temperature_scale (0.5) compose: 0.1x of the nominal amplitude.
    compiled_at = jax.jit(fields.environment_at, static_argnums=(1,))
    temperature, light = compiled_at(state, cfg)
    base_temperature, base_light = fields.environment_at(state, make_cfg())
    assert float(temperature[0]) == pytest.approx(
        0.1 * float(base_temperature[0]), rel=1e-6, abs=1e-7
    )
    assert float(light[0]) == pytest.approx(float(base_light[0]), rel=1e-6)


def test_temperature_scale_scales_the_whole_amplitude():
    nominal = make_cfg()
    scaled = make_cfg(temperature_scale=0.5)
    for tick in (0, 5, 10):
        for y in (0.0, 4.0, 8.0):
            base = float(fields.temperature(nominal, y, tick))
            assert float(fields.temperature(scaled, y, tick)) == pytest.approx(
                0.5 * base, rel=1e-6, abs=1e-7
            ), (tick, y)
    # The in-bush -0.5 offset is part of the amplitude, so it halves too.
    assert float(fields.temperature(scaled, 4.0, 0, True)) == pytest.approx(
        0.5 * float(fields.temperature(nominal, 4.0, 0, True)), rel=1e-6, abs=1e-7
    )
    # environment_at reads the scaled temperature; light is untouched.
    nominal_state = _put_creature(state_of(nominal), 0, [4.0, 2.0])
    scaled_state = _put_creature(state_of(scaled), 0, [4.0, 2.0])
    base_temperature, base_light = fields.environment_at(nominal_state, nominal)
    scaled_temperature, scaled_light = fields.environment_at(scaled_state, scaled)
    assert float(scaled_temperature[0]) == pytest.approx(
        0.5 * float(base_temperature[0]), rel=1e-6, abs=1e-7
    )
    assert float(scaled_light[0]) == pytest.approx(float(base_light[0]), rel=1e-6)


def test_temperature_scale_and_ramp_compose():
    nominal = make_cfg()
    combined = make_cfg(ramp_ticks=10, temperature_scale=0.5)
    # ramp_temperature_multiplier defaults to 0.2: at tick 0 the composed factor
    # is 0.2 * 0.5 = 0.1, halfway through it is 0.6 * 0.5 = 0.3, and after the
    # ramp the calibrated scale 0.5 remains.
    for tick, factor in ((0, 0.1), (5, 0.3), (10, 0.5)):
        base = float(fields.temperature(nominal, 4.0, tick))
        assert float(fields.temperature(combined, 4.0, tick)) == pytest.approx(
            factor * base, rel=1e-6, abs=1e-7
        ), tick
    # Light is affected by neither factor.
    for tick in (0, 5, 10):
        assert float(fields.light(combined, tick)) == pytest.approx(
            float(fields.light(nominal, tick)), rel=1e-6
        )

