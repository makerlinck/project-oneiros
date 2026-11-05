"""Slice layout tests (SANDBOX.md §5.6, §13 test_slice_layout).

Static checks: the slice is assembled only from registered physical sensors, in
the documented order, D = 70, and the designer readouts N, I, X and b never
enter it directly (b only through the physical skin delta-T sensor).  The leak
arm appends exactly five designer summaries, and only there.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from life_sandbox.config import Capacities, Config, LearningConfig, RunConfig, WorldConfig
from life_sandbox.core import rng
from life_sandbox.core.state import empty
from life_sandbox.core.struct import update
from life_sandbox.core.systems import sensing
from life_sandbox.world import fields, vision

EXPECTED_SEGMENTS = (
    ("vision", 45, "none"),
    ("odour", 4, "none"),
    ("touch", 4, "none"),
    ("load", 1, "none"),
    ("proprio", 2, "none"),
    ("interoception", 6, "amplitude"),
    ("skin", 1, "difference"),
    ("light", 1, "none"),
    ("efference", 6, "none"),
)

SILENT = sensing.NoiseScales(
    vision_base=0.0,
    vision_dark=0.0,
    odour=0.0,
    touch=0.0,
    load=0.0,
    proprio=0.0,
    stomach=0.0,
    pain=0.0,
    heartbeat=0.0,
    skin=0.0,
    light=0.0,
)


def make_cfg(readable_leak: bool = False) -> Config:
    return Config.default().replaced(
        world=WorldConfig(width=16.0, height=16.0, cell=1.0),
        capacities=Capacities(creature=2, bush=2, carcass=2, neutral=1),
        run=RunConfig(seed=7, ticks=0),
        learning=LearningConfig(readable_leak=readable_leak),
    )


def put_creature(state, slot, pos, theta=0.0, radius=0.5):
    arena = state.entities.creature
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(True),
        transform=update(
            arena.transform,
            pos=arena.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32)),
            theta=arena.transform.theta.at[slot].set(jnp.asarray(theta, jnp.float32)),
        ),
        shape=update(arena.shape, radius=arena.shape.radius.at[slot].set(radius)),
    )
    return update(state, entities=update(state.entities, creature=arena))


def set_body(state, slot, **values):
    arena = state.entities.creature
    body = arena.body
    for name, value in values.items():
        body = update(body, **{name: getattr(body, name).at[slot].set(value)})
    arena = update(arena, body=body)
    return update(state, entities=update(state.entities, creature=arena))


def set_creature_fields(state, slot, **values):
    """Set a plain creature-arena array field (transform/contact/motor...)."""
    arena = state.entities.creature
    holder = arena
    for path, value in values.items():
        section, name = path.split(".")
        current = getattr(holder, section)
        updated = update(current, **{name: getattr(current, name).at[slot].set(value)})
        holder = update(holder, **{section: updated})
    return update(state, entities=update(state.entities, creature=holder))


def state_with_one_creature(cfg: Config):
    state = empty(cfg, rng.world_key(7))
    state = put_creature(state, 0, [8.0, 8.0], theta=0.0)
    state = update(state, pain_weights=jnp.zeros((9, 4), jnp.float32))
    state = set_body(
        state,
        0,
        G=jnp.asarray([0.2, 0.0, 0.0], jnp.float32),
        N=jnp.asarray([0.1, 0.2, 0.0], jnp.float32),
        I=jnp.asarray(0.9, jnp.float32),
        X=jnp.asarray(0.3, jnp.float32),
        b=jnp.asarray(0.2, jnp.float32),
        phi_hb=jnp.asarray(1.0, jnp.float32),
        S=jnp.asarray(0.0, jnp.float32),
    )
    state = set_creature_fields(
        state,
        0,
        **{
            "transform.v": jnp.asarray(0.25, jnp.float32),
            "transform.omega": jnp.asarray(-0.1, jnp.float32),
            "contact.touch": jnp.asarray([0.1, 0.2, 0.3, 0.4], jnp.float32),
            "motor.last_out": jnp.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], jnp.float32),
        },
    )
    # Linear odour ramp along x so the nose position and bilinear read are analytic.
    rows, cols = cfg.world.grid_shape
    ramp = jnp.broadcast_to(jnp.arange(cols, dtype=jnp.float32)[None, :], (rows, cols))
    odor = jnp.zeros_like(state.fields.odor).at[..., 0].set(ramp)
    state = update(state, fields=update(state.fields, odor=odor))
    return state


def test_registered_segments_and_widths():
    cfg = make_cfg()
    segments = sensing.segments(cfg)
    assert tuple((s.name, s.width, s.reads_body) for s in segments) == EXPECTED_SEGMENTS
    assert sensing.physical_dim(cfg) == 70
    assert sensing.physical_dim(cfg) == cfg.sensors.slice_dim
    assert sensing.offsets(cfg) == {
        "vision": 0,
        "odour": 45,
        "touch": 49,
        "load": 53,
        "proprio": 54,
        "interoception": 56,
        "skin": 62,
        "light": 63,
        "efference": 64,
    }


def test_only_the_pain_bundle_and_skin_are_marked_as_body_readers():
    cfg = make_cfg()
    readers = {segment.name: segment.reads_body for segment in sensing.segments(cfg)}
    assert readers["interoception"] == "amplitude"
    assert readers["skin"] == "difference"
    for name, reads in readers.items():
        if name not in ("interoception", "skin"):
            assert reads == "none", name


def test_slice_is_written_with_the_allocated_width_and_dtype():
    cfg = make_cfg()
    state = state_with_one_creature(cfg)
    out, buffer = sensing.apply(state, cfg)
    assert out.slice.shape == (cfg.capacities.creature, 70)
    assert out.slice.dtype == jnp.float32
    assert int(buffer.count) == 0


def test_physical_sensor_wiring_is_exact():
    cfg = make_cfg()
    state = state_with_one_creature(cfg)
    out, _ = sensing.observe(state, cfg, rng.system_key(state.key, state.tick, "sense"), SILENT)
    starts = sensing.offsets(cfg)
    row = np.asarray(out[0])

    creature = state.entities.creature
    assert row[starts["load"]] == pytest.approx(0.0, abs=1e-6)
    assert row[starts["proprio"]:starts["proprio"] + 2].tolist() == pytest.approx(
        [0.25, -0.1], abs=1e-6
    )
    assert row[starts["touch"]:starts["touch"] + 4].tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.4], abs=1e-6
    )
    assert row[starts["interoception"]] == pytest.approx(0.2, abs=1e-6)
    assert row[starts["interoception"] + 5] == pytest.approx(float(jnp.sin(1.0)), abs=1e-6)
    temperature, luminance = fields.environment_at(state, cfg)
    assert row[starts["skin"]] == pytest.approx(float(temperature[0]) - 0.2, abs=1e-5)
    assert row[starts["light"]] == pytest.approx(float(luminance[0]), abs=1e-5)
    assert row[starts["efference"]:].tolist() == pytest.approx(
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6], abs=1e-6
    )
    # Nose sits at pos + (r + 0.2) * heading = (8.7, 8.0); the grid is x = column,
    # so the bilinear read is 0.8 * 8 + 0.2 * 9 = 8.2 on channel 0.
    expected_odour = [np.log1p(8.2), 0.0, 0.0, 0.0]
    assert row[starts["odour"]:starts["odour"] + 4].tolist() == pytest.approx(
        expected_odour, abs=1e-5
    )
    expected_vision = np.asarray(
        vision.render(
            state,
            cfg,
            rng.subsplit(rng.system_key(state.key, state.tick, "sense"), 3)[0],
            noise_base=0.0,
            noise_dark=0.0,
        )
    )[0].reshape(-1)
    assert row[:45] == pytest.approx(expected_vision.tolist(), abs=1e-6)


def test_nutrient_integrity_and_toxin_never_enter_the_slice():
    cfg = make_cfg()
    base = state_with_one_creature(cfg)
    reference, _ = sensing.apply(base, cfg)

    perturbed = set_body(
        base,
        0,
        N=jnp.asarray([0.3, 0.4, 0.2], jnp.float32),
        I=jnp.asarray(0.4, jnp.float32),
        X=jnp.asarray(0.9, jnp.float32),
    )
    changed, _ = sensing.apply(perturbed, cfg)
    # pain_weights is zero, so neither the pain amplitudes nor any other sensor
    # can see the body readouts: the whole physical slice must be bit-identical.
    assert np.array_equal(np.asarray(reference.slice), np.asarray(changed.slice))


def test_core_temperature_enters_only_through_skin_delta():
    cfg = make_cfg()
    base = state_with_one_creature(cfg)
    reference, _ = sensing.apply(base, cfg)

    warmer = set_body(base, 0, b=jnp.asarray(0.45, jnp.float32))
    changed, _ = sensing.apply(warmer, cfg)

    starts = sensing.offsets(cfg)
    diff = np.asarray(changed.slice[0]) - np.asarray(reference.slice[0])
    skin_index = starts["skin"]
    assert diff[skin_index] == pytest.approx(-0.25, abs=1e-5)
    mask = np.ones(diff.shape, dtype=bool)
    mask[skin_index] = False
    assert np.allclose(diff[mask], 0.0, atol=1e-6)


def test_dead_slots_have_a_zero_slice():
    cfg = make_cfg()
    state = state_with_one_creature(cfg)
    out, _ = sensing.apply(state, cfg)
    assert np.allclose(np.asarray(out.slice[1]), 0.0, atol=1e-6)


def test_leak_dimensions_appear_only_in_the_leak_arm():
    cfg = make_cfg()
    state = state_with_one_creature(cfg)
    assert sensing.slice_dim(cfg) == 70
    plain, _ = sensing.apply(state, cfg)
    assert plain.slice.shape[1] == 70
    assert not np.any(np.asarray(plain.slice[0])[70:])

    leak_cfg = make_cfg(readable_leak=True)
    leak_state = empty(leak_cfg, rng.world_key(7))
    leak_state = update(
        leak_state,
        entities=state.entities,
        comp=state.comp,
        pain_weights=state.pain_weights,
        fields=state.fields,
    )
    assert sensing.slice_dim(leak_cfg) == 75
    leak, _ = sensing.apply(leak_state, leak_cfg)
    assert leak.slice.shape[1] == 75

    starts = sensing.offsets(leak_cfg)
    physical = np.asarray(leak.slice[0])[:70]
    assert sensing.physical_dim(leak_cfg) == 70
    leak_dims = np.asarray(leak.slice[0])[70:]
    assert leak_dims.tolist() == pytest.approx(
        [0.1 + 0.2 + 0.0 - 0.5, 0.9, 0.3, 0.2, 0.0], abs=1e-5
    )
    # The physical part is still the physical sensors.
    assert physical.shape == (70,)
    assert starts["efference"] == 64


def test_leak_only_changes_the_tail():
    cfg = make_cfg()
    state = state_with_one_creature(cfg)
    plain, _ = sensing.apply(state, cfg)

    leak_cfg = make_cfg(readable_leak=True)
    leak_state = empty(leak_cfg, rng.world_key(7))
    leak_state = update(
        leak_state,
        entities=state.entities,
        comp=state.comp,
        pain_weights=state.pain_weights,
        fields=state.fields,
    )
    leak, _ = sensing.apply(leak_state, leak_cfg)
    assert np.allclose(np.asarray(leak.slice)[:, :70], np.asarray(plain.slice), atol=1e-6)


def test_apply_traces_under_jit():
    cfg = make_cfg()
    state = state_with_one_creature(cfg)
    compiled = jax.jit(sensing.apply, static_argnums=(1,))
    out, buffer = compiled(state, cfg)
    out, buffer = compiled(out, cfg)
    assert out.slice.shape == (cfg.capacities.creature, 70)
    assert int(buffer.count) == 0
