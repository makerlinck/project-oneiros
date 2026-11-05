"""Golden vision tests (SANDBOX.md §5.6, §13 test_vision_golden).

Every scenario is rendered twice: once by world.vision and once by the slow
reference renderer below, which re-implements the formulas of §5.6 directly
(no shared code).  Scenarios cover a single circle, occlusion, a wall, night,
inside/outside a bush, held objects and sleep.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from life_sandbox.config import Capacities, Config, RunConfig, WorldConfig
from life_sandbox.core import rng
from life_sandbox.core.components import BUSH, CARCASS, CREATURE, NEUTRAL
from life_sandbox.core.state import empty
from life_sandbox.core.struct import update
from life_sandbox.world import vision

SIZE = 16.0
DAY = 200.0
WALL = np.asarray(vision.WALL_RGB, np.float64)
NO_NOISE = dict(noise_base=0.0, noise_dark=0.0)


def make_cfg(creatures: int = 2) -> Config:
    return Config.default().replaced(
        world=WorldConfig(width=SIZE, height=SIZE, cell=1.0, day_ticks=DAY),
        capacities=Capacities(creature=creatures, bush=2, carcass=2, neutral=1),
        run=RunConfig(seed=3, ticks=0),
    )


def state_of(cfg: Config):
    return empty(cfg, rng.world_key(3))


def put_creature(state, slot, pos, theta=0.0, radius=0.5, rgb=(0.5, 0.5, 0.5), asleep=False):
    arena = state.entities.creature
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(True),
        transform=update(
            arena.transform,
            pos=arena.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32)),
            theta=arena.transform.theta.at[slot].set(jnp.asarray(theta, jnp.float32)),
        ),
        shape=update(
            arena.shape,
            radius=arena.shape.radius.at[slot].set(radius),
            mass=arena.shape.mass.at[slot].set(1.0),
        ),
        look=update(arena.look, rgb=arena.look.rgb.at[slot].set(jnp.asarray(rgb, jnp.float32))),
        body=update(arena.body, asleep=arena.body.asleep.at[slot].set(asleep)),
    )
    return update(state, entities=update(state.entities, creature=arena))


def put_bush(state, slot, pos, radius=1.0, rgb=(0.0, 1.0, 0.0)):
    arena = state.entities.bush
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(True),
        transform=update(
            arena.transform, pos=arena.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32))
        ),
        shape=update(
            arena.shape,
            radius=arena.shape.radius.at[slot].set(radius),
            mass=arena.shape.mass.at[slot].set(1.0),
        ),
        look=update(arena.look, rgb=arena.look.rgb.at[slot].set(jnp.asarray(rgb, jnp.float32))),
    )
    return update(state, entities=update(state.entities, bush=arena))


def put_carcass(state, slot, pos, radius=1.0, rgb=(1.0, 0.0, 0.0)):
    arena = state.entities.carcass
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(True),
        transform=update(
            arena.transform, pos=arena.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32))
        ),
        shape=update(
            arena.shape,
            radius=arena.shape.radius.at[slot].set(radius),
            mass=arena.shape.mass.at[slot].set(1.0),
        ),
        look=update(arena.look, rgb=arena.look.rgb.at[slot].set(jnp.asarray(rgb, jnp.float32))),
    )
    return update(state, entities=update(state.entities, carcass=arena))


def put_neutral(state, slot, pos, radius=0.5, rgb=(0.0, 0.5, 1.0)):
    arena = state.entities.neutral
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(True),
        transform=update(
            arena.transform, pos=arena.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32))
        ),
        shape=update(
            arena.shape,
            radius=arena.shape.radius.at[slot].set(radius),
            mass=arena.shape.mass.at[slot].set(0.2),
        ),
        look=update(arena.look, rgb=arena.look.rgb.at[slot].set(jnp.asarray(rgb, jnp.float32))),
    )
    return update(state, entities=update(state.entities, neutral=arena))


def hold(state, holder, kind, slot, uid):
    creature = state.entities.creature
    holding = creature.grip.holding
    holding = update(
        holding,
        kind=holding.kind.at[holder].set(kind),
        slot=holding.slot.at[holder].set(slot),
        uid=holding.uid.at[holder].set(uid),
    )
    creature = update(creature, grip=update(creature.grip, holding=holding))
    return update(state, entities=update(state.entities, creature=creature))


# --- reference renderer: the §5.6 formulas, written from scratch -------------

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _circles(state, cfg):
    ent = state.entities
    n_creature = cfg.capacities.creature
    n_bush = cfg.capacities.bush
    starts = {
        CREATURE: 0,
        BUSH: n_creature,
        CARCASS: n_creature + n_bush,
        NEUTRAL: n_creature + n_bush + cfg.capacities.carcass,
    }
    arenas = (
        (CREATURE, ent.creature.transform.pos, ent.creature.shape.radius,
         ent.creature.look.rgb, ent.creature.alive),
        (BUSH, ent.bush.transform.pos, ent.bush.shape.radius, ent.bush.look.rgb, ent.bush.alive),
        (CARCASS, ent.carcass.transform.pos, ent.carcass.shape.radius,
         ent.carcass.look.rgb, ent.carcass.alive),
        (NEUTRAL, ent.neutral.transform.pos, ent.neutral.shape.radius,
         ent.neutral.look.rgb, ent.neutral.alive),
    )
    out = []
    for proto, pos, radius, rgb, alive in arenas:
        for slot in range(len(alive)):
            if bool(alive[slot]):
                out.append(
                    (
                        starts[proto] + slot,
                        np.asarray(pos[slot], np.float64),
                        float(radius[slot]),
                        np.asarray(rgb[slot], np.float64),
                    )
                )
    return out


def _ray_circle(origin, direction, center, radius):
    oc = center - origin
    b = float(np.dot(direction, oc))
    quad = float(np.dot(oc, oc)) - radius * radius
    disc = b * b - quad
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    if b + root < 0.0:
        return None
    if quad <= 0.0:
        return 0.0
    return max(b - root, 0.0)


def _ray_wall(origin, direction):
    best = math.inf
    for axis, limit in ((0, SIZE), (1, SIZE)):
        d = direction[axis]
        if abs(d) <= 1e-6:
            continue
        t = (limit - origin[axis]) / d if d > 0.0 else (0.0 - origin[axis]) / d
        if t >= 0.0:
            best = min(best, t)
    return best


def reference_render(state, cfg, slot):
    ent = state.entities
    creature = ent.creature
    origin = np.asarray(creature.transform.pos[slot], np.float64)
    theta = float(creature.transform.theta[slot])
    count = cfg.sensors.vision_pixels
    fov = math.radians(cfg.sensors.fov_deg)
    tick = float(state.tick)

    circles = _circles(state, cfg)

    ignored = {slot}  # the caster itself (creature block starts at 0)
    holding = creature.grip.holding
    kind = int(holding.kind[slot])
    if kind >= 0:
        target_slot = int(holding.slot[slot])
        uid = int(holding.uid[slot])
        n_creature = cfg.capacities.creature
        n_bush = cfg.capacities.bush
        starts = {CREATURE: 0, BUSH: n_creature, CARCASS: n_creature + n_bush}
        starts[NEUTRAL] = n_creature + n_bush + cfg.capacities.carcass
        if kind == CREATURE:
            alive = bool(ent.creature.alive[target_slot])
            target_uid = int(ent.creature.identity.uid[target_slot])
        elif kind == BUSH:
            alive = bool(ent.bush.alive[target_slot])
            target_uid = int(ent.bush.uid[target_slot])
        elif kind == CARCASS:
            alive = bool(ent.carcass.alive[target_slot])
            target_uid = int(ent.carcass.uid[target_slot])
        else:
            alive = bool(ent.neutral.alive[target_slot])
            target_uid = int(ent.neutral.uid[target_slot])
        if alive and target_uid == uid:
            ignored.add(starts[kind] + target_slot)

    bush_positions = np.asarray(ent.bush.transform.pos, np.float64)
    bush_radii = np.asarray(ent.bush.shape.radius, np.float64)
    bush_alive = np.asarray(ent.bush.alive)
    in_bush = False
    for index in range(len(bush_alive)):
        if bush_alive[index] and _dist(origin, bush_positions[index]) <= bush_radii[index]:
            in_bush = True
            ignored.add(cfg.capacities.creature + index)

    luminance = (0.5 + 0.5 * math.cos(2.0 * math.pi * tick / cfg.world.day_ticks)) * (
        0.5 if in_bush else 1.0
    )

    asleep = bool(ent.creature.body.asleep[slot])
    pixels = np.zeros((count, 3), np.float64)
    for i in range(count):
        phi = theta - 0.5 * fov + (i + 0.5) * (fov / count)
        direction = np.asarray([math.cos(phi), math.sin(phi)], np.float64)

        best = math.inf
        colour = None
        for index, center, radius, rgb in circles:
            if index in ignored:
                continue
            t = _ray_circle(origin, direction, center, radius)
            if t is not None and t < best:
                best = t
                colour = rgb
        wall = _ray_wall(origin, direction)
        if wall < best:
            best = wall
            colour = WALL
        if asleep or colour is None or best > vision.R_VIEW:
            pixels[i] = 0.0
            continue
        pixels[i] = colour * math.exp(-best / vision.ATTENUATION) * luminance
    return pixels


def test_single_circle_matches_reference():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [8.0, 8.0], theta=0.0)
    state = put_carcass(state, 0, [12.0, 8.0], radius=1.0, rgb=(1.0, 0.0, 0.0))
    rendered = np.asarray(vision.render(state, cfg, jax.random.key(0), **NO_NOISE))[0]
    reference = reference_render(state, cfg, 0)
    assert np.allclose(rendered, reference, atol=1e-5)
    # The centre pixel sees the sphere head on, attenuated over three units.
    assert rendered[7, 0] == pytest.approx(math.exp(-3.0 / 6.0), abs=1e-5)
    assert rendered[7, 1] == pytest.approx(0.0, abs=1e-6)


def test_occlusion_shows_the_nearest_circle():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [8.0, 8.0], theta=0.0)
    state = put_bush(state, 0, [9.5, 8.0], radius=0.5, rgb=(0.0, 1.0, 0.0))
    state = put_carcass(state, 0, [12.0, 8.0], radius=1.0, rgb=(1.0, 0.0, 0.0))
    rendered = np.asarray(vision.render(state, cfg, jax.random.key(0), **NO_NOISE))[0]
    reference = reference_render(state, cfg, 0)
    assert np.allclose(rendered, reference, atol=1e-5)
    assert rendered[7, 1] == pytest.approx(math.exp(-1.0 / 6.0), abs=1e-5)
    assert rendered[7, 0] == pytest.approx(0.0, abs=1e-6)


def test_wall_is_grey_and_attenuated():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [2.0, 8.0], theta=math.pi)
    rendered = np.asarray(vision.render(state, cfg, jax.random.key(0), **NO_NOISE))[0]
    reference = reference_render(state, cfg, 0)
    assert np.allclose(rendered, reference, atol=1e-5)
    expected = 0.5 * math.exp(-2.0 / 6.0)
    assert rendered[7, :].tolist() == pytest.approx([expected] * 3, abs=1e-5)


def test_night_has_no_colour_signal():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [8.0, 8.0], theta=0.0)
    state = put_carcass(state, 0, [12.0, 8.0], radius=1.0, rgb=(1.0, 0.0, 0.0))
    night = update(state, tick=jnp.asarray(int(DAY) // 2, jnp.int32))
    rendered = np.asarray(vision.render(night, cfg, jax.random.key(0), **NO_NOISE))[0]
    assert np.allclose(rendered, 0.0, atol=1e-6)


def test_sleep_closes_the_field():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [8.0, 8.0], theta=0.0, asleep=True)
    state = put_carcass(state, 0, [12.0, 8.0], radius=1.0, rgb=(1.0, 0.0, 0.0))
    rendered = np.asarray(vision.render(state, cfg, jax.random.key(0), **NO_NOISE))[0]
    assert np.allclose(rendered, 0.0, atol=1e-6)
    # With noise on, a sleeping creature sees noise only.
    noisy = np.asarray(vision.render(state, cfg, jax.random.key(0)))[0]
    assert float(np.abs(noisy).max()) > 0.0


def test_bush_hides_behind_it_but_the_containing_bush_is_ignored():
    cfg = make_cfg()
    # Outside the bush: the bush itself blocks the carcass.
    outside = put_creature(state_of(cfg), 0, [8.6, 8.0], theta=0.0)
    outside = put_bush(outside, 0, [10.0, 8.0], radius=1.0, rgb=(0.0, 1.0, 0.0))
    outside = put_carcass(outside, 0, [14.0, 8.0], radius=1.0, rgb=(1.0, 0.0, 0.0))
    rendered = np.asarray(vision.render(outside, cfg, jax.random.key(0), **NO_NOISE))[0]
    assert np.allclose(rendered, reference_render(outside, cfg, 0), atol=1e-5)
    assert rendered[7, 1] == pytest.approx(math.exp(-0.4 / 6.0), abs=1e-5)  # bush green

    # Inside the bush: it is ignored, so the carcass behind becomes visible and
    # the in-bush shading halves the perceived light.
    inside = put_creature(state_of(cfg), 0, [10.0, 8.0], theta=0.0)
    inside = put_bush(inside, 0, [10.0, 8.0], radius=1.0, rgb=(0.0, 1.0, 0.0))
    inside = put_carcass(inside, 0, [14.0, 8.0], radius=1.0, rgb=(1.0, 0.0, 0.0))
    rendered = np.asarray(vision.render(inside, cfg, jax.random.key(0), **NO_NOISE))[0]
    assert np.allclose(rendered, reference_render(inside, cfg, 0), atol=1e-5)
    assert rendered[7, 0] == pytest.approx(0.5 * math.exp(-3.0 / 6.0), abs=1e-5)


def test_held_entities_are_invisible_to_their_holder():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [8.0, 8.0], theta=0.0)
    state = put_carcass(state, 0, [10.0, 8.0], radius=0.5, rgb=(1.0, 0.0, 0.0))
    state = put_neutral(state, 0, [14.0, 8.0], radius=0.5, rgb=(0.0, 0.5, 1.0))
    state = hold(state, 0, CARCASS, 0, int(state.entities.carcass.uid[0]))
    rendered = np.asarray(vision.render(state, cfg, jax.random.key(0), **NO_NOISE))[0]
    reference = reference_render(state, cfg, 0)
    assert np.allclose(rendered, reference, atol=1e-5)
    # The held carcass at distance 1.5 would win; the neutral object behind it
    # is what the holder actually sees.
    assert rendered[7, 2] == pytest.approx(math.exp(-5.5 / 6.0), abs=1e-5)
    assert rendered[7, 0] == pytest.approx(0.0, abs=1e-6)


def test_pixel_noise_scales_with_light():
    cfg = make_cfg(creatures=8)
    day = state_of(cfg)
    night = state_of(cfg)
    for slot in range(8):
        pos = [2.0 + slot, 8.0]
        day = put_creature(day, slot, pos, theta=0.0)
        night = put_creature(night, slot, pos, theta=0.0)
    night = update(night, tick=jnp.asarray(int(DAY) // 2, jnp.int32))

    rendered = np.asarray(vision.render(day, cfg, jax.random.key(4)))
    reference = np.stack([reference_render(day, cfg, slot) for slot in range(8)])
    residual = rendered - reference
    assert float(residual.std()) == pytest.approx(0.02, rel=0.2)
    assert abs(float(residual.mean())) < 0.005

    darkness = np.asarray(vision.render(night, cfg, jax.random.key(4)))
    assert float(darkness.std()) == pytest.approx(0.10, rel=0.2)
    assert abs(float(darkness.mean())) < 0.02


def test_render_traces_under_jit():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [8.0, 8.0], theta=0.0)
    state = put_carcass(state, 0, [12.0, 8.0], radius=1.0)
    compiled = jax.jit(vision.render, static_argnums=(1,))
    first = compiled(state, cfg, jax.random.key(0))
    second = compiled(state, cfg, jax.random.key(0))
    assert first.shape == (cfg.capacities.creature, cfg.sensors.vision_pixels, 3)
    assert np.array_equal(np.asarray(first), np.asarray(second))

