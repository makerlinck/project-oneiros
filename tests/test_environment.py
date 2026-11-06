"""environment system tests (SANDBOX.md §4.2, §4.4, §4.5, §13 §12.5).

Covers berry regeneration, bush appearance, carcass rot and removal (including
the ledger bookkeeping of §3.5), the toxin schedule (period, composition,
conditional event) and the scripted wander of neutral objects.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from life_sandbox.config import Capacities, Config, RunConfig, WorldConfig
from life_sandbox.core import components as C
from life_sandbox.core import events, rng
from life_sandbox.core.state import empty
from life_sandbox.core.struct import update
from life_sandbox.core.systems import environment
from life_sandbox.world.constants import BERRY_RGB, BUSH_GREEN, INITIAL_COMP

BASE = np.asarray(INITIAL_COMP, np.float64)


def make_cfg(**world_overrides) -> Config:
    world = dataclasses.replace(
        WorldConfig(width=16.0, height=16.0, cell=1.0, l_ticks=10.0, c_switch=1.0),
        **world_overrides,
    )
    return Config.default().replaced(
        world=world,
        capacities=Capacities(creature=2, bush=4, carcass=2, neutral=2),
        run=RunConfig(seed=13, ticks=0),
    )


def state_of(cfg: Config):
    return empty(cfg, rng.world_key(13))


def _full(values, capacity, dtype, fill):
    """Pad or truncate a per-slot vector to the arena capacity."""
    values = jnp.asarray(values, dtype)
    if values.shape[0] >= capacity:
        return values[:capacity]
    tail = jnp.full((capacity - values.shape[0],), fill, dtype)
    return jnp.concatenate([values, tail])


def set_stock(state, stocks, kinds=None, alive=None):
    arena = state.entities.bush
    capacity = arena.alive.shape[0]
    stock = _full(stocks, capacity, jnp.float32, 0.0)
    kind = arena.edible.kind if kinds is None else _full(kinds, capacity, jnp.int32, 0)
    if alive is None:
        # Slots given a stock are made alive by default; the rest stay dead.
        count = len(stocks)
        alive = [slot < count for slot in range(capacity)]
    live = _full(alive, capacity, jnp.bool_, False)
    arena = update(
        arena,
        alive=live,
        edible=update(arena.edible, stock=stock, kind=kind),
    )
    return update(state, entities=update(state.entities, bush=arena))


def set_carcass(state, slot, n, age, alive=True):
    arena = state.entities.carcass
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(alive),
        edible=update(
            arena.edible,
            n=arena.edible.n.at[slot].set(jnp.asarray(n, jnp.float32)),
            age_dead=arena.edible.age_dead.at[slot].set(age),
        ),
    )
    return update(state, entities=update(state.entities, carcass=arena))


def put_creature(state, slot, pos):
    arena = state.entities.creature
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(True),
        transform=update(
            arena.transform, pos=arena.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32))
        ),
        shape=update(arena.shape, radius=arena.shape.radius.at[slot].set(0.5)),
    )
    return update(state, entities=update(state.entities, creature=arena))


def put_neutral(state, slot, pos, heading=0.0, timer=50, held=False):
    arena = state.entities.neutral
    arena = update(
        arena,
        alive=arena.alive.at[slot].set(True),
        transform=update(
            arena.transform,
            pos=arena.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32)),
            theta=arena.transform.theta.at[slot].set(heading),
        ),
        shape=update(arena.shape, radius=arena.shape.radius.at[slot].set(0.4)),
        scripted=update(
            arena.scripted,
            heading=arena.scripted.heading.at[slot].set(heading),
            timer=arena.scripted.timer.at[slot].set(timer),
        ),
    )
    if held:
        arena = update(
            arena,
            held_by=update(
                arena.held_by,
                kind=arena.held_by.kind.at[slot].set(0),
                slot=arena.held_by.slot.at[slot].set(0),
                uid=arena.held_by.uid.at[slot].set(0),
            ),
        )
    return update(state, entities=update(state.entities, neutral=arena))


def at_tick(state, tick):
    return update(state, tick=jnp.asarray(tick, jnp.int32))


def test_berry_regeneration_matches_the_logistic_step():
    cfg = make_cfg()
    state = set_stock(state_of(cfg), [0.3, 0.5, 1.0, 0.0], alive=[True, True, True, True])
    out, _ = environment.apply(state, cfg)
    stock = np.asarray(out.entities.bush.edible.stock)
    expected = [
        0.3 + 0.03 * 0.3 * 0.7 + 1e-4,
        0.5 + 0.03 * 0.5 * 0.5 + 1e-4,
        1.0,
        1e-4,
    ]
    assert stock.tolist() == pytest.approx(expected, abs=1e-7)


def test_dead_bushes_do_not_regenerate():
    cfg = make_cfg()
    state = set_stock(state_of(cfg), [0.3, 0.3], alive=[False, True])
    out, _ = environment.apply(state, cfg)
    stock = np.asarray(out.entities.bush.edible.stock)
    assert stock[0] == pytest.approx(0.3, abs=1e-7)
    assert stock[1] == pytest.approx(0.3 + 0.03 * 0.3 * 0.7 + 1e-4, abs=1e-7)


def test_carcass_rot_turns_plant_and_animal_matter_into_toxin():
    cfg = make_cfg()
    state = set_carcass(state_of(cfg), 0, [0.5, 0.4, 0.1], age=0)
    out, _ = environment.apply(state, cfg)
    carcass = out.entities.carcass
    assert np.asarray(carcass.edible.n[0]).tolist() == pytest.approx(
        [0.4975, 0.398, 0.1 + 0.005 * 0.9], abs=1e-7
    )
    assert int(carcass.edible.age_dead[0]) == 1
    assert bool(carcass.alive[0])


def test_carcass_is_removed_past_600_ticks():
    cfg = make_cfg()
    state = set_carcass(state_of(cfg), 0, [0.5, 0.0, 0.0], age=599)
    state = set_carcass(state, 1, [0.5, 0.0, 0.0], age=600)
    out, _ = environment.apply(state, cfg)
    alive = np.asarray(out.entities.carcass.alive)
    assert bool(alive[0]) is True
    assert bool(alive[1]) is False
    assert int(out.entities.carcass.edible.age_dead[0]) == 600


def test_carcass_is_removed_when_empty():
    cfg = make_cfg()
    state = set_carcass(state_of(cfg), 0, [0.0, 0.0, 0.0], age=5)
    out, _ = environment.apply(state, cfg)
    assert bool(out.entities.carcass.alive[0]) is False


def test_toxin_schedule_switches_every_period():
    cfg = make_cfg(l_ticks=10.0, c_switch=1.0)
    state = at_tick(state_of(cfg), 0)
    out, buffer = environment.apply(state, cfg)
    assert int(buffer.count) == 0
    assert np.allclose(np.asarray(out.comp), BASE, atol=1e-7)

    out, buffer = environment.apply(at_tick(out, 9), cfg)
    assert int(buffer.count) == 0
    assert np.allclose(np.asarray(out.comp), BASE, atol=1e-7)

    switched, buffer = environment.apply(at_tick(out, 10), cfg)
    assert int(buffer.count) == 1
    summary = events.summary(buffer)
    assert summary == {"toxin_switch": 1}
    chosen = int(buffer.uid[0])
    assert 0 <= chosen < BASE.shape[0]
    assert float(buffer.x[0]) == pytest.approx(float(chosen))
    assert int(buffer.tick[0]) == 10
    comp = np.asarray(switched.comp)
    toxic = 0.6 * BASE + 0.4 * np.asarray([0.0, 0.0, 1.0])
    for species in range(BASE.shape[0]):
        expected = toxic[species] if species == chosen else BASE[species]
        assert np.allclose(comp[species], expected, atol=1e-7)


def test_toxin_schedule_uses_the_c_switch_factor():
    cfg = make_cfg(l_ticks=10.0, c_switch=0.5)
    period = 5
    state = at_tick(state_of(cfg), period - 1)
    _, before = environment.apply(state, cfg)
    assert int(before.count) == 0
    _, on = environment.apply(at_tick(state, period), cfg)
    assert int(on.count) == 1


def test_toxin_schedule_is_deterministic():
    cfg = make_cfg()
    state = at_tick(state_of(cfg), 10)
    first, first_buffer = environment.apply(state, cfg)
    second, second_buffer = environment.apply(state, cfg)
    assert np.array_equal(np.asarray(first.comp), np.asarray(second.comp))
    assert int(first_buffer.uid[0]) == int(second_buffer.uid[0])


def test_neutral_objects_walk_at_the_scripted_speed():
    cfg = make_cfg()
    state = put_neutral(state_of(cfg), 0, [8.0, 8.0], heading=0.0, timer=50)
    out, _ = environment.apply(state, cfg)
    neutral = out.entities.neutral
    assert float(neutral.transform.pos[0, 0]) == pytest.approx(8.0 + environment.NEUTRAL_SPEED)
    assert float(neutral.transform.pos[0, 1]) == pytest.approx(8.0, abs=1e-6)
    assert int(neutral.scripted.timer[0]) == 49


def test_neutral_objects_bounce_off_the_walls():
    cfg = make_cfg()
    state = put_neutral(state_of(cfg), 0, [0.45, 8.0], heading=np.pi, timer=50)
    out, _ = environment.apply(state, cfg)
    neutral = out.entities.neutral
    assert float(neutral.transform.pos[0, 0]) > 0.45
    assert float(neutral.transform.pos[0, 0]) >= 0.4 - 1e-6
    # Heading was reflected from -x to +x.
    assert float(np.cos(float(neutral.scripted.heading[0]))) > 0.0


def test_held_neutral_objects_do_not_walk():
    cfg = make_cfg()
    state = put_creature(state_of(cfg), 0, [8.0, 8.0])
    state = put_neutral(state, 0, [8.5, 8.0], heading=0.0, timer=50, held=True)
    out, _ = environment.apply(state, cfg)
    assert float(out.entities.neutral.transform.pos[0, 0]) == pytest.approx(8.5)
    assert float(out.entities.neutral.transform.pos[0, 1]) == pytest.approx(8.0)


def test_environment_is_deterministic():
    cfg = make_cfg()
    state = set_stock(state_of(cfg), [0.3, 0.4, 0.5, 0.6])
    state = set_carcass(state, 0, [0.5, 0.4, 0.1], age=3)
    state = put_neutral(state, 0, [8.0, 8.0], heading=1.0, timer=1)
    state = at_tick(state, 10)
    first, first_buffer = environment.apply(state, cfg)
    second, second_buffer = environment.apply(state, cfg)
    assert np.array_equal(
        np.asarray(first.entities.bush.edible.stock),
        np.asarray(second.entities.bush.edible.stock),
    )
    assert np.array_equal(
        np.asarray(first.entities.neutral.transform.pos),
        np.asarray(second.entities.neutral.transform.pos),
    )
    assert int(first_buffer.count) == int(second_buffer.count)


def test_carcass_rot_records_an_internal_transfer():
    cfg = make_cfg()
    state = set_carcass(state_of(cfg), 0, [0.5, 0.4, 0.1], age=0)
    state = set_carcass(state, 1, [0.2, 0.0, 0.3], age=0)
    assert np.allclose(np.asarray(state.ledger.transfer_from), 0.0)
    out, _ = environment.apply(state, cfg)
    ledger = out.ledger

    expected_from = np.asarray([0.005 * (0.5 + 0.2), 0.005 * 0.4, 0.0])
    expected_to = np.asarray([0.0, 0.0, 0.005 * (0.5 + 0.2 + 0.4)])
    assert np.asarray(ledger.transfer_from).tolist() == pytest.approx(
        expected_from.tolist(), abs=1e-7
    )
    assert np.asarray(ledger.transfer_to).tolist() == pytest.approx(
        expected_to.tolist(), abs=1e-7
    )
    # An internal conversion conserves mass and is not an inflow/outflow (§3.5).
    assert float(np.asarray(ledger.transfer_from).sum()) == pytest.approx(
        float(np.asarray(ledger.transfer_to).sum()), abs=1e-7
    )
    assert np.allclose(np.asarray(ledger.inflow), 0.0, atol=1e-7)
    assert np.allclose(np.asarray(ledger.outflow), 0.0, atol=1e-7)


def test_carcass_removal_books_its_remaining_n_as_decomposition():
    cfg = make_cfg()
    state = set_carcass(state_of(cfg), 0, [0.5, 0.4, 0.1], age=600)  # removed this tick
    state = set_carcass(state, 1, [1.0, 0.0, 0.0], age=0)  # stays alive
    out, _ = environment.apply(state, cfg)

    assert bool(out.entities.carcass.alive[0]) is False
    assert bool(out.entities.carcass.alive[1]) is True
    retained = np.asarray(out.entities.carcass.edible.n[0])
    decomposition = np.asarray(out.ledger.outflow[C.OUTFLOW_DECOMPOSITION])
    assert decomposition.tolist() == pytest.approx(retained.tolist(), abs=1e-7)
    assert float(decomposition.sum()) == pytest.approx(1.0, abs=1e-6)
    # The dead slot keeps its n: the ledger owns the flow, the slot keeps the data.
    assert retained.tolist() == pytest.approx([0.4975, 0.398, 0.1045], abs=1e-7)
    # No other outflow category is touched, and the same tick's rot transfer is
    # still recorded for both living carcasses (the leaving one included).
    other = np.delete(np.asarray(out.ledger.outflow), C.OUTFLOW_DECOMPOSITION, axis=0)
    assert np.allclose(other, 0.0, atol=1e-7)
    assert float(np.asarray(out.ledger.transfer_from).sum()) == pytest.approx(0.0095, abs=1e-7)


def test_empty_carcass_removal_books_near_zero():
    cfg = make_cfg()
    state = set_carcass(state_of(cfg), 0, [0.0, 0.0, 0.0], age=5)
    out, _ = environment.apply(state, cfg)
    assert bool(out.entities.carcass.alive[0]) is False
    assert np.allclose(
        np.asarray(out.ledger.outflow[C.OUTFLOW_DECOMPOSITION]), 0.0, atol=1e-7
    )


def test_already_dead_carcass_books_nothing():
    cfg = make_cfg()
    state = set_carcass(state_of(cfg), 0, [0.5, 0.0, 0.0], age=700, alive=False)
    out, _ = environment.apply(state, cfg)
    assert np.allclose(np.asarray(out.ledger.transfer_from), 0.0, atol=1e-7)
    assert np.allclose(np.asarray(out.ledger.outflow), 0.0, atol=1e-7)


def test_bush_colour_mixes_green_with_the_species_berry_colour():
    cfg = make_cfg()
    state = set_stock(state_of(cfg), [0.0, 0.5, 1.0], kinds=[0, 1, 2])
    arena = environment.bush_appearance(state.entities.bush)
    rgb = np.asarray(arena.look.rgb)
    green = np.asarray(BUSH_GREEN, np.float64)
    berry = np.asarray(BERRY_RGB, np.float64)

    assert rgb[0].tolist() == pytest.approx(green.tolist(), abs=1e-7)  # stock 0
    assert rgb[1].tolist() == pytest.approx(
        (0.5 * green + 0.5 * berry[1]).tolist(), abs=1e-7
    )
    assert rgb[2].tolist() == pytest.approx(berry[2].tolist(), abs=1e-7)  # stock 1


def test_dead_bush_keeps_its_colour():
    cfg = make_cfg()
    state = set_stock(state_of(cfg), [0.5, 0.5], kinds=[0, 0], alive=[False, True])
    arena = state.entities.bush
    rgb = np.asarray(arena.look.rgb).copy()
    rgb[0] = [0.1, 0.2, 0.3]
    arena = update(arena, look=update(arena.look, rgb=jnp.asarray(rgb, jnp.float32)))
    state = update(state, entities=update(state.entities, bush=arena))

    out = environment.bush_appearance(state.entities.bush)
    colours = np.asarray(out.look.rgb)
    assert colours[0].tolist() == pytest.approx([0.1, 0.2, 0.3], abs=1e-7)
    expected_live = 0.5 * np.asarray(BUSH_GREEN, np.float64) + 0.5 * np.asarray(BERRY_RGB[0])
    assert colours[1].tolist() == pytest.approx(expected_live.tolist(), abs=1e-7)


def test_environment_updates_bush_colour_from_the_new_stock():
    cfg = make_cfg()
    state = set_stock(state_of(cfg), [0.0, 0.5, 1.0], kinds=[0, 1, 2])
    out, _ = environment.apply(state, cfg)
    stock = np.asarray(out.entities.bush.edible.stock, np.float64)
    rgb = np.asarray(out.entities.bush.look.rgb, np.float64)
    green = np.asarray(BUSH_GREEN, np.float64)
    berry = np.asarray(BERRY_RGB, np.float64)
    for slot, kind in enumerate((0, 1, 2)):
        expected = green * (1.0 - stock[slot]) + berry[kind] * stock[slot]
        assert rgb[slot].tolist() == pytest.approx(expected.tolist(), abs=1e-6)


def test_apply_traces_under_jit():
    cfg = make_cfg()
    state = put_neutral(state_of(cfg), 0, [8.0, 8.0], heading=0.5, timer=3)
    state = set_carcass(state, 0, [0.5, 0.4, 0.1], age=3)
    state = set_stock(state, [0.5, 0.6], kinds=[0, 1])
    state = at_tick(state, 10)
    compiled = jax.jit(environment.apply, static_argnums=(1,))
    out, buffer = compiled(state, cfg)
    out, buffer = compiled(out, cfg)
    assert out.entities.neutral.transform.pos.shape == (2, 2)
    assert out.ledger.transfer_from.shape == (C.N_NUTRIENTS,)
    assert out.ledger.outflow.shape == (C.N_OUTFLOW_CATEGORIES, C.N_NUTRIENTS)
    assert int(buffer.count) == 1


def test_compiled_ledger_matches_eager():
    cfg = make_cfg()
    state = set_carcass(state_of(cfg), 0, [0.5, 0.4, 0.1], age=600)
    state = set_carcass(state, 1, [0.25, 0.25, 0.1], age=5)
    eager, _ = environment.apply(state, cfg)
    compiled, _ = jax.jit(environment.apply, static_argnums=(1,))(state, cfg)
    for name in ("inflow", "outflow", "transfer_from", "transfer_to", "digested"):
        assert np.array_equal(
            np.asarray(getattr(eager.ledger, name)),
            np.asarray(getattr(compiled.ledger, name)),
        ), name

def _ramp_increment(stock: float) -> float:
    """Berry regeneration increment before the ramp multiplier (§4.2, §9)."""
    return 0.03 * stock * (1.0 - stock) + 1e-4


def test_ramp_boosts_berry_regeneration():
    plain = make_cfg()
    ramped = make_cfg(ramp_ticks=10)  # ramp_berry_multiplier defaults to 3.0
    stocks = [0.3, 0.5, 1.0]

    plain_out, _ = environment.apply(set_stock(state_of(plain), stocks), plain)
    ramp_out, _ = environment.apply(set_stock(state_of(ramped), stocks), ramped)
    middle_out, _ = environment.apply(
        at_tick(set_stock(state_of(ramped), stocks), 5), ramped
    )
    late_out, _ = environment.apply(
        at_tick(set_stock(state_of(ramped), stocks), 10), ramped
    )

    plain_stock = np.asarray(plain_out.entities.bush.edible.stock)
    ramp_stock = np.asarray(ramp_out.entities.bush.edible.stock)
    middle_stock = np.asarray(middle_out.entities.bush.edible.stock)
    late_stock = np.asarray(late_out.entities.bush.edible.stock)
    for slot, start in enumerate(stocks):
        increment = _ramp_increment(start)
        assert plain_stock[slot] == pytest.approx(min(1.0, start + increment), abs=1e-7)
        # tick 0: multiplier 3; tick 5: 3 + (1 - 3) * 0.5 = 2; tick 10: 1.
        assert ramp_stock[slot] == pytest.approx(min(1.0, start + 3.0 * increment), abs=1e-7)
        assert middle_stock[slot] == pytest.approx(min(1.0, start + 2.0 * increment), abs=1e-7)
        assert late_stock[slot] == pytest.approx(min(1.0, start + increment), abs=1e-7)


def test_ramp_blocks_the_toxin_switch_and_keeps_berries_nontoxic():
    # Period 4 is shorter than the 10-tick ramp, so a switch would otherwise
    # fire at ticks 4 and 8.
    cfg = make_cfg(ramp_ticks=10, l_ticks=4.0, c_switch=1.0)
    state = state_of(cfg)
    poisoned = np.asarray(INITIAL_COMP).copy()
    poisoned[0] = [0.51, 0.09, 0.40]
    state = update(state, comp=jnp.asarray(poisoned, jnp.float32))

    for tick in (0, 4, 8):
        out, buffer = environment.apply(at_tick(state, tick), cfg)
        assert int(buffer.count) == 0, tick
        assert np.allclose(np.asarray(out.comp), BASE, atol=1e-7), tick  # no toxin

    # The ramp is over at tick 10; the next period multiple switches normally.
    out, buffer = environment.apply(at_tick(state, 12), cfg)
    assert events.summary(buffer) == {"toxin_switch": 1}
    chosen = int(buffer.uid[0])
    toxic = 0.6 * BASE + 0.4 * np.asarray([0.0, 0.0, 1.0])
    comp = np.asarray(out.comp)
    for species in range(BASE.shape[0]):
        expected = toxic[species] if species == chosen else BASE[species]
        assert np.allclose(comp[species], expected, atol=1e-7), species


def test_ramped_apply_traces_under_jit():
    cfg = make_cfg(ramp_ticks=10, l_ticks=4.0, c_switch=1.0)
    state = set_stock(state_of(cfg), [0.3, 0.5], kinds=[0, 1])
    state = set_carcass(state, 0, [0.5, 0.4, 0.1], age=3)
    state = at_tick(state, 4)  # inside the ramp, on a period multiple

    eager, eager_buffer = environment.apply(state, cfg)
    compiled = jax.jit(environment.apply, static_argnums=(1,))
    out, buffer = compiled(state, cfg)
    assert int(eager_buffer.count) == 0
    assert int(buffer.count) == 0
    assert np.allclose(np.asarray(out.comp), BASE, atol=1e-7)
    assert np.allclose(
        np.asarray(out.entities.bush.edible.stock),
        np.asarray(eager.entities.bush.edible.stock),
        atol=1e-6,
    )
    for name in ("transfer_from", "transfer_to", "outflow"):
        assert np.allclose(
            np.asarray(getattr(out.ledger, name)),
            np.asarray(getattr(eager.ledger, name)),
            atol=1e-9,
        ), name
    out, buffer = compiled(out, cfg)  # still traceable on the next tick
    assert out.entities.bush.edible.stock.shape == (cfg.capacities.bush,)

