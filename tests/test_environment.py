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
from life_sandbox.core import events, ramp, rng
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


def regen_increment(cfg: Config, stock: float) -> float:
    """One-tick berry regeneration increment from the configuration (§4.2)."""
    return cfg.world.bush_regen_rate * stock * (1.0 - stock) + cfg.world.bush_regen_base


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
        0.3 + regen_increment(cfg, 0.3),
        0.5 + regen_increment(cfg, 0.5),
        1.0,
        cfg.world.bush_regen_base,
    ]
    assert stock.tolist() == pytest.approx(expected, abs=1e-7)


def test_dead_bushes_do_not_regenerate():
    cfg = make_cfg()
    state = set_stock(state_of(cfg), [0.3, 0.3], alive=[False, True])
    out, _ = environment.apply(state, cfg)
    stock = np.asarray(out.entities.bush.edible.stock)
    assert stock[0] == pytest.approx(0.3, abs=1e-7)
    assert stock[1] == pytest.approx(0.3 + regen_increment(cfg, 0.3), abs=1e-7)


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
    cfg = make_cfg(l_ticks=10.0, c_switch=1.0)  # period 10
    state = at_tick(state_of(cfg), 0)

    # Tick 0 is a period multiple: the first tick already picks a toxic species.
    out, buffer = environment.apply(state, cfg)
    assert int(buffer.count) == 1
    assert events.summary(buffer) == {"toxin_switch": 1}
    chosen = int(buffer.uid[0])
    assert 0 <= chosen < BASE.shape[0]
    assert float(buffer.x[0]) == pytest.approx(float(chosen))
    assert int(buffer.tick[0]) == 0
    comp = np.asarray(out.comp)
    toxic = 0.6 * BASE + 0.4 * np.asarray([0.0, 0.0, 1.0])  # ramp_ticks = 0 -> alpha = 1
    for species in range(BASE.shape[0]):
        expected = toxic[species] if species == chosen else BASE[species]
        assert np.allclose(comp[species], expected, atol=1e-7)

    # Non-multiple ticks keep the composition and emit no event.
    kept, buffer = environment.apply(at_tick(out, 9), cfg)
    assert int(buffer.count) == 0
    assert np.array_equal(np.asarray(kept.comp), comp)

    # The next period multiple switches again at the full strength.
    switched, buffer = environment.apply(at_tick(out, 10), cfg)
    assert events.summary(buffer) == {"toxin_switch": 1}
    assert int(buffer.tick[0]) == 10
    comp = np.asarray(switched.comp)
    assert np.allclose(comp[:, 2].max(), 0.4, atol=1e-7)


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

def _ramp_increment(cfg: Config, stock: float) -> float:
    """Berry regeneration increment before the ramp multiplier (§4.2, §9)."""
    return regen_increment(cfg, stock)


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
        increment = _ramp_increment(ramped, start)
        assert plain_stock[slot] == pytest.approx(min(1.0, start + increment), abs=1e-7)
        # tick 0: multiplier 3; tick 5: 3 + (1 - 3) * 0.5 = 2; tick 10: 1.
        assert ramp_stock[slot] == pytest.approx(min(1.0, start + 3.0 * increment), abs=1e-7)
        assert middle_stock[slot] == pytest.approx(min(1.0, start + 2.0 * increment), abs=1e-7)
        assert late_stock[slot] == pytest.approx(min(1.0, start + increment), abs=1e-7)


def test_toxin_strength_ramps_from_ten_percent():
    plain = make_cfg()  # ramp_ticks = 0 -> full strength everywhere
    ramped = make_cfg(ramp_ticks=10)
    for tick in (0, 5, 10, 500):
        assert float(ramp.toxin_strength(tick, plain)) == pytest.approx(1.0, abs=1e-6)
    # alpha = 0.1 + 0.9 * fraction (§9 rule fix: never zero)
    for tick, expected in ((0, 0.1), (5, 0.55), (10, 1.0), (20, 1.0)):
        assert float(ramp.toxin_strength(tick, ramped)) == pytest.approx(expected, abs=1e-6)


def _toxic_species(comp: np.ndarray) -> int:
    rows = np.nonzero(comp[:, 2] > 0.0)[0]
    assert len(rows) == 1, comp
    return int(rows[0])


def test_toxin_share_follows_the_ramp_strength():
    # Period 5: switches at ticks 0, 5 and 10, i.e. at ramp fractions 0, 0.5, 1.
    cfg = make_cfg(ramp_ticks=10, l_ticks=5.0, c_switch=1.0)
    state = state_of(cfg)
    for tick, expected_share in ((0, 0.04), (5, 0.22), (10, 0.4)):
        out, buffer = environment.apply(at_tick(state, tick), cfg)
        assert events.summary(buffer) == {"toxin_switch": 1}, tick
        comp = np.asarray(out.comp)
        species = _toxic_species(comp)
        assert comp[species, 2] == pytest.approx(expected_share, abs=1e-7), tick
        expected = (1.0 - expected_share) * BASE[species] + expected_share * np.asarray(
            [0.0, 0.0, 1.0]
        )
        assert comp[species].tolist() == pytest.approx(expected.tolist(), abs=1e-7)
        for other in range(BASE.shape[0]):
            if other != species:
                assert np.allclose(comp[other], BASE[other], atol=1e-7), (tick, other)


def test_non_switch_ticks_keep_the_current_composition():
    cfg = make_cfg(ramp_ticks=10, l_ticks=5.0, c_switch=1.0)
    switched, buffer = environment.apply(at_tick(state_of(cfg), 5), cfg)
    assert events.summary(buffer) == {"toxin_switch": 1}
    kept, buffer = environment.apply(at_tick(switched, 7), cfg)
    assert int(buffer.count) == 0
    assert np.array_equal(np.asarray(kept.comp), np.asarray(switched.comp))
    # tick 10 is a multiple again and refreshes the composition at alpha = 1
    refreshed, buffer = environment.apply(at_tick(kept, 10), cfg)
    assert events.summary(buffer) == {"toxin_switch": 1}
    assert float(np.asarray(refreshed.comp)[:, 2].max()) == pytest.approx(0.4, abs=1e-7)


def test_ramp_zero_keeps_the_full_toxin_share():
    cfg = make_cfg(ramp_ticks=0, l_ticks=5.0, c_switch=1.0)
    for tick in (0, 5, 10):
        assert float(ramp.toxin_strength(tick, cfg)) == pytest.approx(1.0, abs=1e-6)
        out, buffer = environment.apply(at_tick(state_of(cfg), tick), cfg)
        assert events.summary(buffer) == {"toxin_switch": 1}
        comp = np.asarray(out.comp)
        species = _toxic_species(comp)
        old_style = 0.6 * BASE[species] + 0.4 * np.asarray([0.0, 0.0, 1.0])
        assert comp[species].tolist() == pytest.approx(old_style.tolist(), abs=1e-7)


def test_toxin_availability_is_reduced_across_the_ramp():
    # Period 1: the composition is refreshed every tick, so the toxic share can
    # be read as a time series.  The ramp weakens the toxin nutrient instead of
    # removing it (no chronic toxin deficiency, 2026-11-06 review problem 7).
    cfg = make_cfg(ramp_ticks=10, l_ticks=1.0, c_switch=1.0)
    shares = []
    for tick in range(10):
        out, _ = environment.apply(at_tick(state_of(cfg), tick), cfg)
        comp = np.asarray(out.comp)
        shares.append(float(comp[_toxic_species(comp), 2]))
    expected = [0.4 * (0.1 + 0.9 * (tick / 10.0)) for tick in range(10)]
    assert shares == pytest.approx(expected, abs=1e-7)
    # Mean share over the ramp is 0.4 * mean(alpha) = 0.4 * 0.505, about half
    # of the full-strength share; it is never zero.
    assert float(np.mean(shares)) == pytest.approx(0.4 * 0.505, abs=1e-6)
    assert 0.0 < float(np.mean(shares)) < 0.6 * 0.4

    # The full share is back once the ramp is over.
    out, _ = environment.apply(at_tick(state_of(cfg), 10), cfg)
    comp = np.asarray(out.comp)
    assert float(comp[_toxic_species(comp), 2]) == pytest.approx(0.4, abs=1e-7)


def test_ramped_apply_traces_under_jit():
    cfg = make_cfg(
        ramp_ticks=10,
        l_ticks=4.0,
        c_switch=1.0,
        bush_regen_rate=0.06,  # non-default calibration values
        bush_regen_base=2e-4,
    )
    state = set_stock(state_of(cfg), [0.3, 0.5], kinds=[0, 1])
    state = set_carcass(state, 0, [0.5, 0.4, 0.1], age=3)
    state = at_tick(state, 4)  # inside the ramp, on a period multiple

    eager, eager_buffer = environment.apply(state, cfg)
    compiled = jax.jit(environment.apply, static_argnums=(1,))
    out, buffer = compiled(state, cfg)
    assert int(eager_buffer.count) == 1
    assert int(buffer.count) == 1
    assert events.summary(buffer) == {"toxin_switch": 1}
    # Tick 4 of a 10-tick ramp: alpha = 0.1 + 0.9 * 0.4 = 0.46, so the toxic
    # species carries 0.4 * 0.46 = 0.184 of the toxin nutrient.
    expected_share = 0.4 * (0.1 + 0.9 * 0.4)
    comp = np.asarray(out.comp)
    assert float(comp[_toxic_species(comp), 2]) == pytest.approx(expected_share, abs=1e-6)
    assert np.allclose(np.asarray(out.comp), np.asarray(eager.comp), atol=1e-9)
    assert np.allclose(
        np.asarray(out.entities.bush.edible.stock),
        np.asarray(eager.entities.bush.edible.stock),
        atol=1e-6,
    )
    # The jitted path uses the configuration coefficients and the ramp: tick 4
    # of a 10-tick ramp has berry multiplier 3 + (1 - 3) * 0.4 = 2.2.
    multiplier = 3.0 + (1.0 - 3.0) * 0.4
    stock = np.asarray(out.entities.bush.edible.stock)
    for slot, start in enumerate((0.3, 0.5)):
        expected = min(1.0, start + multiplier * regen_increment(cfg, start))
        assert stock[slot] == pytest.approx(expected, abs=1e-6)
    for name in ("transfer_from", "transfer_to", "outflow"):
        assert np.allclose(
            np.asarray(getattr(out.ledger, name)),
            np.asarray(getattr(eager.ledger, name)),
            atol=1e-9,
        ), name
    out, buffer = compiled(out, cfg)  # still traceable on the next tick
    assert out.entities.bush.edible.stock.shape == (cfg.capacities.bush,)


def test_berry_regeneration_reads_the_config_coefficients():
    base_cfg = make_cfg()
    fast_cfg = make_cfg(bush_regen_rate=2.0 * base_cfg.world.bush_regen_rate)
    rich_cfg = make_cfg(bush_regen_base=2.0 * base_cfg.world.bush_regen_base)
    stocks = [0.3, 0.5]

    def grown(cfg):
        out, _ = environment.apply(set_stock(state_of(cfg), stocks), cfg)
        return np.asarray(out.entities.bush.edible.stock)

    base_stock = grown(base_cfg)
    fast_stock = grown(fast_cfg)
    rich_stock = grown(rich_cfg)
    rate = base_cfg.world.bush_regen_rate
    seed = base_cfg.world.bush_regen_base
    for slot, start in enumerate(stocks):
        rate_term = rate * start * (1.0 - start)
        assert base_stock[slot] == pytest.approx(start + rate_term + seed, abs=1e-7)
        # Doubling the rate only doubles the logistic term, not the seed term.
        assert fast_stock[slot] - start == pytest.approx(2.0 * rate_term + seed, abs=1e-7)
        assert rich_stock[slot] - start == pytest.approx(rate_term + 2.0 * seed, abs=1e-7)


def test_berry_regeneration_still_clips_after_the_config_scaling():
    cfg = make_cfg(bush_regen_rate=0.5, bush_regen_base=1e-2)
    state = set_stock(state_of(cfg), [0.99], kinds=[0])
    out, _ = environment.apply(state, cfg)
    assert float(out.entities.bush.edible.stock[0]) == pytest.approx(1.0, abs=1e-7)

