"""Runtime M1 scan parameters: one static compile for config x world batches.

SANDBOX.md §9: the calibration knobs are runtime arrays, so a single compiled
segment runs a batch of (configuration, world) rows.  Structural configuration
(geometry, brain, sensor slices, event capacity, arena capacities) stays static;
the bush arena is compiled at the maximum scanned size and `bush_count` masks
the alive slots.

Acceptance here: 2 configurations x 2 worlds in one vmap segment are bitwise
identical to running each row on its own with the same static configuration
("same scope"), and the static segment factory is built once.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from life_sandbox.config import Capacities, Config, RunConfig, WorldConfig
from life_sandbox.core import components as C
from life_sandbox.core.hashing import state_hash
from life_sandbox.core.struct import update
from life_sandbox.core.params import ScanParams
from life_sandbox.core.step import compile_stats, reset_compile_stats, run, run_batch
from life_sandbox.core.systems import metabolism
from life_sandbox.world.init import initial_state, diet_from_genome

TICKS = 8

# Static structure for the batch: the bush arena is compiled at the maximum
# scanned size (6) while the rows use 3 or 6 alive bushes.
STATIC = Config.default().replaced(
    capacities=Capacities(creature=16, bush=6, carcass=4, neutral=1),
    world=WorldConfig(
        width=16.0,
        height=16.0,
        founders=4,
        event_capacity=32,
    ),
    run=RunConfig(seed=0, ticks=TICKS, segment_ticks=TICKS),
)


def _rows() -> list[ScanParams]:
    """2 configurations x 2 worlds; worlds differ by the init seed only."""
    rows = []
    for config in range(2):
        for _world in range(2):
            mapping = (
                {
                    "base_metabolism": 0.0012,
                    "bush_regen_rate": 0.02,
                    "bush_regen_base": 5e-5,
                    "temperature_scale": 0.3,
                    "ramp_ticks": 6,
                    "founder_reserves": 0.4,
                    "bush_capacity": 3,
                }
                if config == 0
                else {
                    "base_metabolism": 0.004,
                    "bush_regen_rate": 0.05,
                    "bush_regen_base": 2e-4,
                    "temperature_scale": 1.2,
                    "ramp_ticks": 0,
                    "founder_reserves": 0.6,
                    "bush_capacity": 6,
                }
            )
            rows.append(ScanParams.from_mapping(mapping, STATIC))
    return rows


def _stack(values):
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *values)


def _single(index: int):
    return jax.tree.map(lambda leaf: leaf[index], _state_batch_and_params()[0])


def _state_batch_and_params():
    rows = _rows()
    states = [
        initial_state(STATIC, seed=100 + index, params=row)
        for index, row in enumerate(rows)
    ]
    return _stack(states), ScanParams.batched(rows)


def test_initial_state_honours_runtime_bush_count_and_reserves():
    rows = _rows()
    small = initial_state(STATIC, seed=7, params=rows[0])  # config 0: 3 bushes
    large = initial_state(STATIC, seed=7, params=rows[2])  # config 1: 6 bushes
    assert int(jnp.sum(small.entities.bush.alive)) == 3
    assert int(jnp.sum(large.entities.bush.alive)) == 6
    # Runtime founder reserves feed the founder bodies, not the static cfg.
    small_n = jnp.sum(small.entities.creature.body.N[small.entities.creature.alive])
    large_n = jnp.sum(large.entities.creature.body.N[large.entities.creature.alive])
    assert float(small_n) < float(large_n)


def test_batched_config_x_world_matches_single_runs_bitwise():
    states, params = _state_batch_and_params()
    reset_compile_stats(clear_cache=True)

    combined = run_batch(states, STATIC, TICKS, params=params)
    assert compile_stats()["batch_segment"] == 1

    for index in range(4):
        row_state = jax.tree.map(lambda leaf: leaf[index : index + 1], states)
        row_params = ScanParams.broadcast(ScanParams.row(params, index), 1)
        single = run_batch(row_state, STATIC, TICKS, params=row_params)
        assert state_hash(jax.tree.map(lambda leaf: leaf[0], single.states)) == state_hash(
            jax.tree.map(lambda leaf: leaf[index], combined.states)
        )
        assert np.array_equal(single.counts[0], combined.counts[index])
        assert np.array_equal(single.inflow[0], combined.inflow[index])
        assert np.array_equal(single.outflow[0], combined.outflow[index])
        assert np.array_equal(single.transfer_from[0], combined.transfer_from[index])
        assert np.array_equal(single.transfer_to[0], combined.transfer_to[index])
        assert np.array_equal(single.digested[0], combined.digested[index])

    # The four separate calls reused the one compiled segment created for the
    # combined batch (the factory was not invoked again).
    assert compile_stats()["batch_segment"] == 1

    # Different runtime rows really did take different trajectories.
    hashes = {
        state_hash(jax.tree.map(lambda leaf: leaf[i], combined.states)) for i in range(4)
    }
    assert len(hashes) == 4


def test_runtime_params_drive_metabolism_and_ramp():
    mapping = {
        "base_metabolism": 0.0005,
        "bush_regen_rate": 0.03,
        "bush_regen_base": 1e-4,
        "temperature_scale": 1.0,
        "ramp_ticks": 0,
        "founder_reserves": 0.5,
        "bush_capacity": 6,
    }
    row_low = ScanParams.from_mapping(mapping, STATIC)
    row_high = ScanParams.from_mapping({**mapping, "base_metabolism": 0.02}, STATIC)
    state = initial_state(STATIC, seed=11, params=row_low)
    low = run(state, STATIC, TICKS, params=row_low)
    high = run(state, STATIC, TICKS, params=row_high)
    assert state_hash(low.state) != state_hash(high.state)
    # 40x higher base metabolism drains reserves faster over the same segment.
    low_n = float(jnp.sum(low.state.entities.creature.body.N[low.state.entities.creature.alive]))
    high_n = float(jnp.sum(high.state.entities.creature.body.N[high.state.entities.creature.alive]))
    assert high_n < low_n


def test_run_batch_accepts_a_single_world():
    """run_batch promotes one unbatched state to a batch of one (no crash)."""
    state = initial_state(STATIC, seed=3, params=ScanParams.from_config(STATIC))
    result = run_batch(state, STATIC, 4)
    assert result.states.tick.shape == (1,)
    single = run(state, STATIC, 4)
    assert state_hash(jax.tree.map(lambda leaf: leaf[0], result.states)) == state_hash(
        single.state
    )
    assert np.array_equal(result.inflow[0], single.inflow)
    assert np.array_equal(result.outflow[0], single.outflow)


def test_runtime_ramp_scales_whole_energy_but_not_repair():
    """Runtime ramp_ticks scales (base + action costs) * multiplier; repair stays."""
    mapping = {
        "base_metabolism": 0.002,
        "bush_regen_rate": 0.03,
        "bush_regen_base": 1e-4,
        "temperature_scale": 1.0,
        "ramp_ticks": 10,
        "founder_reserves": 0.5,
        "bush_capacity": 6,
    }
    ramped = ScanParams.from_mapping(mapping, STATIC)
    unramped = ScanParams.from_mapping({**mapping, "ramp_ticks": 0}, STATIC)
    state = initial_state(STATIC, seed=5, params=unramped)
    cre = state.entities.creature
    diet = diet_from_genome(cre.genome.g[:1], STATIC)[0]
    reserves = 0.9 * diet  # below each N_cap but above the repair threshold
    cre = update(
        cre,
        body=update(
            cre.body,
            I=cre.body.I.at[0].set(0.5),  # inside (0, 1): repair is paid
            N=cre.body.N.at[0].set(reserves),
        ),
        motor=update(
            cre.motor,
            last_out=cre.motor.last_out.at[0, C.OUT_BITE].set(1.0),
            u_eff=cre.motor.u_eff.at[0].set(jnp.asarray([1.0, 1.0], jnp.float32)),
        ),
    )
    state = update(state, entities=update(state.entities, creature=cre))

    def demand(params):
        out, _ = metabolism.apply(state, STATIC, params)
        drop = state.entities.creature.body.N[0] - out.entities.creature.body.N[0]
        return drop[0] / diet[0]  # plant component: reserves cover the whole demand

    base = 0.002
    energy = base * (1.0 + 1.0 + 0.25 + 1.0)  # base + acc + turn + bite at m=1
    repair = metabolism.REPAIR_RATE
    off = demand(unramped)
    on = demand(ramped)
    assert jnp.allclose(off, energy + repair, atol=1e-6)
    assert jnp.allclose(
        on, energy * STATIC.world.ramp_metabolism_multiplier + repair, atol=1e-6
    )
    # The ramp covers the whole energy scale; the repair charge is unscaled.
    assert jnp.allclose(
        off - on, energy * (1.0 - STATIC.world.ramp_metabolism_multiplier), atol=1e-6
    )


def test_single_run_params_and_none_agree_with_config_defaults():
    state = initial_state(STATIC)
    default = run(state, STATIC, 3)
    explicit = run(state, STATIC, 3, params=ScanParams.from_config(STATIC))
    assert state_hash(default.state) == state_hash(explicit.state)
    assert default.counts == explicit.counts
    assert np.array_equal(default.inflow, explicit.inflow)
    assert np.array_equal(default.outflow, explicit.outflow)
